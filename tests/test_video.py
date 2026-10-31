from pathlib import Path
from types import SimpleNamespace
import hashlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import core
import video_core as video


def face(identity=(1.,0.), box=(20,15,60,65)):
    return SimpleNamespace(normed_embedding=np.asarray(identity,np.float32),bbox=np.asarray(box,np.float32),
                           kps=np.array([[30,30],[50,30],[40,40],[33,50],[47,50]],np.float32))


class TrackingTests(unittest.TestCase):
    def tracker(self):
        return video.PersonTracker(SimpleNamespace(embedding=np.array([1.,0.],np.float32)))

    def test_tracks_identity_when_detection_order_changes_and_other_face_is_larger(self):
        tracker=self.tracker();image=np.zeros((100,100,3),np.uint8)
        wanted=face();stranger=face((0.,1.),(0,0,90,95))
        for faces in ([stranger,wanted],[wanted,stranger],[stranger,wanted]):
            self.assertTrue(np.array_equal(tracker.select(image,faces).bbox,wanted.bbox))

    def test_no_fallback_on_loss_and_ambiguous_reentry(self):
        tracker=self.tracker();image=np.zeros((100,100,3),np.uint8)
        self.assertIsNotNone(tracker.select(image,[face()]))
        self.assertIsNone(tracker.select(image,[face((0.,1.))]))
        self.assertIsNone(tracker.select(image,[face(),face((.999,.01))]))
        self.assertIsNotNone(tracker.select(image,[face()]))

    def test_scene_change_requires_strong_identity_and_resets_alignment_history(self):
        tracker=self.tracker()
        tracker.select(np.zeros((100,100,3),np.uint8),[face()])
        result=tracker.select(np.full((100,100,3),255,np.uint8),[face((.5,.866))])
        self.assertTrue(tracker.cut)
        self.assertIsNone(result)
        self.assertIsNone(tracker.last_kps)

    def test_jitter_damping_preserves_current_relative_feature_geometry(self):
        tracker=self.tracker();image=np.zeros((100,100,3),np.uint8)
        tracker.select(image,[face()])
        current=face();current.kps+=np.array([.8,.3],np.float32)
        result=tracker.select(image,[current])
        np.testing.assert_allclose(result.kps[1:]-result.kps[0],current.kps[1:]-current.kps[0],atol=1e-5)

    def test_video_locks_model_until_reset_without_changing_photo_engine(self):
        engine=video.VideoSwapEngine();engine.begin_segment()
        finish=lambda image,_: image
        with patch.object(core.SwapEngine,'_identity_checked_face',return_value=('first',False)) as choose, \
             patch.object(engine,'_hyperswap_face',return_value='next') as render:
            engine._identity_checked_face(None,None,None,512,'careful',0,lambda _:None,finish)
            self.assertEqual(engine._identity_checked_face(None,None,None,1024,'careful',0,lambda _:None,finish),('next',False))
            self.assertEqual(render.call_args.args[3],512)
            self.assertEqual(choose.call_count,1)
            engine.begin_segment()
            engine._identity_checked_face(None,None,None,768,'careful',0,lambda _:None,finish)
            self.assertEqual(choose.call_count,2)


class VideoTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: video.tools()
        except RuntimeError as exc: raise unittest.SkipTest(str(exc))

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'input.mp4'
        ffmpeg,_=video.tools()
        video.run_tool([ffmpeg,'-v','error','-y','-f','lavfi','-i','color=c=gray:s=96x96:r=8:d=2',
                        '-f','lavfi','-i','sine=frequency=600:sample_rate=48000:duration=2',
                        '-af',"volume=0:enable='lt(t,1)'",'-c:v','libx264','-threads','1','-c:a','aac','-shortest',str(self.path)])
        self.digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
        stat=self.path.stat()
        self.selection=video.PersonSelection(str(self.path.resolve()),(stat.st_size,stat.st_mtime_ns),.5,
            np.array([1.,0.],np.float32),np.zeros((96,96,3),np.uint8),np.array([20,15,60,65]),.5)
        self.engine=SimpleNamespace(begin_segment=lambda:None,source_face=lambda *_:face(),provider='test',
            analyser=SimpleNamespace(get=lambda _: [face()]))
        def swap(image,*_,**kwargs):
            image=image.copy();image[25:50,25:50]=(30,100,220);return image,1,1
        self.engine.swap_array=swap

    def render(self,**kwargs):
        return video.render_video(self.engine,self.root/'source.png',self.path,self.root/'outputs',self.selection,
                                  start=.5,end=1.5,**kwargs)

    def test_real_decode_encode_trim_audio_and_original_preservation(self):
        result=self.render()
        info,stream=video.probe(Path(result['output']),True)
        self.assertEqual(result['swapped'],8)
        self.assertEqual(result['frames'],8)
        self.assertTrue(info.audio)
        self.assertAlmostEqual(info.duration,1.,places=2)
        self.assertEqual((info.width,info.height),(96,96))
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(),self.digest)
        self.assertFalse(list((self.root/'outputs').glob('.swapio-video-*')))
        ffmpeg,_=video.tools()
        audio=video.run_tool([ffmpeg,'-v','error','-i',result['output'],'-map','0:a:0','-f','f32le','-ac','1','-ar','48000','pipe:1'])
        samples=np.frombuffer(audio,np.float32)
        self.assertLess(float(np.sqrt(np.mean(samples[:19000]**2))),.002)
        self.assertGreater(float(np.sqrt(np.mean(samples[34000:43000]**2))),.03)

    def test_cancellation_removes_partial_movie_and_never_claims_save(self):
        state={'stop':False}
        def progress(*_): state['stop']=True
        with self.assertRaises(core.ProcessingCancelled):
            self.render(on_progress=progress,should_stop=lambda:state['stop'])
        self.assertEqual(list((self.root/'outputs').iterdir()),[])

    def test_total_tracking_loss_fails_instead_of_saving_unswapped_movie(self):
        self.engine.analyser.get=lambda _: [face((0.,1.))]
        with self.assertRaisesRegex(RuntimeError,'No confidently matched face'):
            self.render()
        self.assertEqual(list((self.root/'outputs').iterdir()),[])

    def test_silent_rotated_fractional_rate_video_preserves_display_geometry(self):
        ffmpeg,_=video.tools();landscape=self.root/'landscape.mp4'
        video.run_tool([ffmpeg,'-v','error','-y','-f','lavfi','-i','testsrc2=s=96x64:r=30000/1001:d=0.4',
                        '-vf','setsar=4/3','-c:v','libx264','-threads','1',str(landscape)])
        if 'display_rotation' in video.run_tool([ffmpeg,'-hide_banner','-h','full']).decode(errors='replace'):
            video.run_tool([ffmpeg,'-v','error','-y','-display_rotation','90','-i',str(landscape),'-c','copy',str(self.path)])
        else:
            video.run_tool([ffmpeg,'-v','error','-y','-i',str(landscape),'-c','copy',
                            '-metadata:s:v:0','rotate=90',str(self.path)])
        stat=self.path.stat();self.selection.signature=(stat.st_size,stat.st_mtime_ns);self.selection.start=0.
        result=video.render_video(self.engine,self.root/'source.png',self.path,self.root/'outputs',self.selection)
        info,stream=video.probe(Path(result['output']),True)
        self.assertFalse(info.audio)
        self.assertEqual((info.width,info.height),(64,96))
        self.assertEqual(str(info.rate),'30000/1001')
        self.assertEqual(str(info.sar),'3/4')
        self.assertEqual(int(stream['nb_read_frames']),12)

    def test_mux_failure_leaves_no_final_or_partial_movie(self):
        original=video.run_tool
        def fail_mux(command,**kwargs):
            if str(command[-1]).endswith('finished.mp4'):
                raise RuntimeError('audio mux failed')
            return original(command,**kwargs)
        with patch('video_core.run_tool',side_effect=fail_mux):
            with self.assertRaisesRegex(RuntimeError,'audio mux failed'):
                self.render()
        self.assertEqual(list((self.root/'outputs').iterdir()),[])

    def test_selection_invalidated_by_changed_start_and_missing_tool_message(self):
        self.assertFalse(self.selection.valid(self.path,0))
        with patch('video_core.shutil.which',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'FFmpeg and ffprobe'): video.tools()


if __name__=='__main__': unittest.main()
