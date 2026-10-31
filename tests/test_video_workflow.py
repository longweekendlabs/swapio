import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import QThread, QPoint
from PySide6.QtWidgets import QApplication, QTabWidget
import app
import video_core as video

QT_APP=QApplication.instance() or QApplication([])

class UnifiedMediaTests(unittest.TestCase):
    def setUp(self):
        for target,value in [('app.read_state',{}),('app.write_state',None),('app.core.missing_models',[]),('app.DurationWorker.start',None)]:
            mocked=patch(target,return_value=value);mocked.start();self.addCleanup(mocked.stop)
        self.window=app.MainWindow();self.addCleanup(self.window.close)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def add(self,*names):
        paths=[self.root/name for name in names]
        for p in paths:p.touch()
        self.window._add_targets(paths)
        return paths

    def test_controls_stay_fixed_when_resized_without_tabs_or_overlap(self):
        w=self.window;w.show();heights=[]
        self.assertEqual(w.findChildren(QTabWidget),[])
        for height in (670,760,1300):
            w.resize(1180,height)
            for _ in range(3):QT_APP.processEvents()
            heights.append((w.source_button.height(),w.quality_mode.height(),w.preview_button.height()))
            self.assertLessEqual(w.source_view.mapTo(w,QPoint(0,w.source_view.height())).y(),w.source_name.mapTo(w,QPoint(0,0)).y())
            self.assertLessEqual(w.preview_view.mapTo(w,QPoint(0,w.preview_view.height())).y(),w.preview_caption.mapTo(w,QPoint(0,0)).y())
            self.assertLess(w.progress.mapTo(w,QPoint(0,w.progress.height())).y(),w.height())
        self.assertTrue(all(h==heights[0] for h in heights))

    def test_include_videos_locks_careful_only_when_clips_present_and_restores_photo_choice(self):
        w=self.window;w.quality_mode.setCurrentIndex(w.quality_mode.findData('best'))
        w.include_videos.setChecked(True)
        self.assertTrue(w.quality_mode.isEnabled());self.assertEqual(w._quality(),'best')
        photo,clip=self.add('photo.png','clip.mp4')
        self.assertFalse(w.quality_mode.isEnabled());self.assertEqual(w._quality(),'careful')
        w._save_state();self.assertEqual(w._state['quality'],'best')
        w.include_videos.setChecked(False)
        self.assertTrue(w.quality_mode.isEnabled());self.assertEqual(w._quality(),'best')
        self.assertEqual(w._active_targets(),[photo])
        self.assertIn('excluded',w.target_count.text())
        w.include_videos.setChecked(True);w._clear_targets()
        self.assertEqual(w._quality(),'best');self.assertTrue(w.quality_mode.isEnabled())

    def test_ranges_are_per_video_and_controls_ignore_photos(self):
        w=self.window;photo,first,second=self.add('photo.png','one.mp4','two.mp4')
        w.include_videos.setChecked(True);w.target_list.setCurrentRow(1)
        self.assertTrue(w.trim_row.isEnabled())
        w.video_start.setValue(2);w.video_end.setValue(5)
        w.target_list.setCurrentRow(2);self.assertEqual(w.video_start.value(),0)
        w.target_list.setCurrentRow(1);self.assertEqual((w.video_start.value(),w.video_end.value()),(2,5))
        w.video_selections[str(first)]=object();w.video_preview_path=first
        w.video_start.setValue(3)
        self.assertNotIn(str(first),w.video_selections);self.assertIsNone(w.video_preview_path)
        w.target_list.setCurrentRow(0);self.assertFalse(w.trim_row.isEnabled())

    def test_video_only_queue_shows_mp4_and_restores_photo_format_when_excluded(self):
        w=self.window;w.output_format.setCurrentIndex(w.output_format.findData('png'))
        self.add('clip.mp4');w.include_videos.setChecked(True)
        self.assertEqual(w.output_format.currentData(),'mp4');self.assertFalse(w.output_format.isEnabled())
        self.assertEqual(w._output_format(),'png')
        w.include_videos.setChecked(False)
        self.assertEqual(w.output_format.currentData(),'png');self.assertTrue(w.output_format.isEnabled())

    def test_folder_import_finds_both_media_types(self):
        photo=self.root/'photo.jpg';photo.touch();nested=self.root/'nested';nested.mkdir();clip=nested/'clip.mp4';clip.touch()
        animation=nested/'animation.GIF';animation.touch()
        with patch('app.native_directory',return_value=str(self.root)):
            self.window._add_folder()
        self.assertEqual(set(self.window.targets),{photo,clip,animation})

    def test_gifs_use_include_toggle_careful_ranges_and_matching_output_format(self):
        w=self.window;w.quality_mode.setCurrentIndex(w.quality_mode.findData('best'))
        (animation,)=self.add('animation.gif')
        self.assertEqual(w._active_targets(),[])
        self.assertIn('GIF',w.target_count.text())
        w.include_videos.setChecked(True)
        self.assertEqual(w._active_targets(),[animation]);self.assertEqual(w._quality(),'careful')
        self.assertFalse(w.quality_mode.isEnabled());self.assertTrue(w.trim_row.isEnabled())
        self.assertEqual(w.output_format.currentData(),'gif');self.assertFalse(w.output_format.isEnabled())
        self.add('movie.mp4')
        self.assertEqual(w.output_format.currentData(),'motion')
        w.include_videos.setChecked(False)
        self.assertEqual(w._quality(),'best');self.assertEqual(w.output_format.currentData(),'jpg')

    def test_duration_warning_distinguishes_one_and_three_minutes(self):
        a,b,c=self.add('short.mp4','long.mp4','very-long.mp4')
        def info(path,seconds):return video.VideoInfo(path,96,96,Fraction(25),seconds,False)
        with patch('app.QMessageBox.warning') as warning:
            self.window._durations_ready({str(a):info(a,30),str(b):info(b,80),str(c):info(c,200)})
        text=warning.call_args.args[2]
        self.assertNotIn('short.mp4',text);self.assertIn('over 1 minute',text);self.assertIn('over 3 minutes',text)

    def test_preview_time_popup_can_cancel_without_loading_models(self):
        (clip,)=self.add('clip.mp4');w=self.window;w.include_videos.setChecked(True);w.source_path=self.root/'source.png'
        with patch('app.QMessageBox.information',return_value=app.QMessageBox.Cancel) as popup:
            w._preview()
        self.assertIn('2-second',popup.call_args.args[1]);self.assertIn('few minutes',popup.call_args.args[2])
        self.assertIsNone(w.worker)

    def test_device_badge_names_actual_cuda_cpu_fallback_and_idle(self):
        w=self.window
        with patch('app.SettingsDialog._gpu_description',return_value=(True,'available')):
            w._refresh_model_status();self.assertIn('CUDA enabled',w.model_status.text())
            w._busy=True;w.engine.provider='CUDA';w._refresh_model_status()
            self.assertIn('CUDA · Processing',w.model_status.text())
            w.engine.provider='CPU';w._refresh_model_status();self.assertIn('CPU fallback',w.model_status.text())
            self.assertNotIn('Offline',w.model_status.text())
            w._busy=False;w.engine.provider='Not loaded'

    @patch('app.shutil.which',side_effect=lambda name:'/usr/bin/kdialog' if name=='kdialog' else None)
    @patch('app.subprocess.run')
    def test_native_media_picker_includes_photos_and_videos(self,run,_which):
        run.return_value=SimpleNamespace(returncode=0,stdout='/a.jpg\n/b.mp4\n',stderr='')
        self.assertEqual(app.native_media_files(None,'Media','/tmp'),['/a.jpg','/b.mp4'])
        args=run.call_args.args[0]
        self.assertIn('*.mp4',args[-1]);self.assertIn('*.jpg',args[-1]);self.assertIn('--multiple',args)
        self.assertIn('*.gif',args[-1])

class MediaWorkerTests(unittest.TestCase):
    def options(self):
        return dict(video_ranges={},video_selections={'/clip.mp4':SimpleNamespace(valid=lambda *_:True)},
                    source_path=Path('/source.png'),targets=[Path('/clip.mp4')],output_dir=Path('/out'),
                    all_faces=False,quality='careful',output_format='jpg',character_name='',
                    preserve_mouth=False,destination_eyes=False,restoration_strength=0.,skip_completed=True)

    def run_worker(self,worker):
        worker.start()
        while worker.isRunning():QT_APP.processEvents();worker.wait(10)
        QT_APP.processEvents()

    def test_worker_releases_before_notifying_for_success_failure_and_cancel(self):
        for error in (None,RuntimeError('render failed'),app.core.ProcessingCancelled()):
            state={'released':False}
            photo=SimpleNamespace(models=Path('/models'),use_gpu=True,det_size=640,provider='Not loaded',release_models=lambda:None)
            def release():
                self.assertIsNot(QThread.currentThread(),QT_APP.thread());state['released']=True
            movie=SimpleNamespace(provider='CUDA',release_models=release)
            worker=app.MediaWorker(photo,'video_preview',**self.options());results=[]
            def ready(result):self.assertTrue(state['released']);results.append(result)
            worker.batch_ready.connect(ready)
            with patch('app.video.VideoSwapEngine',return_value=movie),patch('app.video.render_video',side_effect=error,return_value={'output':'/out/preview.mp4','provider':'CUDA'}):
                self.run_worker(worker)
            self.assertEqual(len(results),1)
            self.assertEqual(results[0]['stopped'],isinstance(error,app.core.ProcessingCancelled))
            self.assertEqual(results[0]['failed'],int(isinstance(error,RuntimeError) and not isinstance(error,app.core.ProcessingCancelled)))

    def test_photo_video_photo_queue_never_keeps_both_model_sets_loaded(self):
        state={'photo':False,'video':False};order=[]
        def release_photo():state['photo']=False
        def release_video():state['video']=False
        def photo_batch(**kwargs):
            self.assertFalse(state['video']);state['photo']=True;order.append('photo')
            return dict(completed=1,skipped=0,failed=0,outputs=['/out/photo.jpg'],errors=[],provider='CUDA',stopped=False)
        def render(*args,**kwargs):
            self.assertFalse(state['photo']);state['video']=True;order.append('video')
            return {'output':'/out/video.mp4','provider':'CUDA'}
        photo=SimpleNamespace(models=Path('/models'),use_gpu=True,det_size=640,provider='CUDA',release_models=release_photo,batch=photo_batch)
        movie=SimpleNamespace(provider='CUDA',release_models=release_video)
        options=self.options();options['targets']=[Path('/one.jpg'),Path('/clip.mp4'),Path('/two.png')]
        worker=app.MediaWorker(photo,'batch',**options);results=[];worker.batch_ready.connect(results.append)
        with patch('app.video.VideoSwapEngine',return_value=movie),patch('app.video.render_video',side_effect=render):
            self.run_worker(worker)
        self.assertEqual(order,['photo','video','photo']);self.assertEqual(results[0]['completed'],3)
        self.assertFalse(state['photo']);self.assertFalse(state['video'])

    def test_gif_uses_animated_renderer_and_shared_engine_in_mixed_queue(self):
        options=self.options();options['targets']=[Path('/clip.gif'),Path('/clip.mp4')]
        options['video_selections']['/clip.gif']=SimpleNamespace(valid=lambda *_:True)
        photo=SimpleNamespace(models=Path('/models'),use_gpu=True,det_size=640,provider='Not loaded',release_models=lambda:None)
        movie=SimpleNamespace(provider='CUDA',release_models=lambda:None)
        worker=app.MediaWorker(photo,'batch',**options);results=[];worker.batch_ready.connect(results.append)
        with patch('app.video.VideoSwapEngine',return_value=movie) as engine, \
             patch('app.gif.render_gif',return_value={'output':'/out/clip.gif','provider':'CUDA'}) as gif, \
             patch('app.video.render_video',return_value={'output':'/out/clip.mp4','provider':'CUDA'}) as video:
            self.run_worker(worker)
        self.assertEqual(engine.call_count,1);self.assertEqual(gif.call_count,1);self.assertEqual(video.call_count,1)
        self.assertEqual(results[0]['completed'],2);self.assertEqual(results[0]['failed'],0)

if __name__=='__main__':unittest.main()
