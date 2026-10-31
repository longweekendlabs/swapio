from pathlib import Path
from types import SimpleNamespace
import hashlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

import core
import gif_core as gif
import video_core as video
from test_video import face


class GifTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.path = self.root/'input.gif'
        self.make_gif()
        self.engine = SimpleNamespace(begin_segment=lambda: None, source_face=lambda *_: face(),
            ensure_loaded=lambda: None, provider='test', analyser=SimpleNamespace(get=lambda _: [face()]))
        def swap(image, *_, **kwargs):
            result = image.copy(); result[25:40,25:40] = (40,140,230)
            return result, 1, 1
        self.engine.swap_array = swap

    def make_gif(self, delays=(70,230,110,260), loop=2, disposal=2):
        frames = []
        for index in range(len(delays)):
            image = Image.new('RGBA', (97,95))
            ImageDraw.Draw(image).rectangle((10+index*7,10,70+index*7,75), fill=(180,160,120,255))
            frames.append(image)
        options = {} if loop is None else {'loop':loop}
        frames[0].save(self.path,save_all=True,append_images=frames[1:],duration=list(delays),
                       disposal=disposal, optimize=True, **options)
        stat = self.path.stat()
        self.selection = video.PersonSelection(str(self.path.resolve()),(stat.st_size,stat.st_mtime_ns),0.,
            np.array([1.,0.],np.float32),np.zeros((95,97,3),np.uint8),np.array([20,15,60,65]),0.)

    def render(self, **options):
        return gif.render_gif(self.engine,self.root/'source.png',self.path,self.root/'output',self.selection,**options)

    def test_animation_timing_loop_dimensions_transparency_and_original_preserved(self):
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        result = self.render()
        output = Path(result['output']); self.assertEqual(output.suffix,'.gif')
        info = gif.probe(output)
        self.assertEqual(info.delays,(70,230,110,260)); self.assertEqual(info.loop,2)
        self.assertEqual((info.width,info.height),(97,95)); self.assertEqual(result['swapped'],4)
        with Image.open(self.path) as original, Image.open(output) as swapped:
            for index in range(4):
                original.seek(index); swapped.seek(index)
                a,b = np.array(original.convert('RGBA')),np.array(swapped.convert('RGBA'))
                np.testing.assert_array_equal(a[:,:,3],b[:,:,3])
                self.assertTrue(np.any(a[:,:,:3]!=b[:,:,:3]))
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(),digest)

    def test_trim_and_two_second_preview_preserve_variable_delays(self):
        self.make_gif(delays=(700,1200,400,700)); self.selection.start=.25
        result = self.render(start=.25,end=2.75,preview=True)
        self.assertEqual(gif.probe(result['output']).delays,(450,1200,350))
        self.assertEqual(result['duration'],2.)
        self.assertEqual(result['frames'],3)

    def test_disposal_three_and_no_loop_extension_survive_compositing(self):
        self.make_gif(loop=None,disposal=3)
        output = self.render()['output']
        self.assertIsNone(gif.probe(output).loop)
        with Image.open(self.path) as original, Image.open(output) as swapped:
            for index in range(4):
                original.seek(index); swapped.seek(index)
                np.testing.assert_array_equal(np.array(original.convert('RGBA'))[:,:,3],
                                              np.array(swapped.convert('RGBA'))[:,:,3])

    def test_static_gif_and_default_delay(self):
        self.make_gif(delays=(0,),loop=None)
        self.assertEqual(gif.probe(self.path).delays,(100,))
        result = self.render(); self.assertEqual(result['frames'],1)
        self.assertEqual(gif.probe(result['output']).delays,(100,))

    def test_gif_needs_no_ffmpeg_and_can_repeat_infinite_loop(self):
        self.make_gif(loop=0)
        with patch('video_core.tools',side_effect=AssertionError('must not launch FFmpeg')):
            result = self.render()
        self.assertEqual(gif.probe(result['output']).loop,0)

    def test_cancel_and_render_error_remove_partial_output(self):
        for failure in (core.ProcessingCancelled(),RuntimeError('inference failed')):
            with patch.object(self.engine,'swap_array',side_effect=failure):
                with self.assertRaises(type(failure)):
                    self.render()
            self.assertEqual(list((self.root/'output').iterdir()),[])

    def test_unmatched_faces_never_count_as_saved(self):
        self.engine.analyser.get = lambda _: []
        with self.assertRaisesRegex(RuntimeError,'No confidently matched face'):
            self.render()
        self.assertEqual(list((self.root/'output').iterdir()),[])

    def test_invalid_start_and_sub_centisecond_range_fail_without_output(self):
        with self.assertRaisesRegex(RuntimeError,'start changed'):
            self.render(start=.1)
        with self.assertRaisesRegex(ValueError,'at least 0.01'):
            self.render(end=.001)
        self.assertFalse((self.root/'output').exists())

    def test_choose_person_requires_explicit_choice_for_multiple_faces(self):
        self.engine.analyser.get = lambda _: [face((0.,1.),(0,0,90,90)),face()]
        with self.assertRaises(core.ProcessingCancelled):
            gif.choose_person(self.engine,self.path)
        selected = gif.choose_person(self.engine,self.path,selector=lambda *_:1)
        np.testing.assert_allclose(selected.embedding,[1.,0.])
        self.assertTrue(selected.valid(self.path,0.))
        self.assertFalse(selected.valid(self.path,.1))

    def test_validation_failure_never_publishes_a_file(self):
        real_probe = gif.probe
        def probe(path,*args):
            info = real_probe(path,*args)
            if Path(path).name == 'finished.gif':
                from dataclasses import replace
                info = replace(info,loop=99)
            return info
        with patch('gif_core.probe',side_effect=probe):
            with self.assertRaisesRegex(RuntimeError,'validation'):
                self.render()
        self.assertEqual(list((self.root/'output').iterdir()),[])


if __name__ == '__main__':
    unittest.main()
