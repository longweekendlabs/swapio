import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PySide6.QtCore import QThread, QPoint
from PySide6.QtWidgets import QApplication
import app
from video_tab import VideoWorker

QT_APP=QApplication.instance() or QApplication([])


class VideoWorkflowTests(unittest.TestCase):
    @patch('app.read_state',return_value={})
    @patch('app.write_state')
    def test_default_window_has_nonoverlapping_video_controls_and_visible_preview(self,*_):
        window=app.MainWindow();window.resize(1180,820);window.show();window.tabs.setCurrentIndex(1)
        for _ in range(3): QT_APP.processEvents()
        tab=window.video_tab
        self.assertLessEqual(tab.view.mapTo(tab,QPoint(0,tab.view.height())).y(),tab.caption.mapTo(tab,QPoint(0,0)).y())
        viewport=window.video_scroll.viewport()
        self.assertGreaterEqual(tab.preview_button.mapTo(viewport,QPoint(0,0)).y(),0)
        self.assertLessEqual(tab.preview_button.mapTo(viewport,QPoint(0,tab.preview_button.height())).y(),viewport.height())
        window.close()

    @patch('app.read_state',return_value={})
    @patch('app.write_state')
    @patch('app.core.missing_models',return_value=[])
    def test_video_tab_keeps_photo_quality_and_disables_photo_only_options(self,*_):
        window=app.MainWindow()
        window.quality_mode.setCurrentIndex(window.quality_mode.findData('best'))
        window.tabs.setCurrentIndex(1)
        self.assertFalse(window.quality_mode.isEnabled())
        self.assertFalse(window.ask_face.isEnabled())
        self.assertFalse(window.output_format.isEnabled())
        self.assertEqual(window._quality(),'best')
        window.tabs.setCurrentIndex(0)
        self.assertTrue(window.quality_mode.isEnabled())
        self.assertEqual(window._quality(),'best')
        window.close()

    @patch('app.read_state',return_value={})
    @patch('app.write_state')
    @patch('app.core.missing_models',return_value=[])
    def test_ranges_belong_to_clip_and_invalidate_selection_and_preview(self,*_):
        window=app.MainWindow();tab=window.video_tab
        with tempfile.TemporaryDirectory() as temp:
            paths=[Path(temp)/f'{i}.mp4' for i in range(2)]
            for p in paths:p.touch()
            tab.add_paths(paths)
            tab.start.setValue(2);tab.end.setValue(5)
            tab.files.setCurrentRow(1)
            self.assertEqual(tab.start.value(),0)
            tab.files.setCurrentRow(0)
            self.assertEqual((tab.start.value(),tab.end.value()),(2,5))
            tab.selections[str(paths[0])]=object();tab.preview_path=paths[0]
            tab.start.setValue(3)
            self.assertNotIn(str(paths[0]),tab.selections)
            self.assertIsNone(tab.preview_path)
        window.close()

    def test_worker_releases_on_worker_thread_before_every_outcome(self):
        for error in (None,RuntimeError('render failed'),app.core.ProcessingCancelled()):
            with self.subTest(error=error):
                state={'released':False}
                def release():
                    self.assertIsNot(QThread.currentThread(),QT_APP.thread())
                    state['released']=True
                engine=SimpleNamespace(release_models=release)
                selection=SimpleNamespace(valid=lambda *_:True)
                worker=VideoWorker(engine,'preview',[(Path('/test.mp4'),0.,0.)],Path('/source.png'),
                                   Path('/out'),{'/test.mp4':selection},{})
                outcomes=[]
                def receive(result):
                    self.assertTrue(state['released']);outcomes.append(result)
                worker.ready.connect(receive)
                value={'output':'/out/preview.mp4'}
                with patch('video_tab.video.render_video',side_effect=error,return_value=value):
                    worker.start()
                    while worker.isRunning():QT_APP.processEvents();worker.wait(10)
                    QT_APP.processEvents()
                self.assertEqual(len(outcomes),1)
                self.assertEqual(outcomes[0]['stopped'],isinstance(error,app.core.ProcessingCancelled))
                self.assertEqual(len(outcomes[0]['errors']),int(isinstance(error,RuntimeError) and not isinstance(error,app.core.ProcessingCancelled)))

    @patch('app.read_state',return_value={})
    @patch('app.write_state')
    @patch('app.core.missing_models',return_value=[])
    def test_picker_returns_explicit_choice_and_exception_always_unblocks_worker(self,*_):
        window=app.MainWindow();tab=window.video_tab;choices=[];stopped=[]
        tab.worker=SimpleNamespace(_stop=False,submit_face_choice=choices.append,stop=lambda:stopped.append(True))
        tab.face_dialog=lambda *a,**k:SimpleNamespace(exec=lambda:app.QDialog.Accepted,selected_index=1)
        request={'target':'/clip.mp4','image':None,'boxes':[]}
        tab._choose_face(request)
        self.assertEqual(choices,[1])
        def fail(*a,**k): raise RuntimeError('picker failed')
        tab.face_dialog=fail;tab._choose_face(request)
        self.assertEqual(stopped,[True])
        self.assertIsNone(window._face_dialog)
        window.close()

    @patch('app.shutil.which',side_effect=lambda name:'/usr/bin/kdialog' if name=='kdialog' else None)
    @patch('app.subprocess.run')
    def test_video_file_picker_uses_native_multiselect_filter(self,run,_which):
        run.return_value=SimpleNamespace(returncode=0,stdout='/a.mp4\n/b.mov\n',stderr='')
        self.assertEqual(app.native_video_files(None,'Videos','/tmp',True),['/a.mp4','/b.mov'])
        command=run.call_args.args[0]
        self.assertIn('--multiple',command)
        self.assertIn('*.mp4',command[-1])
        self.assertTrue(command[-1].endswith('|Videos'))


if __name__=='__main__':unittest.main()
