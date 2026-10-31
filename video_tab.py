"""Video queue UI; photo controls and inference remain independently usable."""
from pathlib import Path
from threading import Event

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                              QListWidget, QDoubleSpinBox, QMessageBox, QLineEdit, QDialog, QLayout)

import core
import video_core as video


class VideoWorker(QThread):
    log = Signal(str)
    progress = Signal(int, int, dict)
    choice_requested = Signal(object)
    selected = Signal(str, object)
    ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, engine, mode, entries, source_path, output_dir, selections, options):
        super().__init__()
        self.engine, self.mode, self.entries = engine, mode, entries
        self.source_path, self.output_dir = source_path, output_dir
        self.selections, self.options = dict(selections), dict(options)
        self._stop = False
        self._event = Event()
        self._choice = None

    def stop(self):
        self._stop = True
        self._event.set()

    def submit_face_choice(self, choice):
        self._choice = choice
        self._event.set()

    def _select_face(self, path, image, faces):
        if len(faces) == 1:
            return 0
        self._choice = None
        self._event.clear()
        self.choice_requested.emit({"target":path,"image":image,
                                    "boxes":[np.asarray(f.bbox).tolist() for f in faces]})
        while not self._event.wait(.1):
            if self._stop:
                raise core.ProcessingCancelled()
        if self._stop:
            raise core.ProcessingCancelled()
        return self._choice

    def run(self):
        result = {"outputs":[],"errors":[],"stopped":False,"mode":self.mode,"results":[]}
        error = None
        try:
            for index,(path,start,end) in enumerate(self.entries):
                if self._stop:
                    result["stopped"] = True
                    break
                try:
                    selection = self.selections.get(str(path))
                    if selection is None or not selection.valid(path,start):
                        self.log.emit(f"Finding a face in {path.name}...")
                        selection = video.choose_person(self.engine,path,start,end,self._select_face,lambda:self._stop)
                        self.selections[str(path)] = selection
                        self.selected.emit(str(path),selection)
                    if self.mode == "select":
                        break
                    self.log.emit(f"{'Previewing' if self.mode=='preview' else 'Rendering'} {path.name} using Careful...")
                    def progress(done,total,counts):
                        self.progress.emit(done,total,{**counts,"clip":path.name,"clip_index":index+1,
                                                       "clip_total":len(self.entries)})
                    item = video.render_video(self.engine,self.source_path,path,self.output_dir,selection,
                        start=start,end=end,preview=self.mode=="preview",should_stop=lambda:self._stop,
                        on_log=self.log.emit,on_progress=progress,**self.options)
                    result["outputs"].append(item["output"])
                    result["results"].append(item)
                except core.ProcessingCancelled:
                    result["stopped"] = True
                    break
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    result["errors"].append({"path":str(path),"error":message})
                    self.log.emit(f"Failed {path.name}: {message}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                self.engine.release_models()
                self.log.emit("Video models unloaded; GPU memory released.")
            except Exception as exc:
                error = f"Could not release video models: {exc}"
        if error:
            self.failed.emit(error)
        else:
            self.ready.emit(result)


class VideoTab(QWidget):
    def __init__(self, window, *, choose_files, choose_directory, open_path,
                 notify_folder, face_dialog, image_view):
        super().__init__(window)
        self.window = window
        self.choose_files, self.choose_directory = choose_files, choose_directory
        self.open_path, self.notify_folder, self.face_dialog = open_path, notify_folder, face_dialog
        self.paths, self.ranges, self.selections = [], {}, {}
        self.worker = None
        self.preview_path = None
        self._loading = False
        self._row = -1
        self._dialog = None
        root = QVBoxLayout(self)
        root.setSizeConstraint(QLayout.SetMinimumSize)
        root.setContentsMargins(4,8,4,4)
        row = QHBoxLayout()
        self.source_label = QLabel("Choose a source portrait. It is shared with Photos.")
        self.source_label.setWordWrap(True)
        self.source_button = QPushButton("Choose source photo…")
        self.source_button.clicked.connect(window._choose_source)
        row.addWidget(self.source_label,1);row.addWidget(self.source_button)
        self.character_name = QLineEdit()
        self.character_name.setPlaceholderText("Character name (optional)")
        self.character_name.setMaxLength(80)
        self.character_name.textChanged.connect(window.character_name.setText)
        window.character_name.textChanged.connect(self.character_name.setText)
        row.insertWidget(1,self.character_name)
        root.addLayout(row)
        self.help = QLabel("Careful video · Choose one person per clip · Preview 3 seconds before a full render")
        self.help.setWordWrap(True)
        root.addWidget(self.help)
        splitter = QWidget()
        splitter.setFixedHeight(200)
        columns = QHBoxLayout(splitter)
        columns.setContentsMargins(0,0,0,0)
        columns.setSpacing(8)
        left = QWidget(); layout = QVBoxLayout(left);layout.setContentsMargins(0,0,4,0)
        self.files = QListWidget()
        self.files.setMinimumHeight(80)
        self.files.currentRowChanged.connect(self._selection_changed)
        layout.addWidget(self.files,1)
        row = QHBoxLayout()
        self.add_button = QPushButton("Add videos…");self.add_button.clicked.connect(self._add_files)
        self.folder_button = QPushButton("Add folder…");self.folder_button.clicked.connect(self._add_folder)
        self.remove_button = QPushButton("Remove");self.remove_button.clicked.connect(self._remove)
        for button in (self.add_button,self.folder_button,self.remove_button): row.addWidget(button)
        for button in (self.add_button,self.folder_button,self.remove_button): button.setMinimumHeight(30)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.start = QDoubleSpinBox();self.end = QDoubleSpinBox()
        for spin in (self.start,self.end):
            spin.setRange(0,864000);spin.setDecimals(2);spin.setSuffix(" s");spin.setSingleStep(.5)
            spin.valueChanged.connect(self._range_changed)
        self.end.setSpecialValueText("End of clip")
        row.addWidget(QLabel("Start"));row.addWidget(self.start);row.addWidget(QLabel("End"));row.addWidget(self.end)
        layout.addLayout(row)
        self.person_button = QPushButton("Choose / reset person…");self.person_button.clicked.connect(lambda:self._start_job("select"))
        self.person_button.setMinimumHeight(30)
        layout.addWidget(self.person_button)
        right = QWidget();layout = QVBoxLayout(right);layout.setContentsMargins(4,0,0,0)
        self.view = image_view("Choose a video, then choose its person or generate a preview")
        self.view.setMinimumHeight(140)
        layout.addWidget(self.view,1)
        self.caption = QLabel("No preview yet. Uncertain person matches remain unchanged and are counted.")
        self.caption.setWordWrap(True);layout.addWidget(self.caption)
        self.caption.setMinimumHeight(40)
        columns.addWidget(left,1);columns.addWidget(right,1)
        root.addWidget(splitter,1)
        row = QHBoxLayout()
        self.preview_button = QPushButton("Generate 3-second preview");self.preview_button.clicked.connect(lambda:self._start_job("preview"))
        self.play_button = QPushButton("Play preview");self.play_button.clicked.connect(self._play)
        self.run_button = QPushButton("Swap queued videos");self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(lambda:self._start_job("render"))
        self.stop_button = QPushButton("Stop");self.stop_button.clicked.connect(self.stop)
        for button in (self.preview_button,self.play_button,self.run_button,self.stop_button): row.addWidget(button)
        self.stop_button.hide()  # The shared Stop button stays visible below the tabs.
        root.insertLayout(2,row)
        self.device = QLabel("");self.device.setWordWrap(True);root.addWidget(self.device)
        self.set_busy(False)

    def _add_files(self):
        self.add_paths(self.choose_files(self,"Add videos",self.window._dialog_start("video_dir"),True))

    def _add_folder(self):
        folder = self.choose_directory(self,"Add a folder of videos",self.window._dialog_start("video_dir"))
        if folder:
            self.add_paths(sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in video.VIDEO_EXTENSIONS))

    def add_paths(self, paths):
        for item in paths:
            path = Path(item).expanduser().resolve()
            if path.is_file() and path.suffix.lower() in video.VIDEO_EXTENSIONS and path not in self.paths:
                self.paths.append(path);self.files.addItem(path.name)
                self.files.item(len(self.paths)-1).setToolTip(str(path))
                self.window._remember_folder("video_dir",path.parent)
        if self.paths and self.files.currentRow()<0: self.files.setCurrentRow(0)
        self.refresh()

    def _remove(self):
        row = self.files.currentRow()
        if row>=0:
            self._loading = True
            path = self.paths.pop(row);self.selections.pop(str(path),None);self.ranges.pop(str(path),None)
            self.files.takeItem(row);self._loading = False
            self._selection_changed(self.files.currentRow())

    def _selection_changed(self, row):
        if self._loading: return
        self._row = row
        self._loading = True
        values = self.ranges.get(str(self.paths[row]),(0.,0.)) if 0<=row<len(self.paths) else (0.,0.)
        self.start.setValue(values[0]);self.end.setValue(values[1]);self._loading=False
        self.preview_path = None
        selection = self.selections.get(str(self.paths[row])) if 0<=row<len(self.paths) else None
        if selection: self._show_person(selection)
        else:
            self.view.clear_image();self.caption.setText("Choose a person or generate a preview. Start and End apply to this clip.")
        self.refresh()

    def _range_changed(self):
        if self._loading or not 0<=self._row<len(self.paths): return
        key = str(self.paths[self._row])
        self.ranges[key]=(self.start.value(),self.end.value())
        self.selections.pop(key,None);self.preview_path=None
        self.caption.setText("Range changed. The person will be selected again.");self.refresh()

    def _show_person(self, selection):
        image = selection.image.copy()
        import cv2
        l,t,r,b = np.rint(selection.box).astype(int)
        cv2.rectangle(image,(l,t),(r,b),(92,122,255),max(2,round(image.shape[1]/400)))
        self.view.set_image(image)
        self.caption.setText(f"Selected person at {selection.timestamp:.2f} s. Tracking will not fall back to another face.")

    def _selected(self, path, selection):
        self.selections[path]=selection
        if 0<=self._row<len(self.paths) and str(self.paths[self._row])==path: self._show_person(selection)

    def _choose_face(self, request):
        if self.worker is None or self.worker._stop:
            return
        try:
            dialog = self.face_dialog(request['target'],request['image'],request['boxes'],batch=False,parent=self.window)
            self._dialog = self.window._face_dialog = dialog
            answer = dialog.exec()
            self.worker.submit_face_choice(dialog.selected_index if answer==QDialog.Accepted else None)
        except Exception as exc:
            self.worker.stop()
            self.window.log.appendPlainText(f"Video person picker stopped: {exc}")
        finally:
            self._dialog = self.window._face_dialog = None

    def _start_job(self, mode):
        if self.window.worker and self.window.worker.isRunning(): return
        row = self.files.currentRow()
        if row<0 or (mode!='select' and not self.window.source_path): return
        if mode=='select': self.selections.pop(str(self.paths[row]),None)
        selected = self.paths if mode=='render' else [self.paths[row]]
        entries = [(p,*self.ranges.get(str(p),(0.,0.))) for p in selected]
        output = Path(self.window.output_edit.text().strip() or str(self.paths[row].parent.with_name(self.paths[row].parent.name+' - Swapio Output'))).expanduser()
        if mode!='select':
            self.window.output_edit.setText(str(output));self.window.last_output=output
        if mode=='preview': output=output/'.swapio-previews'
        engine = video.VideoSwapEngine(models=self.window.engine.models,use_gpu=self.window.engine.use_gpu,det_size=self.window.engine.det_size)
        options = {'restoration_strength':self.window._restoration_strength(),
                   'destination_eyes':self.window.destination_eyes.isChecked(),
                   'preserve_mouth':self.window.preserve_mouth.isChecked(),
                   'character_name':self.window.character_name.text().strip()}
        self.worker = VideoWorker(engine,mode,entries,self.window.source_path,output,self.selections,options)
        self.window.worker = self.worker
        self.worker.log.connect(self.window.log.appendPlainText)
        self.worker.choice_requested.connect(self._choose_face)
        self.worker.selected.connect(self._selected)
        self.worker.progress.connect(self._progress)
        self.worker.ready.connect(self._ready)
        self.worker.failed.connect(self.window._failed)
        self.worker.finished.connect(self.window._refresh_controls)
        self.window.log.clear();self.window.progress.setRange(0,0)
        self.window.progress.setFormat("Finding selected person…")
        self.window._set_running(True);self.window.status_label.setText("Preparing video…")
        self.worker.start()

    def _progress(self, done, total, counts):
        self.window.progress.setRange(0,max(total,done));self.window.progress.setValue(done)
        self.window.progress.setFormat(f"Clip {counts['clip_index']}/{counts['clip_total']} · frame {done}/{total} · {counts['unchanged']} unchanged")
        self.window.status_label.setText(f"{counts['provider']} · {counts['elapsed']:.0f}s elapsed · ~{counts['eta']:.0f}s left")

    def _ready(self, result):
        self.window._set_running(False)
        self.window._refresh_model_status()
        if result['results']:
            item = result['results'][-1]
            self.view.set_image(item['image'])
            action = "Play preview to review motion and audio." if result['mode']=='preview' else "Saved. Open output folder to review the video."
            self.caption.setText(f"{item['frames']} frames: {item['swapped']} swapped, {item['unchanged']} unchanged.\n{action}")
            self.caption.setToolTip(item['output'])
            if result['mode']=='preview': self.preview_path=Path(item['output'])
            for output in result['outputs']: self.notify_folder(Path(output).parent)
            self.window.open_button.setEnabled(True)
        self.window.progress.setRange(0,1);self.window.progress.setValue(1)
        label = "Stopped" if result['stopped'] else "Finished"
        summary=f"{label}: {len(result['outputs'])} video(s) saved, {len(result['errors'])} failed"
        if result['mode']=='select' and not result['errors'] and not result['stopped']: summary="Person selected. Ready for preview."
        self.window.progress.setFormat(summary)
        self.window.status_label.setText("Stopped" if result['stopped'] else ("Video errors" if result['errors'] else "Video ready"))
        if result['errors']:
            errors='\n\n'.join(f"{Path(e['path']).name}\n{e['error']}" for e in result['errors'])
            QMessageBox.warning(self,'Video processing failed',summary+'\n\n'+errors)
        self.refresh()

    def _play(self):
        if self.preview_path and self.preview_path.is_file() and not self.open_path(self.preview_path):
            QMessageBox.warning(self,'Play preview',f"Could not open the system video player.\n{self.preview_path}")

    def stop(self):
        if self.worker:
            self.worker.stop()
            if self._dialog: self._dialog.reject()
            self.window.status_label.setText("Stopping after this frame…")

    def refresh(self):
        running=bool(self.window.worker and self.window.worker.isRunning())
        selected=bool(self.paths) and self.files.currentRow()>=0
        try:
            paths=video.tools();available=True
            self.device.setText(f"Video tools: {paths[0]} · {paths[1]}\nCareful uses the shared eye, mouth and restoration controls. Video output: MP4 with audio. Variable-rate inputs are exported at their average frame rate.")
        except RuntimeError as exc:
            available=False;self.device.setText(str(exc))
        if self.window.source_path: self.source_label.setText(f"Source: {self.window.source_path.name}")
        ready=not running and selected and available and self.window.permission.isChecked() and not core.missing_models()
        self.person_button.setEnabled(ready)
        self.preview_button.setEnabled(ready and self.window.source_path is not None)
        self.run_button.setEnabled(ready and self.window.source_path is not None)
        self.play_button.setEnabled(not running and self.preview_path is not None)
        self.stop_button.setEnabled(running and isinstance(self.window.worker,VideoWorker))

    def set_busy(self,running):
        for widget in (self.source_button,self.files,self.add_button,self.folder_button,self.remove_button,self.start,self.end):
            widget.setEnabled(not running)
        if hasattr(self.window,'permission'): self.refresh()
