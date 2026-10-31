#!/usr/bin/env python3
"""Swapio - local face swapping for photos, short videos and GIFs."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import core
import video_core as video
import gif_core as gif
from media_tools import native_process_environment

# Load the inference library before Qt installs its Python import feature hook.
# Legacy InsightFace imports augmentation/schema modules that otherwise spend
# minutes in Shiboken source inspection on the first worker-thread import.
# Models/GPU sessions remain lazy, and a missing backend is reported by Preview.
try:
    import insightface
except ImportError:
    pass

from PySide6.QtCore import Qt, QThread, QTimer, QUrl, QSize, Signal
from PySide6.QtGui import QAction, QDesktopServices, QFont, QIcon, QImage, QPixmap, QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QListView,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QDoubleSpinBox,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from about_dialog import AboutDialog
from model_setup_dialog import ModelSetupDialog
from settings_dialog import SettingsDialog
from version import APP_NAME, GITHUB_URL, VERSION, TAGLINE

MOTION_EXTENSIONS = video.VIDEO_EXTENSIONS | gif.GIF_EXTENSIONS
PROJECT_DIR = core.base_dir()
ICON_PATH = PROJECT_DIR / "assets" / "swapio.svg"

BG = "#14161b"
PANEL = "#1e222a"
INPUT = "#272c35"
BORDER = "#333b48"
ACCENT = "#ff7a5c"
ACCENT_H = "#ff9377"
ON_ACCENT = "#241009"
TEXT = "#f3f5f8"
LABEL = "#ccd3db"
MUTED = "#949da8"
GREEN = "#3ddc8f"
RED = "#ff6b6b"
IMAGE_FILTER = "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff|Images"
DIALOG_GEOMETRY = "1000x680"


def _native_result_paths(result: subprocess.CompletedProcess[str]) -> list[str] | None:
    """Return selections, an empty cancellation, or None for a crashed picker."""
    if result.returncode == 0:
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not result.stderr.strip():
        return []
    return None


def state_path() -> Path:
    config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config / "swapio" / "state.json"


def read_state() -> dict:
    try:
        return json.loads(state_path().read_text())
    except Exception:
        return {}


def write_state(data: dict) -> None:
    try:
        path = state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2))
        temp.replace(path)
    except Exception:
        pass


def native_image_files(parent, title: str, start: str, multiple: bool = False,
                       file_filter: str = IMAGE_FILTER) -> list[str]:
    """Open the desktop-native image picker, with a Qt fallback."""
    patterns, label = file_filter.split("|", 1)
    if shutil.which("kdialog"):
        command = ["kdialog", "--geometry", DIALOG_GEOMETRY, "--title", title]
        if multiple:
            command += ["--multiple", "--separate-output"]
        command += ["--getopenfilename", start, file_filter]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=native_process_environment(),
        )
        paths = _native_result_paths(result)
        if paths is not None:
            return paths
    if shutil.which("zenity"):
        command = [
            "zenity",
            "--file-selection",
            "--title",
            title,
            "--filename",
            str(Path(start).expanduser()).rstrip("/") + "/",
            "--file-filter",
            f"{label} | {patterns}",
        ]
        if multiple:
            command += ["--multiple", "--separator", "\n"]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=native_process_environment(),
        )
        paths = _native_result_paths(result)
        if paths is not None:
            return paths
    if multiple:
        paths, _ = QFileDialog.getOpenFileNames(
            parent, title, start, f"{label} ({patterns})"
        )
        return paths
    path, _ = QFileDialog.getOpenFileName(
        parent, title, start, f"{label} ({patterns})"
    )
    return [path] if path else []


def native_video_files(parent, title: str, start: str, multiple: bool = True) -> list[str]:
    from video_core import VIDEO_EXTENSIONS
    patterns = " ".join("*"+ext for ext in sorted(VIDEO_EXTENSIONS))
    return native_image_files(parent, title, start, multiple, patterns+"|Videos")


def native_media_files(parent, title: str, start: str, multiple: bool = True) -> list[str]:
    patterns = ' '.join('*'+ext for ext in sorted(core.IMAGE_EXTS | MOTION_EXTENSIONS))
    return native_image_files(parent,title,start,multiple,patterns+'|Photos and videos')


def native_directory(parent, title: str, start: str) -> str:
    """Open the desktop-native directory picker, with a Qt fallback."""
    if shutil.which("kdialog"):
        result = subprocess.run(
            [
                "kdialog",
                "--geometry",
                DIALOG_GEOMETRY,
                "--title",
                title,
                "--getexistingdirectory",
                start,
            ],
            capture_output=True,
            text=True,
            check=False,
            env=native_process_environment(),
        )
        paths = _native_result_paths(result)
        if paths is not None:
            return paths[0] if paths else ""
    if shutil.which("zenity"):
        result = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--directory",
                "--title",
                title,
                "--filename",
                str(Path(start).expanduser()).rstrip("/") + "/",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=native_process_environment(),
        )
        paths = _native_result_paths(result)
        if paths is not None:
            return paths[0] if paths else ""
    return QFileDialog.getExistingDirectory(parent, title, start)


def native_open_path(path: Path) -> bool:
    """Open a local path with the system desktop outside the bundled Qt runtime."""
    resolved = path.expanduser().resolve()
    candidates = (
        ("kioclient6", ["exec", resolved.as_uri()]),
        ("kioclient5", ["exec", resolved.as_uri()]),
        ("xdg-open", [str(resolved)]),
        ("gio", ["open", str(resolved)]),
    )
    for executable, arguments in candidates:
        command = shutil.which(executable)
        if not command:
            continue
        try:
            subprocess.Popen(
                [command, *arguments],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=native_process_environment(),
                start_new_session=True,
            )
            return True
        except OSError:
            continue
    return False


def notify_output_folder(path: Path) -> bool:
    """Tell KDE directory views about completed writes, including atomic renames."""
    if not sys.platform.startswith("linux"):
        return True
    try:
        from PySide6.QtDBus import QDBusConnection, QDBusMessage

        folder = path.expanduser().resolve()
        if not folder.is_dir():
            return False
        bus = QDBusConnection.sessionBus()
        if not bus.isConnected():
            return False
        message = QDBusMessage.createSignal("/", "org.kde.KDirNotify", "FilesAdded")
        message.setArguments([QUrl.fromLocalFile(str(folder)).toString()])
        return bus.send(message)
    except (ImportError, OSError, RuntimeError):
        return False


def pixmap_from_bgr(image, maximum: int = 900) -> QPixmap:
    height, width = image.shape[:2]
    scale = min(maximum / max(width, height), 1.0)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    qimage = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class ImageView(QLabel):
    def __init__(self, empty_text: str):
        super().__init__(empty_text)
        self._pixmap = None
        self._empty_text = empty_text
        self.setAlignment(Qt.AlignCenter)
        # The source card has more controls than the preview card. A tall fixed
        # minimum made both image canvases paint underneath those controls when
        # the options section grew. Keep the canvases flexible so every card
        # remains inside the shared workflow row.
        self.setMinimumSize(180, 70)
        self.setObjectName("imageView")

    def set_image(self, image) -> None:
        self._pixmap = pixmap_from_bgr(image)
        self._refresh()

    def clear_image(self, text: str | None = None) -> None:
        self._pixmap = None
        self.clear()
        self.setText(text or self._empty_text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap:
            margins = self.contentsMargins()
            self.setPixmap(
                self._pixmap.scaled(
                    max(1, self.width() - margins.left() - margins.right()),
                    max(1, self.height() - margins.top() - margins.bottom()),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )


class FaceSelectionDialog(QDialog):
    STOP_BATCH = 2

    def __init__(self, target, image, boxes, batch=False, parent=None):
        super().__init__(parent)
        self.selected_index = None
        self.setWindowTitle("Choose the face to swap")
        self.resize(780, 640)
        self.setMinimumSize(620, 440)
        layout = QVBoxLayout(self)
        heading = QLabel(Path(target).name)
        heading.setObjectName("aboutName")
        heading.setWordWrap(True)
        layout.addWidget(heading)
        instruction = QLabel("Select a numbered face below. Only that face will be swapped.")
        instruction.setWordWrap(True)
        layout.addWidget(instruction)
        height, width = image.shape[:2]
        scale = min(900 / max(width, height), 1)
        marked = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
        for index, box in enumerate(boxes):
            l, t, r, b = np.rint(np.asarray(box) * scale).astype(int)
            cv2.rectangle(marked, (l, t), (r, b), (92, 122, 255), 2)
            cv2.putText(marked, str(index + 1), (max(0, l), max(20, t - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (92, 122, 255), 2, cv2.LINE_AA)
        overview = ImageView("")
        overview.setMinimumHeight(180)
        overview.set_image(marked)
        layout.addWidget(overview, 1)
        self.faces = QListWidget()
        self.faces.setViewMode(QListView.IconMode)
        self.faces.setResizeMode(QListView.Adjust)
        self.faces.setIconSize(QSize(104, 104))
        self.faces.setGridSize(QSize(124, 136))
        self.faces.setSpacing(5)
        self.faces.setMinimumHeight(150)
        self.faces.setMaximumHeight(260)
        for index, box in enumerate(boxes):
            l, t, r, b = np.asarray(box)
            margin = max(r - l, b - t) * .15
            x0, y0 = max(0, int(l - margin)), max(0, int(t - margin))
            x1, y1 = min(width, int(r + margin)), min(height, int(b + margin))
            item = QListWidgetItem(f"Face {index + 1}")
            if x1 > x0 and y1 > y0:
                item.setIcon(QIcon(pixmap_from_bgr(image[y0:y1, x0:x1], 200)))
            self.faces.addItem(item)
        self.faces.setCurrentRow(-1)
        layout.addWidget(self.faces)
        buttons = QHBoxLayout()
        skip = QPushButton("Skip media" if batch else "Cancel")
        skip.clicked.connect(self.reject)
        buttons.addWidget(skip)
        if batch:
            stop = QPushButton("Stop batch")
            stop.clicked.connect(lambda: self.done(self.STOP_BATCH))
            buttons.addWidget(stop)
        buttons.addStretch()
        self.swap_button = QPushButton("Swap selected face")
        self.swap_button.setObjectName("primary")
        self.swap_button.setDefault(True)
        self.swap_button.setEnabled(False)
        self.swap_button.clicked.connect(self._accept_face)
        self.faces.currentRowChanged.connect(lambda row: self.swap_button.setEnabled(row >= 0))
        self.faces.itemDoubleClicked.connect(lambda _item: self._accept_face())
        buttons.addWidget(self.swap_button)
        layout.addLayout(buttons)
        # QListWidget otherwise selects its first row when it receives the
        # dialog's initial focus, effectively preselecting the largest face.
        skip.setFocus(Qt.OtherFocusReason)

    def _accept_face(self):
        row = self.faces.currentRow()
        if row >= 0:
            self.selected_index = row
            self.accept()


class TaskWorker(QThread):
    log = Signal(str)
    progress = Signal(int, int, dict)
    preview_ready = Signal(object)
    batch_ready = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()
    face_choice_requested = Signal(object)

    def __init__(self, engine: core.SwapEngine, mode: str, *, ask_face=False,
                 face_choices=None, **kwargs):
        super().__init__()
        self.engine = engine
        self.mode = mode
        self.kwargs = kwargs
        self._stop = False
        self.ask_face = ask_face
        self.face_choices = face_choices if face_choices is not None else {}
        self._choice_event = Event()
        self._choice = None

    def stop(self):
        self._stop = True
        self._choice_event.set()

    def submit_face_choice(self, choice):
        self._choice = choice
        self._choice_event.set()

    def _select_face(self, target, image, faces):
        if self._stop:
            raise core.ProcessingCancelled()
        key = core.face_selection_key(target, faces)
        if key in self.face_choices:
            return self.face_choices[key]
        self._choice = None
        self._choice_event.clear()
        self.face_choice_requested.emit({
            "target": str(target), "image": image,
            "boxes": [np.asarray(face.bbox).tolist() for face in faces],
            "worker": self,
        })
        while not self._choice_event.wait(.1):
            if self._stop:
                raise core.ProcessingCancelled()
        if self._stop:
            raise core.ProcessingCancelled()
        if self._choice is not None:
            self.face_choices[key] = self._choice
        return self._choice

    def run(self):
        outcome, payload = self.cancelled, ()
        try:
            if self.ask_face:
                self.kwargs["face_selector"] = self._select_face
            if self.mode == "preview":
                result = self.engine.preview(on_log=self.log.emit, **self.kwargs)
                if not self._stop:
                    outcome, payload = self.preview_ready, (result,)
            else:
                result = self.engine.batch(
                    on_log=self.log.emit,
                    on_progress=self.progress.emit,
                    should_stop=lambda: self._stop,
                    **self.kwargs,
                )
                outcome, payload = self.batch_ready, (result,)
        except (core.FaceSelectionSkipped, core.ProcessingCancelled):
            pass
        except Exception as exc:
            outcome, payload = self.failed, (f"{type(exc).__name__}: {exc}",)
        finally:
            self.last_provider = getattr(self.engine, 'provider', 'Not loaded')
            try:
                if self.engine.release_models():
                    self.log.emit("Processing models unloaded; memory released for other apps.")
            except Exception as exc:
                self.log.emit(f"Could not unload processing models: {type(exc).__name__}: {exc}")
        # Completion dialogs and enabled controls must follow device cleanup.
        # Payloads contain CPU pixels/metadata, so the preview remains available.
        outcome.emit(*payload)


class DurationWorker(QThread):
    ready = Signal(dict)

    def __init__(self, paths):
        super().__init__()
        self.paths = list(paths)

    def run(self):
        results = {}
        for path in self.paths:
            if self.isInterruptionRequested():
                return
            try:
                results[str(path)] = (gif.probe(path, self.isInterruptionRequested) if path.suffix.lower() in gif.GIF_EXTENSIONS
                    else video.probe(path, should_stop=self.isInterruptionRequested)[0])
            except core.ProcessingCancelled:
                return
            except Exception as exc:
                results[str(path)] = str(exc)
        self.ready.emit(results)


class MediaWorker(TaskWorker):
    """One queue and picker, with only one set of GPU models loaded at a time."""
    video_progress = Signal(int, int, dict)
    person_selected = Signal(str, object)

    def __init__(self, engine, mode, *, video_ranges, video_selections, **kwargs):
        super().__init__(engine, mode, **kwargs)
        self.video_ranges = dict(video_ranges)
        self.video_selections = dict(video_selections)
        self.video_engine = None
        self.active_engine = engine

    def _video_person(self, path, start, end):
        selection = self.video_selections.get(str(path))
        if selection is None or not selection.valid(path, start):
            def select(target, image, faces):
                if len(faces)>1:
                    self.face_choices.pop(core.face_selection_key(target,faces),None)
                choice = 0 if len(faces) == 1 else self._select_face(target, image, faces)
                if choice is None and not self._stop:
                    raise core.FaceSelectionSkipped()
                return choice
            transport = gif if path.suffix.lower() in gif.GIF_EXTENSIONS else video
            selection = transport.choose_person(self.video_engine, path, start, end, select, lambda: self._stop)
            self.video_selections[str(path)] = selection
            self.person_selected.emit(str(path), selection)
        return selection

    def run(self):
        outcome, payload = self.cancelled, ()
        result = dict(total=len(self.kwargs['targets']), completed=0, skipped=0, failed=0,
                      stopped=False, outputs=[], errors=[], provider='Not loaded')
        try:
            for index, path in enumerate(self.kwargs['targets']):
                if self._stop:
                    result['stopped'] = True
                    break
                try:
                    if path.suffix.lower() in MOTION_EXTENSIONS:
                        self.engine.release_models()
                        if self.video_engine is None:
                            self.video_engine = video.VideoSwapEngine(models=self.engine.models,
                                use_gpu=self.engine.use_gpu, det_size=self.engine.det_size)
                        self.active_engine = self.video_engine
                        start, end = self.video_ranges.get(str(path), (0., 0.))
                        selection = self._video_person(path, start, end)
                        def progress(done, total, counts):
                            self.video_progress.emit(done, total, {**counts, 'media_index':index+1,
                                'media_total':result['total'], 'name':path.name})
                        preview = self.mode == 'video_preview'
                        render = gif.render_gif if path.suffix.lower() in gif.GIF_EXTENSIONS else video.render_video
                        item = render(self.video_engine, self.kwargs['source_path'], path,
                            self.kwargs['output_dir'], selection, start=start, end=end, preview=preview,
                            character_name=self.kwargs['character_name'],
                            preserve_mouth=self.kwargs['preserve_mouth'],
                            destination_eyes=self.kwargs['destination_eyes'],
                            restoration_strength=self.kwargs['restoration_strength'],
                            should_stop=lambda:self._stop, on_log=self.log.emit, on_progress=progress)
                        result['outputs'].append(item['output'])
                        result['completed'] += 1
                        result['provider'] = item['provider']
                        if preview:
                            result['video_preview'] = item
                    else:
                        if self.video_engine is not None:
                            self.video_engine.release_models()
                        self.active_engine = self.engine
                        options = {k:v for k,v in self.kwargs.items() if k not in {'targets','output_dir'}}
                        part = self.engine.batch(targets=[path], output_dir=self.kwargs['output_dir'],
                            face_selector=self._select_face if self.ask_face else None,
                            should_stop=lambda:self._stop, on_log=self.log.emit, **options)
                        for key in ('completed','skipped','failed'):
                            result[key] += part[key]
                        result['outputs'].extend(part['outputs'])
                        result['errors'].extend(part.get('errors', []))
                        result['provider'] = part['provider']
                        if part['stopped']:
                            result['stopped'] = True
                            break
                except core.FaceSelectionSkipped:
                    result['skipped'] += 1
                    self.log.emit(f'Skipped by choice: {path.name}')
                except core.ProcessingCancelled:
                    result['stopped'] = True
                    break
                except Exception as exc:
                    message = f'{type(exc).__name__}: {exc}'
                    result['failed'] += 1
                    result['errors'].append({'path':str(path),'error':message})
                    self.log.emit(f'Failed {path.name}: {message}')
                self.progress.emit(index+1, result['total'], result.copy())
            outcome, payload = self.batch_ready, (result,)
        except Exception as exc:
            outcome, payload = self.failed, (f'{type(exc).__name__}: {exc}',)
        finally:
            self.last_provider = self.active_engine.provider
            for engine in (self.engine, self.video_engine):
                if engine is not None:
                    try:
                        engine.release_models()
                    except Exception as exc:
                        self.log.emit(f'Could not unload models: {exc}')
            self.log.emit('Processing models unloaded; memory released for other apps.')
        outcome.emit(*payload)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{VERSION} · Face Swap")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1180, 760)
        self.setMinimumSize(930, 650)
        self.source_path: Path | None = None
        self.targets: list[Path] = []
        self.worker: QThread | None = None
        self.face_choices: dict[str, int] = {}
        self._face_dialog = None
        self.last_output: Path | None = None
        self.video_ranges = {}
        self.video_selections = {}
        self.video_info = {}
        self.video_preview_path = None
        self._trim_loading = False
        self._trim_path = None
        self._pending_probes = []
        self.duration_worker = None
        self._video_quality_locked = False
        self._photo_quality = 'careful'
        self._photo_format = 'jpg'
        self._video_format_locked = False
        self._busy = False
        self._state = read_state()
        self.engine = core.SwapEngine(use_gpu=bool(self._state.get("prefer_gpu", True)))
        self._restoring_state = True

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 10)
        root.setSpacing(8)

        root.addLayout(self._build_header())
        root.addWidget(self._build_workflow(), stretch=1)
        options = self._build_options()
        options.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        root.addWidget(options)
        root.addLayout(self._build_actions())
        root.addWidget(self.progress)
        root.addWidget(self.log, stretch=0)
        footer = QHBoxLayout()
        footer.addStretch(1)
        credit = QLabel(f'v{VERSION} · Made with <span style="color:{ACCENT}">♥</span> by Long Weekend Labs')
        credit.setObjectName('madeWithLove')
        credit.setTextFormat(Qt.RichText)
        footer.addWidget(credit)
        root.addLayout(footer)

        self._apply_style()
        self._restore_state()
        self._restoring_state = False
        self._refresh_model_status()
        self._refresh_controls()

    def _build_header(self):
        row = QHBoxLayout()
        row.setSpacing(10)
        logo = QLabel()
        logo.setFixedSize(34,34)
        if ICON_PATH.exists():
            logo.setPixmap(QIcon(str(ICON_PATH)).pixmap(34,34))
        row.addWidget(logo)
        brand = QLabel(APP_NAME)
        brand.setObjectName("brand")
        row.addWidget(brand)
        subtitle = QLabel(TAGLINE)
        subtitle.setObjectName("pageSub")
        subtitle.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        row.addWidget(subtitle,1)
        self.model_status = QPushButton()
        self.model_status.setObjectName("statusPill")
        self.model_status.clicked.connect(self._show_model_setup)
        row.addWidget(self.model_status)
        row.addWidget(self._build_more_menu())
        return row

    def _build_more_menu(self) -> QToolButton:
        button = QToolButton()
        button.setText("☰")
        button.setObjectName("moreButton")
        button.setToolTip("About and project links")
        button.setPopupMode(QToolButton.InstantPopup)

        menu = QMenu(button)
        settings = QAction("Settings…", menu)
        settings.triggered.connect(self._show_settings)
        menu.addAction(settings)
        models = QAction("Model Management…", menu)
        models.triggered.connect(self._show_settings)
        menu.addAction(models)
        menu.addSeparator()
        about = QAction(f"About {APP_NAME}", menu)
        about.triggered.connect(self._show_about)
        menu.addAction(about)
        github = QAction("View on GitHub", menu)
        github.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(GITHUB_URL))
        )
        menu.addAction(github)
        core_models = QAction("Set up core face models…", menu)
        core_models.triggered.connect(self._show_model_setup)
        menu.addAction(core_models)
        menu.addSeparator()
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)
        button.setMenu(menu)
        self.more_button = button
        return button

    def _show_about(self) -> None:
        AboutDialog(self).exec()

    def _show_settings(self) -> None:
        SettingsDialog(self._state, self._save_state, self).exec()
        self.engine.use_gpu = bool(self._state.get('prefer_gpu',True))
        self._refresh_model_status()

    def _show_model_setup(self) -> None:
        dialog = ModelSetupDialog(self)
        dialog.exec()
        self._refresh_model_status()
        self._refresh_controls()

    def _build_workflow(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        source_box = QGroupBox("1 — Source face")
        source_layout = QVBoxLayout(source_box)
        source_layout.setSpacing(8)
        self.source_view = ImageView("Choose one clear source portrait")
        source_layout.addWidget(self.source_view, stretch=1)
        source_footer = QWidget()
        source_footer.setObjectName("cardFooter")
        source_footer.setFixedHeight(112)
        source_footer_layout = QVBoxLayout(source_footer)
        source_footer_layout.setContentsMargins(0, 0, 0, 0)
        source_footer_layout.setSpacing(6)
        self.source_name = QLabel("No source selected")
        self.source_name.setObjectName("fileName")
        self.source_name.setWordWrap(True)
        source_footer_layout.addWidget(self.source_name)
        self.character_name = QLineEdit()
        self.character_name.setPlaceholderText("Character name for output files (optional)")
        self.character_name.setToolTip(
            "Example: Fiona → Fiona_swapped_DDMMYYYY-HHMMSS.png"
        )
        self.character_name.setMaxLength(80)
        self.character_name.textChanged.connect(self._on_character_name_changed)
        source_footer_layout.addWidget(self.character_name)
        self.source_button = QPushButton("Choose source photo…")
        self.source_button.clicked.connect(self._choose_source)
        source_footer_layout.addWidget(self.source_button)
        source_layout.addWidget(source_footer)

        destinations_box = QGroupBox("2 — Destination media")
        destinations_layout = QVBoxLayout(destinations_box)
        destinations_layout.setSpacing(8)
        self.target_list = QListWidget()
        self.target_list.setMinimumHeight(70)
        self.target_list.setAlternatingRowColors(True)
        self.target_list.setTextElideMode(Qt.ElideMiddle)
        self.target_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.target_list.currentRowChanged.connect(self._show_selected_target)
        destinations_layout.addWidget(self.target_list, stretch=1)
        destinations_footer = QWidget()
        destinations_footer.setObjectName("cardFooter")
        destinations_footer.setFixedHeight(112)
        destinations_footer_layout = QVBoxLayout(destinations_footer)
        destinations_footer_layout.setContentsMargins(0, 0, 0, 0)
        destinations_footer_layout.setSpacing(6)
        self.target_count = QLabel("No destination media")
        self.target_count.setObjectName("fileName")
        self.target_count.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        count_row = QHBoxLayout()
        count_row.addWidget(self.target_count,1)
        self.include_videos = QCheckBox('Include videos / GIFs')
        self.include_videos.setToolTip('Enable queued videos and GIF animations. When any are included, all media uses Careful quality.')
        self.include_videos.toggled.connect(self._include_videos_changed)
        count_row.addWidget(self.include_videos)
        destinations_footer_layout.addLayout(count_row)
        target_buttons = QHBoxLayout()
        target_buttons.setSpacing(8)
        add_files = QPushButton("Add media…")
        add_files.clicked.connect(self._add_files)
        add_folder = QPushButton("Add folder…")
        add_folder.clicked.connect(self._add_folder)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_selected)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear_targets)
        for button in (add_files, add_folder, remove, clear):
            button.setProperty("compact", True)
            target_buttons.addWidget(button, stretch=1)
        self.target_buttons = [add_files,add_folder,remove,clear]
        destinations_footer_layout.addLayout(target_buttons)
        self.trim_row = QWidget()
        self.trim_row.setObjectName('cardFooter')
        trim = QHBoxLayout(self.trim_row)
        trim.setContentsMargins(0,0,0,0)
        trim.setSpacing(5)
        self.video_start = QDoubleSpinBox()
        self.video_end = QDoubleSpinBox()
        for spin in (self.video_start,self.video_end):
            spin.setRange(0,864000)
            spin.setDecimals(2)
            spin.setSuffix(' s')
            spin.setSingleStep(.5)
            spin.valueChanged.connect(self._trim_changed)
        self.video_end.setSpecialValueText('End of clip')
        trim.addWidget(QLabel('Clip from'))
        trim.addWidget(self.video_start,1)
        trim.addWidget(QLabel('to'))
        trim.addWidget(self.video_end,1)
        self.trim_row.setEnabled(False)
        destinations_footer_layout.addWidget(self.trim_row)
        destinations_layout.addWidget(destinations_footer)

        preview_box = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.setSpacing(8)
        self.preview_view = ImageView("Select a destination, then click Preview")
        preview_layout.addWidget(self.preview_view, stretch=1)
        preview_footer = QWidget()
        preview_footer.setObjectName("cardFooter")
        preview_footer.setFixedHeight(112)
        preview_footer_layout = QVBoxLayout(preview_footer)
        preview_footer_layout.setContentsMargins(0, 0, 0, 0)
        preview_footer_layout.setSpacing(6)
        self.preview_caption = QLabel("Nothing generated yet")
        self.preview_caption.setObjectName("fileName")
        self.preview_caption.setWordWrap(True)
        preview_footer_layout.addWidget(self.preview_caption)
        preview_footer_layout.addStretch(1)
        self.play_preview_button = QPushButton('Play video preview')
        self.play_preview_button.setEnabled(False)
        self.play_preview_button.clicked.connect(self._play_video_preview)
        preview_footer_layout.addWidget(self.play_preview_button)
        preview_layout.addWidget(preview_footer)

        splitter.addWidget(source_box)
        splitter.addWidget(destinations_box)
        splitter.addWidget(preview_box)
        splitter.setSizes([330, 350, 330])
        return splitter

    def _build_options(self):
        box = QGroupBox("3 — Output and quality")
        grid = QGridLayout(box)
        self.options_grid = grid
        grid.setColumnStretch(1, 1)
        output_label = QLabel("Output folder")
        output_label.setObjectName("fieldLabel")
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Swapped files are written here; originals are never changed")
        self.output_edit.textChanged.connect(self._on_output_changed)
        browse = QPushButton("Browse…")
        self.output_browse = browse
        browse.clicked.connect(self._choose_output)

        faces_label = QLabel("When several faces are found")
        faces_label.setObjectName("fieldLabel")
        self.face_mode = QComboBox()
        self.face_mode.addItems(["Swap the largest face only", "Swap every detected face"])
        self.face_mode.setToolTip("Photo face selection. Videos and GIFs follow one person; multiple-person clips always ask you to choose.")
        self.face_mode.currentIndexChanged.connect(self._save_state)
        self.permission = QCheckBox("I have permission to edit")
        self.permission.stateChanged.connect(self._refresh_controls)

        self.ask_face = QCheckBox("Ask which face to swap when several are detected")
        self.ask_face.setToolTip(
            "Choose one face per photo using numbered thumbnails. Single-face photos "
            "continue automatically. Overrides the largest/every-face setting while enabled. "
            "Choices are remembered during this session. Multiple-person videos and GIFs always ask you to choose one person."
        )
        self.ask_face.toggled.connect(self._face_selection_changed)
        self.reset_faces_button = QPushButton("Reset face choices")
        self.reset_faces_button.clicked.connect(self._reset_face_choices)

        quality_label = QLabel("Processing quality")
        quality_label.setObjectName("fieldLabel")
        self.quality_mode = QComboBox()
        self.quality_mode.addItem(
            "Careful: compare source likeness and recover detail", "careful"
        )
        self.quality_mode.addItem(
            "Best — Careful plus restoration (may alter eyes and lips)",
            "best",
        )
        self.quality_mode.addItem("Balanced — HyperSwap 256", "balanced")
        self.quality_mode.addItem("Fast — InSwapper 128 (draft quality)", "fast")
        self.quality_mode.setToolTip(
            "For smaller faces, Careful and Best compare two replacements against the "
            "source and use the closer match when the difference is clear. Larger faces "
            "keep the detailed rendering. Careful limits restoration "
            "to protect likeness; Best applies your whole-face restoration strength. "
            "Pose and expression can still change with either model."
        )
        self.quality_mode.currentIndexChanged.connect(self._save_state)
        self.output_format = QComboBox()
        self.output_format.addItem("JPEG 98 — smaller, re-encodes the image", "jpg")
        self.output_format.addItem("Lossless PNG — unchanged pixels stay exact", "png")
        self.output_format.addItem('MP4 video · audio kept', 'mp4')
        self.output_format.addItem('GIF animation · loop kept', 'gif')
        self.output_format.addItem('MP4 / GIF · follows input format', 'motion')
        for index in (2,3,4):
            self.output_format.model().item(index).setEnabled(False)
        self.output_format.setToolTip("Choose photo output here. Videos export MP4 with audio when present; GIFs stay GIF animations.")
        self.output_format.currentIndexChanged.connect(self._save_state)
        mouth_label = QLabel("Mouth detail")
        mouth_label.setObjectName("fieldLabel")
        self.preserve_mouth = QCheckBox("Protect open/obstructed target mouth")
        self.preserve_mouth.setChecked(True)
        self.preserve_mouth.setToolTip(
            "For open mouths, tongues or objects at the lips, keeps the target mouth "
            "expression together, including lips and teeth. Closed, unobstructed lips "
            "remain generated by the swap model. Turn off for full model-generated mouth replacement."
        )
        self.preserve_mouth.stateChanged.connect(self._save_state)
        self.destination_eyes = QCheckBox("Keep target eyes and target eye colour")
        self.destination_eyes.setChecked(False)
        self.destination_eyes.setToolTip(
            "Optional: retain the target photo's eyes, eyelids and lashes in the swapped face, "
            "including the target's iris colour. Leave off to keep the model-generated "
            "eyes. The model does not guarantee an exact source iris-colour match."
        )
        self.destination_eyes.stateChanged.connect(self._save_state)

        strength_label = QLabel("Restoration strength")
        strength_label.setObjectName("fieldLabel")
        self.restoration_strength = QSlider(Qt.Horizontal)
        self.restoration_strength.setRange(0, 100)
        self.restoration_strength.setValue(
            int(core.DEFAULT_RESTORATION_STRENGTH * 100)
        )
        self.restoration_strength.setToolTip(
            "Careful: limits eye-detail recovery and detail restoration on stronger "
            "face replacements. Whole-face restoration is capped at 35% and checked "
            "against source likeness. Best: whole-face restoration with checked local "
            "eye-detail recovery. Eye recovery adapts automatically to the face size. "
            "Set to 0 to disable restoration."
        )
        self.restoration_value = QLabel()
        self.restoration_value.setObjectName("fileName")
        self.restoration_strength.valueChanged.connect(self._restoration_changed)
        self.restoration_strength.sliderReleased.connect(self._save_state)
        self._restoration_changed(self.restoration_strength.value())
        self.skip_completed = QCheckBox("Skip unchanged photos already completed")
        self.skip_completed.setChecked(True)
        self.skip_completed.setToolTip(
            "Swapio records successful outputs in the output folder and only processes new or changed photos."
        )
        self.skip_completed.stateChanged.connect(self._save_state)

        # Five fixed control rows; only the media canvases grow with the window.
        output_actions = QWidget()
        output_actions.setObjectName('cardFooter')
        actions = QHBoxLayout(output_actions)
        actions.setContentsMargins(0,0,0,0)
        actions.addWidget(browse)
        actions.addWidget(self.permission)
        face_options = QWidget()
        face_options.setObjectName('cardFooter')
        faces = QHBoxLayout(face_options)
        faces.setContentsMargins(0,0,0,0)
        faces.addWidget(self.face_mode,1)
        self.ask_face.setText('Ask which face')
        faces.addWidget(self.ask_face)
        strength_options = QWidget()
        strength_options.setObjectName('cardFooter')
        strength = QHBoxLayout(strength_options)
        strength.setContentsMargins(0,0,0,0)
        strength.addWidget(self.restoration_value)
        strength.addStretch(1)
        strength.addWidget(self.skip_completed)
        faces_label.setText('Face selection')
        mouth_label.setText('Face detail')
        for row,label,primary,secondary in (
            (0,output_label,self.output_edit,output_actions),
            (1,quality_label,self.quality_mode,self.output_format),
            (2,faces_label,face_options,self.reset_faces_button),
            (3,mouth_label,self.preserve_mouth,self.destination_eyes),
            (4,strength_label,self.restoration_strength,strength_options),
        ):
            grid.addWidget(label,row,0)
            grid.addWidget(primary,row,1)
            grid.addWidget(secondary,row,2)
        return box

    def _build_actions(self):
        row = QHBoxLayout()
        self.preview_button = QPushButton("Preview selected media")
        self.preview_button.clicked.connect(self._preview)
        self.run_button = QPushButton("Swap queued media")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._run_batch)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop)
        self.stop_button.setEnabled(False)
        self.open_button = QPushButton("Open output folder")
        self.open_button.clicked.connect(self._open_output)
        self.open_button.setEnabled(False)
        row.addWidget(self.preview_button)
        row.addWidget(self.run_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.open_button)
        self.details_button = QPushButton('Details')
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self._toggle_details)
        row.addWidget(self.details_button)
        row.addStretch(1)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusText")
        row.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setFormat("Ready")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(80)
        self.log.hide()
        self.log.setPlaceholderText("Processing details will appear here.")
        return row

    def _dialog_start(self, key: str) -> str:
        value = self._state.get(key, "")
        path = Path(value).expanduser() if value else Path.home()
        if not path.is_absolute():
            path = (PROJECT_DIR / path).resolve()
        return str(path if path.exists() else Path.home())

    def _toggle_details(self, shown):
        self.log.setVisible(shown)

    def _remember_folder(self, key: str, path: Path) -> None:
        self._state[key] = str(path.expanduser().resolve())
        self._save_state()

    def _choose_source(self):
        paths = native_image_files(
            self, "Choose source face", self._dialog_start("source_dir")
        )
        if not paths:
            return
        self._set_source(Path(paths[0]))

    def _set_source(self, path: Path):
        image = core.load_image(path)
        if image is None:
            QMessageBox.warning(self, "Unreadable image", f"Could not open:\n{path}")
            return
        self.source_path = path
        source_key = str(path.expanduser().resolve())
        known_names = self._state.get("source_names", {})
        self.character_name.setText(known_names.get(source_key, ""))
        self.source_view.set_image(image)
        self.source_name.setText(path.name)
        self.source_name.setToolTip(str(path))
        self._remember_folder("source_dir", path.parent)
        self.preview_view.clear_image()
        self.video_preview_path = None
        self.play_preview_button.setEnabled(False)
        self.preview_caption.setText("Nothing generated yet")
        self._refresh_controls()

    def _on_character_name_changed(self, name: str) -> None:
        if not self.source_path:
            return
        source_key = str(self.source_path.expanduser().resolve())
        known_names = dict(self._state.get("source_names", {}))
        cleaned = name.strip()
        if cleaned:
            known_names[source_key] = cleaned
        else:
            known_names.pop(source_key, None)
        self._state["source_names"] = known_names
        self._save_state()

    def _add_files(self):
        paths = native_media_files(
            self, "Add destination media", self._dialog_start("target_dir"), multiple=True
        )
        if paths:
            self._add_targets(Path(path) for path in paths)
            self._remember_folder("target_dir", Path(paths[0]).parent)

    def _add_folder(self):
        path = native_directory(
            self, "Add a folder of destination media", self._dialog_start("target_dir")
        )
        if path:
            folder = Path(path)
            self._add_targets(sorted(p for p in folder.rglob('*')
                if p.suffix.lower() in core.IMAGE_EXTS | MOTION_EXTENSIONS))
            self.output_edit.setText(str(core.suggested_output_dir(folder)))
            self._remember_folder("target_dir", folder)

    def _add_targets(self, paths):
        existing = {str(path.resolve()) for path in self.targets}
        added = 0
        videos = []
        for path in paths:
            path = Path(path)
            try:
                resolved = str(path.resolve())
            except OSError:
                continue
            if path.is_file() and path.suffix.lower() in core.IMAGE_EXTS | MOTION_EXTENSIONS and resolved not in existing:
                path = path.resolve()
                self.targets.append(path)
                item = QListWidgetItem(path.name)
                item.setToolTip(str(path))
                item.setData(Qt.UserRole, str(path))
                self.target_list.addItem(item)
                existing.add(resolved)
                added += 1
                if self._is_motion(path):
                    videos.append(path)
        if added and self.target_list.currentRow() < 0:
            self.target_list.setCurrentRow(0)
        if added and not self.output_edit.text().strip():
            self.output_edit.setText(str(self._default_output_dir()))
        self._refresh_target_count()
        self._sync_video_quality()
        self._refresh_controls()
        self._pending_probes.extend(videos)
        self._start_duration_scan()

    @staticmethod
    def _is_motion(path):
        return path is not None and Path(path).suffix.lower() in MOTION_EXTENSIONS

    def _active_targets(self):
        return [p for p in self.targets if not self._is_motion(p) or self.include_videos.isChecked()]

    def _sync_video_quality(self):
        locked = self.include_videos.isChecked() and any(self._is_motion(p) for p in self.targets)
        if locked and not self._video_quality_locked:
            self._photo_quality = self.quality_mode.currentData()
            self.quality_mode.blockSignals(True)
            self.quality_mode.setCurrentIndex(self.quality_mode.findData('careful'))
            self.quality_mode.blockSignals(False)
        elif not locked and self._video_quality_locked:
            self.quality_mode.blockSignals(True)
            self.quality_mode.setCurrentIndex(max(0,self.quality_mode.findData(self._photo_quality)))
            self.quality_mode.blockSignals(False)
        self._video_quality_locked = locked
        self.quality_mode.setToolTip('Included videos and GIFs require Careful for the whole queue. Untick Include videos / GIFs to choose another mode.'
            if locked else 'Choose photo rendering quality. Video and GIF queues use Careful.')

    def _include_videos_changed(self, _checked):
        if not hasattr(self,'quality_mode'):
            return
        self._sync_video_quality()
        self._refresh_target_count()
        self._show_selected_target(self.target_list.currentRow())
        self._save_state()

    def _trim_changed(self):
        if self._trim_loading or self._trim_path is None:
            return
        key = str(self._trim_path)
        self.video_ranges[key] = (self.video_start.value(), self.video_end.value())
        self.video_selections.pop(key,None)
        self.video_preview_path = None
        self.play_preview_button.setEnabled(False)
        self.preview_caption.setText('Range changed. Generate a new 2-second preview.')

    def _start_duration_scan(self):
        if getattr(self,'_closing',False) or not self._pending_probes:
            return
        if self.duration_worker is not None and self.duration_worker.isRunning():
            return
        paths,self._pending_probes = self._pending_probes,[]
        self.duration_worker = DurationWorker(paths)
        self.duration_worker.ready.connect(self._durations_ready)
        self.duration_worker.finished.connect(self._duration_scan_finished)
        self.duration_worker.start()
        self._refresh_controls()

    def _duration_scan_finished(self):
        self._start_duration_scan()
        self._refresh_controls()

    def _durations_ready(self, results):
        warnings = []
        for path,info in results.items():
            if Path(path) not in self.targets:
                continue
            self.video_info[path] = info
            if isinstance(info,str):
                warnings.append(f'{Path(path).name}: could not inspect duration. {info}')
            elif info.duration > 180:
                warnings.append(f'{Path(path).name}: {info.duration/60:.1f} minutes (over 3 minutes).')
            elif info.duration > 60:
                warnings.append(f'{Path(path).name}: {info.duration/60:.1f} minutes (over 1 minute).')
        if self._is_motion(self._selected_target()) and str(self._selected_target()) in results:
            self._show_selected_target(self.target_list.currentRow())
        if warnings:
            QMessageBox.warning(self,'Clip duration', '\n'.join(warnings[:12])
                + ('\nMore long clips are in the queue.' if len(warnings)>12 else '')
                + '\n\nCareful animation rendering can take a long time. Select a clip and adjust Clip from / to for a shorter range. Originals stay untouched.')
        self._refresh_controls()

    def _play_video_preview(self):
        if self.video_preview_path and self.video_preview_path.is_file():
            if not native_open_path(self.video_preview_path):
                QMessageBox.warning(self,'Play preview',f'Could not open your video player.\n{self.video_preview_path}')

    def _default_output_dir(self) -> Path:
        """Choose a sibling folder so recursive imports never ingest outputs."""
        if not self.targets:
            return Path.home() / "Swapio Output"
        return core.suggested_output_dir(self.targets[0].parent)

    def _remove_selected(self):
        row = self.target_list.currentRow()
        if row < 0:
            return
        path = self.targets.pop(row)
        self.video_selections.pop(str(path),None)
        self.video_ranges.pop(str(path),None)
        self.video_info.pop(str(path),None)
        self.target_list.takeItem(row)
        self._sync_video_quality()
        self._refresh_target_count()
        self._refresh_controls()

    def _clear_targets(self):
        self.targets.clear()
        self.video_ranges.clear()
        self.video_selections.clear()
        self.video_info.clear()
        self._sync_video_quality()
        self.target_list.clear()
        self.preview_view.clear_image()
        self.preview_caption.setText("Nothing generated yet")
        self._refresh_target_count()
        self._refresh_controls()

    def _refresh_target_count(self):
        gifs = sum(p.suffix.lower() in gif.GIF_EXTENSIONS for p in self.targets)
        movies = sum(self._is_motion(p) for p in self.targets)-gifs
        photos = len(self.targets)-movies-gifs
        parts = []
        if photos: parts.append(f'{photos} photo'+('s' if photos!=1 else ''))
        for count,kind in ((movies,'video'),(gifs,'GIF')):
            if count: parts.append(f'{count} {kind}'+('s' if count!=1 else '')+(' (excluded)' if not self.include_videos.isChecked() else ''))
        text = ' · '.join(parts) or 'No destination media'
        self.target_count.setText(text)
        self.target_count.setToolTip(text)
        for index,path in enumerate(self.targets):
            item = self.target_list.item(index)
            excluded = self._is_motion(path) and not self.include_videos.isChecked()
            item.setText(('GIF · ' if path.suffix.lower() in gif.GIF_EXTENSIONS else 'Video · ' if self._is_motion(path) else '')+path.name)
            item.setForeground(QColor(MUTED if excluded else TEXT))

    def _show_selected_target(self, row: int):
        if not hasattr(self,'preview_view'):
            return
        target = self.targets[row] if 0<=row<len(self.targets) else None
        if self.worker and self.worker.isRunning():
            return
        self._trim_loading = True
        self._trim_path = target if self._is_motion(target) else None
        start,end = self.video_ranges.get(str(target),(0.,0.))
        info = self.video_info.get(str(target))
        maximum = info.duration if isinstance(info,video.VideoInfo) else 864000
        self.video_start.setMaximum(maximum)
        self.video_end.setMaximum(maximum)
        self.video_start.setValue(start)
        self.video_end.setValue(end)
        self._trim_loading = False
        self.video_preview_path = None
        self.play_preview_button.setEnabled(False)
        if target is None:
            self.preview_view.clear_image()
            self.preview_caption.setText('Nothing generated yet')
            self._refresh_controls()
            return
        self.preview_view.clear_image("Click Preview to generate this result")
        if self._is_motion(target):
            kind = 'GIF' if target.suffix.lower() in gif.GIF_EXTENSIONS else 'video'
            self.play_preview_button.setText(f'Play {kind} preview')
            duration = f'{info.duration:.1f} s · ' if isinstance(info,video.VideoInfo) else ''
            self.preview_caption.setText(duration+(f'2-second {kind} preview · select the person when prompted'
                if self.include_videos.isChecked() else 'Tick Include videos / GIFs to preview or render this clip.'))
        else:
            self.preview_caption.setText(target.name)
        self._refresh_controls()

    def _choose_output(self):
        path = native_directory(
            self, "Choose output folder", self._dialog_start("output_dir")
        )
        if path:
            self.output_edit.setText(path)
            self._remember_folder("output_dir", Path(path))

    def _on_output_changed(self):
        self._save_state()
        self._refresh_controls()

    def _all_faces(self) -> bool:
        return not self.ask_face.isChecked() and self.face_mode.currentIndex() == 1

    def _face_selection_changed(self, _checked):
        self._save_state()
        self._refresh_controls()

    def _reset_face_choices(self):
        self.face_choices.clear()
        self.video_selections.clear()
        self.video_preview_path = None
        self.play_preview_button.setEnabled(False)
        self.preview_view.clear_image("Click Preview to choose a face")
        self.preview_caption.setText("Face choices reset")
        self._refresh_controls()

    def _face_choice_requested(self, request):
        worker = request["worker"]
        if worker is not self.worker or worker._stop:
            return
        try:
            dialog = FaceSelectionDialog(request["target"], request["image"], request["boxes"],
                                         batch=worker.mode == "batch", parent=self)
            self._face_dialog = dialog
            self.status_label.setText("Waiting for face selection…")
            result = dialog.exec()
            if result == FaceSelectionDialog.STOP_BATCH:
                worker.stop()
            else:
                worker.submit_face_choice(dialog.selected_index if result == QDialog.Accepted else None)
            self.status_label.setText("Generating preview…" if worker.mode in {"preview","video_preview"} else "Swapping media…")
        except Exception as exc:
            worker.stop()
            self.log.appendPlainText(f"Face picker stopped: {exc}")
        finally:
            self._face_dialog = None

    def _preview_cancelled(self):
        self._set_running(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Preview cancelled")
        self.status_label.setText("Preview cancelled")
        self._refresh_model_status()

    def _restoration_changed(self, value: int) -> None:
        self.restoration_value.setText(f"{value}%" if value else "off")

    def _restoration_strength(self) -> float:
        return self.restoration_strength.value() / 100.0

    def _quality(self) -> str:
        return self.quality_mode.currentData()

    def _output_format(self) -> str:
        return self._photo_format if self._video_format_locked else self.output_format.currentData()

    def _selected_target(self) -> Path | None:
        row = self.target_list.currentRow()
        return self.targets[row] if 0 <= row < len(self.targets) else None

    def _preview(self):
        target = self._selected_target()
        if not self.source_path or not target:
            return
        if self._is_motion(target):
            self._preview_video(target)
            return
        self.log.clear()
        self.log.appendPlainText(f"Previewing {target.name}...")
        self.progress.setRange(0, 0)
        self.progress.setFormat("Generating preview…")
        self.worker = TaskWorker(
            self.engine,
            "preview",
            ask_face=self.ask_face.isChecked(),
            face_choices=self.face_choices,
            source_path=self.source_path,
            target_path=target,
            all_faces=self._all_faces(),
            quality=self._quality(),
            preserve_mouth=self.preserve_mouth.isChecked(),
            destination_eyes=self.destination_eyes.isChecked(),
            restoration_strength=self._restoration_strength(),
        )
        self.worker.log.connect(self._log_message)
        self.worker.preview_ready.connect(self._preview_ready)
        self.worker.failed.connect(self._failed)
        self.worker.cancelled.connect(self._preview_cancelled)
        self.worker.face_choice_requested.connect(self._face_choice_requested)
        self.worker.finished.connect(self._refresh_controls)
        self._set_running(True)
        self.status_label.setText("Generating preview…")
        self.worker.start()

    def _media_worker(self, mode, targets, output):
        worker = MediaWorker(self.engine, mode, ask_face=self.ask_face.isChecked(),
            face_choices=self.face_choices, video_ranges=self.video_ranges,
            video_selections=self.video_selections, source_path=self.source_path,
            targets=list(targets), output_dir=output, all_faces=self._all_faces(),
            quality=self._quality(), output_format=self._output_format(),
            character_name=self.character_name.text().strip(),
            preserve_mouth=self.preserve_mouth.isChecked(), destination_eyes=self.destination_eyes.isChecked(),
            restoration_strength=self._restoration_strength(), skip_completed=self.skip_completed.isChecked())
        worker.video_progress.connect(self._media_frame_progress)
        worker.person_selected.connect(self._video_person_selected)
        return worker

    def _preview_video(self, target):
        if not self.include_videos.isChecked():
            return
        kind = 'GIF' if target.suffix.lower() in gif.GIF_EXTENSIONS else 'video'
        answer = QMessageBox.information(self,f'2-second {kind} preview',
            f'Swapio will render a 2-second {kind} preview using Careful. This can take a few minutes depending on the face and your device.\n\nUse Play {kind} preview when it finishes to review the animation.',
            QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)
        if answer != QMessageBox.Ok:
            return
        output = Path(self.output_edit.text().strip() or str(self._default_output_dir())).expanduser()
        self.last_output = output
        self.video_preview_path = None
        self.play_preview_button.setEnabled(False)
        self.log.clear()
        self.progress.setRange(0,0)
        self.progress.setFormat('Preparing 2-second video preview…')
        self.worker = self._media_worker('video_preview',[target],output/'.swapio-previews')
        self.worker.log.connect(self._log_message)
        self.worker.batch_ready.connect(self._video_preview_ready)
        self.worker.failed.connect(self._failed)
        self.worker.cancelled.connect(self._preview_cancelled)
        self.worker.face_choice_requested.connect(self._face_choice_requested)
        self.worker.finished.connect(self._refresh_controls)
        self._set_running(True)
        self.status_label.setText(f'Preparing {kind} preview…')
        self.worker.start()

    def _video_person_selected(self, path, selection):
        self.video_selections[path] = selection
        if self._selected_target() == Path(path):
            marked = selection.image.copy()
            l,t,r,b = np.rint(selection.box).astype(int)
            cv2.rectangle(marked,(l,t),(r,b),(92,122,255),2)
            self.preview_view.set_image(marked)
            self.preview_caption.setText('Selected person. Rendering with Careful…')

    def _media_frame_progress(self, done, total, counts):
        self._refresh_model_status()
        self.progress.setRange(0,max(total,done))
        self.progress.setValue(done)
        position = f"{done/1000:.2f}/{total/1000:.2f} s" if counts.get("unit") == "time" else f"frame {done}/{total}"
        self.progress.setFormat(f"Media {counts['media_index']}/{counts['media_total']} · {position} · {counts['unchanged']} unchanged")
        self.status_label.setText(f"{counts['provider']} · ~{counts['eta']:.0f}s left")

    def _video_preview_ready(self, result):
        item = result.get('video_preview')
        if item is None:
            if result['failed']:
                self._failed(result['errors'][0]['error'])
            else:
                self._preview_cancelled()
            return
        self._set_running(False)
        self.video_preview_path = Path(item['output'])
        self.preview_view.set_image(item['image'])
        kind = "GIF" if self.video_preview_path.suffix.lower() in gif.GIF_EXTENSIONS else "Video"
        self.play_preview_button.setText(f"Play {kind.lower() if kind == 'Video' else kind} preview")
        self.preview_caption.setText(f"{kind} preview · {item['duration']:.2f} s · {item['unchanged']} unchanged frame(s)")
        self.preview_caption.setToolTip(item['output'])
        self.play_preview_button.setEnabled(True)
        notify_output_folder(self.video_preview_path.parent)
        self.progress.setRange(0,1)
        self.progress.setValue(1)
        self.progress.setFormat(f'2-second preview ready · {self.play_preview_button.text()}')
        self.status_label.setText('Preview ready')
        self._refresh_model_status()

    def _preview_ready(self, result: dict):
        self._set_running(False)
        self.preview_view.set_image(result["image"])
        target = Path(result["target"]).name
        choice = result.get("target_face_index")
        detail = (f"face {choice + 1} of {result['detected']}" if choice is not None
                  else f"swapped {result['swapped']} face(s)")
        self.preview_caption.setText(
            f"{target} — {detail} · "
            f"{result['quality'].title()} · {result['provider']}"
        )
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFormat("Preview ready")
        self.status_label.setText("Preview ready")
        self._refresh_model_status()

    def _run_batch(self):
        targets = self._active_targets()
        if not targets:
            return
        output_text = self.output_edit.text().strip()
        if not output_text:
            if not self.targets:
                return
            output_text = str(self._default_output_dir())
            self.output_edit.setText(output_text)
        output = Path(output_text).expanduser()
        if output.exists() and not output.is_dir():
            QMessageBox.warning(self, "Output folder", "The output path is not a folder.")
            return
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "Output folder", f"Could not create the output folder:\n{exc}")
            return
        self.last_output = output
        self._notified_saved = 0
        self.log.clear()
        self.progress.setRange(0, len(targets))
        self.progress.setValue(0)
        self.progress.setFormat(f"0/{len(targets)}")
        if any(self._is_motion(p) for p in targets):
            self.worker = self._media_worker('batch',targets,output)
        else:
            self.worker = TaskWorker(
                self.engine,
                "batch",
                ask_face=self.ask_face.isChecked(),
                face_choices=self.face_choices,
                source_path=self.source_path,
                targets=targets,
                output_dir=output,
                all_faces=self._all_faces(),
                quality=self._quality(),
                output_format=self._output_format(),
                character_name=self.character_name.text().strip(),
                preserve_mouth=self.preserve_mouth.isChecked(),
                destination_eyes=self.destination_eyes.isChecked(),
                restoration_strength=self._restoration_strength(),
                skip_completed=self.skip_completed.isChecked(),
            )
        self.worker.log.connect(self._log_message)
        self.worker.progress.connect(self._progress)
        self.worker.batch_ready.connect(self._batch_ready)
        self.worker.failed.connect(self._failed)
        self.worker.face_choice_requested.connect(self._face_choice_requested)
        self.worker.finished.connect(self._refresh_controls)
        self._set_running(True)
        self.status_label.setText("Swapping media…")
        self.worker.start()

    def _progress(self, done: int, total: int, counts: dict):
        self._refresh_model_status()
        completed = counts.get("completed", 0)
        if completed > getattr(self, "_notified_saved", 0) and self.last_output:
            self._notified_saved = completed
            notify_output_folder(self.last_output)
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.progress.setFormat(
            f"{done}/{total} — {counts.get('completed', 0)} saved, "
            f"{counts.get('skipped', 0)} skipped, {counts.get('failed', 0)} failed"
        )

    def _batch_ready(self, result: dict):
        if result["outputs"] and self.last_output:
            if not notify_output_folder(self.last_output):
                self.log.appendPlainText(
                    f"Saved files are on disk, but the desktop folder update could not be sent: {self.last_output}"
                )
        self._set_running(False)
        self.open_button.setEnabled(bool(result["outputs"]) or result["skipped"] > 0)
        label = "Stopped" if result["stopped"] else "Complete"
        self.status_label.setText(
            f"{label} — {result['completed']} saved, {result['skipped']} skipped, "
            f"{result['failed']} failed"
        )
        self.progress.setFormat(
            f"{result['completed']} saved — {result['skipped']} skipped — "
            f"{result['failed']} failed — originals untouched"
        )
        self._refresh_model_status()
        if not result["stopped"]:
            if result["failed"]:
                message = QMessageBox(self)
                message.setIcon(QMessageBox.Warning)
                message.setWindowTitle("Batch finished with errors")
                message.setText(
                    f"Saved {result['completed']} swapped media file(s).\n"
                    f"Skipped {result['skipped']} item(s).\n"
                    f"Failed {result['failed']}. Failed items have no new output."
                )
                errors = result.get("errors", [])
                first = errors[0] if errors else None
                message.setInformativeText(
                    (f"First failure: {Path(first['path']).name}\n{first['error']}\n\n" if first else "")
                    + f"Output: {self.last_output}"
                )
                message.setDetailedText("\n\n".join(
                    f"{error['path']}\n{error['error']}" for error in errors
                ))
                message.exec()
                return
            QMessageBox.information(
                self,
                "Batch complete",
                f"Saved {result['completed']} swapped media file(s).\n"
                f"Skipped {result['skipped']} item(s).\n"
                f"Failed {result['failed']}.\n\n"
                f"Output: {self.last_output}",
            )

    def _stop(self):
        if self.worker:
            self.worker.stop()
            if self._face_dialog is not None:
                self._face_dialog.reject()
            self.status_label.setText("Stopping after the current item/frame…")

    def _failed(self, message: str):
        self._set_running(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Error")
        self.status_label.setText("Error")
        self.log.appendPlainText("ERROR: " + message)
        self._refresh_model_status()
        QMessageBox.critical(self, "Swapio error", message)

    def _open_output(self):
        if self.last_output and self.last_output.is_dir():
            if not native_open_path(self.last_output):
                QMessageBox.warning(
                    self,
                    "Open output folder",
                    f"Could not open the system file manager.\n\nFolder: {self.last_output}",
                )

    def _set_running(self, running: bool):
        self._busy = running
        self._refresh_controls()
        self._refresh_model_status()

    def _log_message(self, message):
        self.log.appendPlainText(message)
        self._refresh_model_status()

    def _refresh_controls(self):
        running = self._busy or bool(self.worker and self.worker.isRunning())
        self._sync_video_quality()
        self.ask_face.setEnabled(not running)
        self.face_mode.setEnabled(not running and not self.ask_face.isChecked())
        self.reset_faces_button.setEnabled(not running and bool(self.face_choices or self.video_selections))
        active = self._active_targets()
        video_only = bool(active) and all(self._is_motion(p) for p in active)
        if video_only:
            if not self._video_format_locked:
                self._photo_format = self.output_format.currentData()
            kinds = {'gif' if p.suffix.lower() in gif.GIF_EXTENSIONS else 'mp4' for p in active}
            format_kind = next(iter(kinds)) if len(kinds) == 1 else 'motion'
            self.output_format.blockSignals(True)
            self.output_format.setCurrentIndex(self.output_format.findData(format_kind))
            self.output_format.blockSignals(False)
        elif not video_only and self._video_format_locked:
            self.output_format.blockSignals(True)
            self.output_format.setCurrentIndex(max(0,self.output_format.findData(self._photo_format)))
            self.output_format.blockSignals(False)
        self._video_format_locked = video_only
        inspecting = any(self._is_motion(p) and str(p) not in self.video_info for p in active)
        ready = (not running and self.source_path is not None and bool(active)
                 and not inspecting and self.permission.isChecked() and not core.missing_models())
        target = self._selected_target()
        self.preview_button.setEnabled(ready and target is not None and target in active)
        self.run_button.setEnabled(ready)
        self.quality_mode.setEnabled(not running and not self._video_quality_locked)
        self.target_list.setEnabled(not running)
        self.include_videos.setEnabled(not running)
        self.source_button.setEnabled(not running)
        for button in self.target_buttons:
            button.setEnabled(not running)
        self.trim_row.setEnabled(not running and self._is_motion(target) and self.include_videos.isChecked())
        self.play_preview_button.setEnabled(not running and self.video_preview_path is not None)
        for widget in (self.output_edit,self.output_browse,self.output_format,self.preserve_mouth,
                       self.destination_eyes,self.restoration_strength,self.skip_completed,self.permission,self.character_name):
            widget.setEnabled(not running)
        self.output_format.setEnabled(not running and not video_only)
        self.stop_button.setEnabled(running)

    def _refresh_model_status(self):
        status_engine = getattr(self.worker,'active_engine',self.worker.engine) if self.worker is not None else self.engine
        running = self._busy or bool(self.worker and self.worker.isRunning())
        provider = status_engine.provider
        last = getattr(self.worker,'last_provider','Not loaded')
        prefer_gpu = bool(self._state.get('prefer_gpu',True))
        cuda_available = SettingsDialog._gpu_description()[0]
        missing = core.missing_models()
        if missing:
            self.model_status.setText('Models missing · Setup')
            self.model_status.setProperty('state','warning')
            self.model_status.setToolTip('Missing: '+', '.join(missing)+'\nClick to set up.')
        else:
            actual = provider if provider != 'Not loaded' else last
            fallback = prefer_gpu and actual == 'CPU'
            if running and provider != 'Not loaded':
                label = ('CPU fallback' if fallback else provider)+' · Processing'
            elif running:
                label = 'Starting CUDA…' if prefer_gpu and cuda_available else 'Starting CPU…'
            else:
                device = 'CPU fallback' if fallback else ('CUDA enabled' if prefer_gpu and cuda_available else 'CPU')
                label = device+' · Idle'
            self.model_status.setText('● '+label)
            self.model_status.setProperty('state','warning' if fallback else 'ready')
            self.model_status.setToolTip(
                f"GPU preference: {'enabled' if prefer_gpu else 'disabled'}. CUDA runtime: {'available' if cuda_available else 'unavailable'}.\n"
                + (f'Current inference: {provider}.' if running and provider!='Not loaded' else f'Last inference: {last}. Models are unloaded while idle and reload automatically for the next job.')
                + '\nCUDA availability is verified when inference starts; any CPU fallback is reported in Details.\n'
                + str(core.model_dir()))
        self.model_status.style().unpolish(self.model_status)
        self.model_status.style().polish(self.model_status)

    def _restore_state(self):
        output = self._state.get("output_dir", "")
        output_path = Path(output).expanduser() if output else None
        if output_path and output_path.is_dir():
            self.output_edit.setText(str(output_path.resolve()))
        self.face_mode.setCurrentIndex(int(self._state.get("face_mode", 0)))
        self.ask_face.setChecked(bool(self._state.get("ask_face", False)))
        quality = self._state.get("quality", "careful")
        quality_index = self.quality_mode.findData(quality)
        self.quality_mode.setCurrentIndex(max(quality_index, 0))
        output_format = self._state.get("output_format", "jpg")
        format_index = self.output_format.findData(output_format)
        self.output_format.setCurrentIndex(max(format_index, 0))
        self.preserve_mouth.setChecked(bool(self._state.get("preserve_mouth", True)))
        self.destination_eyes.setChecked(bool(self._state.get("destination_eyes", False)))
        self.restoration_strength.setValue(
            int(self._state.get(
                "restoration_strength", core.DEFAULT_RESTORATION_STRENGTH * 100
            ))
        )
        self.skip_completed.setChecked(bool(self._state.get("skip_completed", True)))
        self.include_videos.setChecked(bool(self._state.get('include_videos',False)))

    def _save_state(self):
        if self._restoring_state:
            return
        self._state["output_dir"] = self.output_edit.text().strip()
        self._state["face_mode"] = self.face_mode.currentIndex()
        self._state["ask_face"] = self.ask_face.isChecked()
        self._state["quality"] = self._photo_quality if self._video_quality_locked else self._quality()
        self._state['include_videos'] = self.include_videos.isChecked()
        self._state["output_format"] = self._output_format()
        self._state["preserve_mouth"] = self.preserve_mouth.isChecked()
        self._state["destination_eyes"] = self.destination_eyes.isChecked()
        self._state["restoration_strength"] = self.restoration_strength.value()
        self._state["skip_completed"] = self.skip_completed.isChecked()
        for retired in (
            "hair_mode", "custom_hair_color", "hair_strength",
            "skin_match", "skin_strength", "body_skin",
            "skin_tone_match", "skin_tone_strength",
        ):
            self._state.pop(retired, None)
        write_state(self._state)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            answer = QMessageBox.question(
                self,
                "Swap still running",
                "Stop after the current item/frame and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.stop()
            if self._face_dialog is not None:
                self._face_dialog.reject()
            self.worker.wait()
        self._closing = True
        if self.duration_worker is not None and self.duration_worker.isRunning():
            self.duration_worker.requestInterruption()
            self.duration_worker.wait()
        self._save_state()
        event.accept()

    def _apply_style(self):
        self.setStyleSheet(
            f"""
            QWidget {{ background:{BG}; color:{TEXT}; font-size:13px;
                       font-family:'Inter','Segoe UI',system-ui,sans-serif; }}
            QLabel, QCheckBox, QSplitter {{ background:transparent; }}
            QWidget#cardFooter {{ background:transparent; }}
            #brand {{ color:{ACCENT}; font-size:26px; font-weight:800; letter-spacing:0.5px; }}
            #pageSub {{ color:{MUTED}; font-size:12px; }}
            #versionBadge {{ color:{ACCENT}; border:1px solid #694435; border-radius:9px;
                             padding:2px 7px; font-size:10px; font-weight:700; }}
            #madeWithLove {{ color:{MUTED}; font-size:10px; }}
            #aboutName {{ color:{ACCENT}; font-size:24px; font-weight:800; }}
            #aboutVersion {{ color:{LABEL}; font-size:12px; font-weight:600; }}
            #aboutDescription {{ color:{MUTED}; font-size:12px; }}
            #aboutFinePrint {{ color:{MUTED}; font-size:10px; }}
            #modelWarning {{ color:#f4b183; background:#2b211d; border:1px solid #694435;
                             border-radius:7px; padding:9px; font-size:11px; }}
            #aboutDivider {{ color:{BORDER}; background:{BORDER}; max-height:1px; }}
            #fileName {{ color:{MUTED}; font-size:11px; }}
            #fieldLabel {{ color:{LABEL}; font-weight:600; }}
            #statusText {{ color:{MUTED}; }}
            #statusPill {{ border:1px solid {BORDER}; border-radius:12px; padding:5px 10px;
                           background:{PANEL}; color:{MUTED}; font-size:11px; }}
            #statusPill[state="ready"] {{ color:{GREEN}; border-color:#285642; }}
            #statusPill[state="warning"] {{ color:{ACCENT}; border-color:#694435; }}
            QGroupBox {{ border:1px solid {BORDER}; border-radius:10px; margin-top:11px;
                         padding:12px 10px 10px; background:{PANEL}; font-weight:600; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:0 6px;
                                color:{ACCENT}; font-weight:700; }}
            QTabWidget::pane {{ background:{PANEL}; border:1px solid {BORDER};
                                border-radius:9px; top:-1px; }}
            QTabBar {{ background:transparent; }}
            QTabBar::tab {{ background:{BG}; color:{MUTED}; border:1px solid {BORDER};
                            padding:8px 14px; margin-right:3px; border-top-left-radius:7px;
                            border-top-right-radius:7px; }}
            QTabBar::tab:hover {{ color:{TEXT}; background:{INPUT}; }}
            QTabBar::tab:selected {{ color:{ACCENT}; background:{PANEL};
                                     border-bottom-color:{PANEL}; font-weight:700; }}
            QLabel#imageView {{ background:{BG}; border:1px dashed {BORDER}; border-radius:8px;
                                color:{MUTED}; padding:8px; font-weight:400; }}
            QListWidget {{ background:{BG}; alternate-background-color:#191c22;
                           border:1px solid {BORDER}; border-radius:8px; padding:4px; }}
            QListWidget::item {{ padding:7px 8px; border-radius:5px; color:{LABEL}; }}
            QListWidget::item:hover {{ background:{INPUT}; }}
            QListWidget::item:selected {{ background:{INPUT}; color:{ACCENT}; }}
            QLineEdit, QComboBox {{ background:{INPUT}; border:1px solid {BORDER}; border-radius:7px;
                                    padding:7px 9px; color:{TEXT}; selection-background-color:{ACCENT}; }}
            QLineEdit:focus, QComboBox:focus {{ border-color:{ACCENT}; }}
            QComboBox:disabled {{ color:#727c89; background:#191c22; border-color:#2a3039; }}
            QComboBox::drop-down {{ border:none; width:22px; }}
            QPushButton {{ background:{INPUT}; border:1px solid {BORDER}; border-radius:7px;
                           padding:8px 14px; color:{TEXT}; }}
            QPushButton[compact="true"] {{ padding-left:7px; padding-right:7px; }}
            QPushButton:hover {{ background:#2e333d; border-color:#465063; }}
            QPushButton:disabled {{ color:#59616c; background:#181b21; border-color:#242a33; }}
            QPushButton#primary {{ background:{ACCENT}; border-color:{ACCENT}; color:{ON_ACCENT};
                                   font-weight:800; padding-left:20px; padding-right:20px; }}
            QPushButton#primary:hover {{ background:{ACCENT_H}; border-color:{ACCENT_H}; }}
            QPushButton#primary:disabled {{ background:#4a2c24; color:#8a6f66; border-color:#4a2c24; }}
            QToolButton#moreButton {{ background:{INPUT}; border:1px solid {BORDER};
                                      border-radius:7px; color:{TEXT}; font-size:17px;
                                      min-width:34px; min-height:30px; }}
            QToolButton#moreButton:hover {{ background:#2e333d; border-color:#465063; }}
            QToolButton#moreButton::menu-indicator {{ image:none; }}
            QMenu {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:7px;
                     padding:5px; color:{TEXT}; }}
            QMenu::item {{ padding:7px 28px 7px 10px; border-radius:5px; }}
            QMenu::item:selected {{ background:{INPUT}; color:{ACCENT}; }}
            QMenu::separator {{ height:1px; background:{BORDER}; margin:5px 7px; }}
            QPlainTextEdit {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:8px;
                              color:#c3ccd8; padding:7px; font-family:monospace; font-size:11px; }}
            QProgressBar {{ background:{INPUT}; border:1px solid {BORDER}; border-radius:7px;
                            text-align:center; height:20px; color:{TEXT}; }}
            QProgressBar::chunk {{ background:{ACCENT}; border-radius:6px; }}
            QSlider::groove:horizontal {{ background:{BG}; border:1px solid {BORDER};
                                          height:6px; border-radius:3px; }}
            QSlider::sub-page:horizontal {{ background:{ACCENT}; border-radius:3px; }}
            QSlider::handle:horizontal {{ background:{TEXT}; border:1px solid {BORDER};
                                          width:14px; margin:-5px 0; border-radius:7px; }}
            QSlider::groove:horizontal:disabled,
            QSlider::sub-page:horizontal:disabled {{ background:#20242b; }}
            QSlider::handle:horizontal:disabled {{ background:#3a404a; }}
            QCheckBox {{ spacing:8px; color:{LABEL}; }}
            QCheckBox::indicator {{ width:17px; height:17px; border:1px solid {BORDER};
                                    border-radius:4px; background:{INPUT}; }}
            QCheckBox::indicator:checked {{ background:{ACCENT}; border-color:{ACCENT}; }}
            QSplitter::handle {{ background:{BG}; width:8px; }}
            QScrollBar:vertical {{ background:{BG}; width:10px; border:none; }}
            QScrollBar::handle:vertical {{ background:#343b47; border-radius:5px; min-height:28px; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ height:0; }}
            """
        )


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(VERSION)
    app.setOrganizationName("Long Weekend Labs")
    app.setDesktopFileName("swapio")
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    app.setFont(QFont("Inter", 10))
    if "--self-test-swap" in sys.argv:
        from package_diagnostics import self_test_swap

        sys.exit(self_test_swap(sys.argv[1:], TaskWorker))
    if "--self-test-output" in sys.argv:
        import argparse

        parser = argparse.ArgumentParser(description="Check a saved PNG and desktop folder notification.")
        parser.add_argument("--self-test-output", type=Path, required=True, metavar="FOLDER")
        args = parser.parse_args()
        output = core.unique_output_path(args.self_test_output.resolve(), Path("swapio-output-check.png"), suffix=".png")
        core.write_image(output, np.full((32, 32, 3), (60, 122, 255), np.uint8))
        notified = notify_output_folder(output.parent)
        print(json.dumps({"version": VERSION, "output": str(output), "desktop_notified": notified}), flush=True)
        QTimer.singleShot(100, app.quit)
        app.exec()
        sys.exit(0 if notified else 1)
    window = MainWindow()
    window.show()
    if core.missing_models():
        QTimer.singleShot(450, window._show_model_setup)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
