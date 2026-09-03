#!/usr/bin/env python3
"""Swapio - private, offline batch face swapping for still photos."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QFont, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import core
from about_dialog import AboutDialog
from model_setup_dialog import ModelSetupDialog
from settings_dialog import SettingsDialog
from version import APP_NAME, GITHUB_URL, VERSION

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


def native_process_environment() -> dict[str, str]:
    """Return an environment safe for launching system desktop utilities.

    PyInstaller prepends the app bundle to ``LD_LIBRARY_PATH``. That is needed
    by Swapio itself, but it makes KDE tools load Swapio's private Qt libraries
    instead of the matching system Qt build.
    """
    environment = os.environ.copy()
    original_library_path = environment.get("LD_LIBRARY_PATH_ORIG")
    if original_library_path:
        environment["LD_LIBRARY_PATH"] = original_library_path
    elif getattr(sys, "frozen", False):
        environment.pop("LD_LIBRARY_PATH", None)

    if getattr(sys, "frozen", False):
        for variable in (
            "QT_PLUGIN_PATH",
            "QML2_IMPORT_PATH",
            "QML_IMPORT_PATH",
            "QT_QPA_PLATFORM_PLUGIN_PATH",
            "QT_QPA_FONTDIR",
        ):
            original = environment.get(f"{variable}_ORIG")
            if original is not None:
                environment[variable] = original
            else:
                environment.pop(variable, None)
    return environment


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


def native_image_files(parent, title: str, start: str, multiple: bool = False) -> list[str]:
    """Open the desktop-native image picker, with a Qt fallback."""
    if shutil.which("kdialog"):
        command = ["kdialog", "--geometry", DIALOG_GEOMETRY, "--title", title]
        if multiple:
            command += ["--multiple", "--separate-output"]
        command += ["--getopenfilename", start, IMAGE_FILTER]
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
            "Images | *.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff",
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
            parent, title, start, "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)"
        )
        return paths
    path, _ = QFileDialog.getOpenFileName(
        parent, title, start, "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)"
    )
    return [path] if path else []


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


class TaskWorker(QThread):
    log = Signal(str)
    progress = Signal(int, int, dict)
    preview_ready = Signal(object)
    batch_ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, engine: core.SwapEngine, mode: str, **kwargs):
        super().__init__()
        self.engine = engine
        self.mode = mode
        self.kwargs = kwargs
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            if self.mode == "preview":
                result = self.engine.preview(on_log=self.log.emit, **self.kwargs)
                self.preview_ready.emit(result)
            else:
                result = self.engine.batch(
                    on_log=self.log.emit,
                    on_progress=self.progress.emit,
                    should_stop=lambda: self._stop,
                    **self.kwargs,
                )
                self.batch_ready.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{VERSION} — Offline Batch Face Swap")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.resize(1180, 820)
        self.setMinimumSize(930, 650)
        self.source_path: Path | None = None
        self.targets: list[Path] = []
        self.worker: TaskWorker | None = None
        self.last_output: Path | None = None
        self._state = read_state()
        self.engine = core.SwapEngine(use_gpu=bool(self._state.get("prefer_gpu", True)))
        self._restoring_state = True

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(12)

        root.addLayout(self._build_header())
        root.addWidget(self._build_workflow(), stretch=1)
        root.addWidget(self._build_options())
        root.addLayout(self._build_actions())
        root.addWidget(self.progress)
        root.addWidget(self.log, stretch=0)

        self._apply_style()
        self._restore_state()
        self._restoring_state = False
        self._refresh_model_status()
        self._refresh_controls()

    def _build_header(self):
        row = QHBoxLayout()
        titles = QVBoxLayout()
        brand_row = QHBoxLayout()
        brand_row.setSpacing(9)
        brand = QLabel(APP_NAME)
        brand.setObjectName("brand")
        brand_row.addWidget(brand)
        version = QLabel(f"v{VERSION}")
        version.setObjectName("versionBadge")
        brand_row.addWidget(version, alignment=Qt.AlignVCenter)
        brand_row.addStretch(1)
        titles.addLayout(brand_row)
        subtitle = QLabel("One source face. A whole folder of photos. Private and offline.")
        subtitle.setObjectName("pageSub")
        titles.addWidget(subtitle)
        credit = QLabel('Made with <span style="color:#ff7a5c">♥</span> by Long Weekend Labs')
        credit.setTextFormat(Qt.RichText)
        credit.setObjectName("madeWithLove")
        titles.addWidget(credit)
        row.addLayout(titles)
        row.addStretch(1)
        self.model_status = QPushButton()
        self.model_status.setObjectName("statusPill")
        self.model_status.clicked.connect(self._show_model_setup)
        row.addWidget(self.model_status, alignment=Qt.AlignTop)
        row.addWidget(self._build_more_menu(), alignment=Qt.AlignTop)
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
        source_footer.setFixedHeight(96)
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
        source_button = QPushButton("Choose source photo…")
        source_button.clicked.connect(self._choose_source)
        source_footer_layout.addWidget(source_button)
        source_layout.addWidget(source_footer)

        destinations_box = QGroupBox("2 — Destination photos")
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
        destinations_footer.setFixedHeight(96)
        destinations_footer_layout = QVBoxLayout(destinations_footer)
        destinations_footer_layout.setContentsMargins(0, 0, 0, 0)
        destinations_footer_layout.setSpacing(6)
        self.target_count = QLabel("No destination photos")
        self.target_count.setObjectName("fileName")
        destinations_footer_layout.addWidget(self.target_count)
        destinations_footer_layout.addStretch(1)
        target_buttons = QHBoxLayout()
        target_buttons.setSpacing(8)
        add_files = QPushButton("Add photos…")
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
        destinations_footer_layout.addLayout(target_buttons)
        destinations_layout.addWidget(destinations_footer)

        preview_box = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.setSpacing(8)
        self.preview_view = ImageView("Select a destination, then click Preview")
        preview_layout.addWidget(self.preview_view, stretch=1)
        preview_footer = QWidget()
        preview_footer.setObjectName("cardFooter")
        preview_footer.setFixedHeight(96)
        preview_footer_layout = QVBoxLayout(preview_footer)
        preview_footer_layout.setContentsMargins(0, 0, 0, 0)
        preview_footer_layout.setSpacing(6)
        self.preview_caption = QLabel("Nothing generated yet")
        self.preview_caption.setObjectName("fileName")
        self.preview_caption.setWordWrap(True)
        preview_footer_layout.addWidget(self.preview_caption)
        preview_footer_layout.addStretch(1)
        preview_layout.addWidget(preview_footer)

        splitter.addWidget(source_box)
        splitter.addWidget(destinations_box)
        splitter.addWidget(preview_box)
        splitter.setSizes([330, 350, 330])
        return splitter

    def _build_options(self):
        box = QGroupBox("3 — Output and quality")
        grid = QGridLayout(box)
        grid.setColumnStretch(1, 1)
        output_label = QLabel("Output folder")
        output_label.setObjectName("fieldLabel")
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Swapped photos are written here; originals are never changed")
        self.output_edit.textChanged.connect(self._on_output_changed)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._choose_output)
        grid.addWidget(output_label, 0, 0)
        grid.addWidget(self.output_edit, 0, 1)
        grid.addWidget(browse, 0, 2)

        faces_label = QLabel("When several faces are found")
        faces_label.setObjectName("fieldLabel")
        self.face_mode = QComboBox()
        self.face_mode.addItems(["Swap the largest face only", "Swap every detected face"])
        self.face_mode.currentIndexChanged.connect(self._save_state)
        self.permission = QCheckBox("I have permission to modify the selected photos")
        self.permission.stateChanged.connect(self._refresh_controls)
        grid.addWidget(faces_label, 1, 0)
        grid.addWidget(self.face_mode, 1, 1)
        grid.addWidget(self.permission, 1, 2)

        quality_label = QLabel("Processing quality")
        quality_label.setObjectName("fieldLabel")
        self.quality_mode = QComboBox()
        self.quality_mode.addItem(
            "Best — Careful plus face restoration (sharpest eyes and teeth, slowest)",
            "best",
        )
        self.quality_mode.addItem(
            "Careful — adaptive HyperSwap 512–1024 (more detail, slower)", "careful"
        )
        self.quality_mode.addItem("Balanced — HyperSwap 256", "balanced")
        self.quality_mode.addItem("Fast — InSwapper 128 (draft quality)", "fast")
        self.quality_mode.currentIndexChanged.connect(self._save_state)
        self.output_format = QComboBox()
        self.output_format.addItem("Lossless PNG — unchanged pixels stay exact", "png")
        self.output_format.addItem("JPEG 98 — smaller, re-encodes the image", "jpg")
        self.output_format.currentIndexChanged.connect(self._save_state)
        grid.addWidget(quality_label, 2, 0)
        grid.addWidget(self.quality_mode, 2, 1)
        grid.addWidget(self.output_format, 2, 2)
        mouth_label = QLabel("Open-mouth smiles")
        mouth_label.setObjectName("fieldLabel")
        self.preserve_mouth = QCheckBox("Preserve target inner mouth and teeth")
        self.preserve_mouth.setChecked(True)
        self.preserve_mouth.setToolTip(
            "Keeps only teeth, tongue, and the inside of an open mouth; lips and the surrounding face remain swapped."
        )
        self.preserve_mouth.stateChanged.connect(self._save_state)
        grid.addWidget(mouth_label, 3, 0)
        grid.addWidget(self.preserve_mouth, 3, 1)
        eyes_label = QLabel("Eyeballs")
        eyes_label.setObjectName("fieldLabel")
        self.destination_eyes = QCheckBox("Keep the destination's eyes")
        self.destination_eyes.setChecked(True)
        self.destination_eyes.setToolTip(
            "The swapper redraws each eye at 256 pixels, which is where the hard "
            "white glare and the flat iris come from. This puts the destination "
            "photo's real eyeballs back, moved onto the swapped eyelids so they "
            "line up. Lashes, liner and lids stay swapped; eye colour becomes "
            "the destination's."
        )
        self.destination_eyes.stateChanged.connect(self._save_state)
        grid.addWidget(eyes_label, 4, 0)
        grid.addWidget(self.destination_eyes, 4, 1)

        strength_label = QLabel("Restoration strength")
        strength_label.setObjectName("fieldLabel")
        self.restoration_strength = QSlider(Qt.Horizontal)
        self.restoration_strength.setRange(0, 100)
        self.restoration_strength.setValue(
            int(core.DEFAULT_RESTORATION_STRENGTH * 100)
        )
        self.restoration_strength.setToolTip(
            "Best quality only. How much of the restored face replaces the "
            "swapped one. High values repaint skin and eyes hard enough to look "
            "painted; 0 leaves the swap exactly as Careful produced it."
        )
        self.restoration_value = QLabel()
        self.restoration_value.setObjectName("fileName")
        self.restoration_strength.valueChanged.connect(self._restoration_changed)
        self.restoration_strength.sliderReleased.connect(self._save_state)
        grid.addWidget(strength_label, 5, 0)
        grid.addWidget(self.restoration_strength, 5, 1)
        grid.addWidget(self.restoration_value, 5, 2)
        self._restoration_changed(self.restoration_strength.value())
        self.skip_completed = QCheckBox("Skip unchanged photos already completed")
        self.skip_completed.setChecked(True)
        self.skip_completed.setToolTip(
            "Swapio records successful outputs in the output folder and only processes new or changed photos."
        )
        self.skip_completed.stateChanged.connect(self._save_state)
        grid.addWidget(self.skip_completed, 3, 2)

        return box

    def _build_actions(self):
        row = QHBoxLayout()
        self.preview_button = QPushButton("Preview selected photo")
        self.preview_button.clicked.connect(self._preview)
        self.run_button = QPushButton("Swap all photos")
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
        row.addStretch(1)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusText")
        row.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setFormat("Ready")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(112)
        self.log.setPlaceholderText("Processing details will appear here.")
        return row

    def _dialog_start(self, key: str) -> str:
        value = self._state.get(key, "")
        path = Path(value).expanduser() if value else Path.home()
        if not path.is_absolute():
            path = (PROJECT_DIR / path).resolve()
        return str(path if path.exists() else Path.home())

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
        paths = native_image_files(
            self, "Add destination photos", self._dialog_start("target_dir"), multiple=True
        )
        if paths:
            self._add_targets(Path(path) for path in paths)
            self._remember_folder("target_dir", Path(paths[0]).parent)

    def _add_folder(self):
        path = native_directory(
            self, "Add a folder of destination photos", self._dialog_start("target_dir")
        )
        if path:
            folder = Path(path)
            self._add_targets(core.image_files(folder, recursive=True))
            self.output_edit.setText(str(core.suggested_output_dir(folder)))
            self._remember_folder("target_dir", folder)

    def _add_targets(self, paths):
        existing = {str(path.resolve()) for path in self.targets}
        added = 0
        for path in paths:
            path = Path(path)
            try:
                resolved = str(path.resolve())
            except OSError:
                continue
            if path.is_file() and path.suffix.lower() in core.IMAGE_EXTS and resolved not in existing:
                self.targets.append(path)
                item = QListWidgetItem(path.name)
                item.setToolTip(str(path))
                item.setData(Qt.UserRole, str(path))
                self.target_list.addItem(item)
                existing.add(resolved)
                added += 1
        if added and self.target_list.currentRow() < 0:
            self.target_list.setCurrentRow(0)
        if added and not self.output_edit.text().strip():
            self.output_edit.setText(str(self._default_output_dir()))
        self._refresh_target_count()
        self._refresh_controls()

    def _default_output_dir(self) -> Path:
        """Choose a sibling folder so recursive imports never ingest outputs."""
        if not self.targets:
            return Path.home() / "Swapio Output"
        return core.suggested_output_dir(self.targets[0].parent)

    def _remove_selected(self):
        row = self.target_list.currentRow()
        if row < 0:
            return
        self.targets.pop(row)
        self.target_list.takeItem(row)
        self._refresh_target_count()
        self._refresh_controls()

    def _clear_targets(self):
        self.targets.clear()
        self.target_list.clear()
        self.preview_view.clear_image()
        self.preview_caption.setText("Nothing generated yet")
        self._refresh_target_count()
        self._refresh_controls()

    def _refresh_target_count(self):
        count = len(self.targets)
        self.target_count.setText(
            "No destination photos" if not count else f"{count} destination photo{'s' if count != 1 else ''}"
        )

    def _show_selected_target(self, row: int):
        if row < 0 or row >= len(self.targets):
            return
        if self.worker and self.worker.isRunning():
            return
        self.preview_view.clear_image("Click Preview to generate this result")
        self.preview_caption.setText(self.targets[row].name)
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
        return self.face_mode.currentIndex() == 1

    def _restoration_changed(self, value: int) -> None:
        self.restoration_value.setText(f"{value}%" if value else "off")

    def _restoration_strength(self) -> float:
        return self.restoration_strength.value() / 100.0

    def _quality(self) -> str:
        return self.quality_mode.currentData()

    def _output_format(self) -> str:
        return self.output_format.currentData()

    def _selected_target(self) -> Path | None:
        row = self.target_list.currentRow()
        return self.targets[row] if 0 <= row < len(self.targets) else None

    def _preview(self):
        target = self._selected_target()
        if not self.source_path or not target:
            return
        self.log.clear()
        self.log.appendPlainText(f"Previewing {target.name}...")
        self.progress.setRange(0, 0)
        self.progress.setFormat("Generating preview…")
        self.worker = TaskWorker(
            self.engine,
            "preview",
            source_path=self.source_path,
            target_path=target,
            all_faces=self._all_faces(),
            quality=self._quality(),
            preserve_mouth=self.preserve_mouth.isChecked(),
            destination_eyes=self.destination_eyes.isChecked(),
            restoration_strength=self._restoration_strength(),
        )
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.preview_ready.connect(self._preview_ready)
        self.worker.failed.connect(self._failed)
        self._set_running(True)
        self.status_label.setText("Generating preview…")
        self.worker.start()

    def _preview_ready(self, result: dict):
        self._set_running(False)
        self.preview_view.set_image(result["image"])
        target = Path(result["target"]).name
        self.preview_caption.setText(
            f"{target} — swapped {result['swapped']} face(s) · "
            f"{result['quality'].title()} · {result['provider']}"
        )
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setFormat("Preview ready")
        self.status_label.setText("Preview ready")
        self._refresh_model_status()

    def _run_batch(self):
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
        self.log.clear()
        self.progress.setRange(0, len(self.targets))
        self.progress.setValue(0)
        self.progress.setFormat(f"0/{len(self.targets)}")
        self.worker = TaskWorker(
            self.engine,
            "batch",
            source_path=self.source_path,
            targets=list(self.targets),
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
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.progress.connect(self._progress)
        self.worker.batch_ready.connect(self._batch_ready)
        self.worker.failed.connect(self._failed)
        self._set_running(True)
        self.status_label.setText("Swapping photos…")
        self.worker.start()

    def _progress(self, done: int, total: int, counts: dict):
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.progress.setFormat(
            f"{done}/{total} — {counts.get('completed', 0)} saved, "
            f"{counts.get('skipped', 0)} unchanged, {counts.get('failed', 0)} failed"
        )

    def _batch_ready(self, result: dict):
        self._set_running(False)
        self.open_button.setEnabled(bool(result["outputs"]) or result["skipped"] > 0)
        label = "Stopped" if result["stopped"] else "Complete"
        self.status_label.setText(
            f"{label} — {result['completed']} saved, {result['skipped']} unchanged, "
            f"{result['failed']} failed"
        )
        self.progress.setFormat(
            f"{result['completed']} saved — {result['skipped']} unchanged — "
            f"{result['failed']} failed — originals untouched"
        )
        self._refresh_model_status()
        if not result["stopped"]:
            QMessageBox.information(
                self,
                "Batch complete",
                f"Saved {result['completed']} swapped photo(s).\n"
                f"Skipped {result['skipped']} unchanged photo(s).\n"
                f"Failed {result['failed']}.\n\n"
                f"Output: {self.last_output}",
            )

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Stopping after this photo…")

    def _failed(self, message: str):
        self._set_running(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Error")
        self.status_label.setText("Error")
        self.log.appendPlainText("ERROR: " + message)
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
        self.preview_button.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.target_list.setEnabled(not running)
        if not running:
            self._refresh_controls()

    def _refresh_controls(self):
        running = bool(self.worker and self.worker.isRunning())
        ready = (
            not running
            and self.source_path is not None
            and bool(self.targets)
            and self.permission.isChecked()
            and not core.missing_models()
        )
        self.preview_button.setEnabled(ready and self._selected_target() is not None)
        self.run_button.setEnabled(ready)
        self.stop_button.setEnabled(running)

    def _refresh_model_status(self):
        missing = core.missing_models()
        if missing:
            self.model_status.setText("↓ Download models · ~1.2 GB")
            self.model_status.setProperty("state", "warning")
            self.model_status.setToolTip("Missing: " + ", ".join(missing) + "\nClick to set up.")
        elif self.engine.provider != "Not loaded":
            self.model_status.setText(f"● Offline · {self.engine.provider}")
            self.model_status.setProperty("state", "ready")
            self.model_status.setToolTip(str(core.model_dir()))
        else:
            self.model_status.setText("● Offline models ready")
            self.model_status.setProperty("state", "ready")
            self.model_status.setToolTip(str(core.model_dir()))
        self.model_status.style().unpolish(self.model_status)
        self.model_status.style().polish(self.model_status)

    def _restore_state(self):
        output = self._state.get("output_dir", "")
        output_path = Path(output).expanduser() if output else None
        if output_path and output_path.is_dir():
            self.output_edit.setText(str(output_path.resolve()))
        self.face_mode.setCurrentIndex(int(self._state.get("face_mode", 0)))
        quality = self._state.get("quality", "careful")
        quality_index = self.quality_mode.findData(quality)
        self.quality_mode.setCurrentIndex(max(quality_index, 0))
        output_format = self._state.get("output_format", "png")
        format_index = self.output_format.findData(output_format)
        self.output_format.setCurrentIndex(max(format_index, 0))
        self.preserve_mouth.setChecked(bool(self._state.get("preserve_mouth", True)))
        self.destination_eyes.setChecked(bool(self._state.get("destination_eyes", True)))
        self.restoration_strength.setValue(
            int(self._state.get(
                "restoration_strength", core.DEFAULT_RESTORATION_STRENGTH * 100
            ))
        )
        self.skip_completed.setChecked(bool(self._state.get("skip_completed", True)))

    def _save_state(self):
        if self._restoring_state:
            return
        self._state["output_dir"] = self.output_edit.text().strip()
        self._state["face_mode"] = self.face_mode.currentIndex()
        self._state["quality"] = self._quality()
        self._state["output_format"] = self._output_format()
        self._state["preserve_mouth"] = self.preserve_mouth.isChecked()
        self._state["destination_eyes"] = self.destination_eyes.isChecked()
        self._state["restoration_strength"] = self.restoration_strength.value()
        self._state["skip_completed"] = self.skip_completed.isChecked()
        for retired in (
            "hair_mode", "custom_hair_color", "hair_strength",
            "skin_match", "skin_strength", "body_skin",
        ):
            self._state.pop(retired, None)
        write_state(self._state)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            answer = QMessageBox.question(
                self,
                "Swap still running",
                "Stop after the current photo and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.stop()
            self.worker.wait()
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
    window = MainWindow()
    window.show()
    if core.missing_models():
        QTimer.singleShot(450, window._show_model_setup)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
