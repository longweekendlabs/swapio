"""First-run model download and verification UI for Swapio."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

import core
import setup_models

COMPONENTS = (
    "Face detector, mouth landmarks and identity encoder",
    "Fast face swapper",
    "High-quality face swapper",
    "Hair and skin parser",
)


class ModelInstallWorker(QThread):
    progress = Signal(str, str, int)
    completed = Signal()
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, destination: Path):
        super().__init__()
        self.destination = destination
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def _progress(self, component: str, message: str, done: int, total: int) -> None:
        percent = round(done * 100 / total) if total else -1
        if done and total and message == "Downloading":
            message = (
                f"Downloading — {done / 1024 / 1024:.0f} / "
                f"{total / 1024 / 1024:.0f} MB"
            )
        self.progress.emit(component, message, percent)

    def run(self) -> None:
        try:
            setup_models.install_all(
                self.destination,
                on_progress=self._progress,
                should_stop=lambda: self._stop,
            )
            if self._stop:
                self.cancelled.emit()
            else:
                self.completed.emit()
        except setup_models.DownloadCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ModelSetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.destination = core.model_dir()
        self.worker: ModelInstallWorker | None = None
        self.labels: dict[str, QLabel] = {}
        self.setWindowTitle("Set up Swapio models")
        self.setMinimumWidth(570)
        self.setModal(True)
        self._build()
        self._show_existing_state()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(12)

        title = QLabel("Download offline processing models")
        title.setObjectName("aboutName")
        layout.addWidget(title)

        explanation = QLabel(
            "Swapio needs approximately 1.2 GB of pretrained models. They are "
            "downloaded once from their original publishers, verified, and then "
            "used completely offline. Existing verified files are skipped."
        )
        explanation.setObjectName("aboutDescription")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        location = QLabel(f"Saved to: {self.destination}")
        location.setObjectName("fileName")
        location.setWordWrap(True)
        layout.addWidget(location)

        for component in COMPONENTS:
            label = QLabel(f"○  {component} — waiting")
            label.setObjectName("modelComponent")
            self.labels[component] = label
            layout.addWidget(label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Ready to download")
        layout.addWidget(self.progress)

        self.status = QLabel(
            "Model licenses are separate from Swapio and can restrict commercial use."
        )
        self.status.setObjectName("aboutFinePrint")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.acknowledge = QCheckBox(
            "I understand the pretrained model licenses and usage restrictions"
        )
        self.acknowledge.stateChanged.connect(self._refresh_download_button)
        layout.addWidget(self.acknowledge)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_button = QPushButton("Close")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.cancel_button)
        self.download_button = QPushButton("Download missing models")
        self.download_button.setObjectName("primary")
        self.download_button.clicked.connect(self._start)
        buttons.addWidget(self.download_button)
        layout.addLayout(buttons)

    def _show_existing_state(self) -> None:
        missing = set(core.missing_models(self.destination))
        mapping = {
            "Face detector, mouth landmarks and identity encoder": {
                "buffalo_l/2d106det.onnx",
                "buffalo_l/det_10g.onnx",
                "buffalo_l/w600k_r50.onnx",
            },
            "Fast face swapper": {core.SWAPPER_MODEL},
            "High-quality face swapper": {core.HYPERSWAP_MODEL},
            "Hair and skin parser": {core.FACE_PARSER_MODEL},
        }
        for component, files in mapping.items():
            ready = not bool(files & missing)
            state = "✓  Installed" if ready else "○  Not installed"
            self.labels[component].setText(f"{state} — {component}")
        if not missing:
            self.progress.setValue(100)
            self.progress.setFormat("All models are installed")
            self.status.setText("Swapio is ready for fully offline use.")
            self.acknowledge.hide()
            self.download_button.hide()
        else:
            self._refresh_download_button()

    def _refresh_download_button(self) -> None:
        running = bool(self.worker and self.worker.isRunning())
        self.download_button.setEnabled(self.acknowledge.isChecked() and not running)

    def _start(self) -> None:
        self.worker = ModelInstallWorker(self.destination)
        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._completed)
        self.worker.cancelled.connect(self._cancelled)
        self.worker.failed.connect(self._failed)
        self.download_button.setEnabled(False)
        self.acknowledge.setEnabled(False)
        self.cancel_button.setText("Cancel download")
        self.status.setText("Downloading models. You can cancel and resume later.")
        self.worker.start()

    def _on_progress(self, component: str, message: str, percent: int) -> None:
        self.labels[component].setText(f"●  {component} — {message}")
        if percent < 0:
            self.progress.setRange(0, 0)
            self.progress.setFormat(message)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(percent)
            self.progress.setFormat(f"{component}: {percent}%")

    def _completed(self) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.progress.setFormat("Models ready — offline mode enabled")
        self.status.setText("Download complete. Every required model is installed and verified.")
        self.cancel_button.setText("Continue")
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.accept)
        self._show_existing_state()

    def _cancelled(self) -> None:
        self.status.setText("Download cancelled. Already completed files were kept; retry any time.")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Cancelled")
        self.cancel_button.setText("Close")
        self.acknowledge.setEnabled(True)
        self.worker = None
        self._show_existing_state()

    def _failed(self, message: str) -> None:
        self.status.setText(f"Download failed: {message}\nCheck your connection, then retry.")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Download failed")
        self.cancel_button.setText("Close")
        self.acknowledge.setEnabled(True)
        self.worker = None
        self._refresh_download_button()

    def reject(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling after the current download chunk…")
            return
        super().reject()
