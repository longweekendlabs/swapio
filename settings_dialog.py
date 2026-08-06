"""Application preferences and core-model status."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import core


class SettingsDialog(QDialog):
    """One stable home for processing preferences and model status."""

    def __init__(self, state: dict, save_state, parent=None):
        super().__init__(parent)
        self.state = state
        self.save_state = save_state
        self.setWindowTitle("Swapio Settings")
        self.setMinimumSize(680, 500)
        self._build()

    @staticmethod
    def _gpu_description() -> tuple[bool, str]:
        try:
            import onnxruntime as ort

            if "CUDAExecutionProvider" in ort.get_available_providers():
                return True, "NVIDIA CUDA provider available"
        except Exception:
            pass
        return False, "CUDA provider unavailable; CPU fallback will be used"

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        tabs = QTabWidget()
        tabs.addTab(self._settings_tab(), "Settings")
        tabs.addTab(self._models_tab(), "Model Management")
        layout.addWidget(tabs)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        layout.addLayout(row)

    def _settings_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(14, 16, 14, 14)
        title = QLabel("Processing")
        title.setObjectName("aboutName")
        layout.addWidget(title)
        self.prefer_gpu = QCheckBox("Prefer NVIDIA GPU acceleration when available")
        self.prefer_gpu.setChecked(bool(self.state.get("prefer_gpu", True)))
        self.prefer_gpu.stateChanged.connect(self._save_preferences)
        layout.addWidget(self.prefer_gpu)
        gpu_ok, gpu_name = self._gpu_description()
        gpu = QLabel(("✓  " if gpu_ok else "○  ") + gpu_name)
        gpu.setObjectName("aboutDescription")
        layout.addWidget(gpu)
        note = QLabel(
            "Changing this preference takes effect after restarting Swapio. "
            "Careful mode automatically uses 768px or 1024px processing for close-up faces."
        )
        note.setWordWrap(True)
        note.setObjectName("aboutFinePrint")
        layout.addWidget(note)
        layout.addStretch(1)
        return tab

    def _models_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(14, 16, 14, 14)
        title = QLabel("Core face-swap models")
        title.setObjectName("aboutName")
        layout.addWidget(title)
        location = QLabel(f"Stored in: {core.model_dir()}")
        location.setWordWrap(True)
        location.setObjectName("fileName")
        layout.addWidget(location)
        missing = core.missing_models()
        if missing:
            status = QLabel("○ Missing: " + ", ".join(missing))
            button_text = "Download missing core models…"
        else:
            status = QLabel("✓ Detector, identity encoder and both face swappers are installed")
            button_text = "Review core model setup…"
        status.setWordWrap(True)
        status.setObjectName("aboutDescription")
        layout.addWidget(status)
        explanation = QLabel(
            "Only the models required for face detection and swapping are managed here."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("aboutFinePrint")
        layout.addWidget(explanation)
        setup = QPushButton(button_text)
        setup.clicked.connect(self._open_core_setup)
        layout.addWidget(setup)
        layout.addStretch(1)
        return tab

    def _save_preferences(self, *_args) -> None:
        self.state["prefer_gpu"] = self.prefer_gpu.isChecked()
        self.save_state()

    def _open_core_setup(self) -> None:
        parent = self.parent()
        self.accept()
        if parent is not None and hasattr(parent, "_show_model_setup"):
            parent._show_model_setup()
