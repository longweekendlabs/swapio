"""About dialog for Swapio."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

import core
from media_tools import open_web_url
from version import APP_NAME, COPYRIGHT, GITHUB_URL, ORGANIZATION, VERSION, TAGLINE, WEBSITE_URL


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        self.setModal(True)
        self.setFixedWidth(470)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 22)
        layout.setSpacing(11)

        icon_path = core.base_dir() / "assets" / "swapio.svg"
        if icon_path.exists():
            logo = QLabel()
            logo.setPixmap(QIcon(str(icon_path)).pixmap(56, 56))
            logo.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo)

        name = QLabel(APP_NAME)
        name.setObjectName("aboutName")
        name.setAlignment(Qt.AlignCenter)
        layout.addWidget(name)

        version = QLabel(f"Version {VERSION} · {TAGLINE}")
        version.setWordWrap(True)
        version.setObjectName("aboutVersion")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)

        description = QLabel(
            "Replace one face across photos, short videos or GIFs while keeping the work "
            "private on your own computer. Originals are never modified."
        )
        description.setObjectName("aboutDescription")
        description.setAlignment(Qt.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(description)

        licence = QLabel('FREE & OPEN SOURCE · MIT')
        licence.setObjectName('aboutFinePrint')
        licence.setAlignment(Qt.AlignCenter)
        layout.addWidget(licence)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setObjectName("aboutDivider")
        layout.addWidget(divider)

        for url, title in (
            (WEBSITE_URL, 'longweekendlabs.app'),
            (GITHUB_URL, 'GitHub'),
            (GITHUB_URL+'/issues', 'Feedback'),
        ):
            link = QLabel(f'<a href="{url}">{title}</a>')
            link.setAlignment(Qt.AlignCenter)
            link.setTextInteractionFlags(Qt.TextBrowserInteraction)
            link.linkActivated.connect(self._open_link)
            layout.addWidget(link)

        licensing = QLabel(
            "Application code and pretrained face models have separate licenses. "
            "See the README before redistributing models or using them commercially."
        )
        licensing.setObjectName("aboutFinePrint")
        licensing.setAlignment(Qt.AlignCenter)
        licensing.setWordWrap(True)
        layout.addWidget(licensing)

        made = QLabel(
            f'Made with <span style="color:#ff7a5c">♥</span> by {ORGANIZATION}<br>'
            f'<span style="font-size:10px">{COPYRIGHT}</span>'
        )
        made.setTextFormat(Qt.RichText)
        made.setObjectName("madeWithLove")
        made.setAlignment(Qt.AlignCenter)
        layout.addWidget(made)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close = QPushButton("Close")
        close.setFixedWidth(110)
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    def _open_link(self, url):
        if not open_web_url(url):
            QMessageBox.warning(self, 'Open website', f'Could not open your browser.\n{url}')
