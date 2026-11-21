"""FRC Vision Lab - run with:  python main.py"""

import os
import sys

# Faster camera open on Windows (skip the slow Media Foundation hardware-transform probe)
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

from PySide6.QtWidgets import QApplication  # noqa: E402

from frcvision.ui.main_window import MainWindow  # noqa: E402
from frcvision.ui.theme import STYLESHEET  # noqa: E402


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FRC Vision Lab")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
