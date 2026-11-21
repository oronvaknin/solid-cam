"""Dark 'lime' theme inspired by the Limelight web UI."""

BG = "#121417"
PANEL = "#1a1d21"
PANEL_2 = "#21252b"
BORDER = "#2c3138"
TEXT = "#e7eaee"
MUTED = "#8c95a1"
ACCENT = "#9be22d"
ACCENT_DIM = "#5f8f1a"
WARN = "#f5b83d"
BAD = "#ff5d5d"
INFO = "#4cc3ff"
BLUE_ALLIANCE = "#2f7bff"
RED_ALLIANCE = "#ff4040"

STYLESHEET = f"""
* {{ font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 10pt; color: {TEXT}; }}
QMainWindow, QWidget#Root {{ background: {BG}; }}
QWidget {{ background: transparent; }}
QToolTip {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}

QFrame#Header {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1b2116, stop:0.35 {PANEL}, stop:1 {PANEL});
                 border-bottom: 1px solid {BORDER}; }}
QLabel#Logo {{ font-size: 13.5pt; font-weight: 800; letter-spacing: 2px; color: {TEXT}; }}
QLabel#LogoSub {{ color: {MUTED}; font-size: 8.5pt; }}
QLabel#LogoMark {{ border-radius: 8px;
                   background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #c6f36b, stop:1 #5fae12);
                   border: 2px solid #0d1003; }}

QFrame#Sidebar {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
QLabel#NavSection {{ color: #5d6672; font-size: 8pt; font-weight: 700; letter-spacing: 1.5px; padding: 6px 18px 2px 18px; }}
QPushButton#NavButton {{
    text-align: left; padding: 10px 14px; margin: 0 8px; border: none; border-radius: 8px;
    color: {MUTED}; font-size: 10.5pt; font-weight: 600; background: transparent;
}}
QPushButton#NavButton:hover {{ color: {TEXT}; background: {PANEL_2}; }}
QPushButton#NavButton:checked {{ color: {ACCENT}; background: #22301a; }}

QWidget#SegBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QPushButton#Seg {{ background: transparent; border: 1px solid {BORDER}; border-radius: 12px; padding: 3px 12px;
                   color: {MUTED}; font-size: 9pt; font-weight: 600; }}
QPushButton#Seg:hover {{ color: {TEXT}; }}
QPushButton#Seg:checked {{ background: {ACCENT}; color: #0d1003; border-color: {ACCENT}; }}
QPushButton#Overlay {{ background: rgba(20, 24, 28, 210); border: 1px solid {BORDER}; border-radius: 6px;
                       padding: 4px 10px; font-size: 9pt; font-weight: 600; }}
QPushButton#Overlay:hover {{ border-color: {ACCENT}; }}
QPushButton#Overlay:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}
QDialog {{ background: {PANEL}; }}

QFrame#Panel {{ background: {PANEL}; border-left: 1px solid {BORDER}; }}
QFrame#Card {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 8px; }}
QLabel#CardTitle {{ font-weight: 700; font-size: 10.5pt; color: {TEXT}; }}
QLabel#Muted {{ color: {MUTED}; }}
QLabel#Hint {{ color: {MUTED}; font-size: 9pt; }}
QLabel#Big {{ font-size: 20pt; font-weight: 800; }}
QLabel#Value {{ font-family: 'Consolas', 'Cascadia Mono', monospace; font-size: 10pt; }}

QLabel#Chip {{
    background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 11px;
    padding: 3px 10px; font-size: 9pt; color: {TEXT};
}}

QPushButton {{
    background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 12px;
}}
QPushButton:hover {{ border-color: {ACCENT_DIM}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: #59606a; border-color: #262a30; }}
QPushButton#Primary {{ background: {ACCENT}; color: #0d1003; border: none; font-weight: 700; }}
QPushButton#Primary:hover {{ background: #b2f24d; }}
QPushButton#Primary:disabled {{ background: #3a4a22; color: #6d7a5a; }}
QPushButton#Danger {{ color: {BAD}; }}
QPushButton:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {BG}; border: 1px solid {BORDER}; border-radius: 5px; padding: 4px 6px; min-height: 20px;
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL_2}; border: 1px solid {BORDER}; selection-background-color: {ACCENT_DIM}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {BORDER}; background: {BG}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}

QProgressBar {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 5px; text-align: center; height: 16px; font-size: 8.5pt; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 8px 14px; border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QTableWidget {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 6px; gridline-color: {BORDER}; }}
QHeaderView::section {{ background: {PANEL_2}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER}; padding: 4px; font-weight: 600; }}
QTableWidget::item {{ padding: 2px; }}

QListWidget {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 6px; }}
QListWidget::item {{ color: {TEXT}; border-radius: 6px; padding: 2px; }}
QListWidget::item:selected {{ background: {ACCENT_DIM}; }}

QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 30px; }}
QSplitter::handle {{ background: {BORDER}; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
"""
