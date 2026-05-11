# Minimal stub — provides the names that CustomWidgets.py imports at module
# level.  The ToggleSwitch widget does not use either of these; they are only
# consumed by other widgets in that module.
from PyQt6.QtGui import QFont


def default_font(size: int = 10) -> QFont:
    """Return a plain application font at the given point size."""
    f = QFont()
    f.setPointSize(size)
    return f


class ThemeManager:
    """Stub theme manager — no-op for standalone use."""
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def current_theme(self) -> str:
        return "light"
