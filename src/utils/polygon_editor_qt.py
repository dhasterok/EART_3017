"""
polygon_editor_qt.py
--------------------
Qt actions for interactive polygon editing.

Provides
--------
EditMode
    Enum of the six polygon editing interaction modes.

PolygonEditorActions
    QObject that owns the QActions and the snap toggle.  Call
    ``add_to_toolbar(toolbar)`` to embed the whole set into any
    QToolBar; connect the ``mode_changed`` and ``snap_changed`` signals
    to your canvas / application logic.

Usage example
-------------
    icon_dir = "src/resources/icons"
    actions = PolygonEditorActions(icon_dir, parent=self)
    actions.mode_changed.connect(self._on_poly_mode)
    actions.snap_changed.connect(canvas.set_snap_enabled)
    actions.add_to_toolbar(my_toolbar)
"""

from enum import Enum, auto
from pathlib import Path

from PyQt6.QtCore import QObject, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QIcon, QPixmap, QPainter, QKeySequence
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QToolBar


# ── SVG icon helper (works without the Qt SVG image-format plugin) ─────────

def _svg_icon(path: str, size: int = 64) -> QIcon:
    renderer = QSvgRenderer(path)
    if not renderer.isValid():
        return QIcon()
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return QIcon(pm)


# ── editing mode enum ──────────────────────────────────────────────────────

class EditMode(Enum):
    """Polygon editing interaction modes."""
    DRAW          = auto()   # draw a new polygon vertex by vertex
    SELECT        = auto()   # select body; drag a single vertex
    MOVE_BODY     = auto()   # click-drag the whole polygon
    ADD_VERTEX    = auto()   # insert a vertex on an edge
    DELETE_VERTEX = auto()   # remove a single vertex (≥ 3 kept)
    DELETE        = auto()   # remove the whole polygon


# ── actions class ──────────────────────────────────────────────────────────

class PolygonEditorActions(QObject):
    """
    Qt QActions for interactive polygon editing.

    Create one instance per application, connect its signals to your
    canvas, and call :meth:`add_to_toolbar` to insert the action block
    into any ``QToolBar``.

    Signals
    -------
    mode_changed(EditMode)
        Emitted when the user activates a different editing mode via the
        toolbar.
    snap_changed(bool)
        Emitted when the snap toggle is switched.
    """

    mode_changed = pyqtSignal(object)   # EditMode value
    snap_changed = pyqtSignal(bool)

    # (label, mode, shortcut, tooltip, icon filename)
    _MODE_DEFS = [
        ("New\nPolygon",    EditMode.DRAW,          "Ctrl+N",
         "Draw a new polygon — add vertex, RMB/Enter close, Esc cancel",
         "icon-polygon-new-64.svg"),
        ("Move\nPolygon",   EditMode.MOVE_BODY,     "Ctrl+M",
         "Click and drag a whole polygon to a new location",
         "icon-polygon-select-64.svg"),
        ("Delete\nPolygon", EditMode.DELETE,        "Ctrl+D",
         "Click on a polygon to remove it entirely",
         "icon-delete-64.svg"),
        ("Add\nVertex",     EditMode.ADD_VERTEX,    "N",
         "Click on an edge to insert a vertex",
         "icon-add-point-64.svg"),
        ("Move\nVertex",    EditMode.SELECT,        "M",
         "Select body | drag a single vertex to move it",
         "icon-move-point-64.svg"),
        ("Delete\nVertex",  EditMode.DELETE_VERTEX, "D",
         "Click near a vertex to remove it (polygon keeps ≥ 3 vertices)",
         "icon-remove-point-64.svg"),
    ]

    def __init__(self, icon_dir: str, parent=None):
        """
        Parameters
        ----------
        icon_dir : str
            Directory that contains the SVG icon files.
        parent   : QObject, optional
        """
        super().__init__(parent)
        self._icon_dir     = str(icon_dir)
        self._snap_enabled = True
        self._act_snap: QAction | None = None

        # Non-exclusive group — we manage mutual exclusivity manually so that
        # clicking the active action can toggle it off (no mode active).
        self._action_group = QActionGroup(self)
        self._action_group.setExclusive(False)
        self._mode_actions: dict[EditMode, QAction] = {}

        self._build_mode_actions(parent)

    # ── private builders ──────────────────────────────────────────────────

    def _icon(self, filename: str) -> QIcon:
        return _svg_icon(str(Path(self._icon_dir) / filename))

    def _build_mode_actions(self, parent):
        for label, mode, key, tip, icon_file in self._MODE_DEFS:
            act = QAction(label, parent)
            act.setIcon(self._icon(icon_file))
            act.setCheckable(True)
            act.setToolTip(f"{tip}  [{key}]")
            act.setShortcut(QKeySequence(key))
            # triggered(checked) fires with the new check state after the click
            act.triggered.connect(
                lambda checked, m=mode, a=act: self._on_action_triggered(checked, m, a))
            self._action_group.addAction(act)
            self._mode_actions[mode] = act
        # No default selection — start with no mode active
        # (call set_mode externally if a default is wanted)

    def _on_action_triggered(self, checked: bool, mode: EditMode, source: QAction):
        """Handle a polygon mode action being clicked."""
        if checked:
            # Uncheck every other polygon mode action without re-triggering them
            for act in self._mode_actions.values():
                if act is not source and act.isChecked():
                    act.blockSignals(True)
                    act.setChecked(False)
                    act.blockSignals(False)
            self.mode_changed.emit(mode)
        else:
            # Action was unchecked — no polygon mode is now active
            self.mode_changed.emit(None)

    def _make_snap_action(self, parent) -> QAction:
        act = QAction("Snap", parent)
        act.setCheckable(True)
        act.setChecked(True)
        act.setIcon(self._icon("icon-snap-on-point-64.svg"))
        act.setToolTip("Enable vertex snapping  [Ctrl+P]")
        act.setShortcut(QKeySequence("Ctrl+P"))
        act.toggled.connect(self._on_snap_toggled)
        return act

    def _on_snap_toggled(self, checked: bool):
        self._snap_enabled = checked
        filename = ("icon-snap-on-point-64.svg" if checked
                    else "icon-snap-off-point-64.svg")
        self._act_snap.setIcon(self._icon(filename))
        self.snap_changed.emit(checked)

    # ── public API ────────────────────────────────────────────────────────

    @property
    def snap_enabled(self) -> bool:
        """Current state of the snap toggle."""
        return self._snap_enabled

    def deselect_all(self):
        """Uncheck all polygon mode actions without emitting any signal."""
        for act in self._mode_actions.values():
            if act.isChecked():
                act.blockSignals(True)
                act.setChecked(False)
                act.blockSignals(False)

    def set_mode(self, mode: EditMode):
        """
        Programmatically activate a mode action without emitting
        ``mode_changed`` (useful for syncing UI state from application code).
        Unchecks all other polygon mode actions.
        """
        for m, act in self._mode_actions.items():
            target = (m == mode)
            if act.isChecked() != target:
                act.blockSignals(True)
                act.setChecked(target)
                act.blockSignals(False)

    def add_to_toolbar(
        self,
        toolbar: QToolBar,
        include_snap: bool = True,
    ) -> None:
        """
        Add all polygon editing actions (and optionally the snap toggle)
        to *toolbar*.

        The actions are added in definition order (New Polygon → Move
        Polygon → Delete Polygon → Add Vertex → Move Vertex → Delete
        Vertex), followed by the Snap toggle if requested.

        Parameters
        ----------
        toolbar      : target QToolBar
        include_snap : add the Snap checkable action after the mode actions
        """
        for mode in (
            EditMode.DRAW, EditMode.MOVE_BODY, EditMode.DELETE,
            EditMode.ADD_VERTEX, EditMode.SELECT, EditMode.DELETE_VERTEX,
        ):
            toolbar.addAction(self._mode_actions[mode])

        if include_snap:
            self._act_snap = self._make_snap_action(toolbar.parent())
            toolbar.addAction(self._act_snap)
