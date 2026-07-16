from pathlib import Path

class figutils:
    """Utility functions for figures."""

    def __init__(self, figdir: Path = Path("tmp")):
        self._FIGDIR = figdir

    def savefig(self, fig, name):
        """Save a figure into FIGDIR as both PDF (for LaTeX) and PNG (for quick
        preview), using consistent settings."""
        fig.savefig(self._FIGDIR / f"{name}.pdf", dpi=200, bbox_inches="tight")
        fig.savefig(self._FIGDIR / f"{name}.png", dpi=150, bbox_inches="tight")

    @property
    def FIGDIR(self):
        """Get the directory where figures will be saved."""
        return self._FIGDIR

    @FIGDIR.setter
    def FIGDIR(self, path):
        """Set the directory where figures will be saved."""
        self._FIGDIR = Path(path)
        self._FIGDIR.mkdir(parents=True, exist_ok=True)