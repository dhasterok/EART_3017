#!/usr/bin/env python3
"""
Week 5 — Isostasy Demo launcher.

Run from any directory:
    python week_5_isostasy/isostasy_demo.py
"""
import sys
from pathlib import Path

_COURSE = Path(__file__).resolve().parent.parent
if str(_COURSE) not in sys.path:
    sys.path.insert(0, str(_COURSE))

from src.apps.isostasy.isostasy_gui import main

if __name__ == '__main__':
    main()
