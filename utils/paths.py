import os
import sys


def get_base_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.abspath(__file__))


def get_assets_dir():
    base = get_base_dir()
    exedir = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(exedir, "assets"),
        os.path.join(base, "assets"),
        os.path.join(os.path.dirname(base), "assets"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]
