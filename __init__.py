"""Top-level package for LanPaint."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """LanPaint"""
__email__ = "czhengac@connect.ust.hk"
__version__ = "0.0.1"

try:
    from .src.LanPaint.nodes import NODE_CLASS_MAPPINGS
    from .src.LanPaint.nodes import NODE_DISPLAY_NAME_MAPPINGS
except ModuleNotFoundError:
    # Allow importing this package in environments without ComfyUI installed.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./web"
