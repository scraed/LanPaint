"""Basic import tests for LanPaint.

The ComfyUI runtime dependencies (e.g. `comfy`) are intentionally optional for unit tests.
"""


def test_package_imports_without_comfy() -> None:
    import LanPaint

    assert isinstance(LanPaint.NODE_CLASS_MAPPINGS, dict)
    assert isinstance(LanPaint.NODE_DISPLAY_NAME_MAPPINGS, dict)
    assert "LanPaint_KSampler" in LanPaint.NODE_CLASS_MAPPINGS
    assert LanPaint.WEB_DIRECTORY == "./web"
