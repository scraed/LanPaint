"""Top-level package for LanPaint."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """LanPaint"""
__email__ = "czhengac@connect.ust.hk"
__version__ = "0.0.1"


def _install_lightweight_runtime_stubs() -> None:
    """Install lightweight stubs so tooling can import this package without ComfyUI.

    This is used by CI tooling (e.g., comfy-org/node-diff) that imports NODE_CLASS_MAPPINGS
    in an environment where ComfyUI isn't installed.
    """
    import sys
    import types

    # `src/LanPaint/nodes.py` uses `torch.Tensor` in type annotations.
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        torch_mod = types.ModuleType("torch")

        class Tensor:  # noqa: N801 (match torch naming)
            pass

        torch_mod.Tensor = Tensor
        torch_mod.nn = types.SimpleNamespace(functional=types.SimpleNamespace())
        sys.modules["torch"] = torch_mod

    if "comfyui_version" not in sys.modules:
        comfyui_version_mod = types.ModuleType("comfyui_version")
        comfyui_version_mod.__version__ = "0.0.0"
        sys.modules["comfyui_version"] = comfyui_version_mod

    sys.modules.setdefault("nodes", types.ModuleType("nodes"))
    sys.modules.setdefault("latent_preview", types.ModuleType("latent_preview"))

    if "comfy" not in sys.modules:
        comfy_mod = types.ModuleType("comfy")
        comfy_mod.__path__ = []

        comfy_utils_mod = types.ModuleType("comfy.utils")

        def repeat_to_batch_size(tensor, batch_size):  # type: ignore[no-untyped-def]
            if getattr(tensor, "shape", ())[0] == batch_size:
                return tensor
            return tensor

        comfy_utils_mod.repeat_to_batch_size = repeat_to_batch_size

        comfy_samplers_mod = types.ModuleType("comfy.samplers")

        class DummyKSAMPLER:  # noqa: N801 (match ComfyUI naming)
            pass

        comfy_samplers_mod.KSAMPLER = DummyKSAMPLER

        comfy_model_base_mod = types.ModuleType("comfy.model_base")

        class ModelType:  # noqa: N801 (match ComfyUI naming)
            FLUX = "FLUX"
            FLOW = "FLOW"

        class WAN22:  # noqa: N801 (match ComfyUI naming)
            pass

        comfy_model_base_mod.ModelType = ModelType
        comfy_model_base_mod.WAN22 = WAN22

        comfy_mod.utils = comfy_utils_mod
        comfy_mod.samplers = comfy_samplers_mod
        comfy_mod.model_base = comfy_model_base_mod

        sys.modules["comfy"] = comfy_mod
        sys.modules["comfy.utils"] = comfy_utils_mod
        sys.modules["comfy.samplers"] = comfy_samplers_mod
        sys.modules["comfy.model_base"] = comfy_model_base_mod


try:
    from .src.LanPaint.nodes import NODE_CLASS_MAPPINGS
    from .src.LanPaint.nodes import NODE_DISPLAY_NAME_MAPPINGS
except ModuleNotFoundError:
    _install_lightweight_runtime_stubs()
    from .src.LanPaint.nodes import NODE_CLASS_MAPPINGS
    from .src.LanPaint.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
