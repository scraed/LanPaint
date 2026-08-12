import importlib
import sys
import types

import pytest
import torch


def _repeat_to_batch_size(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.repeat((batch_size,) + (1,) * (tensor.ndim - 1))
    repeats = (batch_size + tensor.shape[0] - 1) // tensor.shape[0]
    return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))[:batch_size]


def _import_nodes(monkeypatch, comfyui_version: str):
    comfy_mod = types.ModuleType("comfy")
    comfy_mod.__path__ = []

    comfy_utils_mod = types.ModuleType("comfy.utils")
    comfy_utils_mod.repeat_to_batch_size = _repeat_to_batch_size

    comfy_samplers_mod = types.ModuleType("comfy.samplers")
    class DummyKSAMPLER: ...
    comfy_samplers_mod.KSAMPLER = DummyKSAMPLER

    comfy_model_base_mod = types.ModuleType("comfy.model_base")
    class ModelType:
        FLUX = "FLUX"
        FLOW = "FLOW"

    class WAN22: ...
    comfy_model_base_mod.ModelType = ModelType
    comfy_model_base_mod.WAN22 = WAN22

    comfyui_version_mod = types.ModuleType("comfyui_version")
    comfyui_version_mod.__version__ = comfyui_version

    comfy_mod.utils = comfy_utils_mod
    comfy_mod.samplers = comfy_samplers_mod
    comfy_mod.model_base = comfy_model_base_mod

    monkeypatch.setitem(sys.modules, "comfy", comfy_mod)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_mod)
    monkeypatch.setitem(sys.modules, "comfy.samplers", comfy_samplers_mod)
    monkeypatch.setitem(sys.modules, "comfy.model_base", comfy_model_base_mod)
    monkeypatch.setitem(sys.modules, "nodes", types.ModuleType("nodes"))
    monkeypatch.setitem(sys.modules, "latent_preview", types.ModuleType("latent_preview"))
    monkeypatch.setitem(sys.modules, "comfyui_version", comfyui_version_mod)

    sys.modules.pop("src.LanPaint.nodes", None)
    return importlib.import_module("src.LanPaint.nodes")


@pytest.mark.parametrize("comfyui_version", ["0.5.0", "0.6.0"])
def test_reshape_mask_accepts_bhw_and_5d_output_shape(monkeypatch, comfyui_version: str) -> None:
    lanpaint_nodes = _import_nodes(monkeypatch, comfyui_version)
    input_mask = torch.zeros((1, 4, 4))
    output_shape = (1, 16, 1, 8, 8)

    out = lanpaint_nodes.reshape_mask(input_mask, output_shape, video_inpainting=False)
    assert tuple(out.shape) == output_shape


def test_prepare_mask_accepts_hw_and_moves_device(monkeypatch) -> None:
    lanpaint_nodes = _import_nodes(monkeypatch, "0.5.0")
    input_mask = torch.zeros((4, 4))
    output_shape = (2, 3, 8, 8)

    out = lanpaint_nodes.prepare_mask(input_mask, output_shape, device=torch.device("cpu"), video_inpainting=False)
    assert tuple(out.shape) == output_shape
    assert out.device.type == "cpu"


def test_video_union_covers_picked_frames(monkeypatch) -> None:
    """EXPERIMENTAL order (nearest-exact first, slice-level union): a stroke
    on a frame that the resample picks survives, and the sliding max spreads
    it to neighboring slices."""
    lanpaint_nodes = _import_nodes(monkeypatch, "0.6.0")

    # 8 frames -> 2 slices; nearest-exact picks frames {2, 6}. Strokes on
    # picked frames 2 and 6, on rows/cols the 8->4 grid samples ({1,3,5,7}).
    mask = torch.zeros(8, 8, 8)
    mask[2, 5, 5] = 1.0
    mask[6, 7, 7] = 1.0

    out = lanpaint_nodes.reshape_mask(mask, (1, 16, 2, 4, 4), video_inpainting=True)
    assert tuple(out.shape) == (1, 16, 2, 4, 4)
    # both slices are marked (slice-level union spreads the picked values)
    assert out[0, 0, 0].max().item() == 1.0
    assert out[0, 0, 1].max().item() == 1.0
    # the stroke positions land on the token containing the stroke pixel
    assert out[0, 0, 0, 2, 2].item() == 1.0
    assert out[0, 0, 1, 3, 3].item() == 1.0


def test_video_union_drops_unpicked_frames(monkeypatch) -> None:
    """EXPERIMENTAL order: a stroke on a frame that no slice picks (e.g.
    frame 3, since nearest-exact picks {2, 6}) is silently lost."""
    lanpaint_nodes = _import_nodes(monkeypatch, "0.6.0")

    mask = torch.zeros(8, 8, 8)
    mask[3, 5, 5] = 1.0

    out = lanpaint_nodes.reshape_mask(mask, (1, 16, 2, 4, 4), video_inpainting=True)
    assert out.max().item() == 0.0


def test_video_union_short_sequence(monkeypatch) -> None:
    """A 3-frame sequence downsamples to one slice; the stroke must survive
    if it lands on the frame the resample picks (floor(0.5*3) = frame 1)."""
    lanpaint_nodes = _import_nodes(monkeypatch, "0.6.0")

    mask = torch.zeros(3, 8, 8)
    mask[1, 5, 5] = 1.0

    out = lanpaint_nodes.reshape_mask(mask, (1, 16, 1, 4, 4), video_inpainting=True)
    assert tuple(out.shape) == (1, 16, 1, 4, 4)
    assert out.max().item() == 1.0
