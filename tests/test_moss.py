"""MOSS encoding and native ComfyUI guidance dispatch regressions."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image


def _import_nodes():
    import LanPaint.src.LanPaint.nodes as nodes

    return nodes


class _VAE:
    def encode(self, image):
        self.image = image
        return torch.zeros(image.shape[0], 128, image.shape[1] // 16, image.shape[2] // 16)


def test_all_thin_pixel_positions_survive() -> None:
    vae = _VAE()
    mask = torch.eye(256).reshape(256, 16, 16)
    (result,) = _import_nodes().LanPaint_MOSSImageEncode().encode(torch.zeros(256, 16, 16, 3), vae, mask)
    assert result["noise_mask"].shape == (256, 1, 1, 1)
    assert result["noise_mask"].sum() == 256


def test_dilation_crosses_latent_boundary() -> None:
    encoder = _import_nodes().LanPaint_MOSSImageEncode()
    image = torch.zeros(1, 16, 32, 3)
    mask = torch.zeros(1, 16, 32)
    mask[0, 8, 15] = 1
    (a,) = encoder.encode(image, _VAE(), mask, dilation=0)
    (b,) = encoder.encode(image, _VAE(), mask, dilation=1)
    assert a["noise_mask"].sum() == 1
    assert b["noise_mask"].sum() == 2


@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_empty_full_and_neutralization(fill) -> None:
    image = torch.rand(1, 32, 48, 3)
    vae = _VAE()
    (result,) = _import_nodes().LanPaint_MOSSImageEncode().encode(image, vae, torch.full((1, 32, 48), fill))
    assert torch.equal(result["noise_mask"], torch.full((1, 1, 2, 3), fill))
    assert torch.equal(vae.image, image) if fill == 0 else torch.all(vae.image == 128 / 255)


@pytest.mark.parametrize(
    "image,mask",
    [
        (torch.zeros(1, 31, 32, 3), torch.zeros(1, 31, 32)),
        (torch.zeros(1, 32, 32, 3), torch.zeros(1, 16, 32)),
        (torch.zeros(1, 32, 32, 4), torch.zeros(1, 32, 32)),
        (torch.zeros(1, 32, 32, 3), torch.full((1, 32, 32), float("nan"))),
        (torch.full((1, 32, 32, 3), float("inf")), torch.zeros(1, 32, 32)),
    ],
)
def test_reject_invalid_image_mask(image, mask) -> None:
    with pytest.raises(ValueError):
        _import_nodes().LanPaint_MOSSImageEncode().encode(image, _VAE(), mask)


@pytest.mark.parametrize("shape", [(1, 16, 2, 2), (1, 128, 2, 2, 1), (1, 128, 1, 2)])
def test_reject_wrong_latent(shape) -> None:
    class WrongVAE:
        def encode(self, image):
            return torch.zeros(shape)

    with pytest.raises(ValueError):
        _import_nodes().LanPaint_MOSSImageEncode().encode(torch.zeros(1, 32, 32, 3), WrongVAE(), torch.zeros(1, 32, 32))


def test_source_fill_and_2d_mask() -> None:
    image = torch.rand(1, 32, 32, 3)
    vae = _VAE()
    (result,) = _import_nodes().LanPaint_MOSSImageEncode().encode(image, vae, torch.ones(32, 32), source_fill="source")
    assert torch.equal(vae.image, image) and result["noise_mask"].all()


@pytest.mark.parametrize("options", [{"source_fill": "invalid"}, {"dilation": -1}, {"dilation": 129}, {"dilation": 0.5}])
def test_reject_invalid_options(options) -> None:
    with pytest.raises(ValueError):
        _import_nodes().LanPaint_MOSSImageEncode().encode(torch.zeros(1, 16, 16, 3), _VAE(), torch.zeros(1, 16, 16), **options)


@pytest.fixture
def guidance(monkeypatch):
    nodes = _import_nodes()

    def cfg(model, positive, negative, scale, x, timestep, **kwargs):
        return negative + (positive - negative) * scale

    monkeypatch.setattr(nodes, "cfg_function", cfg, raising=False)
    return nodes


@pytest.mark.parametrize(
    "cfg,big,disable,needed",
    [
        (1.0, -0.5, False, True),
        (1.0, 1.0, False, False),
        (3.0, 3.0, False, True),
        (1.0, 3.0, False, True),
        (1.0, 1.0, True, True),
    ],
)
def test_dual_guidance_branch_optimization(monkeypatch, guidance, cfg, big, disable, needed) -> None:
    seen = []

    def calc(model, conds, x, sigma, options):
        seen.append(conds[1])
        return [torch.ones_like(x) * 2, torch.ones_like(x)]

    monkeypatch.setattr(guidance, "calc_cond_batch", calc, raising=False)
    normal, bidirectional = guidance.sampling_function_LanPaint(
        None,
        torch.zeros(1),
        torch.tensor([0.5]),
        "negative",
        "positive",
        cfg,
        big,
        {"disable_cfg1_optimization": disable},
    )
    assert (seen[0] is not None) == needed
    assert normal.item() == 1 + cfg and bidirectional.item() == 1 + big


def test_native_cond_batch_callback_and_pre_cfg_hook(monkeypatch, guidance) -> None:
    seen = []

    def callback(args):
        seen.append(args)
        return [torch.tensor([2.0]), torch.tensor([1.0])]

    def unexpected(*args):
        pytest.fail("native conditioning callback was bypassed")

    monkeypatch.setattr(guidance, "calc_cond_batch", unexpected, raising=False)
    x, sigma = torch.zeros(1), torch.tensor([0.5])
    options = {"sampler_calc_cond_batch_function": callback, "sampler_pre_cfg_function": [lambda a: [v * 2 for v in a["conds_out"]]]}
    out, big = guidance.sampling_function_LanPaint("model", x, sigma, "negative", "positive", 1.0, -0.5, options)
    assert out.item() == 4 and big.item() == 1
    assert seen[0]["conds"] == ["positive", "negative"]
    assert seen[0]["model_options"] is options
    assert seen[0]["input"] is x and seen[0]["sigma"] is sigma


def test_moss_node_registration() -> None:
    nodes = _import_nodes()
    assert nodes.NODE_CLASS_MAPPINGS["LanPaint_MOSSImageEncode"] is nodes.LanPaint_MOSSImageEncode
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["LanPaint_MOSSImageEncode"] == "LanPaint MOSS Image Encode"
    assert nodes.LanPaint_MOSSImageEncode.CATEGORY == nodes.LanPaint_ImageEncode.CATEGORY
    assert not any(name.startswith("MossLanPaint") for name in nodes.NODE_CLASS_MAPPINGS)


def test_example_follows_repository_conventions() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "examples/Example_33/InPainted_Drag_Me_to_ComfyUI.png"
    with Image.open(path) as image:
        graph = json.loads(image.info["prompt"])
        workflow = json.loads(image.info["workflow"])
    assert workflow == json.loads((root / "example_workflows/MOSS_EncodeDecode_Inpaint.json").read_text())
    assert (root / "example_workflows/MOSS_EncodeDecode_Inpaint.jpg").exists()
    assert not (root / "examples/MOSS").exists()
    assert graph["7"]["class_type"] == "LanPaint_MOSSImageEncode"
    assert graph["12"]["class_type"] == "LanPaint_KSamplerAdvanced"
    assert graph["13"]["class_type"] == "LanPaint_ImageDecode"
    assert graph["13"]["inputs"]["blend_overlap"] == 17
    assert graph["12"]["inputs"]["noise_seed"] == 42
    assert graph["11"]["inputs"]["single_layers"] == "22"
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in graph
    ids = {n["id"]: n for n in workflow["nodes"]}
    for link, source, slot, dest, input_slot, kind in workflow["links"]:
        assert link in ids[source]["outputs"][slot]["links"]
        assert ids[dest]["inputs"][input_slot]["link"] == link
        assert ids[dest]["inputs"][input_slot]["type"] == kind
