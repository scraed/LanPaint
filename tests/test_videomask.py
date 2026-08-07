import importlib
import json
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image

from src.LanPaint.videomask import (
    _edt_2d,
    _shift,
    interpolate_masks,
    load_keyframe_png,
    parse_keyframes_widget,
    resize_masks,
)


def _mask(width=8, height=6, fill=1.0):
    return np.full((height, width), fill, dtype=np.float32)


def _rect(h=6, w=8, y0=0, y1=None, x0=0, x1=None, fill=1.0):
    """A filled rectangle mask of shape (h, w)."""
    if y1 is None:
        y1 = h
    if x1 is None:
        x1 = w
    m = np.zeros((h, w), dtype=np.float32)
    m[y0:y1, x0:x1] = fill
    return m


# --- interpolation contract (shared with the frontend preview) ----------------

def test_single_keyframe_masks_only_its_frame() -> None:
    out = interpolate_masks({7: _mask()}, 12)
    assert out.shape == (12, 6, 8)
    # the mask exists only at the keyframe itself, nowhere else
    assert (out[7] == 1.0).all()
    for t in range(12):
        if t != 7:
            assert (out[t] == 0.0).all()


def test_frames_outside_keyframe_window_are_empty() -> None:
    out = interpolate_masks({10: _mask(), 20: _mask()}, 30)
    # frames before the first keyframe and after the last have NO mask
    assert (out[:10] == 0.0).all()
    assert (out[21:] == 0.0).all()
    # the keyframe frames themselves keep their masks
    assert (out[10] == 1.0).all()
    assert (out[20] == 1.0).all()
    # frames between the keyframes morph (uniform fills: essentially full)
    assert out[15].max() > 0.9


def test_exact_keyframes_are_preserved() -> None:
    # exact keyframe frames return the ORIGINAL painted mask, not the sigmoid
    left = _rect(x1=4, fill=0.25)
    right = _rect(x0=4, fill=0.75)
    out = interpolate_masks({10: left, 20: right}, 30)
    assert np.array_equal(out[10], left)
    assert np.array_equal(out[20], right)
    # outside the window: no mask
    assert (out[0] == 0.0).all()
    assert (out[29] == 0.0).all()


def test_edt_matches_bruteforce() -> None:
    rng = np.random.RandomState(0)
    for (h, w) in [(16, 16), (8, 15), (17, 5)]:
        mask = rng.rand(h, w) > 0.5
        d = _edt_2d(mask)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        yy, xx = np.mgrid[0:h, 0:w]
        bf = np.min((yy[:, :, None] - ys) ** 2 + (xx[:, :, None] - xs) ** 2, axis=2)
        assert np.abs(d**2 - bf).max() < 1e-6


def test_shift_places_pixels_at_the_right_offset() -> None:
    field = np.arange(24, dtype=np.float64).reshape(4, 6)
    out = _shift(field, dy=1, dx=2)
    assert (out[1:, 2:] == field[:3, :4]).all()
    assert (out[0] == 0).all() and (out[:, :2] == 0).all()
    out2 = _shift(field, dy=-1, dx=-1)
    assert (out2[:3, :5] == field[1:, 1:]).all()
    assert (out2[3] == 0).all() and (out2[:, 5] == 0).all()


def test_sdf_morph_translation_moves_centroid_linearly() -> None:
    left = _rect(h=20, w=20, y0=5, y1=15, x0=2, x1=7)
    right = _rect(h=20, w=20, y0=5, y1=15, x0=13, x1=18)
    out = interpolate_masks({0: left, 10: right}, 11)
    centroids = []
    for t in range(11):
        ys, xs = np.where(out[t] > 0.5)
        centroids.append(xs.mean())
    assert centroids[0] == pytest.approx(4.0)
    assert centroids[10] == pytest.approx(15.0)
    # the shape slides (no collapse): mid centroid is the exact midpoint
    assert abs(centroids[5] - (centroids[0] + centroids[10]) / 2) <= 1.0
    # every intermediate frame has a solid shape (no vanish gap)
    for t in range(1, 10):
        assert (out[t] > 0.5).sum() > 0


def test_sdf_morph_keeps_soft_edges() -> None:
    # between a translating pair, intermediate frames have a soft transition
    # band (not a binary cut)
    left = _rect(h=20, w=20, y0=5, y1=15, x0=2, x1=7)
    right = _rect(h=20, w=20, y0=5, y1=15, x0=13, x1=18)
    out = interpolate_masks({0: left, 10: right}, 11)
    for t in range(1, 10):
        soft = ((out[t] > 0.05) & (out[t] < 0.95)).sum()
        assert soft > 0


def _level_area(frame: np.ndarray) -> int:
    return int((frame > 0.5).sum())


def test_empty_keyframe_morphs_to_shape_monotonically() -> None:
    empty = np.zeros((10, 10), dtype=np.float32)
    rect = _rect(h=10, w=10, y0=2, y1=8, x0=2, x1=8)
    out = interpolate_masks({0: empty, 4: rect}, 5)
    assert np.array_equal(out[0], empty)
    assert np.array_equal(out[4], rect)
    # the SHAPE grows monotonically (level-set area); per-pixel values may dip
    # slightly at edge pixels where the soft sigmoid meets the hard keyframe edge
    areas = [_level_area(out[t]) for t in range(5)]
    assert areas == sorted(areas)
    assert areas[0] == 0 and areas[4] == 36
    # the center fills before the corner
    for t in range(1, 4):
        assert out[t, 5, 5] > out[t, 0, 0]


def test_shape_morphs_to_empty_monotonically() -> None:
    empty = np.zeros((10, 10), dtype=np.float32)
    rect = _rect(h=10, w=10, y0=2, y1=8, x0=2, x1=8)
    out = interpolate_masks({0: rect, 4: empty}, 5)
    areas = [_level_area(out[t]) for t in range(5)]
    assert areas == sorted(areas, reverse=True)
    assert areas[0] == 36 and areas[4] == 0


def test_full_keyframe_morphs_out_to_shape() -> None:
    full = np.ones((10, 10), dtype=np.float32)
    rect = _rect(h=10, w=10, y0=2, y1=8, x0=2, x1=8)
    out = interpolate_masks({0: full, 4: rect}, 5)
    assert np.array_equal(out[0], full)
    assert np.array_equal(out[4], rect)
    # the full-field sentinel keeps everything positive early on; by the last
    # intermediate frame the corners have emptied while the center stays filled
    assert out[3, 0, 0] < out[3, 5, 5]


def test_sdf_morph_no_nan_or_overflow() -> None:
    full = np.ones((6, 8), dtype=np.float32)
    empty = np.zeros((6, 8), dtype=np.float32)
    out = interpolate_masks({0: empty, 1: full, 2: empty}, 3)
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_determinism_same_keyframes_same_output() -> None:
    rng = np.random.RandomState(42)
    a = rng.rand(6, 8).astype(np.float32)
    b = rng.rand(6, 8).astype(np.float32)
    out1 = interpolate_masks({0: a, 5: b}, 6)
    out2 = interpolate_masks({0: a, 5: b}, 6)
    assert np.array_equal(out1, out2)


def test_nonadjacent_and_multiple_keyframes() -> None:
    empty = np.zeros((6, 8), dtype=np.float32)
    rect = _rect(x1=4)
    out = interpolate_masks({5: empty, 15: rect, 25: empty}, 30)
    assert (out[0] == 0.0).all()  # before the first keyframe: no mask
    assert np.array_equal(out[5], empty)  # exact keyframe
    assert np.array_equal(out[15], rect)  # exact keyframe
    assert (out[29] == 0.0).all()  # after the last keyframe: no mask
    # intermediate frames morph (soft values, never outside [0, 1])
    mid = out[10]
    assert mid.min() >= 0.0 and mid.max() <= 1.0


def test_interpolate_requires_keyframes_and_positive_count() -> None:
    with pytest.raises(ValueError):
        interpolate_masks({}, 5)
    with pytest.raises(ValueError):
        interpolate_masks({0: _mask()}, 0)


def test_interpolate_respects_keyframe_shape() -> None:
    a = np.zeros((4, 8), dtype=np.float32)
    b = np.ones((4, 8), dtype=np.float32)
    out = interpolate_masks({0: a, 10: b}, 11)
    assert out.shape == (11, 4, 8)


# --- keyframe PNG loading and resizing ----------------------------------------

def test_load_keyframe_png_reads_alpha_channel(tmp_path) -> None:
    # the editor saves the mask in the alpha channel (white RGB + varying alpha)
    rgba = np.zeros((6, 8, 4), dtype=np.uint8)
    rgba[:, :, 0:3] = 255
    rgba[:, 4:, 3] = 255  # right half masked
    Image.fromarray(rgba, "RGBA").save(tmp_path / "kf.png")
    arr = load_keyframe_png(str(tmp_path / "kf.png"))
    assert arr.shape == (6, 8)
    assert (arr[0, :4] == 0).all() and (arr[0, 4:] == 1).all()


def test_load_keyframe_png_grayscale_and_range(tmp_path) -> None:
    path = tmp_path / "kf.png"
    Image.fromarray((np.arange(256, dtype=np.uint8) * 255 // 255)).resize(
        (8, 8), Image.NEAREST
    )
    im = np.zeros((6, 8), dtype=np.uint8)
    im[:, 4:] = 255
    Image.fromarray(im).save(path)
    arr = load_keyframe_png(str(path))
    assert arr.shape == (6, 8)
    assert arr.dtype == np.float32
    assert arr.max() == pytest.approx(1.0)
    assert arr.min() == pytest.approx(0.0)
    assert (arr[0, :4] == 0).all() and (arr[0, 4:] == 1).all()


def test_load_keyframe_png_resizes_to_requested_size(tmp_path) -> None:
    path = tmp_path / "kf.png"
    Image.fromarray(np.full((6, 8), 255, dtype=np.uint8)).save(path)
    arr = load_keyframe_png(str(path), size=(16, 12))
    assert arr.shape == (12, 16)
    assert arr.max() == pytest.approx(1.0)


def test_resize_masks_bilinear_keeps_soft_edges(tmp_path) -> None:
    m = np.zeros((1, 4, 4), dtype=np.float32)
    m[0, 1:3, 1:3] = 1.0
    out = resize_masks(m, (8, 8))
    assert out.shape == (1, 8, 8)
    # the upscale creates a transition zone: some pixel is strictly between 0 and 1
    assert ((out[0] > 0.0) & (out[0] < 1.0)).any()


def test_resize_masks_identity_when_size_matches() -> None:
    m = np.full((3, 6, 8), 0.5, dtype=np.float32)
    assert resize_masks(m, (8, 6)) is m


# --- keyframes widget parsing -------------------------------------------------

def test_parse_keyframes_widget() -> None:
    assert parse_keyframes_widget(None) == {}
    assert parse_keyframes_widget("") == {}
    assert parse_keyframes_widget("not json") == {}
    assert parse_keyframes_widget("[1, 2]") == {}
    assert parse_keyframes_widget(json.dumps({"0": "a.png", "42": "b.png"})) == {0: "a.png", 42: "b.png"}
    # malformed entries are dropped, valid ones kept
    assert parse_keyframes_widget(json.dumps({"x": 1, "3": None})) == {}


# --- node behavior through the stub environment -------------------------------

def _repeat_to_batch_size(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.repeat((batch_size,) + (1,) * (tensor.ndim - 1))
    repeats = (batch_size + tensor.shape[0] - 1) // tensor.shape[0]
    return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))[:batch_size]


def _import_nodes(monkeypatch, tmp_path):
    comfy_mod = types.ModuleType("comfy")
    comfy_mod.__path__ = []

    comfy_utils_mod = types.ModuleType("comfy.utils")
    comfy_utils_mod.repeat_to_batch_size = _repeat_to_batch_size

    folder_paths_mod = types.ModuleType("folder_paths")
    folder_paths_mod.get_input_directory = lambda: str(tmp_path)
    folder_paths_mod.get_annotated_filepath = lambda f, subfolder=None, type=None: str(tmp_path / f)
    folder_paths_mod.exists_annotated_filepath = lambda f: (tmp_path / f).exists()

    comfy_samplers_mod = types.ModuleType("comfy.samplers")
    comfy_samplers_mod.KSAMPLER = type("KSAMPLER", (), {})

    comfy_model_base_mod = types.ModuleType("comfy.model_base")
    comfy_model_base_mod.ModelType = types.SimpleNamespace(FLUX="FLUX", FLOW="FLOW")
    comfy_model_base_mod.WAN22 = type("WAN22", (), {})

    comfyui_version_mod = types.ModuleType("comfyui_version")
    comfyui_version_mod.__version__ = "0.6.0"

    comfy_mod.utils = comfy_utils_mod
    comfy_mod.samplers = comfy_samplers_mod
    comfy_mod.model_base = comfy_model_base_mod

    monkeypatch.setitem(sys.modules, "comfy", comfy_mod)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_mod)
    monkeypatch.setitem(sys.modules, "comfy.samplers", comfy_samplers_mod)
    monkeypatch.setitem(sys.modules, "comfy.model_base", comfy_model_base_mod)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_mod)
    monkeypatch.setitem(sys.modules, "nodes", types.ModuleType("nodes"))
    monkeypatch.setitem(sys.modules, "latent_preview", types.ModuleType("latent_preview"))
    monkeypatch.setitem(sys.modules, "comfyui_version", comfyui_version_mod)

    sys.modules.pop("src.LanPaint.nodes", None)
    return importlib.import_module("src.LanPaint.nodes")


def _save_keyframe(tmp_path, name, value=1.0, w=8, h=6):
    im = Image.fromarray((np.full((h, w), value * 255, dtype=np.uint8)))
    im.save(tmp_path / name)
    return name


def _run(nodes, video="video.mp4", keyframes="{}", audio_mask="[]", monkeypatch=None, count=5, dims=(8, 6), fps=24.0):
    """Run the node with a stubbed VideoFromFile (no real container)."""

    class FakeVideoFromFile:
        def __init__(self, path):
            self.path = path

        def get_frame_count(self):
            return count

        def get_dimensions(self):
            return dims

        def get_fps(self):
            return fps

    monkeypatch.setattr(nodes, "VideoFromFile", FakeVideoFromFile)
    return nodes.LanPaint_VideoMaskEditor().run(video=video, keyframes=keyframes, audio_mask=audio_mask)


def test_node_interpolates_keyframes_to_frame_count(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    _save_keyframe(tmp_path, "k0.png", 0.0)
    _save_keyframe(tmp_path, "k2.png", 1.0)
    widget = json.dumps({"0": "k0.png", "2": "k2.png"})

    out_video, mask, _ = _run(nodes, keyframes=widget, monkeypatch=monkeypatch)
    assert out_video.path.endswith("video.mp4")  # the VIDEO reference passes through
    assert mask.shape == (5, 6, 8)
    assert mask[0].max() == pytest.approx(0.0)
    assert mask[2].max() == pytest.approx(1.0)
    assert mask[1].max() == pytest.approx(0.5)  # interpolated halfway
    assert mask[4].max() == pytest.approx(0.0)  # after the last keyframe: no mask


def test_node_upscales_keyframes_to_video_size(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    _save_keyframe(tmp_path, "k0.png", 1.0, w=4, h=4)
    mask = _run(nodes, keyframes=json.dumps({"0": "k0.png"}), count=3, dims=(8, 8), monkeypatch=monkeypatch)[1]
    assert mask.shape == (3, 8, 8)
    assert mask[0].max() == pytest.approx(1.0)


def test_node_keyframe_beyond_video_length_makes_empty_mask(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    _save_keyframe(tmp_path, "k9.png", 1.0)
    # the only keyframe (9) is beyond the 5-frame video: no frame is a
    # keyframe or between keyframes -> the mask is empty
    mask = _run(nodes, keyframes=json.dumps({"9": "k9.png"}), monkeypatch=monkeypatch)[1]
    assert mask.shape == (5, 6, 8)
    assert mask.max() == pytest.approx(0.0)


def test_node_missing_keyframe_files_yield_empty_mask(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    mask = _run(nodes, keyframes=json.dumps({"0": "gone.png"}), monkeypatch=monkeypatch)[1]
    assert mask.shape == (5, 6, 8)
    assert mask.max() == pytest.approx(0.0)


def test_node_without_keyframes_yields_empty_mask(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    out_video, mask, _ = _run(nodes, monkeypatch=monkeypatch)
    assert out_video is not None
    assert mask.shape == (5, 6, 8)
    assert mask.max() == pytest.approx(0.0)


def test_node_requires_a_video(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    node = nodes.LanPaint_VideoMaskEditor()
    with pytest.raises(ValueError):
        node.run(video=None)


def test_node_has_no_data_inputs(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    types = nodes.LanPaint_VideoMaskEditor.INPUT_TYPES()
    assert "optional" not in types or not types.get("optional")
    assert list(types["required"]) == ["video", "keyframes", "audio_mask"]
    assert nodes.LanPaint_VideoMaskEditor.RETURN_NAMES == ("video", "mask", "audio_mask")


# --- audio mask intervals ----------------------------------------------------

def test_node_audio_intervals_to_frame_mask(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    # 100 frames at 24 fps: [1.0, 2.0]s covers frames 24..47
    video, mask, audio = _run(nodes, count=100, fps=24.0,
                              audio_mask='[{"start": 1.0, "end": 2.0}]',
                              monkeypatch=monkeypatch)
    assert audio.shape == (100,)
    assert (audio[24:48] == 1.0).all()
    assert (audio[:24] == 0.0).all() and (audio[48:] == 0.0).all()


def test_node_audio_multiple_intervals(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    _, _, audio = _run(nodes, count=100, fps=24.0,
                       audio_mask='[{"start": 1.0, "end": 1.5}, {"start": 3.0, "end": 4.0}]',
                       monkeypatch=monkeypatch)
    assert (audio[24:36] == 1.0).all()
    assert (audio[72:96] == 1.0).all()
    assert (audio[36:72] == 0.0).all()


def test_node_audio_no_intervals_keeps_everything(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    _, _, audio = _run(nodes, count=100, fps=24.0, audio_mask="[]",
                       monkeypatch=monkeypatch)
    assert (audio == 0.0).all()  # default: keep the original track


def test_node_audio_interval_clamped_to_video_length(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    # interval beyond the video end: clamped; inverted interval: ignored
    _, _, audio = _run(nodes, count=100, fps=24.0,
                       audio_mask='[{"start": 90.0, "end": 95.0}, {"start": 5.0, "end": 4.0}]',
                       monkeypatch=monkeypatch)
    assert (audio[100:] == 0.0).all()
    assert (audio == 0.0).all()


def test_encoder_has_no_mask_params(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    keys = set(nodes.LanPaint_MiniMaxAudioEncode.INPUT_TYPES()["required"])
    assert "mask_start" not in keys and "mask_end" not in keys
    assert keys == {"audio", "vae"}


def test_reshape_audio_mask_1d_to_tokens(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    # [F=100] audio mask at video rate -> the audio latent [1, 32, 2, 40]
    mask = torch.zeros(100)
    mask[50:60] = 1.0
    out = nodes.reshape_mask(mask, (1, 32, 2, 40), video_inpainting=False)
    assert out.shape == (1, 32, 2, 40)
    assert (out == 0.0).float().mean() > 0.7  # only the interval region is on
    assert out[0, 0, 0].max() == 1.0
    # nearest mapping: frame 50 of 100 lands near token 20 of 40
    assert out[0, 0, 0, 20] == 1.0 and out[0, 0, 1, 20] == 1.0


# --- temporal union in reshape_mask (window-4 max + nearest-exact) ---------

def _reshape(nodes, mask, output_shape):
    return nodes.reshape_mask(mask, output_shape, video_inpainting=True)


def test_union_window4_max_takes_the_union(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    # frame 3 (window 0: frames 0-3) and frame 5 (window 1: frames 4-7)
    mask = torch.zeros(8, 6, 8)
    mask[3, 2, 3] = 1.0
    mask[5, 4, 5] = 1.0
    out = _reshape(nodes, mask, (1, 24, 2, 6, 8))
    assert out.shape == (1, 24, 2, 6, 8)
    assert out[0, 0, 0, 2, 3] == 1.0  # window 0: union of frames 0-3
    assert out[0, 0, 1, 4, 5] == 1.0  # window 1: union of frames 4-7
    assert out[0, 0, 0].max() == 1.0 and out[0, 0, 1].max() == 1.0


def test_union_sparse_stroke_survives(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    # a stroke on a single frame must NOT average away: its token regenerates
    mask = torch.zeros(16, 4, 4)
    mask[7, 1, 1] = 1.0
    out = _reshape(nodes, mask, (1, 24, 4, 4, 4))
    assert out.shape == (1, 24, 4, 4, 4)
    assert out.max() == 1.0  # the painted frame's token is fully regenerated
    assert out[0, 0, 0].max() == 0.0  # windows without paint stay 0


def test_union_resamples_to_latent_shape_nearest(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    # 8 frames -> 2 windows -> nearest-exact to the latent (T=2, /2 spatial)
    mask = torch.zeros(8, 8, 6)
    mask[5, 4:6, 2:4] = 1.0  # window 1 (frames 4-7), a small stroke
    out = _reshape(nodes, mask, (1, 24, 2, 4, 3))
    assert out.shape == (1, 24, 2, 4, 3)
    assert out[0, 0, 0].max() == 0.0
    assert out[0, 0, 1].max() == 1.0
    # nearest-exact spatial: the painted pixel survives at its mapped location
    assert out[0, 0, 1, 2, 1] == 1.0


def test_union_124_frames_to_37_tokens(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    mask = torch.zeros(124, 864, 480)
    mask[60, 100:140, 100:140] = 1.0  # a brush stroke on a single frame
    out = _reshape(nodes, mask, (1, 24, 37, 30, 54))
    assert out.shape == (1, 24, 37, 30, 54)
    assert out.max() == 1.0  # the paint is covered by some token
    # most tokens are empty (no window-4 smear)
    assert (out == 0.0).float().mean() > 0.95


def test_union_leaves_static_single_frame_mask(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    mask = torch.ones(1, 6, 8)  # static image mask over the whole video
    out = _reshape(nodes, mask, (1, 24, 37, 3, 4))
    assert out.shape == (1, 24, 37, 3, 4)
    assert out.max() == 1.0  # duplicated to every token


def test_union_skips_non_video_masks(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    # video_inpainting=False -> the 2D image branch (unchanged behavior)
    mask = torch.ones(4, 6, 8)
    out = nodes.reshape_mask(mask, (1, 4, 6, 8), video_inpainting=False)
    assert out.shape == (1, 4, 6, 8)
    assert out.max() == 1.0


def test_node_audio_fps_falls_back_to_get_frame_rate(monkeypatch, tmp_path) -> None:
    nodes = _import_nodes(monkeypatch, tmp_path)
    # newer ComfyUI: VideoFromFile has get_frame_rate (a Fraction) instead of get_fps
    class FakeVideoFromFile:
        def __init__(self, path):
            self.path = path

        def get_frame_count(self):
            return 100

        def get_dimensions(self):
            return (8, 6)

        def get_frame_rate(self):
            from fractions import Fraction
            return Fraction(24, 1)

    monkeypatch.setattr(nodes, "VideoFromFile", FakeVideoFromFile)
    _, _, audio = nodes.LanPaint_VideoMaskEditor().run(
        video="video.mp4", keyframes="{}",
        audio_mask='[{"start": 1.0, "end": 2.0}]')
    # 24 fps: [1.0, 2.0]s covers frames 24..47
    assert (audio[24:48] == 1.0).all()
    assert (audio[:24] == 0.0).all() and (audio[48:] == 0.0).all()
