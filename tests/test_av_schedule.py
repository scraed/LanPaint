import importlib
import sys
import types

import pytest
import torch

from src.LanPaint.lanpaint import LanPaint as LanPaintEngine


def _repeat_to_batch_size(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.repeat((batch_size,) + (1,) * (tensor.ndim - 1))
    repeats = (batch_size + tensor.shape[0] - 1) // tensor.shape[0]
    return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))[:batch_size]


def _import_nodes(monkeypatch):
    comfy_mod = types.ModuleType("comfy")
    comfy_mod.__path__ = []

    comfy_utils_mod = types.ModuleType("comfy.utils")
    comfy_utils_mod.repeat_to_batch_size = _repeat_to_batch_size

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
    monkeypatch.setitem(sys.modules, "nodes", types.ModuleType("nodes"))
    monkeypatch.setitem(sys.modules, "latent_preview", types.ModuleType("latent_preview"))
    monkeypatch.setitem(sys.modules, "comfyui_version", comfyui_version_mod)

    sys.modules.pop("src.LanPaint.nodes", None)
    return importlib.import_module("src.LanPaint.nodes")


class _FakeDiffusionModel:
    sigma_shift_video = 12.0
    sigma_shift_audio = 3.0


class _FakeModelPatcher:
    def __init__(self, model):
        self.model = model


def _two_stream_shapes():
    return [(1, 24, 37, 30, 54), (1, 32, 2, 207)]


def _minimax_model():
    model = type("FakeMiniMaxH3", (), {})()
    model.diffusion_model = _FakeDiffusionModel()
    return model


# --- detection: only MiniMax H3 AV packs get the audio schedule -----------------

def test_detect_returns_none_for_single_stream_latent(monkeypatch) -> None:
    nodes = _import_nodes(monkeypatch)
    patcher = _FakeModelPatcher(_minimax_model())
    assert nodes._detect_minimax_h3_audio(patcher, {}, [(1, 24, 37, 30, 54)]) is None
    assert nodes._detect_minimax_h3_audio(patcher, {}, None) is None


def test_detect_returns_none_for_model_without_audio_schedule(monkeypatch) -> None:
    nodes = _import_nodes(monkeypatch)
    patcher = _FakeModelPatcher(object())  # no diffusion_model / no shift attrs
    assert nodes._detect_minimax_h3_audio(patcher, {}, _two_stream_shapes()) is None


def test_detect_returns_layout_for_minimax_av_pack(monkeypatch) -> None:
    nodes = _import_nodes(monkeypatch)
    patcher = _FakeModelPatcher(_minimax_model())
    layout = nodes._detect_minimax_h3_audio(patcher, {}, _two_stream_shapes())
    assert layout == (_two_stream_shapes(), 12.0, 3.0)


def test_detect_honors_sigma_shift_node_overrides(monkeypatch) -> None:
    nodes = _import_nodes(monkeypatch)
    patcher = _FakeModelPatcher(_minimax_model())
    options = {"transformer_options": {"minimax_h3_sigma_shift_video": 10.0, "minimax_h3_sigma_shift_audio": 2.5}}
    layout = nodes._detect_minimax_h3_audio(patcher, options, _two_stream_shapes())
    assert layout == (_two_stream_shapes(), 10.0, 2.5)


def test_guarded_imports_fall_back_in_stub_env(monkeypatch) -> None:
    nodes = _import_nodes(monkeypatch)
    assert nodes.time_shift_sigma is None  # comfy.ldm.minimax not importable here


# --- per-stream schedule blending in the paint loop -----------------------------

class _DummySampling:
    """Emulates the flow model_sampling: CONST.noise_scaling is
    sigma * (s * noise) + (1 - sigma) * latent and reshape_sigma requires a
    scalar sigma."""

    noise_scale = 1.0

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):  # type: ignore[no-untyped-def]
        assert sigma.numel() == 1, "noise_scaling requires a scalar sigma"
        return sigma * (self.noise_scale * noise) + (1.0 - sigma) * latent_image


class _DummyModel:
    def __init__(self) -> None:
        self.inner_model = self
        self.model_sampling = _DummySampling()
        self.last_input = None

    def __call__(self, x, sigma, model_options=None, seed=None):  # type: ignore[no-untyped-def]
        self.last_input = x
        return x, x


def _engine(n_steps: int = 1):  # type: ignore[no-untyped-def]
    return LanPaintEngine(
        _DummyModel(),
        NSteps=n_steps,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )


def _flat_pack_inputs():  # type: ignore[no-untyped-def]
    x = torch.zeros(1, 1, 8)  # flat pack: 5 video rows + 3 audio rows
    latent_image = torch.zeros_like(x)
    noise = torch.ones_like(x)
    sigma = torch.tensor([0.5])
    latent_mask = torch.zeros_like(x)  # 1 = keep -> regenerate everywhere by default
    current_times = (torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([0.5]))  # video: VE, abt, flow
    current_times_audio = (torch.tensor([0.25]), torch.tensor([0.9]), torch.tensor([0.2]))  # audio: VE, abt, flow
    audio_indicator = torch.zeros(1, 1, 8)
    audio_indicator[..., 5:] = 1.0
    return x, latent_image, noise, sigma, latent_mask, current_times, current_times_audio, audio_indicator


def test_audio_rows_get_audio_schedule_parameters() -> None:
    engine = _engine()
    captured = {}

    def fake_prepare_step_size(current_times, step_size, sigma_x, sigma_y):  # type: ignore[no-untyped-def]
        captured["abt"] = current_times[1]
        captured["step_size"] = step_size
        abt = current_times[1]
        ones = torch.ones_like(abt)
        return (current_times[0], abt, ones, ones, ones, ones, torch.zeros_like(abt), torch.zeros_like(abt), torch.zeros_like(abt), torch.zeros_like(abt))

    engine.prepare_step_size = fake_prepare_step_size  # type: ignore[method-assign]

    x, latent_image, noise, sigma, latent_mask, current_times, current_times_audio, audio_indicator = _flat_pack_inputs()
    engine(x, latent_image, noise, sigma, latent_mask, current_times,
           model_options=None, seed=0, n_steps=1,
           current_times_audio=current_times_audio, audio_indicator=audio_indicator)

    abt = captured["abt"].flatten()
    assert abt[0] == 0.5            # video rows keep the video schedule
    assert abt[-1] == 0.9           # audio rows run on the audio schedule
    step = captured["step_size"].flatten()
    assert step[0] == pytest.approx(0.2 * (1 - 0.5))   # video step size
    assert step[-1] == pytest.approx(0.2 * (1 - 0.9))  # audio step size


def test_without_audio_context_schedule_stays_uniform() -> None:
    engine = _engine()
    captured = {}

    def fake_prepare_step_size(current_times, step_size, sigma_x, sigma_y):  # type: ignore[no-untyped-def]
        captured["abt"] = current_times[1]
        abt = current_times[1]
        ones = torch.ones_like(abt)
        return (current_times[0], abt, ones, ones, ones, ones, torch.zeros_like(abt), torch.zeros_like(abt), torch.zeros_like(abt), torch.zeros_like(abt))

    engine.prepare_step_size = fake_prepare_step_size  # type: ignore[method-assign]

    x, latent_image, noise, sigma, latent_mask, current_times, _, _ = _flat_pack_inputs()
    engine(x, latent_image, noise, sigma, latent_mask, current_times,
           model_options=None, seed=0, n_steps=1)

    abt = captured["abt"].flatten()
    assert abt[0] == 0.5
    assert abt[-1] == 0.5  # uniform: the video schedule everywhere


def test_replace_step_uses_audio_sigma_for_audio_rows() -> None:
    # n_steps=0 with keep-everywhere mask: the only transform applied is the
    # replace step, and the final model call sees its result (dummy returns x).
    engine = _engine(n_steps=0)
    x, latent_image, noise, sigma, _, current_times, current_times_audio, audio_indicator = _flat_pack_inputs()
    latent_mask = torch.ones_like(x)  # keep everywhere -> replace step applies

    engine(x, latent_image, noise, sigma, latent_mask, current_times,
           model_options=None, seed=0, n_steps=0,
           current_times_audio=current_times_audio, audio_indicator=audio_indicator)

    inp = engine.inner_model.last_input.flatten()
    # replace step = sigma_eff * (s * noise) + (1 - sigma_eff) * latent, with
    # noise = 1 and latent = 0 -> the effective flow sigma itself
    assert inp[0] == pytest.approx(0.5)  # video rows: sigma_v
    assert inp[-1] == pytest.approx(0.2)  # audio rows: sigma_audio


class _OffsetModel:
    """Returns x + offset for both heads, so the flat-grid target deviation is known."""

    def __init__(self, offset):  # type: ignore[no-untyped-def]
        self.inner_model = self
        self.model_sampling = _DummySampling()
        self.offset = offset

    def __call__(self, x, sigma, model_options=None, seed=None):  # type: ignore[no-untyped-def]
        return x + self.offset, x + self.offset


def _score_engine(offset: float = 2.0):  # type: ignore[no-untyped-def]
    engine = LanPaintEngine(
        _OffsetModel(torch.tensor(offset)),
        NSteps=1, Friction=15.0, Lambda=1.0, Beta=1.0, StepSize=0.2,
        IS_FLOW=True,
    )
    engine.img_dim_size = 3
    return engine


def test_score_model_corrects_audio_target_only() -> None:
    engine = _score_engine()
    ai = torch.zeros(1, 1, 8)
    ai[..., 5:] = 1.0
    engine.audio_indicator = ai
    engine.audio_correction = (1.0 - ai) + 0.625 * ai  # c = sigma_a/(sigma_v*slope_a) at sigma_v=0.5

    x_t = torch.zeros(1, 1, 8)
    y = torch.zeros(1, 1, 8)
    mask = torch.zeros(1, 1, 8)  # regenerate everywhere -> score_x branch
    abt = torch.full((1, 1, 8), 0.5)
    sigma = torch.ones(1, 1, 8)
    tflow = engine.add_none_dims(torch.tensor([0.5]))

    score = engine.score_model(x_t, y, mask, abt, sigma, tflow, model_options=None, seed=0)
    s = score.flatten()
    # x = 0, x0_flat = 2 -> corrected x0 = 2*corr -> score = 2*corr
    assert s[0] == pytest.approx(2.0)    # video rows: corr = 1 -> untouched
    assert s[-1] == pytest.approx(1.25)  # audio rows: 2 * 0.625


def test_score_model_without_audio_context_uses_flat_target() -> None:
    engine = _score_engine()
    x_t = torch.zeros(1, 1, 8)
    y = torch.zeros(1, 1, 8)
    mask = torch.zeros(1, 1, 8)
    abt = torch.full((1, 1, 8), 0.5)
    sigma = torch.ones(1, 1, 8)
    tflow = engine.add_none_dims(torch.tensor([0.5]))

    score = engine.score_model(x_t, y, mask, abt, sigma, tflow, model_options=None, seed=0)
    s = score.flatten()
    assert s[0] == pytest.approx(2.0)
    assert s[-1] == pytest.approx(2.0)  # uniform flat target: no correction


# --- add_none_dims equivalence ------------------------------------------------

def test_add_none_dims_equivalent_for_scalar_inputs() -> None:
    engine = _engine()
    for img_dim_size in (3, 4, 5):
        engine.img_dim_size = img_dim_size
        t = torch.zeros(1)
        old = t[(slice(None),) + (None,) * (img_dim_size - 1)]
        new = engine.add_none_dims(t.clone())
        assert tuple(new.shape) == tuple(old.shape)
        # the while-loop additionally handles 0-dim inputs, which the
        # tuple-index form rejects in modern torch
        assert tuple(engine.add_none_dims(torch.zeros(())).shape) == (1,) * img_dim_size


def test_add_none_dims_passes_per_row_tensors_through() -> None:
    engine = _engine()
    engine.img_dim_size = 3
    t = torch.zeros(1, 1, 8)
    assert tuple(engine.add_none_dims(t).shape) == (1, 1, 8)


# --- prepare_step_size with per-row parameters --------------------------------

def test_prepare_step_size_handles_per_row_parameters() -> None:
    engine = _engine()
    engine.img_dim_size = 3
    sigma = torch.ones(1, 1, 8)
    abt = torch.full((1, 1, 8), 0.5)
    abt[..., 5:] = 0.9
    step_size = torch.full((1, 1, 8), 0.1)
    step_size[..., 5:] = 0.02
    sigma_x = torch.ones(1, 1, 8)
    sigma_y = torch.ones(1, 1, 8)

    out = engine.prepare_step_size((sigma, abt, sigma), step_size, sigma_x, sigma_y)
    for t in out:
        assert t.ndim == 3  # everything stays per-row broadcastable

    _, abt_out, dtx, dty, Gamma_x, Gamma_y, A_x, A_y, D_x, D_y = out
    # the A * dt cancellation: A_x = 1/(1-abt), dt = dtx/2 = step_size * sigma_x
    # -> A_x * dt = step_size / (1-abt) * (1-abt) * ... bounded per row
    adt = (A_x * dtx).flatten()
    assert adt[0] == pytest.approx(0.2)   # 1/(1-0.5) * 0.1
    assert adt[-1] == pytest.approx(0.2)  # 1/(1-0.9) * 0.02 -- bounded invariant
