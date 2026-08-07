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


class StubAudioVAE:
    """Emulates the sd.VAE wrapper: channels-last audio in/out, converts to
    [B, C, L] for the raw VAE internally (as sd.py:1040/1216 do)."""

    audio_sample_rate = 32000

    def __init__(self):
        self.last_input = None

    def encode(self, waveform):
        self.last_input = waveform
        w = waveform.movedim(-1, 1)  # [B, L, C] -> [B, C, L]
        t = max(1, w.shape[-1] // 800)
        return torch.zeros(w.shape[0], 32, 2, t)

    def decode(self, z):
        return torch.zeros(z.shape[0], z.shape[-1] * 800, 2)  # [B, L, C]


def _audio(samples: int, channels: int = 2, sample_rate: int = 32000):
    return {"waveform": torch.zeros(1, channels, samples), "sample_rate": sample_rate}


def test_encode_produces_correct_latent_and_no_mask(monkeypatch) -> None:
    nodes = _import_nodes(monkeypatch)
    vae = StubAudioVAE()
    out = nodes.LanPaint_MiniMaxAudioEncode().encode(_audio(320000), vae)
    latent = out[0]
    assert latent["samples"].shape == (1, 32, 2, 400)
    # pure encoder: masks come from the video mask editor via SetLatentNoiseMask
    assert "noise_mask" not in latent


def test_encode_resamples_and_upmixes_mono(monkeypatch) -> None:
    nodes = _import_nodes(monkeypatch)
    if nodes.torchaudio is None:
        pytest.skip("torchaudio not available")
    vae = StubAudioVAE()
    mono_44k = _audio(320000, channels=1, sample_rate=44100)
    nodes.LanPaint_MiniMaxAudioEncode().encode(mono_44k, vae)
    # wrapper receives channels-last stereo at its own sample rate
    assert vae.last_input.shape[-1] == 2                       # mono upmixed to stereo (channels last)
    assert 200000 < vae.last_input.shape[1] < 320000           # resampled 44.1k -> 32k shortens


def test_decode_returns_waveform_and_sample_rate(monkeypatch) -> None:
    nodes = _import_nodes(monkeypatch)
    vae = StubAudioVAE()
    audio = nodes.LanPaint_MiniMaxAudioDecode().decode({"samples": torch.zeros(1, 32, 2, 400)}, vae)
    assert audio[0]["waveform"].shape == (1, 2, 320000)
    assert audio[0]["sample_rate"] == 32000


def test_decode_handles_nested_av_latent(monkeypatch) -> None:
    nodes = _import_nodes(monkeypatch)
    vae = StubAudioVAE()

    class FakeNested:
        is_nested = True

        def __init__(self, tensors):
            self.tensors = tensors

        def unbind(self):
            return self.tensors

    audio = nodes.LanPaint_MiniMaxAudioDecode().decode({"samples": FakeNested([torch.zeros(1, 24, 4, 8, 8), torch.zeros(1, 32, 2, 400)])}, vae)
    assert audio[0]["waveform"].shape == (1, 2, 320000)
