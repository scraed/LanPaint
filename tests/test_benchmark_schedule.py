import torch

from tests.benchmark_lpips import _make_mask, _sigmas_and_timesteps


def test_sigmas_schedule_descends_and_ends_at_zero() -> None:
    sigmas, timesteps = _sigmas_and_timesteps(
        inference_steps=20,
        train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    )

    assert sigmas.numel() == timesteps.numel() + 1
    assert sigmas[-1].item() == 0.0
    assert torch.all(sigmas[:-1] > 0.0)
    assert torch.all(sigmas[:-1] >= sigmas[1:])
    assert timesteps[0].item() >= timesteps[-1].item()


def test_make_mask_shapes() -> None:
    mask = _make_mask("box", 64, 64)
    assert tuple(mask.shape) == (1, 3, 64, 64)
    assert mask.dtype == torch.float32

