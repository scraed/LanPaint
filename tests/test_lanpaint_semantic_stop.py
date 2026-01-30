import torch

from src.LanPaint.lanpaint import LanPaint as LanPaintEngine


class _DummySampling:
    def noise_scaling(self, sigma, noise, latent_image):  # type: ignore[no-untyped-def]
        return latent_image + noise * sigma


class _DummyModel:
    def __init__(self) -> None:
        self.inner_model = self
        self.model_sampling = _DummySampling()

    def __call__(self, x, sigma, model_options=None, seed=None):  # type: ignore[no-untyped-def]
        return x, x


def _inputs():  # type: ignore[no-untyped-def]
    x = torch.zeros((1, 4, 8, 8))
    latent_image = torch.zeros_like(x)
    noise = torch.ones_like(x)
    sigma = torch.tensor([1.0])

    latent_mask = torch.zeros_like(x)
    current_times = (sigma, torch.tensor([0.5]), torch.tensor([0.0]))
    return x, latent_image, noise, sigma, latent_mask, current_times


def test_default_semantic_stop_triggers_at_patience_without_custom_distance_fn() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    calls = {"langevin": 0, "with_score": 0, "without_score": 0}

    def fake_langevin(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        if score is None:
            calls["without_score"] += 1
        else:
            calls["with_score"] += 1
        return x_t, args

    engine.langevin_dynamics = fake_langevin  # type: ignore[method-assign]

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 1e-6,
            "patience": 2,
        }
    }

    x, latent_image, noise, sigma, latent_mask, current_times = _inputs()
    engine(x, latent_image, noise, sigma, latent_mask, current_times, model_options=model_options, seed=0, n_steps=10)

    assert calls["langevin"] == 3
    assert calls["with_score"] == 3
    assert calls["without_score"] == 0


def test_semantic_stop_is_disabled_when_no_inpaint_region() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    calls = {"langevin": 0, "with_score": 0, "without_score": 0}

    def fake_langevin(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        if score is None:
            calls["without_score"] += 1
        else:
            calls["with_score"] += 1
        return x_t, args

    engine.langevin_dynamics = fake_langevin  # type: ignore[method-assign]

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 1e-6,
            "patience": 1,
        }
    }

    x, latent_image, noise, sigma, latent_mask, _ = _inputs()
    current_times = (sigma, torch.tensor([0.5]), torch.tensor([0.0]))
    no_inpaint_mask = torch.ones_like(latent_mask)
    engine(x, latent_image, noise, sigma, no_inpaint_mask, current_times, model_options=model_options, seed=0, n_steps=10)

    assert calls["langevin"] == 10
    assert calls["with_score"] == 10
    assert calls["without_score"] == 0
