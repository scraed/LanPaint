import torch
import torch.nn.functional as F

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


def test_semantic_stop_triggers_deterministically_at_min_steps() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    calls = {"langevin": 0, "distance": 0, "with_score": 0, "without_score": 0}

    def fake_langevin(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        if score is None:
            calls["without_score"] += 1
        else:
            calls["with_score"] += 1
        return x_t, args

    def distance_fn(prev_x, cur_x, ctx):  # type: ignore[no-untyped-def]
        assert ctx["step"] == calls["distance"]
        calls["distance"] += 1
        return 0.0

    engine.langevin_dynamics = fake_langevin  # type: ignore[method-assign]

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 1e-6,
            "min_steps": 3,
            "patience": 1,
            "distance_fn": distance_fn,
        }
    }

    x, latent_image, noise, sigma, latent_mask, current_times = _inputs()
    engine(
        x,
        latent_image,
        noise,
        sigma,
        latent_mask,
        current_times,
        model_options=model_options,
        seed=0,
        n_steps=10,
    )

    assert calls["langevin"] == 3
    assert calls["with_score"] == 3
    assert calls["without_score"] == 0
    assert calls["distance"] == 3


def test_default_semantic_stop_triggers_at_min_steps_without_custom_distance_fn() -> None:
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
            "min_steps": 3,
            "patience": 1,
        }
    }

    x, latent_image, noise, sigma, latent_mask, current_times = _inputs()
    engine(x, latent_image, noise, sigma, latent_mask, current_times, model_options=model_options, seed=0, n_steps=10)

    assert calls["langevin"] == 3
    assert calls["with_score"] == 3
    assert calls["without_score"] == 0


def test_semantic_stop_is_disabled_for_very_late_steps() -> None:
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
            "min_steps": 2,
            "patience": 1,
        }
    }

    x, latent_image, noise, sigma, latent_mask, current_times = _inputs()

    # Use a constant per-step x_t update (simulating Langevin noise) while keeping
    # an x0-like tensor perfectly stable. Early-stop should ignore the x_t noise.
    delta = 0.01

    def fake_langevin_delta(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        if score is None:
            calls["without_score"] += 1
        else:
            calls["with_score"] += 1
        prev_x0 = None if args is None else args[2]
        x0 = x_t.detach().clone() * 0 if prev_x0 is None else prev_x0
        return x_t + delta, (None, None, x0)

    engine.langevin_dynamics = fake_langevin_delta  # type: ignore[method-assign]

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 1e-12,
            "min_steps": 2,
            "patience": 1,
        }
    }

    calls["langevin"] = 0
    current_times = (sigma, torch.tensor([0.5]), torch.tensor([0.0]))
    engine(
        x,
        latent_image,
        noise,
        sigma,
        latent_mask,
        current_times,
        model_options=model_options,
        seed=0,
        n_steps=10,
    )
    assert calls["langevin"] == 3
    assert calls["with_score"] == 3
    assert calls["without_score"] == 0

    calls["langevin"] = 0
    calls["with_score"] = 0
    calls["without_score"] = 0
    current_times = (sigma, torch.tensor([0.99]), torch.tensor([0.0]))
    engine(
        x,
        latent_image,
        noise,
        sigma,
        latent_mask,
        current_times,
        model_options=model_options,
        seed=0,
        n_steps=10,
    )
    assert calls["langevin"] == 4
    assert calls["with_score"] == 4
    assert calls["without_score"] == 0


def test_semantic_stop_is_disabled_for_very_early_steps() -> None:
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
            "min_steps": 2,
            "patience": 1,
        }
    }

    x, latent_image, noise, sigma, latent_mask, _ = _inputs()
    current_times = (sigma, torch.tensor([0.0]), torch.tensor([0.0]))
    engine(
        x,
        latent_image,
        noise,
        sigma,
        latent_mask,
        current_times,
        model_options=model_options,
        seed=0,
        n_steps=10,
    )

    assert calls["langevin"] == 10
    assert calls["with_score"] == 10
    assert calls["without_score"] == 0


def test_default_semantic_stop_is_mask_normalized() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    calls = {"langevin": 0, "with_score": 0, "without_score": 0}
    delta = 0.01

    def fake_langevin_delta(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        if score is None:
            calls["without_score"] += 1
        else:
            calls["with_score"] += 1
        prev_x0 = None if args is None else args[2]
        x0 = x_t.detach().clone() * 0 if prev_x0 is None else prev_x0
        x0 = x0 + delta * (1 - mask)
        return x_t, (None, None, x0)

    engine.langevin_dynamics = fake_langevin_delta  # type: ignore[method-assign]

    # Choose a threshold that would previously depend on how much of the tensor is masked.
    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": (delta * delta) * 0.2,
            "min_steps": 2,
            "patience": 1,
        }
    }

    x, latent_image, noise, sigma, latent_mask, current_times = _inputs()
    current_times = (sigma, torch.tensor([0.0]), torch.tensor([0.0]))

    # Small hole (unknown region) -> mask mostly 1, with a small 0 patch.
    small_hole_mask = torch.ones_like(latent_mask)
    small_hole_mask[:, :, 3:5, 3:5] = 0.0

    calls["langevin"] = 0
    calls["with_score"] = 0
    calls["without_score"] = 0
    engine(
        x,
        latent_image,
        noise,
        sigma,
        small_hole_mask,
        current_times,
        model_options=model_options,
        seed=0,
        n_steps=10,
    )
    assert calls["langevin"] == 10
    assert calls["with_score"] == 10
    assert calls["without_score"] == 0

    # Larger hole (unknown region).
    large_hole_mask = torch.ones_like(latent_mask)
    large_hole_mask[:, :, 1:7, 1:7] = 0.0

    calls["langevin"] = 0
    calls["with_score"] = 0
    calls["without_score"] = 0
    engine(
        x,
        latent_image,
        noise,
        sigma,
        large_hole_mask,
        current_times,
        model_options=model_options,
        seed=0,
        n_steps=10,
    )
    assert calls["langevin"] == 10
    assert calls["with_score"] == 10
    assert calls["without_score"] == 0


def test_semantic_stop_does_not_stop_while_ring_changes() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    batch = 1
    channels = 4
    height = 64
    width = 64

    x = torch.zeros((batch, channels, height, width))
    latent_image = torch.zeros_like(x)
    noise = torch.ones_like(x)
    sigma = torch.tensor([1.0])

    # latent_mask uses LanPaint convention: 1=known region, 0=to-inpaint.
    latent_mask = torch.ones_like(x)
    latent_mask[:, :, 12:52, 12:52] = 0.0
    inpaint_mask = 1.0 - latent_mask

    # Replicate the "ring" construction in LanPaint: a dilated mask boundary
    # intersected with the inpaint region.
    mask_f = latent_mask.to(dtype=torch.float32)
    dilated = F.max_pool2d(mask_f, kernel_size=11, stride=1, padding=5)
    ring_mask = (dilated - mask_f).clamp(min=0.0, max=1.0) * inpaint_mask.to(dtype=torch.float32)

    ring_area = float(torch.sum(ring_mask).item())
    inpaint_area = float(torch.sum(inpaint_mask).item())
    assert ring_area > 0.0
    assert ring_area < inpaint_area

    threshold = 1e-6
    ratio = inpaint_area / ring_area
    delta_sq = threshold * ratio * 0.9
    delta = float(delta_sq**0.5)

    calls = {"langevin": 0, "with_score": 0, "without_score": 0}

    def fake_langevin_delta(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        if score is None:
            calls["without_score"] += 1
        else:
            calls["with_score"] += 1

        # Simulate a case where the interior is stable, but the boundary ring keeps changing.
        step = float(calls["langevin"])
        x0 = ring_mask.to(dtype=x_t.dtype) * (step * delta)
        return x_t, (None, None, x0)

    engine.langevin_dynamics = fake_langevin_delta  # type: ignore[method-assign]

    current_times = (sigma, torch.tensor([0.0]), torch.tensor([0.0]))
    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": threshold,
            "min_steps": 2,
            "patience": 1,
        }
    }

    engine(
        x,
        latent_image,
        noise,
        sigma,
        latent_mask,
        current_times,
        model_options=model_options,
        seed=0,
        n_steps=10,
    )

    # If ring changes are respected, we should never stop early.
    assert calls["langevin"] == 10
    assert calls["with_score"] == 10
    assert calls["without_score"] == 0

    # If ring changes were ignored, inpaint-average MSE would fall below the threshold
    # and the sampler would stop early. This test ensures we keep running the model
    # while the boundary is still changing.
