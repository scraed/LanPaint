import torch

from src.LanPaint.lanpaint import LanPaint as LanPaintEngine
from src.LanPaint.types import LangevinState


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


def test_semantic_stop_triggers_deterministically_at_patience() -> None:
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
            "patience": 2,
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


def test_semantic_stop_maps_min_steps_into_patience_floor() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    calls = {"langevin": 0, "distance": 0}

    def fake_langevin(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        return x_t, args

    def distance_fn(prev_x, cur_x, ctx):  # type: ignore[no-untyped-def]
        calls["distance"] += 1
        return 0.0

    engine.langevin_dynamics = fake_langevin  # type: ignore[method-assign]

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 1e-6,
            "patience": 1,
            "min_steps": 5,
            "distance_fn": distance_fn,
        }
    }

    x, latent_image, noise, sigma, latent_mask, current_times = _inputs()
    engine(x, latent_image, noise, sigma, latent_mask, current_times, model_options=model_options, seed=0, n_steps=10)

    # Legacy 'min_steps' is mapped into the effective patience floor, so stopping cannot happen before 5 steps.
    assert calls["langevin"] == 5
    assert calls["distance"] == 5


def test_semantic_stop_patience_is_not_abt_boosted() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    calls = {"langevin": 0, "distance": 0}

    def fake_langevin(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        return x_t, args

    def distance_fn(prev_x, cur_x, ctx):  # type: ignore[no-untyped-def]
        calls["distance"] += 1
        return 0.0

    engine.langevin_dynamics = fake_langevin  # type: ignore[method-assign]

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 1e-6,
            "patience": 1,
            "distance_fn": distance_fn,
        }
    }

    x, latent_image, noise, sigma, latent_mask, _ = _inputs()

    calls["langevin"] = 0
    calls["distance"] = 0
    current_times = (sigma, torch.tensor([0.5]), torch.tensor([0.0]))
    engine(x, latent_image, noise, sigma, latent_mask, current_times, model_options=model_options, seed=0, n_steps=10)
    assert calls["langevin"] == 2
    assert calls["distance"] == 2

    calls["langevin"] = 0
    calls["distance"] = 0
    current_times = (sigma, torch.tensor([0.99]), torch.tensor([0.0]))
    engine(x, latent_image, noise, sigma, latent_mask, current_times, model_options=model_options, seed=0, n_steps=10)
    assert calls["langevin"] == 2
    assert calls["distance"] == 2


def test_semantic_stop_is_not_disabled_below_old_min_abt_gate() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    calls = {"langevin": 0, "distance": 0}

    def fake_langevin(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        return x_t, args

    def distance_fn(prev_x, cur_x, ctx):  # type: ignore[no-untyped-def]
        calls["distance"] += 1
        return 0.0

    engine.langevin_dynamics = fake_langevin  # type: ignore[method-assign]

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 1e-6,
            "patience": 1,
            "distance_fn": distance_fn,
        }
    }

    x, latent_image, noise, sigma, latent_mask, _ = _inputs()
    current_times = (sigma, torch.tensor([0.1]), torch.tensor([0.0]))
    engine(x, latent_image, noise, sigma, latent_mask, current_times, model_options=model_options, seed=0, n_steps=10)

    assert calls["langevin"] == 2
    assert calls["distance"] == 2


def test_semantic_stop_distance_fn_two_arg_order_is_cur_prev() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    calls = {"langevin": 0, "distance": 0}

    def fake_langevin(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        return x_t + 1.0, args

    def distance_fn(cur_x, prev_x):  # type: ignore[no-untyped-def]
        assert float(torch.mean(cur_x).item()) > float(torch.mean(prev_x).item())
        calls["distance"] += 1
        return 0.0

    engine.langevin_dynamics = fake_langevin  # type: ignore[method-assign]

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 1e-6,
            "patience": 2,
            "distance_fn": distance_fn,
        }
    }

    x, latent_image, noise, sigma, latent_mask, current_times = _inputs()
    engine(x, latent_image, noise, sigma, latent_mask, current_times, model_options=model_options, seed=0, n_steps=10)

    assert calls["langevin"] == 3
    assert calls["distance"] == 3


def test_semantic_stop_distance_fn_accepts_third_param_not_named_ctx() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    calls = {"langevin": 0, "distance": 0}

    def fake_langevin(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        return x_t, args

    def distance_fn(prev_x, cur_x, meta):  # type: ignore[no-untyped-def]
        assert isinstance(meta, dict)
        assert meta["step"] == calls["distance"]
        calls["distance"] += 1
        return 0.0

    engine.langevin_dynamics = fake_langevin  # type: ignore[method-assign]

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 1e-6,
            "patience": 2,
            "distance_fn": distance_fn,
        }
    }

    x, latent_image, noise, sigma, latent_mask, current_times = _inputs()
    engine(x, latent_image, noise, sigma, latent_mask, current_times, model_options=model_options, seed=0, n_steps=10)

    assert calls["langevin"] == 3
    assert calls["distance"] == 3


def test_semantic_stop_distance_fn_accepts_kw_only_ctx() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    calls = {"langevin": 0, "distance": 0}

    def fake_langevin(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        return x_t, args

    def distance_fn(prev_x, cur_x, *, ctx):  # type: ignore[no-untyped-def]
        assert isinstance(ctx, dict)
        assert ctx["step"] == calls["distance"]
        calls["distance"] += 1
        return 0.0

    engine.langevin_dynamics = fake_langevin  # type: ignore[method-assign]

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 1e-6,
            "patience": 2,
            "distance_fn": distance_fn,
        }
    }

    x, latent_image, noise, sigma, latent_mask, current_times = _inputs()
    engine(x, latent_image, noise, sigma, latent_mask, current_times, model_options=model_options, seed=0, n_steps=10)

    assert calls["langevin"] == 3
    assert calls["distance"] == 3


def test_semantic_stop_distance_fn_accepts_scalar_tensor_return() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    calls = {"langevin": 0, "distance": 0}

    def fake_langevin(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        return x_t, args

    def distance_fn(prev_x, cur_x, ctx):  # type: ignore[no-untyped-def]
        calls["distance"] += 1
        return torch.tensor(0.0)

    engine.langevin_dynamics = fake_langevin  # type: ignore[method-assign]

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 1e-6,
            "patience": 2,
            "distance_fn": distance_fn,
        }
    }

    x, latent_image, noise, sigma, latent_mask, current_times = _inputs()
    engine(x, latent_image, noise, sigma, latent_mask, current_times, model_options=model_options, seed=0, n_steps=10)

    assert calls["langevin"] == 3
    assert calls["distance"] == 3


def test_semantic_stop_distance_fn_rejects_multi_element_tensor_return() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    def distance_fn(prev_x, cur_x, ctx):  # type: ignore[no-untyped-def]
        return torch.tensor([0.0, 0.0])

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 1e-6,
            "patience": 2,
            "distance_fn": distance_fn,
        }
    }

    x, latent_image, noise, sigma, latent_mask, current_times = _inputs()

    try:
        engine(x, latent_image, noise, sigma, latent_mask, current_times, model_options=model_options, seed=0, n_steps=10)
    except TypeError:
        return
    raise AssertionError("expected TypeError for multi-element tensor distance")


def test_semantic_stop_fallback_wrapper_does_not_swallow_internal_typeerror() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    class DistanceFn:  # type: ignore[no-untyped-def]
        __signature__ = "broken"

        def __call__(self, prev_x, cur_x, ctx):
            raise TypeError("internal")

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 1e-6,
            "patience": 1,
            "distance_fn": DistanceFn(),
        }
    }

    x, latent_image, noise, sigma, latent_mask, current_times = _inputs()

    try:
        engine(x, latent_image, noise, sigma, latent_mask, current_times, model_options=model_options, seed=0, n_steps=10)
    except TypeError:
        return
    raise AssertionError("expected TypeError from distance_fn to propagate")


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


def test_semantic_stop_is_more_conservative_near_tail_abt() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    calls = {"langevin": 0, "with_score": 0, "without_score": 0}

    x, latent_image, noise, sigma, latent_mask, current_times = _inputs()

    # Construct an x0 delta (MSE) that is below the mid-step effective threshold
    # but above the near-tail effective threshold. This ensures early-stop is
    # more conservative near the tail via threshold scaling.
    delta_sq = 1e-7
    delta = float(delta_sq**0.5)

    def fake_langevin_delta(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        if score is None:
            calls["without_score"] += 1
        else:
            calls["with_score"] += 1

        step = float(calls["langevin"])
        x0 = torch.zeros_like(x_t) + (step * delta)

        # Ensure the first iteration does not look stable due to x_t noise metric.
        return x_t + (1.0 if step <= 1 else 0.0), LangevinState(v=None, C=None, x0=x0)

    engine.langevin_dynamics = fake_langevin_delta  # type: ignore[method-assign]

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 1e-6,
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
    assert calls["langevin"] == 10
    assert calls["with_score"] == 10
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
        return x_t, LangevinState(v=None, C=None, x0=x0)

    engine.langevin_dynamics = fake_langevin_delta  # type: ignore[method-assign]

    # Choose a threshold that would diverge if the metric was accidentally area-scaled
    # (sum instead of mean) over the inpaint region.
    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": 0.005,
            "patience": 1,
        }
    }

    x, latent_image, noise, sigma, latent_mask, current_times = _inputs()
    current_times = (sigma, torch.tensor([0.5]), torch.tensor([0.0]))

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
    assert calls["langevin"] == 2
    assert calls["with_score"] == 2
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
    assert calls["langevin"] == 2
    assert calls["with_score"] == 2
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


def test_semantic_stop_drift_guard_prevents_premature_stop() -> None:
    engine = LanPaintEngine(
        _DummyModel(),
        NSteps=10,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.2,
    )

    calls = {"langevin": 0, "with_score": 0, "without_score": 0}

    threshold = 2e-5
    delta_sq = 1e-5
    delta = float(delta_sq**0.5)

    def fake_langevin_drift(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        if score is None:
            calls["without_score"] += 1
        else:
            calls["with_score"] += 1

        step = float(calls["langevin"])
        x0 = torch.zeros_like(x_t) + (step * delta)

        # Ensure the first iteration does not look stable due to x_t noise metric.
        return x_t + (1.0 if step <= 1 else 0.0), LangevinState(v=None, C=None, x0=x0)

    engine.langevin_dynamics = fake_langevin_drift  # type: ignore[method-assign]

    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": threshold,
            "patience": 3,
        }
    }

    x, latent_image, noise, sigma, latent_mask, _ = _inputs()
    current_times = (sigma, torch.tensor([0.5]), torch.tensor([0.0]))
    engine(x, latent_image, noise, sigma, latent_mask, current_times, model_options=model_options, seed=0, n_steps=10)

    assert calls["langevin"] == 10
    assert calls["with_score"] == 10
    assert calls["without_score"] == 0


def test_semantic_stop_does_not_stop_while_boundary_changes() -> None:
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

    # Construct a 4-neighbor boundary: unknown pixels adjacent to known pixels.
    known = latent_mask > 0.5
    neighbor_known = torch.zeros_like(known)
    neighbor_known[:, :, 1:, :] |= known[:, :, :-1, :]
    neighbor_known[:, :, :-1, :] |= known[:, :, 1:, :]
    neighbor_known[:, :, :, 1:] |= known[:, :, :, :-1]
    neighbor_known[:, :, :, :-1] |= known[:, :, :, 1:]

    boundary_mask = (inpaint_mask > 0.5) & neighbor_known
    boundary_weight = boundary_mask.to(dtype=torch.float32)

    boundary_area = float(torch.sum(boundary_weight).item())
    inpaint_area = float(torch.sum(inpaint_mask).item())
    assert boundary_area > 0.0
    assert boundary_area < inpaint_area

    threshold = 1e-6
    ratio = inpaint_area / boundary_area
    delta_sq = threshold * ratio * 0.9
    delta = float(delta_sq**0.5)

    calls = {"langevin": 0, "with_score": 0, "without_score": 0}

    def fake_langevin_delta(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        if score is None:
            calls["without_score"] += 1
        else:
            calls["with_score"] += 1

        # Simulate a case where the interior is stable, but the boundary keeps changing.
        step = float(calls["langevin"])
        x0 = boundary_weight.to(dtype=x_t.dtype) * (step * delta)
        return x_t, LangevinState(v=None, C=None, x0=x0)

    engine.langevin_dynamics = fake_langevin_delta  # type: ignore[method-assign]

    current_times = (sigma, torch.tensor([0.5]), torch.tensor([0.0]))
    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": threshold,
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


def test_semantic_stop_boundary_guard_is_adjacency_only() -> None:
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
    h0, h1 = 12, 52
    w0, w1 = 12, 52
    latent_mask[:, :, h0:h1, w0:w1] = 0.0
    inpaint_mask = 1.0 - latent_mask

    # Pixels at distance 3 from the boundary (inside the old 5px ring, but not in the 1px adjacency boundary).
    offset = 3
    changed_mask = torch.zeros_like(inpaint_mask)
    changed_mask[:, :, h0 + offset : h0 + offset + 1, w0 + offset : w1 - offset] = 1.0
    changed_mask[:, :, h1 - offset - 1 : h1 - offset, w0 + offset : w1 - offset] = 1.0
    changed_mask[:, :, h0 + offset + 1 : h1 - offset - 1, w0 + offset : w0 + offset + 1] = 1.0
    changed_mask[:, :, h0 + offset + 1 : h1 - offset - 1, w1 - offset - 1 : w1 - offset] = 1.0

    # Reference old "ring": all pixels within 5px of the boundary inside the hole.
    ring_width = 5
    ring5_mask = torch.zeros_like(inpaint_mask)
    ring5_mask[:, :, h0 : h0 + ring_width, w0:w1] = 1.0
    ring5_mask[:, :, h1 - ring_width : h1, w0:w1] = 1.0
    ring5_mask[:, :, h0 + ring_width : h1 - ring_width, w0 : w0 + ring_width] = 1.0
    ring5_mask[:, :, h0 + ring_width : h1 - ring_width, w1 - ring_width : w1] = 1.0

    changed_area = float(torch.sum(changed_mask).item())
    ring5_area = float(torch.sum(ring5_mask).item())
    inpaint_area = float(torch.sum(inpaint_mask).item())
    assert changed_area > 0.0
    assert ring5_area > changed_area
    assert inpaint_area > ring5_area

    threshold = 1e-6
    # Choose a delta so that the inpaint-average MSE falls below threshold, but the old 5px ring-average stays above.
    delta_sq = threshold * (ring5_area / changed_area) * 1.1
    assert (delta_sq * changed_area / inpaint_area) < threshold
    assert (delta_sq * changed_area / ring5_area) > threshold
    delta = float(delta_sq**0.5)

    calls = {"langevin": 0, "with_score": 0, "without_score": 0}

    def fake_langevin_delta(x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):  # type: ignore[no-untyped-def]
        calls["langevin"] += 1
        if score is None:
            calls["without_score"] += 1
        else:
            calls["with_score"] += 1

        step = float(calls["langevin"])
        x0 = changed_mask.to(dtype=x_t.dtype) * (step * delta)
        return x_t, LangevinState(v=None, C=None, x0=x0)

    engine.langevin_dynamics = fake_langevin_delta  # type: ignore[method-assign]

    current_times = (sigma, torch.tensor([0.5]), torch.tensor([0.0]))
    model_options = {
        "lanpaint_semantic_stop": {
            "threshold": threshold,
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

    # With an adjacency-only boundary guard, these changes should not prevent early-stop.
    assert calls["langevin"] == 2
    assert calls["with_score"] == 2
    assert calls["without_score"] == 0


def test_semantic_stop_distance_fn_dispatch_variants() -> None:
    # 1. Standard 3-arg
    def dist3(p, c, ctx):
        dist3.called = True
        return 0.0

    dist3.called = False

    # 2. 3-arg with different name
    def dist3_alt(p, c, context):
        dist3_alt.called = True
        return 0.0

    dist3_alt.called = False

    # 3. 2-arg
    def dist2(p, c):
        dist2.called = True
        return 0.0

    dist2.called = False

    # 4. Keyword-only ctx
    def dist_kw(p, c, *, ctx):
        dist_kw.called = True
        return 0.0

    dist_kw.called = False

    # 5. Varargs
    def dist_varargs(p, c, *args):
        dist_varargs.called = True
        return 0.0

    dist_varargs.called = False

    # 6. Varkwargs
    def dist_varkw(p, c, **kwargs):
        dist_varkw.called = True
        assert "ctx" in kwargs
        return 0.0

    dist_varkw.called = False

    variants = [dist3, dist3_alt, dist2, dist_kw, dist_varargs, dist_varkw]

    for fn in variants:
        engine = LanPaintEngine(_DummyModel(), NSteps=2, Friction=15.0, Lambda=1.0, Beta=1.0, StepSize=0.2)
        model_options = {
            "lanpaint_semantic_stop": {
                "threshold": 1e-6,
                "patience": 1,
                "distance_fn": fn,
            }
        }
        x, latent_image, noise, sigma, latent_mask, current_times = _inputs()
        engine(x, latent_image, noise, sigma, latent_mask, current_times, model_options=model_options, seed=0, n_steps=2)
        assert fn.called, f"Function {fn.__name__} was not called"
