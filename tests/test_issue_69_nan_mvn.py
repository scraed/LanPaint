import torch
from torch.distributions import Distribution

from src.LanPaint.lanpaint import LanPaint


def test_langevin_dynamics_does_not_crash_on_nan_score() -> None:
    """Regression test for issue #69: NaNs used to crash MultivariateNormal(validate_args=True)."""

    torch.manual_seed(0)

    prev_validate_args = Distribution._validate_args
    Distribution.set_default_validate_args(True)
    try:
        lanpaint = LanPaint(
            Model=None,
            NSteps=1,
            Friction=1.0,
            Lambda=1.0,
            Beta=1.0,
            StepSize=0.1,
        )

        x_t = torch.zeros((1, 16, 1, 16, 16), dtype=torch.float32)
        lanpaint.img_dim_size = x_t.ndim

        mask = torch.zeros_like(x_t)
        batch = x_t.shape[0]
        abt = torch.full((batch,), 0.5, dtype=x_t.dtype)
        current_times = (
            torch.full((batch,), 0.5, dtype=x_t.dtype),
            abt,
            torch.zeros((batch,), dtype=x_t.dtype),
        )

        step_size = lanpaint.add_none_dims(torch.full((batch,), 0.1, dtype=x_t.dtype))

        def nan_score(x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x, float("nan"))

        out, _ = lanpaint.langevin_dynamics(
            x_t=x_t,
            score=nan_score,
            mask=mask,
            step_size=step_size,
            current_times=current_times,
            sigma_x=lanpaint.add_none_dims(lanpaint.sigma_x(abt)),
            sigma_y=lanpaint.add_none_dims(lanpaint.sigma_y(abt)),
            args=None,
        )
    finally:
        Distribution.set_default_validate_args(prev_validate_args)

    assert torch.isfinite(out).all()

