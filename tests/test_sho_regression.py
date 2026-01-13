import torch
from torch.distributions import Distribution

from src.LanPaint.utils import StochasticHarmonicOscillator


def test_dynamics_does_not_pass_nan_loc_to_multivariatenormal() -> None:
    """Regression test: previously NaNs in mean crashed MultivariateNormal when validate_args=True."""

    torch.manual_seed(0)

    prev_validate_args = Distribution._validate_args
    Distribution.set_default_validate_args(True)
    try:
        osc = StochasticHarmonicOscillator(
            Gamma=torch.tensor(1.0),
            A=torch.tensor(1.0),
            C=torch.tensor(float("nan")),
            D=torch.tensor(1.0),
        )

        y0 = torch.zeros((1, 16, 1, 8, 8), dtype=torch.float32)
        v0 = torch.zeros_like(y0)
        t = torch.tensor(0.1)

        y1, v1 = osc.dynamics(y0=y0, v0=v0, t=t)
    finally:
        Distribution.set_default_validate_args(prev_validate_args)

    assert torch.isfinite(y1).all()
    assert torch.isfinite(v1).all()

