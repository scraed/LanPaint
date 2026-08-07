import torch
from unittest.mock import MagicMock
from src.LanPaint.lanpaint import LanPaint


def test_langevin_dynamics_uses_first_order_scheme() -> None:
    """The second-order damped-oscillator scheme is disabled; langevin_dynamics
    always runs the first-order (overdamped) scheme, whose state carries no
    velocity and whose output stays finite."""
    torch.manual_seed(0)
    # Setup minimal LanPaint instance
    lp = LanPaint(Model=MagicMock(), NSteps=10, Friction=1.0, Lambda=1.0, Beta=1.0, StepSize=0.1)
    # Dummy inputs
    # Shape: (Batch, Channel, Height, Width)
    x_t = torch.randn(1, 4, 8, 8)
    lp.img_dim_size = 4
    mask = torch.zeros_like(x_t)
    # Simple score function
    def score(x):
        return torch.zeros_like(x)
    step_size = torch.tensor([0.1])
    # (sigma, abt, flow_t)
    current_times = (torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([0.5]))
    # Execute langevin_dynamics (first-order overdamped scheme directly)
    x_out, args_out = lp.langevin_dynamics(x_t, score, mask, step_size, current_times, sigma_y=1.0)
    assert hasattr(args_out, "v")
    assert hasattr(args_out, "C")
    assert hasattr(args_out, "x0")
    assert args_out.v is None  # first-order scheme: no velocity state
    assert args_out[1] is args_out.C
    assert args_out[2] is args_out.x0
    # Verify result is finite (the overdamped update is numerically robust)
    assert torch.isfinite(x_out).all(), "Output contains NaNs"
