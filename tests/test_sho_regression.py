import torch
from unittest.mock import MagicMock, patch
from src.LanPaint.lanpaint import LanPaint


def test_langevin_dynamics_fallback_on_nan() -> None:
    """Test that langevin_dynamics falls back to overdamped dynamics if damped dynamics produces NaNs."""
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
    # Mock StochasticHarmonicOscillator to return NaNs
    # We patch it where it is used (imported) in lanpaint.py
    with patch("src.LanPaint.lanpaint.StochasticHarmonicOscillator") as MockSHO:
        mock_instance = MockSHO.return_value
        # Configure dynamics to return NaNs
        nan_tensor = torch.full_like(x_t, float('nan'))
        mock_instance.dynamics.return_value = (nan_tensor, nan_tensor)
        # Execute langevin_dynamics
        # This should try run_damped -> get NaNs -> raise ValueError -> catch -> run_overdamped
        x_out, args_out = lp.langevin_dynamics(x_t, score, mask, step_size, current_times, sigma_y=1.0)
        assert hasattr(args_out, "v")
        assert hasattr(args_out, "C")
        assert hasattr(args_out, "x0")
        assert args_out[0] is args_out.v
        assert args_out[1] is args_out.C
        assert args_out[2] is args_out.x0
        v_out = args_out[0]
        # Verify that SHO was initialized and dynamics called
        MockSHO.assert_called()
        mock_instance.dynamics.assert_called()
        # Verify result is finite (indicating fallback to overdamped logic was successful)
        assert torch.isfinite(x_out).all(), "Output contains NaNs, fallback failed"
        assert v_out is None or torch.isfinite(v_out).all()


def test_langevin_dynamics_fallback_on_exception() -> None:
    """Test that langevin_dynamics falls back to overdamped dynamics if damped dynamics raises Exception."""
    torch.manual_seed(0)
    lp = LanPaint(Model=MagicMock(), NSteps=10, Friction=1.0, Lambda=1.0, Beta=1.0, StepSize=0.1)
    x_t = torch.randn(1, 4, 8, 8)
    lp.img_dim_size = 4
    mask = torch.zeros_like(x_t)
    score = lambda x: torch.zeros_like(x)
    step_size = torch.tensor([0.1])
    current_times = (torch.tensor([0.5]), torch.tensor([0.5]), torch.tensor([0.5]))
    with patch("src.LanPaint.lanpaint.StochasticHarmonicOscillator") as MockSHO:
        mock_instance = MockSHO.return_value
        # Configure dynamics to raise Exception
        mock_instance.dynamics.side_effect = RuntimeError("Simulation exploded")
        x_out, args_out = lp.langevin_dynamics(x_t, score, mask, step_size, current_times, sigma_y=1.0)
        assert hasattr(args_out, "v")
        assert hasattr(args_out, "C")
        assert hasattr(args_out, "x0")
        assert args_out[0] is args_out.v
        assert args_out[1] is args_out.C
        assert args_out[2] is args_out.x0
        v_out = args_out[0]
        assert torch.isfinite(x_out).all()
        assert v_out is None or torch.isfinite(v_out).all()
