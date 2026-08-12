"""Tests for the MinStepFrac tail ramp (inner-step count reduction).

The ramp makes sure only one knob changes at a time: while the remaining
noise fraction (1 - abt) is above the threshold the step size scales inside
LanPaint and the count is constant; below it the step size is pinned and
the count ramps down linearly.
"""


def _import_nodes():
    import LanPaint.src.LanPaint.nodes as nodes  # type: ignore[attr-defined]

    return nodes


def test_disabled_returns_full_count() -> None:
    nodes = _import_nodes()
    assert nodes.min_step_frac_effective_steps(5, 0.1, 0.0) == 5  # min_frac 0 = off
    assert nodes.min_step_frac_effective_steps(5, 0.01, 0.0) == 5


def test_above_fraction_returns_full_count() -> None:
    nodes = _import_nodes()
    assert nodes.min_step_frac_effective_steps(5, 0.2, 0.05) == 5
    assert nodes.min_step_frac_effective_steps(5, 0.05, 0.05) == 5  # boundary kept


def test_below_fraction_ramps_linear() -> None:
    nodes = _import_nodes()
    # frac / min_frac: 0.04/0.05 = 0.8 -> 5*0.8 = 4
    assert nodes.min_step_frac_effective_steps(5, 0.04, 0.05) == 4
    # 0.025/0.05 = 0.5 -> 2.5 -> round -> 2
    assert nodes.min_step_frac_effective_steps(5, 0.025, 0.05) == 2
    # 0.005/0.05 = 0.1 -> 0.5 -> round(0.5) = 0 (banker's), max(0, ...) anyway
    assert nodes.min_step_frac_effective_steps(5, 0.005, 0.05) == 0
    assert nodes.min_step_frac_effective_steps(5, 0.0, 0.05) == 0


def test_zero_steps_stays_zero() -> None:
    nodes = _import_nodes()
    assert nodes.min_step_frac_effective_steps(0, 0.01, 0.05) == 0
