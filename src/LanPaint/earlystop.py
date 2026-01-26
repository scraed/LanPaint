import inspect
from typing import Any, Callable, Optional

import torch

from .types import LangevinState


def _clamp01(val: float) -> float:
    if val <= 0.0:
        return 0.0
    if val >= 1.0:
        return 1.0
    return val


def _abt_scale(abt_val: float) -> float:
    """
    Smooth, parameter-free scale based on outer-step noise level.

    - 0 at abt=0/1 (disable at extreme noise / extreme tail)
    - 1 at abt=0.5 (mid-schedule)
    """
    abt_val = _clamp01(abt_val)
    return _clamp01(4.0 * abt_val * (1.0 - abt_val))


def _boundary_weight(latent_mask: torch.Tensor, inpaint_weight: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Return a 4-neighbor boundary weight: unknown pixels adjacent to known pixels.

    This replaces the previous dilation-based "ring" (kernel/padding) and has no tunable hyperparameters.
    """
    if latent_mask.dim() != 4:
        return None

    known = latent_mask > 0.5
    neighbor_known = torch.zeros_like(known)
    neighbor_known[:, :, 1:, :] |= known[:, :, :-1, :]
    neighbor_known[:, :, :-1, :] |= known[:, :, 1:, :]
    neighbor_known[:, :, :, 1:] |= known[:, :, :, :-1]
    neighbor_known[:, :, :, :-1] |= known[:, :, :, 1:]

    boundary = (~known) & neighbor_known
    return boundary.to(dtype=torch.float32) * inpaint_weight


def _weighted_mse(t1: torch.Tensor, t2: torch.Tensor, weight: torch.Tensor) -> float:
    diff_sq = (t1.to(dtype=torch.float32) - t2.to(dtype=torch.float32)) ** 2
    denom = torch.sum(weight) + 1e-12
    return float((torch.sum(diff_sq * weight) / denom).item())


class LanPaintEarlyStopper:
    """
    Per-step early-stop logic for LanPaint inner (Langevin) iterations.
    """

    @classmethod
    def from_options(
        cls,
        *,
        model_options: Optional[dict],
        latent_mask: torch.Tensor,
        abt: torch.Tensor,
        default_threshold: float,
        default_patience: int,
        default_distance_fn: Optional[Callable[..., Any]],
    ) -> Optional["LanPaintEarlyStopper"]:
        semantic_stop = model_options.get("lanpaint_semantic_stop") if isinstance(model_options, dict) else None

        threshold = float(default_threshold)
        patience = int(default_patience)
        distance_fn = default_distance_fn
        # distance_fn contract: return None (use default metric) or a scalar (Python number / 0-d (1-element) torch.Tensor)

        if isinstance(semantic_stop, dict):
            threshold = float(semantic_stop.get("threshold", threshold))
            patience = int(semantic_stop.get("patience", patience))
            distance_fn = semantic_stop.get("distance_fn", distance_fn)

            # Backward compatibility: map legacy 'min_steps' to a patience floor so it is not an independent knob.
            if patience > 0:
                min_steps = semantic_stop.get("min_steps")
                if min_steps is not None:
                    try:
                        min_steps_int = int(min_steps)
                    except (TypeError, ValueError):
                        min_steps_int = 0
                    if min_steps_int > 1:
                        patience = max(patience, min_steps_int - 1)

        enabled_early_stop = (threshold > 0.0) and (patience > 0)
        # Require N+1 consecutive stable checks:
        # - the first stable step sets patience_counter to 1
        # - `patience=1` therefore stops after 2 stable steps
        patience_eff = max(1, patience) + 1
        threshold_eff = threshold
        inpaint_weight = ring_weight = trace = abt_val = None

        if enabled_early_stop:
            try:
                abt_val = float(torch.mean(abt).item())
            except (TypeError, ValueError):
                abt_val = 0.0

            threshold_eff = threshold * _abt_scale(abt_val)
            if threshold_eff <= 0.0:
                enabled_early_stop = False
            else:
                inpaint_weight = (1 - latent_mask).to(dtype=torch.float32)
                if float(torch.sum(inpaint_weight).item()) < 1e-6:
                    enabled_early_stop = False
                else:
                    ring_weight = _boundary_weight(latent_mask, inpaint_weight)
                    if isinstance(model_options, dict):
                        trace = model_options.get("lanpaint_semantic_trace")

        if not enabled_early_stop:
            return None

        # Pre-fetch trace keys to avoid repeated dict lookups
        bench_case_id = bench_outer_step = bench_timestep = None
        if isinstance(trace, list) and isinstance(model_options, dict):
            bench_case_id = model_options.get("bench_case_id")
            bench_outer_step = model_options.get("bench_outer_step")
            bench_timestep = model_options.get("bench_timestep")

        return cls(
            enabled=enabled_early_stop,
            threshold=threshold,
            threshold_eff=threshold_eff,
            patience_eff=patience_eff,
            inpaint_weight=inpaint_weight,
            ring_weight=ring_weight,
            distance_fn=distance_fn,
            trace=trace,
            bench_case_id=bench_case_id,
            bench_outer_step=bench_outer_step,
            bench_timestep=bench_timestep,
            abt_val=abt_val,
        )

    def __init__(
        self,
        *,
        enabled: bool,
        threshold: float,
        threshold_eff: float,
        patience_eff: int,
        inpaint_weight: Optional[torch.Tensor],
        ring_weight: Optional[torch.Tensor],
        distance_fn: Optional[Callable[..., Any]] = None,
        trace: Optional[list] = None,
        bench_case_id: Any = None,
        bench_outer_step: Any = None,
        bench_timestep: Any = None,
        abt_val: Optional[float] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.threshold = float(threshold)
        self.threshold_eff = float(threshold_eff)
        self.patience_eff = int(patience_eff)

        self.inpaint_weight = inpaint_weight
        self.ring_weight = ring_weight

        self.trace = trace
        self.bench_case_id = bench_case_id
        self.bench_outer_step = bench_outer_step
        self.bench_timestep = bench_timestep
        self.abt_val = abt_val

        self.patience_counter = 0
        self.x0_anchor = None

        self._dist_wrapper = self._wrap_distance_fn(distance_fn) if self.enabled else None

    @property
    def has_custom_distance_fn(self) -> bool:
        return self._dist_wrapper is not None

    @staticmethod
    def _wrap_distance_fn(distance_fn: Optional[Callable[..., Any]]):
        """
        Wrap a user-provided `distance_fn` into a normalized callable: fn(prev, cur, ctx) -> dist|None.

        Supported signatures:
        - 3+ positional (or *args): `distance_fn(prev, cur, ctx)`
        - explicit / **kwargs ctx: `distance_fn(prev, cur, ctx=ctx)`
        - default 2-arg: `distance_fn(cur, prev)`

        Return contract: None (use default metric) or a scalar (Python number / 0-d (1-element) torch.Tensor).
        """
        if not callable(distance_fn):
            return None

        try:
            sig = inspect.signature(distance_fn)
            params = list(sig.parameters.values())

            has_ctx_param = "ctx" in sig.parameters
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
            has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)

            pos_params = [
                p
                for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]

            if len(pos_params) >= 3 or has_var_pos:
                # 3-arg positional: fn(prev, cur, ctx)
                return lambda p, c, ctx: distance_fn(p, c, ctx)
            if has_ctx_param or has_var_kw:
                # keyword ctx: fn(prev, cur, ctx=ctx)
                return lambda p, c, ctx: distance_fn(p, c, ctx=ctx)

            # Default 2-arg: fn(cur, prev)
            return lambda p, c, ctx: distance_fn(c, p)
        except (ValueError, TypeError):
            # Fallback for built-ins or complex callables.
            def fallback_wrapper(p, c, ctx):
                try:
                    return distance_fn(p, c, ctx)
                except TypeError as e:
                    tb = e.__traceback__
                    if tb is not None and tb.tb_frame.f_code is not fallback_wrapper.__code__:
                        raise
                    return distance_fn(c, p)

            return fallback_wrapper

    def step(
        self,
        *,
        i: int,
        n_steps: int,
        x_t_before: torch.Tensor,
        x_t_after: torch.Tensor,
        x_t_prev_for_custom: Optional[torch.Tensor],
        prev_args: Any,
        args: Any,
        ctx: dict,
    ) -> bool:
        if not self.enabled:
            return False

        # 'inpaint_weight' is guaranteed to be set when enabled is True in the caller.
        inpaint = self.inpaint_weight
        if inpaint is None:
            return False

        dist = None
        custom_dist = False
        dist_inpaint = dist_ring = dist_drift = x0_prev = x0_cur = None

        if self._dist_wrapper is not None:
            dist = self._dist_wrapper(x_t_prev_for_custom, x_t_after, ctx)
            if dist is not None:
                if isinstance(dist, torch.Tensor):
                    if dist.numel() != 1:
                        raise TypeError("distance_fn must return None or a scalar / 0-d (1-element) tensor")
                    dist = float(dist.item())
                else:
                    dist = float(dist)
        custom_dist = dist is not None

        if dist is None:
            def _get_x0(arg: Any) -> Optional[torch.Tensor]:
                if isinstance(arg, LangevinState):
                    return arg.x0
                if isinstance(arg, tuple) and len(arg) >= 3:
                    return arg[2]
                return None

            x0_prev = _get_x0(prev_args)
            x0_cur = _get_x0(args)

            if x0_prev is not None and x0_cur is not None:
                dist_inpaint = _weighted_mse(x0_cur, x0_prev, inpaint)
                dist_ring = _weighted_mse(x0_cur, x0_prev, self.ring_weight) if self.ring_weight is not None else None
                dist = dist_inpaint if dist_ring is None else max(dist_inpaint, dist_ring)
            else:
                dist_inpaint = _weighted_mse(x_t_after, x_t_before, inpaint)
                dist = dist_inpaint

        threshold_used = self.threshold if custom_dist else self.threshold_eff

        # Drift guard (only for default metric with x0_cur).
        if x0_cur is not None and not custom_dist:
            if dist <= threshold_used:
                if self.x0_anchor is None:
                    self.x0_anchor = x0_cur.detach()
                else:
                    drift_inpaint = _weighted_mse(x0_cur, self.x0_anchor, inpaint)
                    drift_ring = _weighted_mse(x0_cur, self.x0_anchor, self.ring_weight) if self.ring_weight is not None else None
                    dist_drift = drift_inpaint if drift_ring is None else max(drift_inpaint, drift_ring)
                    dist = max(dist, dist_drift)
            else:
                self.x0_anchor = None

        if dist <= threshold_used:
            self.patience_counter += 1
        else:
            self.patience_counter = 0
            self.x0_anchor = None

        should_stop = self.patience_counter >= self.patience_eff

        if isinstance(self.trace, list):
            self.trace.append(
                {
                    "case_id": self.bench_case_id,
                    "outer_step": self.bench_outer_step,
                    "bench_timestep": self.bench_timestep,
                    "inner_step": i + 1,
                    "dist": dist,
                    "dist_inpaint": None if dist_inpaint is None else float(dist_inpaint),
                    "dist_ring": None if dist_ring is None else float(dist_ring),
                    "dist_drift": None if dist_drift is None else float(dist_drift),
                    "threshold": float(threshold_used),
                    "threshold_eff": float(self.threshold_eff),
                    "patience_counter": int(self.patience_counter),
                    "patience_eff": int(self.patience_eff),
                    "abt": None if self.abt_val is None else float(self.abt_val),
                    "custom_dist": bool(custom_dist),
                    "stopped": bool(should_stop),
                }
            )

        return bool(should_stop)

