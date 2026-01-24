import inspect
from typing import Any, Callable, Optional

import torch


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
        semantic_stop = None
        if isinstance(model_options, dict):
            semantic_stop = model_options.get("lanpaint_semantic_stop")

        threshold = float(default_threshold)
        patience = int(default_patience)
        distance_fn = default_distance_fn
        # distance_fn contract: return None (use default metric) or a scalar (Python number / 0-d (1-element) torch.Tensor)

        if isinstance(semantic_stop, dict):
            threshold = float(semantic_stop.get("threshold", threshold))
            patience = int(semantic_stop.get("patience", patience))
            distance_fn = semantic_stop.get("distance_fn", distance_fn)

        enabled_early_stop = (threshold > 0.0) and (patience > 0)
        patience = max(1, patience)
        patience_eff = patience + 1
        threshold_eff = threshold
        inpaint_weight = None
        ring_weight = None
        trace = None
        abt_val = None

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
                ring_weight = _boundary_weight(latent_mask, inpaint_weight)
                if isinstance(model_options, dict):
                    trace = model_options.get("lanpaint_semantic_trace")

        if not enabled_early_stop:
            return None

        # Pre-fetch trace keys to avoid repeated dict lookups
        bench_case_id = None
        bench_outer_step = None
        bench_timestep = None
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
        dist_inpaint = None
        dist_ring = None
        dist_drift = None
        x0_prev = None
        x0_cur = None

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
            if isinstance(prev_args, tuple) and len(prev_args) >= 3:
                x0_prev = prev_args[2]
            if isinstance(args, tuple) and len(args) >= 3:
                x0_cur = args[2]

            if x0_prev is not None and x0_cur is not None:
                diff_sq = (x0_cur.to(dtype=torch.float32) - x0_prev.to(dtype=torch.float32)) ** 2
                denom = torch.sum(inpaint) + 1e-12
                dist_inpaint = (torch.sum(diff_sq * inpaint) / denom).item()
                dist = float(dist_inpaint)
                if self.ring_weight is not None:
                    ring_denom = torch.sum(self.ring_weight) + 1e-12
                    dist_ring = (torch.sum(diff_sq * self.ring_weight) / ring_denom).item()
                    dist = max(float(dist_inpaint), float(dist_ring))
            else:
                diff_sq = (x_t_after.to(dtype=torch.float32) - x_t_before.to(dtype=torch.float32)) ** 2
                denom = torch.sum(inpaint) + 1e-12
                dist = (torch.sum(diff_sq * inpaint) / denom).item()
                dist_inpaint = dist

        threshold_used = self.threshold if custom_dist else self.threshold_eff

        # Drift guard (only for default metric with x0_cur).
        if x0_cur is not None and not custom_dist:
            if float(dist) <= threshold_used:
                if self.x0_anchor is None:
                    self.x0_anchor = x0_cur.detach()
                else:
                    diff_sq = (x0_cur.to(dtype=torch.float32) - self.x0_anchor.to(dtype=torch.float32)) ** 2
                    denom = torch.sum(inpaint) + 1e-12
                    drift_inpaint = (torch.sum(diff_sq * inpaint) / denom).item()
                    dist_drift = float(drift_inpaint)
                    if self.ring_weight is not None:
                        ring_denom = torch.sum(self.ring_weight) + 1e-12
                        drift_ring = (torch.sum(diff_sq * self.ring_weight) / ring_denom).item()
                        dist_drift = max(float(drift_inpaint), float(drift_ring))
                    dist = max(float(dist), float(dist_drift))
            else:
                self.x0_anchor = None

        if float(dist) <= threshold_used:
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
                    "dist": float(dist),
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

