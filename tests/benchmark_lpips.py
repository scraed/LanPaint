"""LanPaintBench-style LPIPS benchmark for GitHub Actions.

Supports:
- A fast dummy-mode sanity benchmark.
- A slow e2e-ish mode using LanPaintBench's `guided_diffusion` ImageNet-256 checkpoint.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _make_mask(mask_type: str, height: int, width: int) -> torch.Tensor:
    """Return a (1, 3, H, W) mask where 1=known region, 0=to-inpaint."""
    if height <= 0 or width <= 0:
        raise ValueError("height/width must be positive")

    mask = torch.ones((1, 3, height, width), dtype=torch.float32)

    if mask_type == "half":
        mask[:, :, :, width // 2 :] = 0.0
        return mask

    if mask_type == "box":
        h0, h1 = height // 4, (3 * height) // 4
        w0, w1 = width // 4, (3 * width) // 4
        mask[:, :, h0:h1, w0:w1] = 0.0
        return mask

    if mask_type == "outpaint":
        mask = torch.zeros((1, 3, height, width), dtype=torch.float32)
        h0, h1 = height // 4, (3 * height) // 4
        w0, w1 = width // 4, (3 * width) // 4
        mask[:, :, h0:h1, w0:w1] = 1.0
        return mask

    if mask_type == "checkerboard":
        grid = max(1, min(height, width) // 16)
        mask_2d = torch.zeros((height, width), dtype=torch.float32)
        for row in range(height):
            row_parity = (row // grid) % 2
            start = 0 if row_parity == 0 else 1
            for block in range(start, (width + grid - 1) // grid, 2):
                c0 = block * grid
                c1 = min(width, (block + 1) * grid)
                mask_2d[row, c0:c1] = 1.0
        return mask_2d.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    raise ValueError(f"unknown mask type: {mask_type}")


class _DummySampling:
    def noise_scaling(self, sigma, noise, latent_image):  # type: ignore[no-untyped-def]
        # Variance exploding convention (ComfyUI/k-diffusion style).
        return latent_image + noise * sigma


class _DummyModel:
    def __init__(self) -> None:
        self.inner_model = self
        self.model_sampling = _DummySampling()

    def __call__(self, x, sigma, model_options=None, seed=None):  # type: ignore[no-untyped-def]
        # Cheap "denoiser" so the benchmark exercises LanPaint math deterministically on CPU.
        sigma_ = sigma.reshape((sigma.shape[0],) + (1,) * (x.ndim - 1))
        x0 = x / (1.0 + sigma_**2)
        return x0, x0


def _load_images(*, dataset: str, split: str, subset: int, seed: int, shuffle_buffer: int) -> list[torch.Tensor]:
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split, streaming=True)
    try:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    except Exception:
        pass

    images: list[torch.Tensor] = []
    for idx, item in enumerate(ds):
        if idx >= subset:
            break
        image = item["image"].convert("RGB")
        arr = np.asarray(image, dtype=np.uint8)
        img = torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0
        images.append(img)
    return images


def _run_dummy_bench(  # type: ignore[no-untyped-def]
    LanPaintClass,
    images: list[torch.Tensor],
    device: torch.device,
    *,
    n_steps: int,
    sigma_value: float,
    mask_type: str,
    seed: int,
) -> float:
    import lpips

    lpips_fn = lpips.LPIPS(net="alex").to(device)

    sigma = torch.tensor([sigma_value], device=device)
    abt = 1.0 / (1.0 + sigma**2)
    flow_t = torch.zeros_like(sigma)
    current_times = (sigma, abt, flow_t)

    engine = LanPaintClass(
        Model=_DummyModel(),
        NSteps=n_steps,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.1,
    )

    scores: list[float] = []
    for idx, img in enumerate(images):
        img = img.unsqueeze(0).to(device)
        latent_image = img.clone()

        height, width = img.shape[-2:]
        latent_mask = _make_mask(mask_type, height, width).to(device)

        gen = torch.Generator(device=device.type).manual_seed(seed + idx)
        noise = torch.randn(img.shape, generator=gen, device=device)
        x = noise * sigma.reshape((1,) + (1,) * (img.ndim - 1))

        output = engine(
            x=x,
            latent_image=latent_image,
            noise=noise,
            sigma=sigma,
            latent_mask=latent_mask,
            current_times=current_times,
            model_options={},
            seed=seed,
            n_steps=n_steps,
        ).clamp(-1.0, 1.0)

        dist = lpips_fn(output, img)
        scores.append(float(dist.item()))

    return float(np.mean(scores))


def _ddpm_alpha_bar(*, timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def _sigmas_and_timesteps(
    *,
    inference_steps: int,
    train_timesteps: int,
    beta_start: float,
    beta_end: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inference_steps <= 0:
        raise ValueError("inference_steps must be > 0")
    if train_timesteps <= 1:
        raise ValueError("train_timesteps must be > 1")

    alpha_bar = _ddpm_alpha_bar(timesteps=train_timesteps, beta_start=beta_start, beta_end=beta_end)
    t = torch.linspace(train_timesteps - 1, 0, inference_steps, dtype=torch.float64).round().to(torch.long)
    t = torch.unique_consecutive(t)
    if t.numel() < 2:
        raise ValueError("inference_steps too small after rounding")

    abt = alpha_bar[t].clamp(min=1e-12, max=1.0)
    sigmas = torch.sqrt((1.0 - abt) / abt).to(torch.float32)
    sigmas = torch.cat([sigmas, sigmas.new_zeros(1)], dim=0)
    return sigmas, t.to(torch.long)


class _GuidedDiffusionSampling:
    def noise_scaling(self, sigma, noise, latent_image):  # type: ignore[no-untyped-def]
        return latent_image + noise * sigma


class _GuidedDiffusionModel:
    def __init__(self, *, unet, device: torch.device) -> None:  # type: ignore[no-untyped-def]
        self.inner_model = self
        self.model_sampling = _GuidedDiffusionSampling()
        self._unet = unet
        self._device = device

    @torch.no_grad()
    def __call__(self, x, sigma, model_options=None, seed=None):  # type: ignore[no-untyped-def]
        if not isinstance(model_options, dict) or "bench_timestep" not in model_options:
            raise ValueError("model_options['bench_timestep'] is required for guided_diffusion mode")

        t_int = int(model_options["bench_timestep"])
        t = torch.full((x.shape[0],), t_int, device=self._device, dtype=torch.long)

        sigma_ = sigma.reshape((sigma.shape[0],) + (1,) * (x.ndim - 1))
        abt = 1.0 / (1.0 + sigma_**2)

        x_vp = x / torch.sqrt(1.0 + sigma_**2)
        eps = self._unet(x_vp, t)
        eps = eps[:, :3, :, :]

        sqrt_abt = torch.sqrt(abt)
        sqrt_1m_abt = torch.sqrt(1.0 - abt)
        x0 = (x_vp - sqrt_1m_abt * eps) / sqrt_abt
        return x0, x0


def _load_guided_diffusion_unet(*, lanpaintbench: Path, checkpoint: Path, device: torch.device):  # type: ignore[no-untyped-def]
    sys.path.insert(0, str(lanpaintbench.resolve()))
    from guided_diffusion.unet import UNetModel  # type: ignore[import-not-found]

    image_size = 256
    num_channels = 256
    num_head_channels = 64
    num_res_blocks = 2
    resblock_updown = True
    learn_sigma = True
    class_cond = False
    use_checkpoint = False
    attention_resolutions = "32,16,8"
    use_scale_shift_norm = True
    use_fp16 = False

    channel_mult = (1, 1, 2, 2, 4, 4)
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    model = UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=0.0,
        channel_mult=channel_mult,
        num_classes=(1000 if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=1,
        num_head_channels=num_head_channels,
        num_heads_upsample=-1,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=False,
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _run_guided_diffusion_e2e_bench(  # type: ignore[no-untyped-def]
    LanPaintClass,
    images: list[torch.Tensor],
    device: torch.device,
    *,
    lanpaintbench: Path,
    checkpoint: Path,
    seed: int,
    mask_type: str,
    sample_steps: int,
    inner_steps: int,
    beta_start: float,
    beta_end: float,
    train_timesteps: int,
    semantic_stop: dict[str, float | int] | None = None,
) -> tuple[float, dict[str, float]]:
    import lpips

    unet = _load_guided_diffusion_unet(lanpaintbench=lanpaintbench, checkpoint=checkpoint, device=device)
    model = _GuidedDiffusionModel(unet=unet, device=device)

    sigmas, timesteps = _sigmas_and_timesteps(
        inference_steps=sample_steps,
        train_timesteps=train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
    )
    sigmas = sigmas.to(device=device)
    timesteps = timesteps.to(device=device)

    lpips_fn = lpips.LPIPS(net="alex").to(device)

    scores: list[float] = []
    start = time.perf_counter()

    engine = LanPaintClass(
        Model=model,
        NSteps=inner_steps,
        Friction=15.0,
        Lambda=1.0,
        Beta=1.0,
        StepSize=0.1,
    )

    for idx, img in enumerate(images):
        img = img.unsqueeze(0).to(device)

        height, width = img.shape[-2:]
        latent_mask = _make_mask(mask_type, height, width).to(device)
        inpaint_mask = 1.0 - latent_mask

        latent_image = img * latent_mask

        gen = torch.Generator(device=device.type).manual_seed(seed + idx)
        noise = torch.randn(img.shape, generator=gen, device=device)

        x = latent_image + noise * sigmas[0].reshape((1,) + (1,) * (img.ndim - 1))

        for step_idx in range(sigmas.shape[0] - 1):
            sigma = sigmas[step_idx : step_idx + 1]
            sigma_next = sigmas[step_idx + 1 : step_idx + 2]
            t = timesteps[min(step_idx, timesteps.shape[0] - 1)].item()

            abt = 1.0 / (1.0 + sigma**2)
            flow_t = torch.sqrt(1.0 - abt) / (torch.sqrt(1.0 - abt) + torch.sqrt(abt))
            current_times = (sigma, abt, flow_t)

            model_options = {"bench_timestep": int(t)}
            if semantic_stop is not None:
                model_options["lanpaint_semantic_stop"] = semantic_stop

            x0 = engine(
                x=x,
                latent_image=latent_image,
                noise=noise,
                sigma=sigma,
                latent_mask=latent_mask,
                current_times=current_times,
                model_options=model_options,
                seed=seed,
                n_steps=inner_steps,
            )

            if float(sigma_next.item()) == 0.0:
                x = x0
                break

            sigma_ = sigma.reshape((sigma.shape[0],) + (1,) * (x.ndim - 1))
            sigma_next_ = sigma_next.reshape((sigma_next.shape[0],) + (1,) * (x.ndim - 1))
            d = (x - x0) / sigma_
            x = x + d * (sigma_next_ - sigma_)

        x = x.clamp(-1.0, 1.0)
        dist = lpips_fn(x * inpaint_mask, img * inpaint_mask)
        scores.append(float(dist.item()))

    elapsed = time.perf_counter() - start
    stats = {"seconds_total": elapsed, "seconds_per_image": elapsed / max(1, len(images))}
    return float(np.mean(scores)), stats


def main() -> None:
    parser = argparse.ArgumentParser(description="LanPaint LPIPS benchmark (ImageNet-1k-256x256 subset)")
    parser.add_argument("--src", type=Path, required=True, help="Path to LanPaint src directory (contains LanPaint/)")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--mode", type=str, default="dummy", choices=["dummy", "guided_diffusion_e2e"])

    parser.add_argument("--dataset", type=str, default="benjamin-paine/imagenet-1k-256x256")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)

    parser.add_argument("--mask-type", type=str, default="box", choices=["half", "box", "outpaint", "checkerboard"])
    parser.add_argument("--n-steps", type=int, default=5, help="LanPaint inner iterations per call (dummy mode)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Variance-exploding sigma value (dummy mode)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])

    parser.add_argument("--lanpaintbench", type=Path, default=None, help="Path to LanPaintBench checkout (guided_diffusion_e2e)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to 256x256_diffusion_uncond.pt (guided_diffusion_e2e)")
    parser.add_argument("--sample-steps", type=int, default=20, help="Outer diffusion steps (guided_diffusion_e2e)")
    parser.add_argument("--train-timesteps", type=int, default=1000, help="Training timesteps for beta schedule (guided_diffusion_e2e)")
    parser.add_argument("--beta-start", type=float, default=0.0001, help="DDPM beta_start (guided_diffusion_e2e)")
    parser.add_argument("--beta-end", type=float, default=0.02, help="DDPM beta_end (guided_diffusion_e2e)")
    parser.add_argument("--min-seconds", type=float, default=0.0, help="Fail if benchmark runtime is below this (guided_diffusion_e2e)")
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.0,
        help="Enable LanPaint semantic early-stop if > 0 (guided_diffusion_e2e)",
    )
    parser.add_argument("--semantic-patience", type=int, default=1, help="Semantic early-stop patience (guided_diffusion_e2e)")
    parser.add_argument("--semantic-min-steps", type=int, default=1, help="Semantic early-stop min_steps (guided_diffusion_e2e)")
    args = parser.parse_args()

    if args.subset <= 0:
        raise SystemExit("--subset must be > 0")
    if args.n_steps <= 0:
        raise SystemExit("--n-steps must be > 0")
    if args.shuffle_buffer <= 0:
        raise SystemExit("--shuffle-buffer must be > 0")
    if args.semantic_patience <= 0:
        raise SystemExit("--semantic-patience must be > 0")
    if args.semantic_min_steps <= 0:
        raise SystemExit("--semantic-min-steps must be > 0")

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    sys.path.insert(0, str(args.src.resolve()))
    from LanPaint.lanpaint import LanPaint as LanPaintEngine

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    images = _load_images(
        dataset=args.dataset,
        split=args.split,
        subset=args.subset,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )

    stats: dict[str, float] = {}
    if args.mode == "dummy":
        score = _run_dummy_bench(
            LanPaintEngine,
            images,
            device,
            n_steps=args.n_steps,
            sigma_value=args.sigma,
            mask_type=args.mask_type,
            seed=args.seed,
        )
    else:
        if args.lanpaintbench is None or args.checkpoint is None:
            raise SystemExit("--lanpaintbench and --checkpoint are required for guided_diffusion_e2e mode")

        semantic_stop = None
        if float(args.semantic_threshold) > 0.0:
            semantic_stop = {
                "threshold": float(args.semantic_threshold),
                "patience": int(args.semantic_patience),
                "min_steps": int(args.semantic_min_steps),
            }

        score, stats = _run_guided_diffusion_e2e_bench(
            LanPaintEngine,
            images,
            device,
            lanpaintbench=args.lanpaintbench,
            checkpoint=args.checkpoint,
            seed=args.seed,
            mask_type=args.mask_type,
            sample_steps=args.sample_steps,
            inner_steps=args.n_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            train_timesteps=args.train_timesteps,
            semantic_stop=semantic_stop,
        )
        if args.min_seconds > 0.0 and stats.get("seconds_total", 0.0) < args.min_seconds:
            raise SystemExit(f"benchmark too fast: {stats.get('seconds_total', 0.0):.1f}s < {args.min_seconds:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "lpips": score,
                "stats": stats,
                "config": {
                    "mode": args.mode,
                    "dataset": args.dataset,
                    "split": args.split,
                    "subset": args.subset,
                    "seed": args.seed,
                    "shuffle_buffer": args.shuffle_buffer,
                    "mask_type": args.mask_type,
                    "n_steps": args.n_steps,
                    "sigma": args.sigma,
                    "sample_steps": args.sample_steps,
                    "train_timesteps": args.train_timesteps,
                    "beta_start": args.beta_start,
                    "beta_end": args.beta_end,
                    "lanpaintbench": str(args.lanpaintbench) if args.lanpaintbench is not None else None,
                    "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
                    "device": device_str,
                    "semantic_threshold": args.semantic_threshold,
                    "semantic_patience": args.semantic_patience,
                    "semantic_min_steps": args.semantic_min_steps,
                },
            },
            f,
            indent=2,
        )

    print(f"LPIPS: {score:.6f}")


if __name__ == "__main__":
    main()
