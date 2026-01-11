"""Lightweight CI benchmark inspired by LanPaintBench.

Runs LPIPS on a tiny deterministic ImageNet-1k-256x256 subset and compares base vs head.
"""

import argparse
import json
import sys
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


def _run_bench(  # type: ignore[no-untyped-def]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="LanPaint LPIPS benchmark (ImageNet-1k-256x256 subset)")
    parser.add_argument("--src", type=Path, required=True, help="Path to LanPaint src directory (contains LanPaint/)")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")

    parser.add_argument("--dataset", type=str, default="benjamin-paine/imagenet-1k-256x256")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)

    parser.add_argument("--mask-type", type=str, default="box", choices=["half", "box", "outpaint", "checkerboard"])
    parser.add_argument("--n-steps", type=int, default=5, help="LanPaint inner iterations per call")
    parser.add_argument("--sigma", type=float, default=1.0, help="Variance-exploding sigma value")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    if args.subset <= 0:
        raise SystemExit("--subset must be > 0")
    if args.n_steps <= 0:
        raise SystemExit("--n-steps must be > 0")
    if args.shuffle_buffer <= 0:
        raise SystemExit("--shuffle-buffer must be > 0")

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

    score = _run_bench(
        LanPaintEngine,
        images,
        device,
        n_steps=args.n_steps,
        sigma_value=args.sigma,
        mask_type=args.mask_type,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "lpips": score,
                "config": {
                    "dataset": args.dataset,
                    "split": args.split,
                    "subset": args.subset,
                    "seed": args.seed,
                    "shuffle_buffer": args.shuffle_buffer,
                    "mask_type": args.mask_type,
                    "n_steps": args.n_steps,
                    "sigma": args.sigma,
                    "device": device_str,
                },
            },
            f,
            indent=2,
        )

    print(f"LPIPS: {score:.6f}")


if __name__ == "__main__":
    main()
