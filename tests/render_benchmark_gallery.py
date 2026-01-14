import argparse
import json
from pathlib import Path

import numpy as np


def _load_cases(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list):
        return {}
    out: dict[int, dict] = {}
    for row in cases:
        if not isinstance(row, dict) or "case_id" not in row:
            continue
        out[int(row["case_id"])] = row
    return out


def _open_rgb(path: Path):  # type: ignore[no-untyped-def]
    from PIL import Image

    return Image.open(path).convert("RGB")


def _open_mask(path: Path):  # type: ignore[no-untyped-def]
    from PIL import Image

    return Image.open(path).convert("L")


def _diff_heatmap(off_rgb, on_rgb, inpaint_mask_l):  # type: ignore[no-untyped-def]
    from PIL import Image

    off = np.asarray(off_rgb, dtype=np.int16)
    on = np.asarray(on_rgb, dtype=np.int16)
    mask = (np.asarray(inpaint_mask_l, dtype=np.uint8) > 127).astype(np.uint8)

    diff = np.abs(off - on).astype(np.uint8)
    diff = diff.max(axis=2)  # (H,W)
    diff = diff * mask

    # Scale for visibility; clamp to uint8.
    diff = np.clip(diff.astype(np.int16) * 4, 0, 255).astype(np.uint8)
    return Image.fromarray(diff, mode="L").convert("RGB")


def _hstack(images):  # type: ignore[no-untyped-def]
    from PIL import Image

    if not images:
        raise ValueError("no images to stack")
    w, h = images[0].size
    out = Image.new("RGB", (w * len(images), h))
    for i, im in enumerate(images):
        out.paste(im, (i * w, 0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a small visual gallery comparing two benchmark runs")
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--current-json", type=Path, required=True)
    parser.add_argument("--baseline-images", type=Path, required=True)
    parser.add_argument("--current-images", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--metric", type=str, default="lpips_inpaint_ctx")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    b_cases = _load_cases(args.baseline_json)
    c_cases = _load_cases(args.current_json)
    shared = sorted(set(b_cases.keys()) & set(c_cases.keys()))
    if not shared:
        raise SystemExit("no shared cases found in JSON files")

    deltas: list[tuple[float, int]] = []
    for case_id in shared:
        try:
            b_val = float(b_cases[case_id][args.metric])
            c_val = float(c_cases[case_id][args.metric])
        except Exception:
            continue
        deltas.append((c_val - b_val, case_id))

    if not deltas:
        raise SystemExit(f"no comparable case metric '{args.metric}' found")

    deltas.sort(reverse=True, key=lambda x: x[0])
    selected = [case_id for _, case_id in deltas[: max(1, args.top_k)]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []

    for case_id in selected:
        case_name = f"{case_id:04d}"
        b_dir = args.baseline_images / case_name
        c_dir = args.current_images / case_name

        ref = _open_rgb(b_dir / "reference.png")
        masked = _open_rgb(b_dir / "masked_input.png")
        off = _open_rgb(b_dir / "output.png")
        on = _open_rgb(c_dir / "output.png")
        inpaint_mask = _open_mask(b_dir / "inpaint_mask.png")

        diff = _diff_heatmap(off, on, inpaint_mask)
        strip = _hstack([ref, masked, off, on, diff])

        out_name = f"{case_name}_compare.png"
        strip.save(args.out_dir / out_name)
        rendered.append(out_name)

    index = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'><title>LanPaint benchmark gallery</title></head>",
            "<body>",
            "<h2>LanPaint benchmark gallery</h2>",
            "<p>Columns: reference | masked_input | head_off | head_on | diff(head_off vs head_on)</p>",
            *[
                f"<div><img src='{name}' style='image-rendering:pixelated;max-width:100%'></div><hr>"
                for name in rendered
            ],
            "</body></html>",
            "",
        ]
    )
    (args.out_dir / "index.html").write_text(index, encoding="utf-8")


if __name__ == "__main__":
    main()
