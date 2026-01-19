"""Compare two LPIPS benchmark JSON outputs (baseline vs current)."""

import argparse
import json
import math
import os
import warnings
from pathlib import Path


def _append_step_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with Path(summary_path).open("a", encoding="utf-8", errors="ignore") as f:
        f.write(markdown)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0.0:
        return min(values)
    if q >= 1.0:
        return max(values)
    ordered = sorted(values)
    # Nearest-rank quantile.
    idx = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return float(ordered[idx])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--max-rel-diff", type=float, default=0.01, help="Fail if (current-baseline)/baseline exceeds this")
    parser.add_argument("--max-abs-diff", type=float, default=0.0, help="Fail if current-baseline exceeds this")
    parser.add_argument(
        "--case-metric",
        type=str,
        default="lpips_inpaint_ctx",
        help="Per-case metric key to compare when 'cases' are present",
    )
    parser.add_argument("--max-case-mean-delta", type=float, default=None)
    parser.add_argument("--max-case-p95-delta", type=float, default=None)
    parser.add_argument("--max-case-max-delta", type=float, default=None)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))

    b_score = float(baseline["lpips"])
    c_score = float(current["lpips"])

    b_stats = baseline.get("stats") if isinstance(baseline, dict) else None
    c_stats = current.get("stats") if isinstance(current, dict) else None
    b_seconds = float(b_stats.get("seconds_total", 0.0)) if isinstance(b_stats, dict) else 0.0
    c_seconds = float(c_stats.get("seconds_total", 0.0)) if isinstance(c_stats, dict) else 0.0
    b_spi = float(b_stats.get("seconds_per_image", 0.0)) if isinstance(b_stats, dict) else 0.0
    c_spi = float(c_stats.get("seconds_per_image", 0.0)) if isinstance(c_stats, dict) else 0.0

    diff = c_score - b_score
    rel_diff = diff / b_score if b_score != 0.0 else float("inf")

    lines = [
        f"Baseline LPIPS: {b_score:.6f}",
        f"Current LPIPS:  {c_score:.6f}",
        f"Difference:     {diff:+.6f}",
    ]
    if b_score != 0.0:
        lines.append(f"Relative diff:   {rel_diff:+.2%}")
    print("\n".join(lines))

    summary = "\n".join(
        [
            "### LPIPS benchmark",
            "",
            "| Metric | Baseline | Current | Diff |",
            "|---|---:|---:|---:|",
            f"| LPIPS | {b_score:.6f} | {c_score:.6f} | {diff:+.6f} |",
            f"| Runtime (s) | {b_seconds:.1f} | {c_seconds:.1f} | {(c_seconds - b_seconds):+.1f} |",
            f"| Sec/img | {b_spi:.2f} | {c_spi:.2f} | {(c_spi - b_spi):+.2f} |",
            "",
        ]
    )
    _append_step_summary(summary)

    b_cases = baseline.get("cases") if isinstance(baseline, dict) else None
    c_cases = current.get("cases") if isinstance(current, dict) else None
    if isinstance(b_cases, list) and isinstance(c_cases, list) and b_cases and c_cases:
        b_by_id = {int(row["case_id"]): row for row in b_cases if isinstance(row, dict) and "case_id" in row}
        c_by_id = {int(row["case_id"]): row for row in c_cases if isinstance(row, dict) and "case_id" in row}
        shared_ids = sorted(set(b_by_id.keys()) & set(c_by_id.keys()))
        if shared_ids:
            deltas: list[float] = []
            b_vals: list[float] = []
            c_vals: list[float] = []
            for case_id in shared_ids:
                b_row = b_by_id[case_id]
                c_row = c_by_id[case_id]
                try:
                    b_val = float(b_row[args.case_metric])
                    c_val = float(c_row[args.case_metric])
                except Exception as e:
                    warnings.warn(f"Skipping case {case_id} due to error: {e}")
                    continue
                b_vals.append(b_val)
                c_vals.append(c_val)
                deltas.append(c_val - b_val)

            if deltas:
                mean_delta = float(sum(deltas) / len(deltas))
                p95_delta = _quantile(deltas, 0.95)
                max_delta = max(deltas)
                b_mean = float(sum(b_vals) / len(b_vals))
                c_mean = float(sum(c_vals) / len(c_vals))

                print("")
                print(f"Case metric: {args.case_metric}")
                print(f"Baseline mean: {b_mean:.6f}")
                print(f"Current mean:  {c_mean:.6f}")
                print(f"Mean delta:    {mean_delta:+.6f}")
                print(f"P95 delta:     {p95_delta:+.6f}")
                print(f"Max delta:     {max_delta:+.6f}")

                case_summary = "\n".join(
                    [
                        "### Per-case deltas",
                        "",
                        "| Metric | Mean Δ | P95 Δ | Max Δ | Cases |",
                        "|---|---:|---:|---:|---:|",
                        f"| `{args.case_metric}` | {mean_delta:+.6f} | {p95_delta:+.6f} | {max_delta:+.6f} | {len(deltas)} |",
                        "",
                    ]
                )
                _append_step_summary(case_summary)

                if args.max_case_mean_delta is not None and mean_delta > float(args.max_case_mean_delta):
                    print("FAIL: per-case mean delta exceeds threshold")
                    raise SystemExit(1)
                if args.max_case_p95_delta is not None and p95_delta > float(args.max_case_p95_delta):
                    print("FAIL: per-case p95 delta exceeds threshold")
                    raise SystemExit(1)
                if args.max_case_max_delta is not None and max_delta > float(args.max_case_max_delta):
                    print("FAIL: per-case max delta exceeds threshold")
                    raise SystemExit(1)

    if diff > args.max_abs_diff:
        if b_score == 0.0 or rel_diff > args.max_rel_diff:
            print("FAIL: LPIPS regression exceeds configured threshold(s)")
            raise SystemExit(1)

    print("PASS: LPIPS within configured threshold(s)")


if __name__ == "__main__":
    main()
