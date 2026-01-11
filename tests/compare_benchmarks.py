"""Compare two LPIPS benchmark JSON outputs (baseline vs current)."""

import argparse
import json
import os
from pathlib import Path


def _append_step_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with Path(summary_path).open("a", encoding="utf-8", errors="ignore") as f:
        f.write(markdown)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--max-rel-diff", type=float, default=0.01, help="Fail if (current-baseline)/baseline exceeds this")
    parser.add_argument("--max-abs-diff", type=float, default=0.0, help="Fail if current-baseline exceeds this")
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))

    b_score = float(baseline["lpips"])
    c_score = float(current["lpips"])
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
            "",
        ]
    )
    _append_step_summary(summary)

    if diff > args.max_abs_diff:
        if b_score == 0.0 or rel_diff > args.max_rel_diff:
            print("FAIL: LPIPS regression exceeds configured threshold(s)")
            raise SystemExit(1)

    print("PASS: LPIPS within configured threshold(s)")


if __name__ == "__main__":
    main()
