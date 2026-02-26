#!/usr/bin/env python3
"""Generate up to five daily Stage 2 submission candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CandidateStats:
    file: str
    strategy: str
    rows: int
    pred_mean: float
    pred_std: float
    pred_min: float
    pred_max: float


def _build_predictions(base: pd.Series, strategy: str, rng: np.random.Generator) -> np.ndarray:
    x = base.to_numpy(dtype=float)
    if strategy == "baseline_050":
        preds = np.full_like(x, 0.50)
    elif strategy == "constant_052":
        preds = np.full_like(x, 0.52)
    elif strategy == "constant_048":
        preds = np.full_like(x, 0.48)
    elif strategy == "jitter_sd002":
        preds = 0.50 + rng.normal(0.0, 0.02, size=x.shape[0])
    elif strategy == "jitter_sd005":
        preds = 0.50 + rng.normal(0.0, 0.05, size=x.shape[0])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return np.clip(preds, 0.001, 0.999)


def generate_candidates(
    sample_submission: Path,
    output_dir: Path,
    date_tag: str,
    max_files: int,
    seed: int,
) -> list[CandidateStats]:
    if max_files < 1 or max_files > 5:
        raise ValueError("max_files must be between 1 and 5")
    df = pd.read_csv(sample_submission)
    if "ID" not in df.columns or "Pred" not in df.columns:
        raise ValueError("Sample submission must contain ID and Pred columns")

    strategies = [
        "baseline_050",
        "constant_052",
        "constant_048",
        "jitter_sd002",
        "jitter_sd005",
    ][:max_files]

    run_dir = output_dir / date_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    stats: list[CandidateStats] = []

    for i, strategy in enumerate(strategies, start=1):
        out_df = df[["ID"]].copy()
        out_df["Pred"] = _build_predictions(df["Pred"], strategy, rng)
        file_name = f"{date_tag}_sub{i:02d}_{strategy}.csv"
        out_path = run_dir / file_name
        out_df.to_csv(out_path, index=False, float_format="%.6f")
        stats.append(
            CandidateStats(
                file=str(out_path),
                strategy=strategy,
                rows=int(out_df.shape[0]),
                pred_mean=float(out_df["Pred"].mean()),
                pred_std=float(out_df["Pred"].std()),
                pred_min=float(out_df["Pred"].min()),
                pred_max=float(out_df["Pred"].max()),
            )
        )

    manifest = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_sample_submission": str(sample_submission),
        "max_files": max_files,
        "seed": seed,
        "candidates": [asdict(s) for s in stats],
        "kaggle_submit_hint": "kaggle competitions submit -c march-machine-learning-mania-2026 -f <file.csv> -m \"note\"",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(s) for s in stats]).to_csv(run_dir / "manifest.csv", index=False)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate up to five daily submission candidates.")
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("input") / "SampleSubmissionStage2.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("submissions"))
    parser.add_argument(
        "--date-tag",
        type=str,
        default=datetime.now().strftime("%Y%m%d"),
        help="Tag used in output folder/file names (default: today, local time).",
    )
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    stats = generate_candidates(
        sample_submission=args.sample_submission,
        output_dir=args.output_dir,
        date_tag=args.date_tag,
        max_files=args.max_files,
        seed=args.seed,
    )
    print(f"Generated {len(stats)} submission candidate files in {args.output_dir / args.date_tag}")


if __name__ == "__main__":
    main()
