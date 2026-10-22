#!/usr/bin/env python3
"""Generate up to five daily Stage 2 submission candidates."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CandidateStats:
    file: str
    strategy: str
    rows: int
    pred_mean: float
    pred_std: float
    pred_min: float
    pred_max: float


def _read_submission(path: Path) -> tuple[list[str], list[float]]:
    ids: list[str] = []
    preds: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "Pred"]:
            raise ValueError(f"{path} must contain exactly ID and Pred columns")
        for row in reader:
            ids.append(row["ID"])
            preds.append(float(row["Pred"]))
    return ids, preds


def _write_submission(path: Path, ids: list[str], preds: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Pred"])
        for game_id, pred in zip(ids, preds):
            writer.writerow([game_id, f"{pred:.6f}"])


def _stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance ** 0.5, min(values), max(values)


def _build_predictions(base: list[float], strategy: str, rng: random.Random) -> list[float]:
    if strategy == "base":
        preds = base
    elif strategy == "base_pull_50":
        preds = [0.85 * value + 0.15 * 0.50 for value in base]
    elif strategy == "base_push":
        preds = [0.50 + 1.08 * (value - 0.50) for value in base]
    elif strategy == "jitter_sd002":
        preds = [value + rng.gauss(0.0, 0.02) for value in base]
    elif strategy == "jitter_sd005":
        preds = [value + rng.gauss(0.0, 0.05) for value in base]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return [min(0.999, max(0.001, value)) for value in preds]


def generate_candidates(
    sample_submission: Path,
    base_submission: Path | None,
    output_dir: Path,
    date_tag: str,
    max_files: int,
    seed: int,
) -> list[CandidateStats]:
    if max_files < 1 or max_files > 5:
        raise ValueError("max_files must be between 1 and 5")

    sample_ids, sample_preds = _read_submission(sample_submission)
    base_ids, base_preds = (sample_ids, sample_preds)
    if base_submission is not None:
        base_ids, base_preds = _read_submission(base_submission)
        if base_ids != sample_ids:
            raise ValueError("Base submission IDs must match the sample submission IDs exactly")

    strategies = [
        "base",
        "base_pull_50",
        "base_push",
        "jitter_sd002",
        "jitter_sd005",
    ][:max_files]

    run_dir = output_dir / date_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    stats: list[CandidateStats] = []

    for index, strategy in enumerate(strategies, start=1):
        preds = _build_predictions(base_preds, strategy, rng)
        file_name = f"{date_tag}_sub{index:02d}_{strategy}.csv"
        out_path = run_dir / file_name
        _write_submission(out_path, sample_ids, preds)
        pred_mean, pred_std, pred_min, pred_max = _stats(preds)
        stats.append(
            CandidateStats(
                file=str(out_path),
                strategy=strategy,
                rows=len(preds),
                pred_mean=pred_mean,
                pred_std=pred_std,
                pred_min=pred_min,
                pred_max=pred_max,
            )
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_sample_submission": str(sample_submission),
        "base_submission": str(base_submission) if base_submission else None,
        "max_files": max_files,
        "seed": seed,
        "candidates": [asdict(item) for item in stats],
        "kaggle_submit_hint": 'kaggle competitions submit -c march-machine-learning-mania-2026 -f <file.csv> -m "note"',
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (run_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(stats[0]).keys()))
        writer.writeheader()
        for item in stats:
            writer.writerow(asdict(item))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate up to five daily submission candidates.")
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("input") / "SampleSubmissionStage2.csv",
    )
    parser.add_argument(
        "--base-submission",
        type=Path,
        default=None,
        help="Optional base submission to perturb instead of using the sample baseline.",
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
        base_submission=args.base_submission,
        output_dir=args.output_dir,
        date_tag=args.date_tag,
        max_files=args.max_files,
        seed=args.seed,
    )
    print(f"Generated {len(stats)} submission candidate files in {args.output_dir / args.date_tag}")


if __name__ == "__main__":
    main()
