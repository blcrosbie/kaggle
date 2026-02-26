#!/usr/bin/env python3
"""Summarize competition rules and docs into an actionable brief."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_line(text: str, pattern: str, default: str = "Not found") -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return default
    return match.group(0).strip()


def build_competition_brief(about_dir: Path, reports_dir: Path) -> Path:
    rules_text = _read_text(about_dir / "rules.md")
    overview_text = _read_text(about_dir / "overview.md")
    data_text = _read_text(about_dir / "data-description.md")
    overview_html = _read_text(about_dir / "overview.html")

    timeline_lines = re.findall(
        r"^[A-Za-z]+\s+\d{1,2},\s+\d{4}.*$", overview_text, flags=re.MULTILINE
    )
    timeline_block = (
        "\n".join(f"- {line.strip()}" for line in timeline_lines)
        if timeline_lines
        else "- Timeline lines not parsed from overview.md"
    )

    rules_bits = {
        "team_limit": _extract_line(rules_text, r"maximum Team size is five.*"),
        "daily_submission_limit": _extract_line(
            rules_text, r"maximum of five \(5\) Submissions per day.*"
        ),
        "final_submission_limit": _extract_line(
            rules_text, r"select up to two \(2\) Final Submissions.*"
        ),
        "external_data": _extract_line(
            rules_text, r"You may use data other than the Competition Data.*"
        ),
    }

    eval_line = _extract_line(overview_text, r"Submissions are evaluated on the Brier score.*")
    format_line = _extract_line(
        overview_text, r"you must predict the probability that the team with the lower TeamId.*"
    )
    refresh_line = _extract_line(
        overview_text, r"leaderboard.*only meaningful once the 2026 tournaments begin.*"
    )

    data_section_lines = re.findall(r"^Data Section \d+.*$", data_text, flags=re.MULTILINE)

    brief_lines = [
        "# Competition Brief",
        "",
        "## Core constraints",
        f"- {rules_bits['daily_submission_limit']}",
        f"- {rules_bits['final_submission_limit']}",
        f"- {rules_bits['team_limit']}",
        f"- {rules_bits['external_data']}",
        "",
        "## Evaluation and submission format",
        f"- {eval_line}",
        "- Single submission must include both men's and women's matchups.",
        f"- {format_line}",
        "- Required output columns: `ID,Pred` with `ID` like `2026_1101_1102`.",
        f"- {refresh_line}",
        "",
        "## Timeline (from overview.md)",
        timeline_block,
        "",
        "## Data inventory highlights",
        f"- Data sections documented: {len(data_section_lines)}",
        "- Files are split by prefixes: `M*` men, `W*` women, shared files without prefix.",
        "- Historical season depth reaches back to 1985 in core men's files and 1998 in core women's files.",
        "- Detailed box-score files begin later (men 2003, women 2010).",
        "",
        "## Notes",
        "- `about/overview.html` is primarily Kaggle page shell markup (scripts/meta), with no additional competition-rule content beyond metadata/title.",
        "- Keep external data/tools reasonably accessible and low-cost per rules.",
        "",
        "## Immediate execution plan",
        "1. Profile all CSVs and confirm season coverage, missingness, and key joins.",
        "2. Build a baseline model from regular season + tournament history.",
        "3. Generate 5 daily candidate submissions and track outcomes.",
        "4. Promote top daily candidates to the 2 final submissions near deadline.",
    ]

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "competition_brief.md"
    out_path.write_text("\n".join(brief_lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize competition docs into a brief.")
    parser.add_argument("--about-dir", type=Path, default=Path("about"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    out_path = build_competition_brief(args.about_dir, args.reports_dir)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
