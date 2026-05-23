#!/usr/bin/env python3
"""Run MemoNoveltyAgent for a single paper title."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AGENT_DIR = ROOT / "MemoNoveltyAgent"
sys.path.insert(0, str(AGENT_DIR))

from Main import load_config, run_single_paper_pipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate a novelty analysis report for one paper.")
    parser.add_argument("paper_name", help="Exact paper title or arXiv id to analyze.")
    parser.add_argument("--config", default=str(AGENT_DIR / "config.json"), help="Path to config.json.")
    parser.add_argument("--database-dir", help="Override local PDF database directory.")
    parser.add_argument("--output-dir", help="Override output root for generated artifacts.")
    parser.add_argument("--max-total-papers", type=int, help="Target number of main+reference PDFs to collect.")
    parser.add_argument("--force", action="store_true", help="Re-run even when cached intermediate files exist.")
    parser.add_argument("--no-validation", action="store_true", help="Skip validation/polish stages.")
    args = parser.parse_args()

    config = load_config(args.config)
    config["paper_name"] = args.paper_name

    if args.database_dir:
        config.setdefault("paths", {})["database_dir"] = args.database_dir
    if args.max_total_papers:
        config["max_total_papers"] = args.max_total_papers
    if args.output_dir:
        out = Path(args.output_dir)
        paths = config.setdefault("paths", {})
        paths["result_dir"] = str(out / "pipeline_state")
        paths["innovation_points_dir"] = str(out / "innovation_points")
        paths["paper_summary_dir"] = str(out / "paper_summaries")
        paths["compare_result_dir"] = str(out / "comparisons")
        paths["reports_dir"] = str(out / "reports")
        paths["draft_reports_dir"] = str(out / "draft_reports")
        paths["validated_reports_dir"] = str(out / "validated_reports")
        paths["polished_reports_dir"] = str(out / "polished_reports")

    report_path = run_single_paper_pipeline(
        config,
        paper_name=args.paper_name,
        paper_folder_path=None,
        enable_validation=not args.no_validation,
        force_rerun=args.force,
    )
    if not report_path:
        raise SystemExit(1)
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
