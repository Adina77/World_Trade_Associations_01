#!/usr/bin/env python3
"""
error_redo.py

Re-processes a selected set of page images with the model currently set in
ocr_pipeline.py, then rebuilds the full pipeline output.

Workflow
────────
1. Read the pages to re-process from ocr_output/pages_to_redo.txt
   (generated and ranked by ocr_error_check.py).

2. Remove those pages from progress.jsonl so the pipeline treats them as
   unprocessed.

3. Run ocr_pipeline.py  →  OCR only the removed pages; rebuild associations_raw.csv.

4. Run ocr_cleanup.py   →  Fix newlines, merge split entries;
                            produce associations_cleaned.csv.

5. Run ocr_error_check.py  →  Report remaining issues and refresh pages_to_redo.txt.

Repeat as needed, each time working on the highest-issue pages first.

Usage
─────
  python error_redo.py                  # process ALL pages listed in pages_to_redo.txt
  python error_redo.py --top 5          # process only the 5 highest-issue pages
  python error_redo.py --min-issues 20  # process only pages with ≥ 20 total issues
  python error_redo.py --top 10 --min-issues 5   # both filters applied

Selecting pages manually (original workflow)
  Edit pages_to_redo.txt to keep only the lines you want, then run without flags.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
OUTPUT_DIR      = BASE_DIR / "ocr_output"
CHECKPOINT_FILE = OUTPUT_DIR / "progress.jsonl"
FAILED_FILE     = OUTPUT_DIR / "failed_pages.jsonl"
REDO_PAGES_FILE = OUTPUT_DIR / "pages_to_redo.txt"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_redo_pages() -> list[tuple[str, int]]:
    """
    Read pages from pages_to_redo.txt.
    Returns a list of (filename, issue_count) pairs in file order (highest first).
    issue_count is 0 if the line has no parseable count.
    """
    if not REDO_PAGES_FILE.exists():
        sys.exit(
            f"ERROR: {REDO_PAGES_FILE.name} not found.\n"
            "Run ocr_error_check.py first to generate it."
        )
    pages = []
    with open(REDO_PAGES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if not tokens or not tokens[0].endswith(".jpg"):
                continue
            filename = tokens[0]
            # Parse issue count from comment: "# 156 issues  [...]"
            m = re.search(r"#\s*(\d+)\s+issues", line)
            count = int(m.group(1)) if m else 0
            pages.append((filename, count))
    return pages


def remove_from_checkpoint(pages: set[str]) -> int:
    """Rewrite progress.jsonl without entries for the specified pages."""
    if not CHECKPOINT_FILE.exists():
        return 0
    kept_lines = []
    removed = 0
    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rec = json.loads(stripped)
            if rec["page"] in pages:
                removed += 1
            else:
                kept_lines.append(stripped)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")
    return removed


def remove_from_failed_log(pages: set[str]):
    """Remove stale failure records for pages we are about to retry."""
    if not FAILED_FILE.exists():
        return
    kept_lines = []
    with open(FAILED_FILE, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rec = json.loads(stripped)
                if rec["page"] not in pages:
                    kept_lines.append(stripped)
    with open(FAILED_FILE, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")


def run_script(script_name: str, label: str, extra_args: list[str] | None = None) -> int:
    """Run a sibling Python script and return its exit code."""
    script = BASE_DIR / script_name
    cmd    = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'─'*60}")
    print(f"  Running {label} ...")
    print(f"{'─'*60}")
    result = subprocess.run(cmd)
    return result.returncode


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-process high-issue pages and rebuild the pipeline output."
    )
    parser.add_argument(
        "--top", type=int, metavar="N",
        help="Process only the top N highest-issue pages from pages_to_redo.txt"
    )
    parser.add_argument(
        "--min-issues", type=int, metavar="N", dest="min_issues",
        help="Process only pages with at least N total issues"
    )
    args = parser.parse_args()

    all_entries = load_redo_pages()   # list of (filename, issue_count)
    if not all_entries:
        sys.exit("No pages found in pages_to_redo.txt — nothing to do.")

    # Apply --min-issues filter first (preserves ranking order)
    if args.min_issues is not None:
        filtered = [(p, n) for p, n in all_entries if n >= args.min_issues]
    else:
        filtered = list(all_entries)

    # Apply --top N slice
    if args.top is not None:
        filtered = filtered[: args.top]

    if not filtered:
        sys.exit(
            f"No pages passed the filter "
            f"(total in file: {len(all_entries)}, "
            f"--top={args.top}, --min-issues={args.min_issues})."
        )

    pages   = [p for p, _ in filtered]
    page_set = set(pages)
    SEP2 = "═" * 60
    print(SEP2)
    print("  ERROR REDO")
    print(SEP2)
    if args.top or args.min_issues:
        filter_desc = []
        if args.top:        filter_desc.append(f"--top {args.top}")
        if args.min_issues: filter_desc.append(f"--min-issues {args.min_issues}")
        print(f"  Filter:              {', '.join(filter_desc)}")
        print(f"  Pages in file:       {len(all_entries)}")
    print(f"  Pages to re-process: {len(pages)}")
    for p, n in filtered:
        count_str = f"  ({n} issues)" if n else ""
        print(f"    {p}{count_str}")
    print()

    # Step 1 — remove from checkpoint so pipeline picks them up
    removed = remove_from_checkpoint(page_set)
    remove_from_failed_log(page_set)
    print(f"Removed {removed} checkpoint entry/entries for {len(page_set)} page(s).")
    print("The pipeline will re-process only these pages.")

    # Step 2 — re-run OCR pipeline (only the selected pages)
    rc = run_script("ocr_pipeline.py", "OCR pipeline", extra_args=["--pages"] + pages)
    if rc != 0:
        print(f"\nWARNING: ocr_pipeline.py exited with code {rc}.")
        print("Some pages may not have been processed. Check failed_pages.jsonl.")

    # Step 3 — rebuild cleaned CSV
    run_script("ocr_cleanup.py", "Cleanup (rebuild associations_cleaned.csv)")

    # Step 4 — refresh error report and pages_to_redo.txt
    run_script("ocr_error_check.py", "Error check (refresh pages_to_redo.txt)")

    print(f"\n{SEP2}")
    print("  REDO COMPLETE")
    print(SEP2)
    print("  Review the error check output above.")
    print("  Edit pages_to_redo.txt and run error_redo.py again to continue.")


if __name__ == "__main__":
    main()
