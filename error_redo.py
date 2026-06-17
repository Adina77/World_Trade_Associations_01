#!/usr/bin/env python3
"""
error_redo.py

Re-processes a selected set of page images with the model currently set in
ocr_pipeline.py, then rebuilds the full pipeline output.

Workflow
────────
1. Read the pages to re-process from ocr_output/pages_to_redo.txt
   (generated and ranked by ocr_error_check.py).
   Edit that file first — remove pages you do not want to re-scan, or
   reduce it to just the top N highest-issue pages.

2. Remove those pages from progress.jsonl so the pipeline treats them as
   unprocessed.

3. Run ocr_pipeline.py  →  OCR only the removed pages; rebuild associations_raw.csv.

4. Run ocr_cleanup.py   →  Fix newlines, remove boundary duplicates;
                            produce associations_cleaned.csv.

5. Run ocr_error_check.py  →  Report remaining issues and refresh pages_to_redo.txt.

Repeat as needed, each time working on the highest-issue pages first.

Usage
─────
  1. Run ocr_error_check.py to generate pages_to_redo.txt.
  2. Edit pages_to_redo.txt — keep only the pages you want to redo this round.
  3. python error_redo.py
"""

import json
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

def load_redo_pages() -> list[str]:
    """Read page names from pages_to_redo.txt, skipping comment lines."""
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
            # Each data line is "  image000XX.jpg  # N issues" — take first token
            page = line.split()[0]
            if page.endswith(".jpg"):
                pages.append(page)
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


def run_script(script_name: str, label: str) -> int:
    """Run a sibling Python script and return its exit code."""
    script = BASE_DIR / script_name
    print(f"\n{'─'*60}")
    print(f"  Running {label} ...")
    print(f"{'─'*60}")
    result = subprocess.run([sys.executable, str(script)])
    return result.returncode


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    pages = load_redo_pages()
    if not pages:
        sys.exit("No pages found in pages_to_redo.txt — nothing to do.")

    page_set = set(pages)
    SEP2 = "═" * 60
    print(SEP2)
    print("  ERROR REDO")
    print(SEP2)
    print(f"  Pages to re-process: {len(pages)}")
    for p in pages:
        print(f"    {p}")
    print()

    # Step 1 — remove from checkpoint so pipeline picks them up
    removed = remove_from_checkpoint(page_set)
    remove_from_failed_log(page_set)
    print(f"Removed {removed} checkpoint entry/entries for {len(page_set)} page(s).")
    print("The pipeline will re-process only these pages.")

    # Step 2 — re-run OCR pipeline
    rc = run_script("ocr_pipeline.py", "OCR pipeline")
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
