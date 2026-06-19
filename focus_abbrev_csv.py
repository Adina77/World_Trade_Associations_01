#!/usr/bin/env python3
"""
Generate focus_abbreviations.csv from the focus_abbrev_progress.jsonl checkpoint.

Run this to rebuild the CSV without repeating API calls, e.g. after manually
editing the .jsonl file to fix a mis-read entry.

Usage:  python focus_abbrev_csv.py
"""

import csv
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "ocr_output"
CHECKPOINT = OUTPUT_DIR / "focus_abbrev_progress.jsonl"
FINAL_CSV  = OUTPUT_DIR / "focus_abbreviations.csv"

ABBREV_PAGES = [
    "image00013.jpg",
    "image00014.jpg",
    "image00015.jpg",
    "image00016.jpg",
]


def build_csv():
    if not CHECKPOINT.exists():
        print(f"ERROR: {CHECKPOINT} not found. Run focus_abbrev_ocr.py first.")
        return

    page_order   = {name: i for i, name in enumerate(ABBREV_PAGES)}
    rows_by_page = {}

    with open(CHECKPOINT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["page"] in page_order:
                # Keep only the last record per page (in case of re-runs)
                rows_by_page[rec["page"]] = rec["pairs"]

    all_pairs = []
    for page in ABBREV_PAGES:
        for pair in rows_by_page.get(page, []):
            all_pairs.append({
                "abbreviation": pair.get("abbreviation", ""),
                "full_term":    pair.get("full_term",    ""),
                "source_page":  page,
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["abbreviation", "full_term", "source_page"]
    with open(FINAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_pairs)

    done_pages = [p for p in ABBREV_PAGES if p in rows_by_page]
    missing    = [p for p in ABBREV_PAGES if p not in rows_by_page]

    print(f"Pages included:  {len(done_pages)}/4  "
          f"({', '.join(done_pages) if done_pages else 'none'})")
    if missing:
        print(f"WARNING — missing pages not yet processed: {', '.join(missing)}")
    print(f"Saved {len(all_pairs)} abbreviation pairs → {FINAL_CSV}")


if __name__ == "__main__":
    build_csv()
