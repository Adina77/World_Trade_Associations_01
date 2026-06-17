#!/usr/bin/env python3
"""
ocr_cleanup.py

Post-processing cleanup for associations_raw.csv.  Run this once after the
pipeline finishes and produces the raw CSV.  Output is associations_cleaned.csv.

Step 1 — Fix embedded newlines
  Some model responses embed actual newline characters (\\n) inside field
  values (most commonly in address or name).  These break LibreOffice Calc
  and any CSV parser that does not handle RFC-4180 multi-line fields.
  Every embedded newline is replaced with a single space.

Step 2 — Remove boundary duplicates
  When the same association entry is the last entry on one page and the
  first entry on the next, the pipeline extracts it twice: same ID, same
  name, but different source_page.  The second occurrence (from the higher-
  numbered page) is removed; the first is kept.

  Entries with the same ID but DIFFERENT names are NOT auto-removed — those
  are misread IDs and need manual review.  ocr_error_check.py reports them.

Usage:
    python ocr_cleanup.py
"""

import csv
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "ocr_output"
INPUT_CSV  = OUTPUT_DIR / "associations_raw.csv"
OUTPUT_CSV = OUTPUT_DIR / "associations_cleaned.csv"

FIELDNAMES = ["id", "country", "name", "address", "focus", "source_page"]


def fix_newlines(value: str) -> str:
    return value.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')


def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"ERROR: {INPUT_CSV} not found — run ocr_pipeline.py first.")

    # ── Load ─────────────────────────────────────────────────────────────────
    with open(INPUT_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows):,} rows from {INPUT_CSV.name}\n")

    # ── Step 1: fix embedded newlines ────────────────────────────────────────
    newline_fixes: list[tuple[str, str]] = []
    for row in rows:
        for field in FIELDNAMES:
            val = row.get(field, "")
            if '\n' in val or '\r' in val:
                newline_fixes.append((row['id'], field))
                row[field] = fix_newlines(val)

    print(f"Step 1 — Embedded newlines fixed: {len(newline_fixes)}")
    for rid, field in newline_fixes:
        print(f"  id={rid}  field={field!r}")
    print()

    # ── Step 2: remove boundary duplicates ───────────────────────────────────
    # Group by ID.  If all entries for an ID share the same name → boundary
    # duplicate; keep the one from the lowest-numbered source page.
    # If entries differ in name → misread IDs; keep all for manual review.
    id_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        id_groups[row['id']].append(row)

    kept: list[dict] = []
    removed_boundary = 0
    misread_kept = 0

    for id_val in sorted(id_groups):
        group = id_groups[id_val]
        if len(group) == 1:
            kept.append(group[0])
            continue

        names = {r['name'] for r in group}
        if len(names) == 1:
            # Same name on multiple pages → keep earliest page
            best = min(group, key=lambda r: r['source_page'])
            kept.append(best)
            removed_boundary += len(group) - 1
        else:
            # Different names → misread IDs; keep all, flag for manual review
            kept.extend(group)
            misread_kept += len(group)

    print(f"Step 2 — Boundary duplicates (same name, different page):")
    print(f"  Removed: {removed_boundary} rows")
    if misread_kept:
        print(f"  Kept {misread_kept} rows with misread IDs (different names) — "
              f"see ocr_error_check.py for details")
    print()

    # ── Write output ─────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)

    SEP = "═" * 60
    print(SEP)
    print("  CLEANUP SUMMARY")
    print(SEP)
    print(f"  Input rows:             {len(rows):>7,}")
    print(f"  Newline fixes:          {len(newline_fixes):>7,}")
    print(f"  Boundary dups removed:  {removed_boundary:>7,}")
    print(f"  Output rows:            {len(kept):>7,}")
    print(f"  Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
