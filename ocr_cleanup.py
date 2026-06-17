#!/usr/bin/env python3
"""
ocr_cleanup.py

Post-processing cleanup for associations_raw.csv.  Run this once after the
pipeline finishes and produces the raw CSV.  Output is associations_cleaned.csv.

Step 1 — Fix embedded newlines
  Some model responses embed actual newline characters inside field values
  (most commonly in address or name).  These break LibreOffice Calc and any
  CSV parser that does not handle RFC-4180 multi-line fields.
  Every embedded newline is replaced with a single space.

Step 2 — Merge split-entry duplicates
  When an association entry spans two pages, the pipeline extracts two partial
  records that share the same ID:
    • Page N  — has the name + start of address (no Focus, no ID visible)
    • Page N+1 — has the address continuation + Focus + ID

  These are merged into one complete record:
    name    → from the page where the name is visible (non-empty)
    address → both halves joined with  " | "
    focus   → from the later page (where Focus appears)
    id      → from the later page (confirmed by the text)

  Entries with the same ID but two DIFFERENT non-empty names are NOT merged —
  those are misread IDs and need manual review (ocr_error_check.py reports them).
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


def merge_group(group: list[dict]) -> dict:
    """
    Merge two (or more) partial records for the same split entry into one.

    Sort by source_page so page-N comes before page-N+1.
    - name    : first non-empty name
    - country : first non-empty country
    - address : see below
    - focus   : from the last record (where Focus appears in the text)
    - id      : from the last record (confirmed ID)
    - source_page : from the last record

    Address strategy — two distinct cases:
      Boundary duplicate: both models read the same complete address text
        (entry appeared at the bottom of page N AND the top of page N+1).
        Detected when both address strings share a long common prefix (≥40 chars).
        Action: keep the longer string (more complete).
      True split: one model read the first half, the other the second half.
        Detected when the address strings start differently.
        Action: join with ' | ' to preserve both halves.
    """
    ordered = sorted(group, key=lambda r: r['source_page'])
    last    = ordered[-1]

    name    = next((r['name']    for r in ordered if r['name'].strip()),    '')
    country = next((r['country'] for r in ordered if r['country'].strip()), '')

    addr_parts = [r['address'].strip() for r in ordered if r['address'].strip()]

    if len(addr_parts) == 0:
        address = ''
    elif len(addr_parts) == 1:
        address = addr_parts[0]
    else:
        a1, a2 = addr_parts[0], addr_parts[-1]
        PREFIX_LEN = 40
        # If both addresses start with the same text → same complete address
        # extracted twice (boundary duplicate) → keep the longer one
        if a1[:PREFIX_LEN] == a2[:PREFIX_LEN]:
            address = a1 if len(a1) >= len(a2) else a2
        else:
            # Different starts → genuine split-entry halves → join them
            address = ' | '.join(addr_parts)

    return {
        'id':          last['id'],
        'country':     country,
        'name':        name,
        'address':     address,
        'focus':       last['focus'],
        'source_page': last['source_page'],
    }


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

    # ── Step 2: merge split-entry duplicates ─────────────────────────────────
    # Group by ID.
    # If all entries for an ID have the same non-empty name (or one name is "")
    #   → split-entry pair: merge into one complete record.
    # If two+ entries have DIFFERENT non-empty names
    #   → misread IDs: keep all, flag for manual review.
    id_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        id_groups[row['id']].append(row)

    kept: list[dict] = []
    merged_count  = 0   # number of extra rows collapsed by merging
    misread_kept  = 0

    for id_val in sorted(id_groups):
        group = id_groups[id_val]
        if len(group) == 1:
            kept.append(group[0])
            continue

        non_empty_names = {r['name'] for r in group if r['name'].strip()}

        if len(non_empty_names) <= 1:
            # All entries share one name (or one entry has name="")  → merge
            kept.append(merge_group(group))
            merged_count += len(group) - 1
        else:
            # Different non-empty names → misread IDs; keep all for review
            kept.extend(group)
            misread_kept += len(group)

    print(f"Step 2 — Split-entry duplicates merged: {merged_count} extra row(s) collapsed")
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
    print(f"  Input rows:               {len(rows):>7,}")
    print(f"  Newline fixes:            {len(newline_fixes):>7,}")
    print(f"  Split-entry rows merged:  {merged_count:>7,}")
    print(f"  Output rows:              {len(kept):>7,}")
    print(f"  Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
