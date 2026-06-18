#!/usr/bin/env python3
"""
ocr_cleanup.py

Post-processing cleanup for associations_raw.csv.  Run this once after the
pipeline finishes and produces the raw CSV.  Output is associations_cleaned.csv.

Step 0 — Filter duplicate-scan pages
  Some book pages were scanned twice, producing two image files that cover
  the same content.  Both files were OCR'd, producing slightly different
  text that evades the name-match deduplication in Step 2 and shows up as
  misread-ID duplicates in the error check.

  If ocr_output/duplicate_pages.txt exists (written by find_duplicate_pages.py),
  any row whose source_page is listed there is removed before further processing.

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

Step 3 — Clean country field
  The page heading format is "Belgium: Syndicat 05273 — 05460". The model
  should extract only the text before the colon as the country name, but
  sometimes includes the word(s) after the colon (the start of an association
  name), producing values like "Belgium: European" or "U.S.A.: Wisconsin".

  Fix: strip everything from the first colon onward.  All legitimate country
  names in this dataset use commas (e.g. "Korea, Republic") but not colons,
  so the colon is a reliable artifact signal.

  One entry (id 23393) has "Wisconsin" as the country — the model misread the
  heading entirely.  This is corrected to "U.S.A." as a hardcoded override.
"""

import csv
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR           = Path(__file__).parent / "ocr_output"
INPUT_CSV            = OUTPUT_DIR / "associations_raw.csv"
OUTPUT_CSV           = OUTPUT_DIR / "associations_cleaned.csv"
DUPLICATE_PAGES_FILE = OUTPUT_DIR / "duplicate_pages.txt"

FIELDNAMES = ["id", "country", "name", "address", "focus", "source_page"]


COUNTRY_OVERRIDES = {
    "Wisconsin": "U.S.A.",   # id 23393: model misread the page heading entirely
}


def load_ignored_pages() -> set[str]:
    """Return image filenames listed in duplicate_pages.txt (if the file exists)."""
    if not DUPLICATE_PAGES_FILE.exists():
        return set()
    ignored = set()
    with open(DUPLICATE_PAGES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0]
            if token.endswith(".jpg"):
                ignored.add(token)
    return ignored


def fix_newlines(value: str) -> str:
    return value.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')


def clean_country(value: str) -> str:
    """Strip colon artifacts from heading bleed-through, e.g. 'Belgium: European' → 'Belgium'."""
    value = COUNTRY_OVERRIDES.get(value, value)
    if ':' in value:
        value = value.split(':')[0].strip()
    return value


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

    # ── Step 0: filter duplicate-scan pages ──────────────────────────────────
    ignored = load_ignored_pages()
    if ignored:
        before = len(rows)
        rows = [r for r in rows if r.get('source_page', '') not in ignored]
        removed = before - len(rows)
        print(f"Step 0 — Duplicate-scan pages removed: {removed} row(s) "
              f"({len(ignored)} page(s) in duplicate_pages.txt)")
        for p in sorted(ignored):
            print(f"  {p}")
    else:
        print("Step 0 — duplicate_pages.txt not found; no pages filtered")
    print()

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

    # ── Step 3: clean country field ───────────────────────────────────────────
    country_fixes: list[tuple[str, str, str]] = []   # (id, old_value, new_value)
    for row in kept:
        original = row['country']
        cleaned  = clean_country(original)
        if cleaned != original:
            country_fixes.append((row['id'], original, cleaned))
            row['country'] = cleaned

    print(f"Step 3 — Country field cleaned: {len(country_fixes)} row(s) fixed")
    for rid, old, new in country_fixes[:20]:
        print(f"  id={rid}  {old!r}  →  {new!r}")
    if len(country_fixes) > 20:
        print(f"  ... and {len(country_fixes) - 20} more")
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
    print(f"  Country field fixes:      {len(country_fixes):>7,}")
    print(f"  Output rows:              {len(kept):>7,}")
    print(f"  Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
