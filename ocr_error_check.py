#!/usr/bin/env python3
"""
ocr_error_check.py

Validates the raw OCR output CSV for completeness after the pipeline finishes.

Reads associations_raw.csv, sorts entries by the 5-digit sequential ID, and
checks that every integer in the range [min_id, max_id] is present.  Any gap
is reported together with the source page most likely responsible, so we know
exactly which page image to re-run through a better model.

Also checks for:
  - Duplicate IDs (same ID extracted from more than one page)
  - Malformed IDs (not exactly 5 digits)
  - Pages in the data range that were never processed

Falls back to progress.jsonl if the CSV has not been built yet (e.g. because
a JSON parse error on some pages prevented build_csv() from completing).

"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# ── Paths (mirrors ocr_pipeline.py) ─────────────────────────────────────────
OUTPUT_DIR      = Path(__file__).parent / "ocr_output"
FINAL_CSV       = OUTPUT_DIR / "associations_raw.csv"
CHECKPOINT_FILE = OUTPUT_DIR / "progress.jsonl"
IMAGE_DIR       = Path(__file__).parent / "WorldGuideTrade_bookpages"

FIRST_PAGE = "image00023.jpg"
LAST_PAGE  = "image00477.jpg"


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_from_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id":          row["id"].strip(),
                "source_page": row["source_page"].strip(),
            })
    return rows


def load_from_checkpoint(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for entry in rec["entries"]:
                id_val = entry.get("id", "").strip()
                if id_val:
                    rows.append({"id": id_val, "source_page": rec["page"]})
    return rows


def pages_processed_from_checkpoint(path: Path) -> set[str]:
    """All page filenames recorded in the checkpoint (including empty-result pages)."""
    processed = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                processed.add(json.loads(line)["page"])
    return processed


def pages_in_range(first: str, last: str) -> list[str]:
    """Sorted list of image filenames within the data-page range."""
    if not IMAGE_DIR.exists():
        return []
    all_images = sorted(p.name for p in IMAGE_DIR.glob("*.jpg"))
    try:
        start = all_images.index(first)
        end   = all_images.index(last) + 1
    except ValueError:
        return []
    return all_images[start:end]


# ── Gap analysis ─────────────────────────────────────────────────────────────

def find_responsible_page(
    gap_start: int,
    gap_end:   int,
    id_to_page: dict[int, str],
    page_list:  list[str],
) -> str:
    """
    Return the page(s) most likely responsible for a missing ID range.

    Looks at the nearest successfully extracted IDs below and above the gap:
    - Same page on both sides → that page missed those entries.
    - Adjacent pages → boundary; report both.
    - Pages in between → list the intermediate pages (likely unprocessed).
    """
    lower_page = upper_page = None
    for offset in range(1, 100_000):
        if lower_page is None and (gap_start - offset) in id_to_page:
            lower_page = id_to_page[gap_start - offset]
        if upper_page is None and (gap_end + offset) in id_to_page:
            upper_page = id_to_page[gap_end + offset]
        if lower_page is not None and upper_page is not None:
            break

    if lower_page is None and upper_page is None:
        return "unknown"
    if lower_page is None:
        return upper_page
    if upper_page is None:
        return lower_page
    if lower_page == upper_page:
        return lower_page

    # Span multiple pages — list the pages between the neighbours
    if page_list:
        try:
            lo_idx = page_list.index(lower_page)
            hi_idx = page_list.index(upper_page)
            span = page_list[lo_idx : hi_idx + 1]
            if len(span) <= 4:
                return "  /  ".join(span)
            return f"{span[0]}  …  {span[-1]}  ({len(span)} pages)"
        except ValueError:
            pass
    return f"{lower_page}  /  {upper_page}"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Choose data source
    if FINAL_CSV.exists():
        print(f"Source: {FINAL_CSV}\n")
        entries = load_from_csv(FINAL_CSV)
        source_label = "CSV"
    elif CHECKPOINT_FILE.exists():
        print(f"CSV not found — falling back to {CHECKPOINT_FILE}\n")
        entries = load_from_checkpoint(CHECKPOINT_FILE)
        source_label = "checkpoint JSONL"
    else:
        sys.exit(
            "ERROR: Neither associations_raw.csv nor progress.jsonl found in ocr_output/\n"
            "Run the pipeline first."
        )

    print(f"Rows loaded: {len(entries):,}")

    # 2. Sort by ID (the CSV is already sorted, but re-sort to be safe)
    entries.sort(key=lambda r: r["id"])

    # 3. Validate IDs
    id_to_page:  dict[int, str]        = {}           # int id → first source page
    extra_pages: dict[int, list[str]]  = defaultdict(list)  # duplicates
    bad_format:  list[dict]            = []

    for row in entries:
        raw = row["id"]
        if not raw.isdigit() or len(raw) != 5:
            bad_format.append(row)
            continue
        num = int(raw)
        if num in id_to_page:
            extra_pages[num].append(row["source_page"])
        else:
            id_to_page[num] = row["source_page"]

    if not id_to_page:
        sys.exit("ERROR: No valid 5-digit IDs found — check the data source.")

    min_id = min(id_to_page)
    max_id = max(id_to_page)
    expected_count = max_id - min_id + 1
    missing_ids = sorted(set(range(min_id, max_id + 1)) - id_to_page.keys())

    # Group consecutive missing IDs into gap ranges
    gaps: list[tuple[int, int]] = []
    if missing_ids:
        gs = ge = missing_ids[0]
        for m in missing_ids[1:]:
            if m == ge + 1:
                ge = m
            else:
                gaps.append((gs, ge))
                gs = ge = m
        gaps.append((gs, ge))

    # 4. Unprocessed pages
    all_pages = pages_in_range(FIRST_PAGE, LAST_PAGE)
    if all_pages and CHECKPOINT_FILE.exists():
        processed_pages = pages_processed_from_checkpoint(CHECKPOINT_FILE)
        unprocessed = [p for p in all_pages if p not in processed_pages]
    else:
        processed_pages = set()
        unprocessed = []

    # ── Report ───────────────────────────────────────────────────────────────
    SEP  = "─" * 70
    SEP2 = "═" * 70

    print(SEP2)
    print("  OCR COMPLETENESS REPORT")
    print(SEP2)
    print(f"  Source:             {source_label}")
    print(f"  ID range in data:   {min_id:05d} – {max_id:05d}")
    print(f"  IDs expected:       {expected_count:,}")
    print(f"  IDs found:          {len(id_to_page):,}")
    print(f"  Missing IDs:        {len(missing_ids):,}")
    print(f"  Duplicate IDs:      {len(extra_pages):,}")
    print(f"  Bad-format IDs:     {len(bad_format):,}")
    if all_pages:
        print(f"  Pages in range:     {len(all_pages):,}")
        print(f"  Pages processed:    {len(processed_pages):,}")
        print(f"  Pages unprocessed:  {len(unprocessed):,}")
    print(SEP2)
    print()

    any_issue = gaps or unprocessed or extra_pages or bad_format

    if not any_issue:
        print("✓  All checks passed — every ID is present and sequential, no duplicates.")
        return

    # ── Missing ID gaps ───────────────────────────────────────────────────────
    if gaps:
        print(f"MISSING ID GAPS  ({len(gaps)} gap(s), {len(missing_ids)} IDs missing)")
        print(SEP)
        for gs, ge in gaps:
            count = ge - gs + 1
            id_str = f"{gs:05d}" if count == 1 else f"{gs:05d} – {ge:05d}  ({count} IDs)"
            page   = find_responsible_page(gs, ge, id_to_page, all_pages)
            print(f"  Missing: {id_str}")
            print(f"    → Re-run page: {page}")
        print()

    # ── Unprocessed pages ─────────────────────────────────────────────────────
    if unprocessed:
        print(f"UNPROCESSED PAGES  ({len(unprocessed)} page(s) never sent to the API)")
        print(SEP)
        for p in unprocessed:
            print(f"  {p}")
        print()

    # ── Duplicate IDs ─────────────────────────────────────────────────────────
    if extra_pages:
        print(f"DUPLICATE IDs  ({len(extra_pages)} ID(s) extracted from multiple pages)")
        print(SEP)
        for num in sorted(extra_pages):
            pages = [id_to_page[num]] + extra_pages[num]
            print(f"  {num:05d}  →  {', '.join(pages)}")
        print()

    # ── Bad-format IDs ────────────────────────────────────────────────────────
    if bad_format:
        shown = bad_format[:20]
        print(f"BAD-FORMAT IDs  ({len(bad_format)} row(s) where 'id' is not exactly 5 digits)")
        print(SEP)
        for row in shown:
            print(f"  id={row['id']!r:14s}  source_page={row['source_page']}")
        if len(bad_format) > 20:
            print(f"  … and {len(bad_format) - 20} more")
        print()

    # ── Consolidated re-run list ───────────────────────────────────────────────
    rerun_pages: set[str] = set(unprocessed)
    for gs, ge in gaps:
        page_str = find_responsible_page(gs, ge, id_to_page, all_pages)
        for part in page_str.replace(" … ", "  /  ").split("  /  "):
            part = part.strip()
            if part and part != "unknown" and not part.startswith("("):
                rerun_pages.add(part)

    if rerun_pages:
        print("PAGES TO RE-RUN  (consolidated)")
        print(SEP)
        for p in sorted(rerun_pages):
            print(f"  {p}")
        print()
        print(f"Total pages to re-run: {len(rerun_pages)}")


if __name__ == "__main__":
    main()
