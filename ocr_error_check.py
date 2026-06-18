#!/usr/bin/env python3
"""
ocr_error_check.py

Validates OCR output for completeness and flags pages to re-run.

Prefers associations_cleaned.csv (post-cleanup); falls back to
associations_raw.csv, then progress.jsonl.

Checks:
  - Sequential ID completeness (gaps)
  - Duplicate IDs — categorised as boundary dups (same name) or
    misread IDs (different names)
  - Malformed IDs (not exactly 5 digits)
  - Pages in the data range never processed

Outputs ocr_output/pages_to_redo.txt: source pages ranked by issue
count (misread-ID duplicates + bad-format IDs), ready for error_redo.py.
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = Path(__file__).parent / "ocr_output"
CLEANED_CSV     = OUTPUT_DIR / "associations_cleaned.csv"
FINAL_CSV       = OUTPUT_DIR / "associations_raw.csv"
CHECKPOINT_FILE = OUTPUT_DIR / "progress.jsonl"
IMAGE_DIR       = Path(__file__).parent / "WorldGuideTrade_bookpages"
REDO_PAGES_FILE = OUTPUT_DIR / "pages_to_redo.txt"

FIRST_PAGE = "image00023.jpg"
LAST_PAGE  = "image00400.jpg"


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_from_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id":          row["id"].strip(),
                "name":        row.get("name", "").strip(),
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
                    rows.append({
                        "id":          id_val,
                        "name":        entry.get("name", "").strip(),
                        "source_page": rec["page"],
                    })
    return rows


def pages_processed_from_checkpoint(path: Path) -> set[str]:
    processed = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                processed.add(json.loads(line)["page"])
    return processed


def pages_in_range(first: str, last: str) -> list[str]:
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


def find_responsible_pages(
    gap_start:       int,
    gap_end:         int,
    id_to_page:      dict[int, str],
    page_list:       list[str],
    processed_pages: set[str],
) -> list[str]:
    """
    Return the page filename(s) most likely responsible for a missing-ID gap,
    as a list suitable for issue counting.

    When the gap falls between two different pages, unprocessed pages inside
    the span are returned (they're the clear cause).  If all pages in the span
    were processed, both neighboring pages are returned because we can't tell
    which one missed the entries.
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
        return []
    if lower_page is None:
        return [upper_page]
    if upper_page is None:
        return [lower_page]
    if lower_page == upper_page:
        return [lower_page]

    if page_list:
        try:
            lo_idx = page_list.index(lower_page)
            hi_idx = page_list.index(upper_page)
            span   = page_list[lo_idx : hi_idx + 1]
            unprocessed_in_span = [p for p in span if p not in processed_pages]
            if unprocessed_in_span:
                return unprocessed_in_span   # unprocessed pages caused the gap
        except ValueError:
            pass
    # All pages in span were processed — blame both neighbors
    return [lower_page, upper_page]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Choose data source (prefer cleaned, then raw, then checkpoint)
    if CLEANED_CSV.exists():
        print(f"Source: {CLEANED_CSV.name}\n")
        entries = load_from_csv(CLEANED_CSV)
        source_label = "associations_cleaned.csv"
    elif FINAL_CSV.exists():
        print(f"Source: {FINAL_CSV.name}  (run ocr_cleanup.py for a cleaned version)\n")
        entries = load_from_csv(FINAL_CSV)
        source_label = "associations_raw.csv"
    elif CHECKPOINT_FILE.exists():
        print(f"No CSV found — falling back to {CHECKPOINT_FILE.name}\n")
        entries = load_from_checkpoint(CHECKPOINT_FILE)
        source_label = "progress.jsonl"
    else:
        sys.exit(
            "ERROR: No data source found in ocr_output/\n"
            "Run ocr_pipeline.py first."
        )

    print(f"Rows loaded: {len(entries):,}")
    entries.sort(key=lambda r: r["id"])

    # 2. Parse and validate IDs
    # id_to_page: first occurrence of each numeric ID → source page
    # dup_entries: all entries (including first) for IDs that appear more than once
    id_to_page:  dict[int, str]         = {}
    dup_entries: dict[int, list[dict]]  = defaultdict(list)
    bad_format:  list[dict]             = []

    for row in entries:
        raw = row["id"]
        if not raw.isdigit() or len(raw) != 5:
            bad_format.append(row)
            continue
        num = int(raw)
        dup_entries[num].append(row)
        if num not in id_to_page:
            id_to_page[num] = row["source_page"]

    # Separate true duplicates from singles
    true_dups = {k: v for k, v in dup_entries.items() if len(v) > 1}

    # Categorise duplicates
    boundary_dups: list[int] = []   # same name → page-boundary overlap
    misread_dups:  list[int] = []   # different names → model misread an ID

    for num, group in true_dups.items():
        names = {r["name"] for r in group}
        if len(names) == 1:
            boundary_dups.append(num)
        else:
            misread_dups.append(num)

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

    # 3. Unprocessed pages
    all_pages = pages_in_range(FIRST_PAGE, LAST_PAGE)
    if all_pages and CHECKPOINT_FILE.exists():
        processed_pages = pages_processed_from_checkpoint(CHECKPOINT_FILE)
        unprocessed = [p for p in all_pages if p not in processed_pages]
    else:
        processed_pages = set()
        unprocessed = []

    # 4. Per-page issue counts: missing IDs + misread dups + bad-format IDs
    #    Track each type separately so the ranked list can show a breakdown.
    page_gap_count: dict[str, int] = defaultdict(int)  # missing IDs attributed here
    page_dup_count: dict[str, int] = defaultdict(int)  # misread-ID duplicate rows
    page_bad_count: dict[str, int] = defaultdict(int)  # bad-format ID rows

    for gs, ge in gaps:
        gap_size = ge - gs + 1
        for page in find_responsible_pages(gs, ge, id_to_page, all_pages, processed_pages):
            page_gap_count[page] += gap_size

    for num in misread_dups:
        for row in true_dups[num]:
            page_dup_count[row["source_page"]] += 1

    for row in bad_format:
        page_bad_count[row["source_page"]] += 1

    all_issue_pages = set(page_gap_count) | set(page_dup_count) | set(page_bad_count)
    page_issues: dict[str, int] = {
        p: page_gap_count[p] + page_dup_count[p] + page_bad_count[p]
        for p in all_issue_pages
    }

    ranked_pages = sorted(page_issues, key=lambda p: page_issues[p], reverse=True)

    # ── Print report ─────────────────────────────────────────────────────────
    SEP  = "─" * 70
    SEP2 = "═" * 70

    print(SEP2)
    print("  OCR COMPLETENESS REPORT")
    print(SEP2)
    print(f"  Source:               {source_label}")
    print(f"  ID range in data:     {min_id:05d} – {max_id:05d}")
    print(f"  IDs expected:         {expected_count:,}")
    print(f"  IDs found (unique):   {len(id_to_page):,}")
    print(f"  Missing IDs:          {len(missing_ids):,}")
    print(f"  Duplicate IDs:        {len(true_dups):,}  "
          f"({len(boundary_dups)} boundary, {len(misread_dups)} misread)")
    print(f"  Bad-format IDs:       {len(bad_format):,}")
    if all_pages:
        print(f"  Pages in range:       {len(all_pages):,}")
        print(f"  Pages processed:      {len(processed_pages):,}")
        print(f"  Pages unprocessed:    {len(unprocessed):,}")
    print(f"  Pages with issues:    {len(page_issues):,}  "
          f"(missing IDs + duplicates + bad-format — candidates for re-scan)")
    print(SEP2)
    print()

    any_issue = gaps or unprocessed or true_dups or bad_format

    if not any_issue:
        print("✓  All checks passed — every ID is present and sequential, no duplicates.")
        if REDO_PAGES_FILE.exists():
            REDO_PAGES_FILE.unlink()
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

    # ── Boundary duplicates ───────────────────────────────────────────────────
    if boundary_dups:
        print(f"BOUNDARY DUPLICATES  ({len(boundary_dups)} — same name, different page)")
        print(SEP)
        print("  These should have been removed by ocr_cleanup.py.")
        print("  If present, re-run ocr_cleanup.py.")
        for num in sorted(boundary_dups)[:10]:
            pages = [r["source_page"] for r in true_dups[num]]
            print(f"  {num:05d}  →  {', '.join(pages)}")
        if len(boundary_dups) > 10:
            print(f"  … and {len(boundary_dups) - 10} more")
        print()

    # ── Misread-ID duplicates ─────────────────────────────────────────────────
    if misread_dups:
        print(f"MISREAD-ID DUPLICATES  ({len(misread_dups)} — different names sharing the same ID)")
        print(SEP)
        print("  One entry has a correct ID; the other has a misread ID.")
        print("  Re-scanning the source pages with a better model should fix these.")
        print()
        for num in sorted(misread_dups):
            group = true_dups[num]
            print(f"  ID {num:05d}:")
            for r in group:
                print(f"    [{r['source_page']}]  {r['name'][:60]}")
        print()

    # ── Bad-format IDs ────────────────────────────────────────────────────────
    if bad_format:
        print(f"BAD-FORMAT IDs  ({len(bad_format)} row(s) where 'id' is not exactly 5 digits)")
        print(SEP)
        for row in bad_format[:20]:
            print(f"  id={row['id']!r:14s}  [{row['source_page']}]  {row['name'][:50]}")
        if len(bad_format) > 20:
            print(f"  … and {len(bad_format) - 20} more")
        print()

    # ── Pages ranked by issue count ───────────────────────────────────────────
    if ranked_pages:
        print(f"PAGES TO RE-SCAN  (ranked by total issue count — missing IDs + dups + bad-format)")
        print(SEP)
        print(f"  {'Page':<25}  {'Total':>5}  Breakdown")
        print(f"  {'─'*25}  {'─'*5}  {'─'*30}")
        for p in ranked_pages:
            parts = []
            if page_gap_count[p]: parts.append(f"{page_gap_count[p]} missing")
            if page_dup_count[p]: parts.append(f"{page_dup_count[p]} dups")
            if page_bad_count[p]: parts.append(f"{page_bad_count[p]} bad-format")
            print(f"  {p:<25}  {page_issues[p]:>5}  {', '.join(parts)}")
        print()

    # ── Write pages_to_redo.txt ───────────────────────────────────────────────
    with open(REDO_PAGES_FILE, "w", encoding="utf-8") as f:
        f.write(f"# pages_to_redo.txt — generated by ocr_error_check.py on {date.today()}\n")
        f.write(f"# Pages ranked by total issue count (missing IDs + misread dups + bad-format).\n")
        f.write(f"# Edit this file to select which pages to re-scan, then run error_redo.py.\n")
        f.write(f"# Lines starting with # are ignored.  Total pages: {len(ranked_pages)}\n")
        f.write("#\n")
        f.write(f"# {'Page':<25}  {'Total':>5}  Breakdown\n")
        for p in ranked_pages:
            parts = []
            if page_gap_count[p]: parts.append(f"{page_gap_count[p]} missing")
            if page_dup_count[p]: parts.append(f"{page_dup_count[p]} dups")
            if page_bad_count[p]: parts.append(f"{page_bad_count[p]} bad-format")
            f.write(f"  {p:<25}  # {page_issues[p]:>4} issues  [{', '.join(parts)}]\n")

    print(f"Wrote {len(ranked_pages)} pages → {REDO_PAGES_FILE.name}")
    print("Edit that file to choose which pages to re-scan, then run error_redo.py.")


if __name__ == "__main__":
    main()
