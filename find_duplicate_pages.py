#!/usr/bin/env python3
"""
find_duplicate_pages.py

Scan a set of image files looking ONLY at the running page header
(book page number + association ID range) to identify duplicate scans —
two image files that cover the same printed book page.

By default, scans the pages listed in ocr_output/pages_to_redo.txt
(both copies of any duplicate page should appear there, because the
inconsistent OCR results show up as misread-ID duplicates in the error check).

Pass --all to scan every image in the FIRST_PAGE-LAST_PAGE range instead
(~378 API calls; only needed if not all duplicates are on the redo list).

Output:  ocr_output/duplicate_pages.txt
  Lists the later-numbered copy of each duplicate pair — these are the files
  to IGNORE in the pipeline and cleaned CSV.
  Review the file before using it: faint or cut-off headers may be misread.

Usage
─────
  python find_duplicate_pages.py           # scan pages_to_redo.txt  (default)
  python find_duplicate_pages.py --all     # scan all ~378 data pages
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import google.genai as genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

BASE_DIR   = Path(__file__).parent
IMAGE_DIR  = BASE_DIR / "WorldGuideTrade_bookpages"
OUTPUT_DIR = BASE_DIR / "ocr_output"
REDO_FILE  = OUTPUT_DIR / "pages_to_redo.txt"
OUT_FILE   = OUTPUT_DIR / "duplicate_pages.txt"

FIRST_PAGE = "image00023.jpg"
LAST_PAGE  = "image00400.jpg"

# Header-only extraction is a trivial task — cheapest model is fine.
MODEL = "gemini-3.1-flash-lite"

# Rate between API calls (seconds).  Flash-lite has a generous free quota;
# raise to 4–5 if you see 429 rate-limit errors.
CALL_DELAY = 5

# Height (pixels) of the header strip cropped from each image before sending.
# The top 250 px captures the running header on a ~2800×3900 scan.
HEADER_HEIGHT_PX = 250

_SAFETY_OFF = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",       threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]
JSON_CONFIG = types.GenerateContentConfig(
    response_mime_type="application/json",
    safety_settings=_SAFETY_OFF,
)

HEADER_PROMPT = """\
Look ONLY at the very top of this scanned book page — the single running
header line.  Ignore everything below it.

The header typically looks like one of these formats:
  "Japan: Kansai    12650 — 12700"        (country, then ID range)
  "12650 —    Japan: Kansai    12700"     (ID range split around country)

Extract:
  book_page — the small printed page number (2–3 digits) that appears in
              the outer top corner of the page (top-left or top-right).
              This is the physical book page number, NOT an association ID.
  id_start  — the lower 5-digit association ID shown in the header range.
  id_end    — the higher 5-digit association ID shown in the header range.

Return ONLY valid JSON in this exact format (use null for any value you
cannot find with confidence):
{"book_page": 190, "id_start": 12650, "id_end": 12700}
"""


# ── Page list helpers ─────────────────────────────────────────────────────────

def load_problem_pages() -> list[str]:
    """Read image filenames from pages_to_redo.txt (skip comment lines)."""
    if not REDO_FILE.exists():
        sys.exit(
            f"ERROR: {REDO_FILE.name} not found.\n"
            "Run ocr_error_check.py first to generate it, or use --all."
        )
    pages = []
    with open(REDO_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0]
            if token.endswith(".jpg"):
                pages.append(token)
    return pages


def load_all_pages() -> list[str]:
    """Return all image filenames in the FIRST_PAGE – LAST_PAGE range."""
    all_images = sorted(p.name for p in IMAGE_DIR.glob("*.jpg"))
    try:
        start = all_images.index(FIRST_PAGE)
        end   = all_images.index(LAST_PAGE) + 1
    except ValueError:
        sys.exit(f"ERROR: Could not locate {FIRST_PAGE} or {LAST_PAGE} in {IMAGE_DIR}")
    return all_images[start:end]


# ── Header extraction ─────────────────────────────────────────────────────────

def extract_header(client, image_path: Path) -> dict:
    """
    Crop the header strip from one image and ask the model for
    book_page, id_start, id_end.  Returns a dict with those keys
    (int or None) and optionally an 'error' key.
    """
    img = Image.open(image_path)
    crop_h = min(HEADER_HEIGHT_PX, img.height)
    header_strip = img.crop((0, 0, img.width, crop_h))

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[HEADER_PROMPT, header_strip],
            config=JSON_CONFIG,
        )
        if response.text is None:
            return {"book_page": None, "id_start": None, "id_end": None,
                    "error": "API returned None (safety block?)"}
        data = json.loads(response.text)
        return {
            "book_page": data.get("book_page"),
            "id_start":  data.get("id_start"),
            "id_end":    data.get("id_end"),
        }
    except Exception as e:
        return {"book_page": None, "id_start": None, "id_end": None,
                "error": str(e)}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect duplicate-scan pages by comparing running page headers."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scan ALL data pages instead of just pages_to_redo.txt"
    )
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("ERROR: GOOGLE_API_KEY not found. Add it to the .env file.")

    pages = load_all_pages() if args.all else load_problem_pages()
    if not pages:
        sys.exit("No pages found to scan.")

    client = genai.Client(api_key=api_key)

    mode_label = "all data pages" if args.all else "pages_to_redo.txt"
    print(f"Scanning {len(pages)} page header(s)  [{mode_label}]")
    print(f"Model: {MODEL}   Header strip: top {HEADER_HEIGHT_PX} px")
    print()

    # ── Scan ─────────────────────────────────────────────────────────────────
    results: dict[str, dict] = {}

    for i, page_name in enumerate(pages, start=1):
        img_path = IMAGE_DIR / page_name
        if not img_path.exists():
            print(f"  [{i:3d}/{len(pages)}]  {page_name}  — file not found, skipping")
            continue

        print(f"  [{i:3d}/{len(pages)}]  {page_name} ...", end="  ", flush=True)
        info = extract_header(client, img_path)
        results[page_name] = info

        if "error" in info:
            print(f"ERROR: {info['error']}")
        else:
            bp_str = str(info["book_page"]) if info["book_page"] is not None else "?"
            id_str = (f"{info['id_start']}–{info['id_end']}"
                      if info["id_start"] and info["id_end"] else "?–?")
            print(f"book_page={bp_str:<5}  ids={id_str}")

        if i < len(pages):
            time.sleep(CALL_DELAY)

    # ── Duplicate detection ───────────────────────────────────────────────────
    # Use BOTH book_page AND (id_start, id_end) as independent duplicate signals.
    # Two images are duplicates if they share book_page OR share the same ID range.
    # A union-find merges groups from either signal, so a missed page number won't
    # cause a true duplicate to be overlooked (and vice versa).

    scanned = list(results.keys())

    # Union-Find ──────────────────────────────────────────────────────────────
    parent = {p: p for p in scanned}

    def uf_find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    def uf_union(x: str, y: str):
        parent[uf_find(y)] = uf_find(x)

    # Signal 1 — same book_page
    by_book_page: dict[int, list[str]] = defaultdict(list)
    for page_name, info in results.items():
        bp = info.get("book_page")
        if bp is not None:
            by_book_page[bp].append(page_name)

    for imgs in by_book_page.values():
        for img in imgs[1:]:
            uf_union(imgs[0], img)

    # Signal 2 — same (id_start, id_end) pair (both must be non-null)
    by_id_range: dict[tuple, list[str]] = defaultdict(list)
    for page_name, info in results.items():
        s, e = info.get("id_start"), info.get("id_end")
        if s is not None and e is not None:
            by_id_range[(s, e)].append(page_name)

    for imgs in by_id_range.values():
        for img in imgs[1:]:
            uf_union(imgs[0], img)

    # Collect connected components
    components: dict[str, list[str]] = defaultdict(list)
    for p in scanned:
        components[uf_find(p)].append(p)

    # Only groups of 2+ are duplicates; sort each so the first = lowest filename = keeper
    dup_groups: list[list[str]] = [
        sorted(grp) for grp in components.values() if len(grp) > 1
    ]

    # For each pair in a group, record which signal(s) triggered the match
    def match_signals(a: str, b: str) -> str:
        ia, ib = results[a], results[b]
        sigs = []
        if (ia.get("book_page") is not None and
                ia.get("book_page") == ib.get("book_page")):
            sigs.append(f"book_page={ia['book_page']}")
        if (ia.get("id_start") is not None and ia.get("id_end") is not None and
                ia.get("id_start") == ib.get("id_start") and
                ia.get("id_end")   == ib.get("id_end")):
            sigs.append(f"IDs {ia['id_start']}–{ia['id_end']}")
        return " + ".join(sigs) if sigs else "transitively linked"

    # A page is "unclear" only if it contributed to NO signal at all
    unclear = [
        p for p, info in results.items()
        if info.get("book_page") is None
        and (info.get("id_start") is None or info.get("id_end") is None)
        and "error" not in info
    ]
    errors = [p for p, info in results.items() if "error" in info]

    pages_to_ignore: list[str] = [img for grp in dup_groups for img in grp[1:]]

    # ── Report ────────────────────────────────────────────────────────────────
    SEP  = "─" * 70
    SEP2 = "═" * 70

    print()
    print(SEP2)
    print("  DUPLICATE PAGE DETECTION REPORT")
    print(SEP2)
    print(f"  Pages scanned:           {len(results):>4}")
    print(f"  Duplicate groups found:  {len(dup_groups):>4}")
    print(f"  Pages to ignore:         {len(pages_to_ignore):>4}")
    print(f"  Unclear headers:         {len(unclear):>4}  (no usable signal detected)")
    print(f"  Errors:                  {len(errors):>4}")
    print(SEP2)

    if dup_groups:
        print()
        print("DUPLICATE GROUPS  (keep first, ignore rest)")
        print("  Confidence — both signals: page number AND ID range matched")
        print("               one signal:  only one matched (review these)")
        print(SEP)
        for grp in dup_groups:
            keeper = grp[0]
            info_k = results[keeper]
            bp_str = str(info_k.get("book_page")) if info_k.get("book_page") is not None else "?"
            id_str = (f"{info_k.get('id_start')}–{info_k.get('id_end')}"
                      if info_k.get("id_start") and info_k.get("id_end") else "?")
            print(f"  Book page {bp_str:<5}  IDs {id_str}")
            print(f"    KEEP:   {keeper}")
            for dup in grp[1:]:
                sigs = match_signals(keeper, dup)
                confidence = "both signals" if "+" in sigs else "one signal — review"
                print(f"    IGNORE: {dup}  [{confidence}: {sigs}]")
        print()

    if unclear:
        print(f"UNCLEAR HEADERS  (neither book_page nor ID range detected — review manually)")
        print(SEP)
        for p in unclear:
            print(f"  {p}")
        print()

    if errors:
        print(f"ERRORS  ({len(errors)} page(s) could not be scanned)")
        print(SEP)
        for p in errors:
            print(f"  {p}:  {results[p]['error']}")
        print()

    # ── Write duplicate_pages.txt ─────────────────────────────────────────────
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# duplicate_pages.txt — generated by find_duplicate_pages.py on {date.today()}\n")
        f.write( "# Image files that are duplicate scans of a book page already captured\n")
        f.write( "# by another image file.  These pages are excluded from the pipeline\n")
        f.write( "# and from the cleaned CSV.\n")
        f.write( "#\n")
        f.write( "# Detection used two independent signals: book_page number AND ID range.\n")
        f.write( "# 'both signals' = high confidence.  'one signal' = review recommended.\n")
        f.write( "# To add pages manually: one filename per line; # lines are comments.\n")
        f.write( "#\n")
        if dup_groups:
            for grp in dup_groups:
                keeper = grp[0]
                for dup in grp[1:]:
                    sigs = match_signals(keeper, dup)
                    confidence = "both signals" if "+" in sigs else "one signal — review"
                    f.write(f"  {dup}  # duplicate of {keeper}  [{confidence}: {sigs}]\n")
        else:
            f.write("# No duplicates detected in the scanned pages.\n")
        if unclear:
            f.write("#\n")
            f.write("# Pages where no header signal was detected — review manually:\n")
            for p in unclear:
                info = results[p]
                parts = []
                if info.get("book_page") is not None:
                    parts.append(f"book_page={info['book_page']}")
                if info.get("id_start") is not None:
                    parts.append(f"id_start={info['id_start']}")
                if info.get("id_end") is not None:
                    parts.append(f"id_end={info['id_end']}")
                detail = ", ".join(parts) if parts else "nothing detected"
                f.write(f"# ??  {p}  ({detail})\n")

    print(f"Wrote {OUT_FILE.name}  ({len(pages_to_ignore)} page(s) to ignore)")
    print()
    if pages_to_ignore:
        print("Next steps:")
        print("  1. Review duplicate_pages.txt and correct any misidentifications.")
        print("  2. python ocr_pipeline.py   — rebuilds associations_raw.csv (skips ignored pages)")
        print("  3. python ocr_cleanup.py    — rebuilds associations_cleaned.csv")
        print("  4. python ocr_error_check.py")
    else:
        print("No duplicates detected among the scanned pages.")
        if not args.all:
            print("If issues persist, try:  python find_duplicate_pages.py --all")


if __name__ == "__main__":
    main()
