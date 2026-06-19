#!/usr/bin/env python3
"""
OCR pipeline for the Focus Term Abbreviation pages (image00013-image00016).

Sends each page to the Gemini API and extracts abbreviation → full-term pairs.
Progress is saved to a .jsonl checkpoint after every page, so the script can
be interrupted and restarted safely. Writes the final CSV when all pages are done.

Dependencies:  pip install google-genai Pillow python-dotenv
Usage:
    python focus_abbrev_ocr.py              # gemini-2.5-flash (default)
    python focus_abbrev_ocr.py --lite       # gemini-3.1-flash-lite + thinking (2048 tokens)
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import google.genai as genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

API_KEY = os.environ["GOOGLE_API_KEY"]

# Default model — good quality, has free tier; higher rate limit on paid tier.
MODEL_FLASH = "gemini-2.5-flash"

# --lite model — 3rd-gen vision adjusts resolution to content (more input tokens
# but better accuracy on small-font pages); thinking budget raises quality above
# the default "efficient" mode that cuts corners on dense text.
# Higher free-tier rate limit than 2.5-flash.
MODEL_LITE   = "gemini-3.1-flash-lite"
THINKING_BUDGET = 2048

_SAFETY_OFF = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",       threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

# Used with MODEL_FLASH: forces JSON output directly.
JSON_CONFIG = types.GenerateContentConfig(
    response_mime_type="application/json",
    safety_settings=_SAFETY_OFF,
)

# Used with MODEL_LITE: thinking enabled, no forced JSON mime type.
# parse_response() strips any markdown fences the model adds.
THINKING_CONFIG = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
    safety_settings=_SAFETY_OFF,
)

# Fallback when JSON_CONFIG returns response.text = None
FALLBACK_CONFIG = types.GenerateContentConfig(safety_settings=_SAFETY_OFF)

IMAGE_DIR  = Path(__file__).parent / "WorldGuideTrade_bookpages"
OUTPUT_DIR = Path(__file__).parent / "ocr_output"
CHECKPOINT = OUTPUT_DIR / "focus_abbrev_progress.jsonl"
FINAL_CSV  = OUTPUT_DIR / "focus_abbreviations.csv"

# The four pages that carry the abbreviation list, in page order.
ABBREV_PAGES = [
    "image00013.jpg",
    "image00014.jpg",
    "image00015.jpg",
    "image00016.jpg",
]

DELAY         = 1
MAX_RETRIES   = 4
RETRY_BACKOFF = 10   # seconds; doubles each attempt (10, 20, 40 …)

PROMPT = """\
This is a scanned page from the front matter of "World Guide to Trade Associations".
The page is titled "Abbreviations for Areas of Specialization" and contains a lookup
table mapping abbreviated focus terms to their full expanded forms.

PAGE LAYOUT
───────────
The page has FOUR columns arranged as two independent side-by-side pairs:

  Left pair:   column 1 (abbreviation)  |  column 2 (full term)
  Right pair:  column 3 (abbreviation)  |  column 4 (full term)

Each row in a pair is a single abbreviation mapped to its full expanded term.
The rows of the left pair are completely independent of the rows of the right
pair. The two pairs simply sit next to each other on the page.

WHAT TO IGNORE
──────────────
• The page header: either the bold title "Abbreviations for Areas of
  Specialization" (on the first page) or the running header line at the very
  top with the horizontal rule beneath it (on subsequent pages). Skip entirely.
• The roman numeral page number at the bottom of the page. Skip it.

EXTRACTION RULES
────────────────
1. Process the LEFT pair first (columns 1 and 2), reading top to bottom.
2. Then process the RIGHT pair (columns 3 and 4), reading top to bottom.
3. Keep each abbreviation paired with its matching full term from the same row.
4. Do NOT shuffle or interleave rows between the left pair and the right pair.
5. Preserve the abbreviation exactly as printed (mixed case, ampersands, spaces).
6. Preserve the full term exactly as printed, including any line-continuation
   that wraps onto the next line within the same cell.

OUTPUT FORMAT
─────────────
Return ONLY a raw JSON array — no explanation, no markdown, no code fences.
Each element is an object with exactly two keys:

  "abbreviation" : text from column 1 or column 3
  "full_term"    : text from column 2 or column 4

Example (illustrating the first few rows from the first page):
[
  {"abbreviation": "Abras", "full_term": "Abrasives"},
  {"abbreviation": "Account", "full_term": "Accounting"},
  {"abbreviation": "Accum", "full_term": "Accumulators"},
  {"abbreviation": "Acids", "full_term": "Acids"}
]
"""


def load_done() -> set:
    """Return the set of page filenames already written to the checkpoint."""
    if not CHECKPOINT.exists():
        return set()
    done = set()
    with open(CHECKPOINT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["page"])
    return done


def save_page(page_name: str, pairs: list):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "a", encoding="utf-8") as f:
        f.write(json.dumps({"page": page_name, "pairs": pairs}, ensure_ascii=False) + "\n")


def is_transient_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(t in msg for t in ("503", "500", "502", "504", "429",
                                   "unavailable", "overloaded", "quota"))


def parse_response(text: str) -> list:
    """Strip any markdown fences the model may have added, then parse JSON."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text.strip(), flags=re.MULTILINE)
    return json.loads(text.strip())


def build_csv():
    """Merge all checkpointed pages in book order and write the CSV."""
    page_order   = {name: i for i, name in enumerate(ABBREV_PAGES)}
    rows_by_page = {}

    with open(CHECKPOINT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["page"] in page_order:
                rows_by_page[rec["page"]] = rec["pairs"]

    all_pairs = []
    for page in ABBREV_PAGES:
        for pair in rows_by_page.get(page, []):
            all_pairs.append({
                "abbreviation": pair.get("abbreviation", ""),
                "full_term":    pair.get("full_term",    ""),
                "source_page":  page,
            })

    fieldnames = ["abbreviation", "full_term", "source_page"]
    with open(FINAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_pairs)

    missing = [p for p in ABBREV_PAGES if p not in rows_by_page]
    if missing:
        print(f"  WARNING — pages not yet processed: {', '.join(missing)}")
    print(f"Saved {len(all_pairs)} abbreviation pairs → {FINAL_CSV}")


def main():
    parser = argparse.ArgumentParser(
        description="OCR the Focus Term Abbreviation pages (image00013-image00016)."
    )
    parser.add_argument(
        "--lite", action="store_true",
        help=(
            f"Use {MODEL_LITE} with thinking_budget={THINKING_BUDGET} instead of "
            f"{MODEL_FLASH}. Better for dense small-font pages; higher free-tier rate limit."
        ),
    )
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: GOOGLE_API_KEY not set. Add it to the .env file.")
        return

    if args.lite:
        model  = MODEL_LITE
        config = THINKING_CONFIG
        print(f"Model: {model}  (thinking_budget={THINKING_BUDGET})")
    else:
        model  = MODEL_FLASH
        config = JSON_CONFIG
        print(f"Model: {model}")

    client = genai.Client(api_key=API_KEY)
    done   = load_done()
    todo   = [p for p in ABBREV_PAGES if p not in done]

    print(f"Pages total:       {len(ABBREV_PAGES)}")
    print(f"Already processed: {len(done)}")
    print(f"To process:        {len(todo)}")
    print()

    for i, page_name in enumerate(todo, 1):
        img_path = IMAGE_DIR / page_name
        if not img_path.exists():
            print(f"  [{i}/{len(todo)}]  {page_name}  — FILE NOT FOUND, skipping")
            continue

        print(f"  [{i}/{len(todo)}]  {page_name}", end="  ...  ", flush=True)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                img      = Image.open(img_path)
                response = client.models.generate_content(
                    model=model, contents=[PROMPT, img], config=config
                )
                if response.text is None:
                    # JSON mime type and thinking can silently conflict; try without.
                    print("config returned None — retrying with FALLBACK_CONFIG ...",
                          end="  ", flush=True)
                    response = client.models.generate_content(
                        model=model, contents=[PROMPT, img], config=FALLBACK_CONFIG
                    )
                if response.text is None:
                    raise ValueError("response.text is None after fallback")

                pairs = parse_response(response.text)
                save_page(page_name, pairs)
                print(f"{len(pairs)} pairs")
                break

            except json.JSONDecodeError as e:
                print(f"JSON parse error — {e}")
                break

            except Exception as e:
                if is_transient_error(e) and attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                    print(f"\n    server error (attempt {attempt}/{MAX_RETRIES}): {e}")
                    print(f"    retrying in {wait}s ...", flush=True)
                    time.sleep(wait)
                    print(f"  [{i}/{len(todo)}]  {page_name}", end="  ...  ", flush=True)
                else:
                    print(f"ERROR: {e}")
                    break

        if i < len(todo):
            time.sleep(DELAY)

    print()
    print("Building CSV ...")
    build_csv()
    print("Done.")


if __name__ == "__main__":
    main()
