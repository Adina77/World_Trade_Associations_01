#!/usr/bin/env python3
"""
OCR pipeline for World Guide to Trade Associations.

Sends each scanned page image to the Gemini API and extracts structured
association data (country, name, address, focus, ID number).

Dependencies — install in your virtualenv:
    pip install google-genai Pillow python-dotenv

Usage:
    python ocr_pipeline.py

Progress is saved after every page, so the script can be safely interrupted
and restarted. Pages already processed are skipped automatically.
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

load_dotenv()   # Reads the .env file in the project folder (if present)


# Configuration — edit these before running 

# Set the API_KEY in the .env file (see .env.example). Put .env in .gitignore!
# Be sure to use correct key, paid tier for models that require it.
API_KEY = os.environ["GOOGLE_API_KEY"]   

# Bulk-run model (gemini-3.1-flash-lite, thinking budget 2048 tokens):
MODEL = "gemini-3.1-flash-lite"

# Re-run with better model for problem pages
# MODEL = "gemini-3.5-flash"
# MODEL = "gemini-2.5-pro"

# Primary config: force JSON output + disable safety filters.
# Needed because legitimate entries (e.g. "Armament Engineers", focus "Drugs"
# for a pharmaceutical body) trip the weapons/substances filters.
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

# Fallback config: safety off but NO forced JSON mime type.
# Used when JSON_CONFIG returns response.text=None (the JSON-mode enforcement
# and the safety refusal interact and produce a silent null response on some pages).
# The regex parser in parse_response() handles the output instead.
FALLBACK_CONFIG = types.GenerateContentConfig(safety_settings=_SAFETY_OFF)

# Thinking config: used with gemini-3.1-flash-lite for bulk processing.
# No JSON mime type — parse_response() handles stripping any markdown fences.
THINKING_CONFIG = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=2048),
    safety_settings=_SAFETY_OFF,
)

IMAGE_DIR = Path(__file__).parent / "WorldGuideTrade_bookpages"

OUTPUT_DIR           = Path(__file__).parent / "ocr_output"
CHECKPOINT_FILE      = OUTPUT_DIR / "progress.jsonl"   # One JSON record per page
FINAL_CSV            = OUTPUT_DIR / "associations_raw.csv"
FAILED_PAGES_FILE    = OUTPUT_DIR / "failed_pages.jsonl"   # Append-only log of failures
FAILED_SUMMARY_FILE  = OUTPUT_DIR / "failed_pages_summary.txt"  # Human-readable, regenerated each run
DUPLICATE_PAGES_FILE = OUTPUT_DIR / "duplicate_pages.txt"  # Written by find_duplicate_pages.py

# Data pages only — front matter ends at image00022, indexes begin at image00401.
FIRST_PAGE = "image00023.jpg"
LAST_PAGE  = "image00400.jpg"

# Seconds to wait between API calls.
# Raise this to ~4-8 if you see rate-limit errors.
# Should not be an issue for paid tier; free tier may hit rate limit.
DELAY = 1

# Retry settings for transient server errors (503, 500, 429, etc.)
MAX_RETRIES  = 4    # total attempts per page (1 original + 3 retries)
RETRY_BACKOFF = 10  # seconds before first retry; doubles each attempt (10, 20, 40 …)

PROMPT = """\
This is a scanned page from a printed reference book called \
"World Guide to Trade Associations" (published 2002).
The page is set in five columns of small print.

Extract every trade association entry visible on this page.

──────────────────────────────────────
HOW THE BOOK IS STRUCTURED
──────────────────────────────────────
Country identification — TWO sources, use both
  1. Running page header: Every page has a header line at the very top showing
     the current country, in a format like:
         "France: Syndicat   05273 — 05460"
         "05461 —   France: Syndicat"
     The country name is ONLY the text before the colon. The word(s) after the
     colon are the beginning of an association name — do NOT include the colon
     or anything after it in the country field.
     Use this country name as the default for all entries on the page.

  2. Section headers in the body: Country names also appear as bold headings
     within the column text when a new country section begins mid-page.
     When you see one, switch the country for all subsequent entries.

  IMPORTANT: Each page is processed independently with no knowledge of prior
  pages. Always read the running page header at the top to establish the country,
  even if no section header appears in the body text.

Entry structure
  Each entry ends with a 5-digit sequential ID number (e.g. 06012, 15334)
  at the right edge of the column, typically preceded by dots (.....) or spaces.
  This is a unique identification number that is assigned to a SINGLE association.
  It must fall within the range that is shown in the header and must not be assinged
  to more than one association.

  A typical entry looks like:
      <Association Name>              ← one to three lines, sometimes bilingual
      <Street address, City>
      - T: (phone); Fax: (fax)       ← optional: phone, fax
      - Founded: YYYY; Members: N    ← optional: year founded, member count
      - Focus: <industry/sector>     ← always present; the sector classification
      Periodicals <pub name> (freq)  ← OPTIONAL: some entries list publications here
      ........ 12345                 ← 5-digit entry ID

  IMPORTANT: Some entries include a "Periodicals" line between the Focus field
  and the ID number, listing journals or newsletters the association publishes
  (e.g. "Periodicals Annual Report (yearly) - Newsletter (monthly)").
  This Periodicals line is NOT part of the Focus field. Stop the focus text
  before any "Periodicals" content. Do not include publication names in focus.

Association name changes: 
Occasionally an entry is just an indication that an 
association has changed its name, and the actual entry is under the new name 
elsewhere. This can be detected because the entry consists of two association names
separated by a right-pointing arrow. There is no ID number for these cases and 
they should be ignored. They must not be assigned an ID number and must not be
extracted into the output file.

Split entries — what to do when an association spans two pages:
  IDs are strictly sequential. The page header shows the range ONLY for
  entries that both start AND end on this page.

  Case 1 — Incomplete LAST entry (cut off at the bottom of the page):
    You may see an association name (and perhaps part of an address) near
    the bottom of the last column, but the entry is cut off — no Focus
    line and no ID number appear before the page ends.
    → DO include it as an association. Assign the next integer after the last complete
      entry's ID on this page (it will not appear in the header range).
    → Leave focus as "" if the Focus line is not visible on this page.

  Case 2 — Continuation FIRST entry (cut off at the top of the page):
    You may see what looks like a mid-entry continuation at the top of
    the first column — an address line or a Focus line with no association
    name above it.  The name started on the previous page.
    → DO include it. Use the ID that appears at the end of this entry
      (it is the first ID in this page's header range).
    → Leave name as "" if the association name is truly not visible on
      this page — it will be merged with the previous page's entry later.

──────────────────────────────────────
FIELDS TO EXTRACT
──────────────────────────────────────
For each entry return these five fields:

  country  — name of the country from the nearest section header above this entry
  id       — the 5-digit number at the end of the entry (string; keep leading zeros)
  name     — the full association name
  address  — everything between the name and the Focus field
              (street, city, phone, fax, president, founded date, member count, etc.)
  focus    — the text that follows "Focus:" — the industry or sector description only;
              stop before any "Periodicals" line that may follow

──────────────────────────────────────
OUTPUT FORMAT
──────────────────────────────────────
Return ONLY a raw JSON array. No explanation, no markdown, no code fences.

Example:
[
  {
    "country": "Germany",
    "id": "06012",
    "name": "Verband der Deutschen Lederwarenindustrie e.V.",
    "address": "Postfach 1207, 63002 Offenbach - T: (069) 800985; Fax: 800986",
    "focus": "Leather Goods"
  },
  {
    "country": "Germany",
    "id": "06013",
    "name": "Zentralverband des Deutschen Bäckerhandwerks",
    "address": "Neustädtische Kirchstr. 7A, 10117 Berlin - T: (030) 206 64 50",
    "focus": "Bakery"
  }
]

If this page is an index page (alphabetical index, subject index, table of
contents, or any non-listing page), return an empty array: []
"""


def load_done() -> set:
    """Return the set of image filenames already written to the checkpoint file."""
    if not CHECKPOINT_FILE.exists():
        return set()
    done = set()
    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["page"])
    return done


def save_page(page_name: str, entries: list):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"page": page_name, "entries": entries}, ensure_ascii=False) + "\n")


def log_failure(page_name: str, error_type: str, error_msg: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {"page": page_name, "error_type": error_type, "error": error_msg}
    with open(FAILED_PAGES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_transient_error(e: Exception) -> bool:
    """Return True for server-side errors that are worth retrying."""
    msg = str(e).lower()
    return any(token in msg for token in ("503", "500", "502", "504", "429",
                                          "unavailable", "overloaded", "quota",
                                          "safety_blocked"))


def parse_response(text: str) -> list:
    """Strip markdown code fences if the model added them, then parse JSON."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text.strip())


def load_ignored_pages() -> set[str]:
    """Return the set of image filenames listed in duplicate_pages.txt (if it exists)."""
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


def valid_pages() -> set[str]:
    """Return data-range pages, excluding any known duplicate scans."""
    all_images = sorted(p.name for p in IMAGE_DIR.glob("*.jpg"))
    try:
        start = all_images.index(FIRST_PAGE)
        end   = all_images.index(LAST_PAGE) + 1
    except ValueError:
        return set()
    in_range = set(all_images[start:end])
    ignored  = load_ignored_pages()
    if ignored:
        in_range -= ignored
    return in_range


def build_csv():
    """Read checkpointed pages within the data range and write a single sorted CSV."""
    in_range = valid_pages()
    rows = []
    skipped_pages = set()
    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["page"] not in in_range:
                if rec["entries"]:   # only warn if the out-of-range page had entries
                    skipped_pages.add(rec["page"])
                continue
            for entry in rec["entries"]:
                entry.setdefault("country", "")
                entry.setdefault("id", "")
                entry.setdefault("name", "")
                entry.setdefault("address", "")
                entry.setdefault("focus", "")
                entry["source_page"] = rec["page"]
                # Sanitize embedded newlines so every CSV parser handles the file correctly
                for field in ("country", "id", "name", "address", "focus"):
                    entry[field] = entry[field].replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
                rows.append(entry)

    if skipped_pages:
        print(f"  Skipped {len(skipped_pages)} out-of-range page(s) with entries: "
              f"{', '.join(sorted(skipped_pages))}")

    # Sort by the 5-digit entry ID so the CSV is in book order
    rows.sort(key=lambda r: r.get("id", "99999"))

    fieldnames = ["id", "country", "name", "address", "focus", "source_page"]
    with open(FINAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows):,} entries → {FINAL_CSV}")


def build_failed_summary():
    """Write a human-readable list of pages that failed and have not yet been processed successfully."""
    done = load_done()

    if not FAILED_PAGES_FILE.exists():
        return

    # Read all logged failures; keep only the most-recent record per page.
    latest = {}
    with open(FAILED_PAGES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                latest[rec["page"]] = rec

    # Pages that have since been processed successfully are no longer "failed".
    still_failed = {p: v for p, v in latest.items() if p not in done}

    with open(FAILED_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(f"Failed pages — {len(still_failed)} not yet successfully processed\n")
        f.write("=" * 70 + "\n\n")
        for page in sorted(still_failed):
            info = still_failed[page]
            f.write(f"{page}  [{info['error_type']}]  {info['error']}\n")

    if still_failed:
        print(f"  {len(still_failed)} page(s) failed — see {FAILED_SUMMARY_FILE}")
    else:
        print("  No outstanding failed pages.")


def main():
    parser = argparse.ArgumentParser(description="OCR pipeline for World Guide to Trade Associations.")
    parser.add_argument(
        "--pages", nargs="+", metavar="PAGE",
        help="Process only these specific image filenames (used by error_redo.py)"
    )
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: GOOGLE_API_KEY not found. Create a .env file containing:\n  GOOGLE_API_KEY=your_key_here")
        return

    client = genai.Client(api_key=API_KEY)

    all_images = sorted(IMAGE_DIR.glob("*.jpg"))
    if not all_images:
        print(f"No JPG files found in {IMAGE_DIR}")
        return

    # Trim to data-page range only (no front matter or back-matter indexes)
    names = [p.name for p in all_images]
    try:
        start = names.index(FIRST_PAGE)
        end   = names.index(LAST_PAGE) + 1
    except ValueError as e:
        print(f"ERROR: boundary page not found — {e}")
        return
    images = all_images[start:end]

    done = load_done()

    if args.pages:
        # Targeted mode: process only the explicitly requested pages
        requested = set(args.pages)
        todo = [p for p in images if p.name in requested]
        not_found = requested - {p.name for p in todo}
        if not_found:
            print(f"WARNING: {len(not_found)} requested page(s) not found in image directory: "
                  f"{', '.join(sorted(not_found))}")
    else:
        todo = [p for p in images if p.name not in done]

    print(f"Pages found:       {len(images):>5}")
    print(f"Already processed: {len(done):>5}")
    print(f"To process:        {len(todo):>5}")
    print()

    for i, img_path in enumerate(todo, 1):
        print(f"[{i:>4}/{len(todo)}]  {img_path.name}", end="  ...  ", flush=True)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                img      = Image.open(img_path)
                response = client.models.generate_content(model=MODEL, contents=[PROMPT, img], config=THINKING_CONFIG)
                # response = client.models.generate_content(model=MODEL, contents=[PROMPT, img], config=JSON_CONFIG)
                if response.text is None:
                    # Retry once without thinking config in case of a silent null.
                    print("THINKING_CONFIG returned None — retrying with FALLBACK_CONFIG ...", end="  ", flush=True)
                    response = client.models.generate_content(model=MODEL, contents=[PROMPT, img], config=FALLBACK_CONFIG)
                if response.text is None:
                    block_reason = "unknown"
                    try:
                        block_reason = str(response.prompt_feedback)
                    except Exception:
                        pass
                    raise ValueError(f"safety_blocked: response.text is None after fallback — {block_reason}")
                entries  = parse_response(response.text)
                save_page(img_path.name, entries)
                print(f"{len(entries)} entries")
                break

            except json.JSONDecodeError as e:
                # Model returned unparseable JSON.  Do NOT save to checkpoint so
                # the page will be retried on the next run (possibly with a better model).
                print(f"JSON parse error — logged to failed_pages")
                log_failure(img_path.name, "json_parse_error", str(e))
                break

            except Exception as e:
                if is_transient_error(e) and attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                    print(f"\n       server error (attempt {attempt}/{MAX_RETRIES}): {e}")
                    print(f"       retrying in {wait}s ...", flush=True)
                    time.sleep(wait)
                    print(f"[{i:>4}/{len(todo)}]  {img_path.name}", end="  ...  ", flush=True)
                else:
                    # Non-transient error or retries exhausted.  Not saved to checkpoint
                    # so the page will be retried on the next run.
                    print(f"ERROR: {e}")
                    log_failure(img_path.name, "api_error", str(e))
                    break

        if i < len(todo):
            time.sleep(DELAY)

    print()
    if args.pages:
        remaining = valid_pages() - load_done()
        if remaining:
            print(f"WARNING: {len(remaining)} page(s) still unprocessed in the data range — "
                  f"skipping CSV rebuild to avoid overwriting with incomplete data.")
            print("Run ocr_pipeline.py without --pages to process all remaining pages first.")
            print("Done.")
            return
    print("Building final CSV ...")
    build_csv()
    build_failed_summary()
    print("Done.")


if __name__ == "__main__":
    main()
