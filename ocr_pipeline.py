#!/usr/bin/env python3
"""
OCR pipeline for World Guide to Trade Associations.

Sends each scanned page image to the Gemini API and extracts structured
association data (country, name, address, focus, ID number).

Dependencies — install in your virtualenv:
    pip install google-generativeai Pillow python-dotenv

Usage:
    python ocr_pipeline.py

Progress is saved after every page, so the script can be safely interrupted
and restarted. Pages already processed are skipped automatically.
"""

import csv
import json
import os
import re
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

load_dotenv()   # Reads the .env file in the project folder (if present)


# Configuration — edit these before running 

API_KEY = os.environ["GOOGLE_API_KEY"]   # Set this in your .env file (see .env.example)

MODEL = "gemini-3.5-flash"      # Check for updates on models and pricing

IMAGE_DIR = Path(__file__).parent / "WorldGuideTrade_bookpages"

OUTPUT_DIR      = Path(__file__).parent / "ocr_output"
CHECKPOINT_FILE = OUTPUT_DIR / "progress.jsonl"   # One JSON record per page
FINAL_CSV       = OUTPUT_DIR / "associations_raw.csv"

# Data pages only — front matter ends at image00022, indexes begin at image00478.
FIRST_PAGE = "image00023.jpg"
LAST_PAGE  = "image00477.jpg"

# Seconds to wait between API calls.
# Raise this to ~4-8 if you see rate-limit errors; lower to 2 if processing is slow.
DELAY = 1

PROMPT = """\
This is a scanned page from a printed reference book called \
"World Guide to Trade Associations" (published circa 2001).
The page is set in five columns of small print.

Extract every trade association entry visible on this page.

──────────────────────────────────────
HOW THE BOOK IS STRUCTURED
──────────────────────────────────────
Country headers
  Country names (e.g. "France", "Germany", "South Africa") appear as bold
  section headings. Every entry below a country heading belongs to that country
  until the next heading appears. Headers can start mid-column.

Entry structure
  Each entry ends with a 5-digit sequential ID number (e.g. 06012, 15334)
  at the right edge of the column, typically preceded by dots (.....) or spaces.

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


def parse_response(text: str) -> list:
    """Strip markdown code fences if the model added them, then parse JSON."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text.strip())


def build_csv():
    """Read all checkpointed pages and write a single sorted CSV."""
    rows = []
    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for entry in rec["entries"]:
                entry.setdefault("country", "")
                entry.setdefault("id", "")
                entry.setdefault("name", "")
                entry.setdefault("address", "")
                entry.setdefault("focus", "")
                entry["source_page"] = rec["page"]
                rows.append(entry)

    # Sort by the 5-digit entry ID so the CSV is in book order
    rows.sort(key=lambda r: r.get("id", "99999"))

    fieldnames = ["id", "country", "name", "address", "focus", "source_page"]
    with open(FINAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows):,} entries → {FINAL_CSV}")


def main():
    if not API_KEY:
        print("ERROR: GOOGLE_API_KEY not found. Create a .env file containing:\n  GOOGLE_API_KEY=your_key_here")
        return

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL)

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
    todo = [p for p in images if p.name not in done]

    print(f"Pages found:       {len(images):>5}")
    print(f"Already processed: {len(done):>5}")
    print(f"To process:        {len(todo):>5}")
    print()

    for i, img_path in enumerate(todo, 1):
        print(f"[{i:>4}/{len(todo)}]  {img_path.name}", end="  ...  ", flush=True)

        try:
            img      = Image.open(img_path)
            response = model.generate_content([PROMPT, img])
            entries  = parse_response(response.text)
            save_page(img_path.name, entries)
            print(f"{len(entries)} entries")

        except json.JSONDecodeError as e:
            # Model returned something unparseable; save blank so we skip on retry
            print(f"JSON parse error ({e}) — saving blank")
            save_page(img_path.name, [])

        except Exception as e:
            # Don't save to checkpoint so the page is retried on the next run
            print(f"ERROR: {e}")

        if i < len(todo):
            time.sleep(DELAY)

    print()
    print("All pages processed. Building final CSV ...")
    build_csv()
    print("Done.")


if __name__ == "__main__":
    main()
