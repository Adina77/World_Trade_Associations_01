#!/usr/bin/env python3
"""
Test reasoning configurations on image00050.jpg and compare against the 3.5 Flash baseline.

Three configurations tested:
  1. gemini-2.5-flash      — capped thinking budget (1024 tokens): avoids 10-min waits
  2. gemini-3.1-flash-lite — high thinking: more reasoning to fix ID-reading errors
  3. gemini-3.1-flash-lite — medium thinking: balance of speed and accuracy

Baseline: the 3.5 Flash result for image00050.jpg already in progress.jsonl.
Full outputs saved to ocr_output/model_comparison/ for diffing.
"""

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

BASE_DIR   = Path(__file__).parent
IMAGE_PATH = BASE_DIR / "WorldGuideTrade_bookpages" / "image00050.jpg"
CHECKPOINT = BASE_DIR / "ocr_output" / "progress.jsonl"
OUT_DIR    = BASE_DIR / "ocr_output" / "model_comparison"

TEST_PAGE = "image00050.jpg"

CONFIGS = [
    # {
    #     "name":   "gemini-2.5-flash",
    #     "desc":   "capped thinking (1024 tokens)",
    #     "slug":   "flash25-capped1024",
    #     "config": types.GenerateContentConfig(
    #         thinking_config=types.ThinkingConfig(thinking_budget=1024)
    #     ),
    # },
    # {
    #     "name":   "gemini-3.1-flash-lite",
    #     "desc":   "high thinking (8192 tokens)",
    #     "slug":   "flash31lite-high",
    #     "config": types.GenerateContentConfig(
    #         thinking_config=types.ThinkingConfig(thinking_budget=8192)
    #     ),
    # },
    {
        "name":   "gemini-3.1-flash-lite",
        "desc":   "medium thinking (2048 tokens)",
        "slug":   "flash31lite-medium",
        "config": types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=2048)
        ),
    },
]

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
     The country name is the word(s) before the colon. Use this as the default
     country for all entries on the page.

  2. Section headers in the body: Country names also appear as bold headings
     within the column text when a new country section begins mid-page.
     When you see one, switch the country for all subsequent entries.

  IMPORTANT: Each page is processed independently with no knowledge of prior
  pages. Always read the running page header at the top to establish the country,
  even if no section header appears in the body text.

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


def parse_response(text: str) -> list:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text.strip())


def load_baseline(page_name: str) -> list | None:
    """Return the existing entries for this page from progress.jsonl, or None."""
    if not CHECKPOINT.exists():
        return None
    with open(CHECKPOINT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec["page"] == page_name:
                    return rec["entries"]
    return None


def country_counts(entries: list) -> dict:
    counts: dict = {}
    for e in entries:
        c = e.get("country", "unknown")
        counts[c] = counts.get(c, 0) + 1
    return counts


def print_sample(entries: list, n: int = 3):
    for entry in entries[:n]:
        print(f"  [{entry.get('id')}] ({entry.get('country', '?')}) {entry.get('name', '')[:52]}")
        print(f"       focus: {entry.get('focus', '')}")
    if len(entries) > n:
        last = entries[-1]
        print(f"  ...")
        print(f"  [{last.get('id')}] ({last.get('country', '?')}) {last.get('name', '')[:52]}  (last entry)")


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found. Create a .env file with GOOGLE_API_KEY=...")
        return

    if not IMAGE_PATH.exists():
        print(f"ERROR: Image not found at {IMAGE_PATH}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Baseline ────────────────────────────────────────────────────────────
    baseline = load_baseline(TEST_PAGE)
    if baseline is None:
        print(f"WARNING: No baseline found for {TEST_PAGE} in progress.jsonl")
    else:
        print(f"\n{'='*62}")
        print(f"  BASELINE: gemini-3.5-flash  (progress.jsonl line 28)")
        print(f"  {len(baseline)} entries total")
        print(f"{'='*62}")
        for country, count in country_counts(baseline).items():
            print(f"  {country}: {count} entries")
        print()
        print_sample(baseline)
        (OUT_DIR / "baseline_flash35.json").write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── Configurations ───────────────────────────────────────────────────────
    client = genai.Client(api_key=api_key)
    img = Image.open(IMAGE_PATH)

    all_results: dict = {}

    for cfg in CONFIGS:
        label = f"{cfg['name']} ({cfg['desc']})"
        print(f"\n{'='*62}")
        print(f"  Running {label} ...")
        print(f"{'='*62}")
        start = time.time()
        try:
            response = client.models.generate_content(
                model=cfg["name"],
                contents=[PROMPT, img],
                config=cfg["config"],
            )
            elapsed = time.time() - start
            entries = parse_response(response.text)
            all_results[cfg["slug"]] = (label, entries)

            out_file = OUT_DIR / f"{cfg['slug']}.json"
            out_file.write_text(
                json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            print(f"  {len(entries)} entries  [{elapsed:.1f}s]  → saved to {out_file.name}")
            for country, count in country_counts(entries).items():
                print(f"  {country}: {count} entries")
            print()
            print_sample(entries)

        except json.JSONDecodeError as e:
            elapsed = time.time() - start
            print(f"  JSON parse error after {elapsed:.1f}s: {e}")
            raw_file = OUT_DIR / f"{cfg['slug']}_raw.txt"
            raw_file.write_text(response.text, encoding="utf-8")
            print(f"  Raw response saved to {raw_file.name}")

        except Exception as e:
            elapsed = time.time() - start
            print(f"  ERROR after {elapsed:.1f}s: {e}")

        if cfg is not CONFIGS[-1]:
            time.sleep(1)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  SUMMARY")
    print(f"{'='*62}")
    if baseline is not None:
        print(f"  gemini-3.5-flash (baseline):              {len(baseline):3d} entries")
    for _, (label, entries) in all_results.items():
        print(f"  {label:<45s} {len(entries):3d} entries")
    print(f"\n  Full JSON outputs saved to: {OUT_DIR}")
    print(f"  To diff: diff {OUT_DIR}/flash31lite-high.json {OUT_DIR}/flash31lite-medium.json")


if __name__ == "__main__":
    main()
