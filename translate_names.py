#!/usr/bin/env python3
"""
Translate trade association names to English using the Gemini API.

Reads associations_cleaned.csv, adds a name_English column after the name column,
and writes associations_with_english.csv.

Rules applied in order (cheapest first):
  1. Names containing " / " already have an English translation after the slash
     (e.g. "Fédération / Federation of..."). These are extracted without any API call.
  2. All other names are sent to Gemini in batches of BATCH_SIZE. Gemini returns
     the name unchanged if it is already in English, or an English translation otherwise.

Progress is checkpointed to translation_progress.jsonl after every batch, so the
script can be safely interrupted (Ctrl+C) and restarted — already-translated names
are never re-sent to the API.

"""

import csv
import json
import os
import re
import time
from pathlib import Path

import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["GOOGLE_API_KEY"]

# Flash model is sufficient for translation — much cheaper than 2.5-pro.
# Swap to "gemini-2.5-pro" if translation quality is unsatisfactory.
MODEL = "gemini-2.5-flash"

BATCH_SIZE = 500  # names per API call — ~42 calls total for 21k names
DELAY      = 10  # seconds between calls — 13 sec stays at ~4.6 RPM, safely under the 5 RPM limit

OUTPUT_DIR      = Path(__file__).parent / "ocr_output"
INPUT_CSV       = OUTPUT_DIR / "associations_cleaned.csv"
OUTPUT_CSV      = OUTPUT_DIR / "associations_with_english.csv"
CHECKPOINT_FILE = OUTPUT_DIR / "translation_progress.jsonl"

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

PROMPT_TEMPLATE = """\
Below is a JSON array of trade association \
entries from a 2002 international reference book. Each entry has a "country" field \
(tells you the language to expect) and a "name" field (the association name to translate).

For each entry, return an English translation of the name:
- If the name is already in English, return it unchanged.
- If the name is in another language, use the country to confirm the source language \
and translate to English.
- Keep standard organisational abbreviations (e.g. "e.V.", "GmbH", "a.s.b.l.", "S.A.") \
unchanged.
- Do not add any explanation.

Return ONLY a JSON array of strings — one translated name per entry, in the same order \
as the input.

Input:
{entries_json}
"""


def load_checkpoint() -> dict[str, str]:
    """Return a mapping of id → name_English for all already-translated rows."""
    if not CHECKPOINT_FILE.exists():
        return {}
    result: dict[str, str] = {}
    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                result.update(json.loads(line))
    return result


def save_checkpoint(id_to_english: dict[str, str]) -> None:
    """Append one batch's results to the checkpoint file."""
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(id_to_english, ensure_ascii=False) + "\n")


def extract_after_slash(name: str) -> str | None:
    """Return the part after the last ' / ' if the name is bilingual, else None."""
    if " / " in name:
        return name.rsplit(" / ", 1)[1].strip()
    return None


def translate_batch(client, batch: list[dict]) -> list[str] | None:
    """Send one batch of {country, name} dicts to Gemini. Returns English strings, or None on failure."""
    entries = [{"country": r["country"], "name": r["name"]} for r in batch]
    prompt = PROMPT_TEMPLATE.format(entries_json=json.dumps(entries, ensure_ascii=False))
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=JSON_CONFIG,
        )
        if response.text is None:
            print("  WARNING: Gemini returned None — skipping batch")
            return None

        text = response.text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
        result = json.loads(text.strip())

        if not isinstance(result, list) or len(result) != len(batch):
            print(f"  WARNING: unexpected response length ({len(result)} vs {len(batch)}) — skipping")
            return None

        return [str(r) for r in result]

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main() -> None:
    if not API_KEY:
        print("ERROR: GOOGLE_API_KEY not found. Create a .env file containing:\n  GOOGLE_API_KEY=your_key_here")
        return

    # ── Load input CSV ──────────────────────────────────────────────────────
    rows: list[dict] = []
    with open(INPUT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"Loaded {len(rows):,} rows from {INPUT_CSV.name}")

    # ── Load existing checkpoint ────────────────────────────────────────────
    done: dict[str, str] = load_checkpoint()
    print(f"Already translated (from checkpoint): {len(done):,}")

    # ── Apply slash-extraction rule (free — no API call) ───────────────────
    slash_count = 0
    for row in rows:
        if row["id"] in done:
            continue
        english = extract_after_slash(row["name"])
        if english:
            done[row["id"]] = english
            slash_count += 1
    print(f"Extracted from ' / ' pattern:         {slash_count:,}")

    # ── Identify rows still needing Gemini ─────────────────────────────────
    todo = [r for r in rows if r["id"] not in done]
    print(f"Need Gemini translation:              {len(todo):,}")
    print()

    # ── Call Gemini in batches ─────────────────────────────────────────────
    if todo:
        client = genai.Client(api_key=API_KEY)
        batches = [todo[i : i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]

        for i, batch in enumerate(batches, 1):
            ids = [r["id"] for r in batch]
            print(f"[{i:>4}/{len(batches)}]  {len(batch)} names ...", end="  ", flush=True)

            translations = translate_batch(client, batch)
            if translations is not None:
                batch_result = dict(zip(ids, translations))
                done.update(batch_result)
                save_checkpoint(batch_result)
                print("done")
            else:
                print("SKIPPED (re-run to retry)")

            if i < len(batches):
                time.sleep(DELAY)

    # ── Write output CSV ───────────────────────────────────────────────────
    print()
    print(f"Writing {OUTPUT_CSV.name} ...")
    fieldnames = ["id", "country", "name", "name_English", "address", "focus", "source_page"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row["name_English"] = done.get(row["id"], "")
            writer.writerow(row)

    missing = sum(1 for r in rows if not done.get(r["id"]))
    print(f"Saved {len(rows):,} rows → {OUTPUT_CSV.name}")
    if missing:
        print(f"  {missing:,} rows have no translation — re-run to retry skipped batches.")
    else:
        print("  All rows translated.")
    print("Done.")


if __name__ == "__main__":
    main()
