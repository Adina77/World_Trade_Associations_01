#!/usr/bin/env python3
"""
Add expanded focus terms to associations_raw.csv.

Pass 1 (always): character-level fuzzy matching (rapidfuzz Levenshtein ratio)
    against focus_abbreviations.csv.  Handles typical OCR character errors.

Pass 2 (optional, --llm): sends low-confidence rows to gemini-3.1-flash-lite,
    including the association name and top-5 fuzzy candidates as context.

Outputs:
    ocr_output/associations_focus.csv   — full dataset with focus_full, focus_score,
                                          focus_flag added
    ocr_output/focus_low_confidence.csv — only the rows that stayed uncertain after
                                          all passes (for manual review)

Dependencies:  pip install rapidfuzz google-genai python-dotenv
Usage:
    python ocr_associations_focus.py                   # fuzzy pass only
    python ocr_associations_focus.py --llm             # fuzzy + LLM review (threshold 80)
    python ocr_associations_focus.py --threshold 85    # fuzzy + LLM review (custom threshold 85, implies --llm)
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

from rapidfuzz import fuzz, process as fuzz_process
from dotenv import load_dotenv

load_dotenv()

# ── paths ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR       = Path(__file__).parent / "ocr_output"
INPUT_CSV        = OUTPUT_DIR / "associations_with_english.csv"
ABBREV_CSV       = OUTPUT_DIR / "focus_abbreviations.csv"
OUTPUT_CSV       = OUTPUT_DIR / "associations_focus.csv"
LOW_CONF_CSV     = OUTPUT_DIR / "focus_low_confidence.csv"

# ── fuzzy-pass settings ────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 80   # rows below this go to low-confidence / LLM pass
TOP_N_CANDIDATES  = 5    # how many fuzzy candidates for focus term to pass to the LLM

# ── LLM settings ──────────────────────────────────────────────────────────────

LLM_MODEL      = "gemini-3.1-flash-lite"
LLM_BATCH_SIZE = 25     # rows per API call
LLM_DELAY      = 1      # seconds between API calls
LLM_MAX_RETRIES   = 4
LLM_RETRY_BACKOFF = 10

# flag values written to the focus_flag column
FLAG_EXACT    = "exact"          # case-insensitive exact match in the abbreviation table
FLAG_FUZZY    = "fuzzy"          # fuzzy match at or above threshold — accepted automatically
FLAG_LLM      = "llm"            # was low_confidence after fuzzy; resolved by the LLM pass
FLAG_LOW_CONF = "low_confidence" # below threshold and still unresolved: either --llm was not
                                 # used, or the LLM returned NO_MATCH; needs manual review
FLAG_EMPTY    = "empty"          # focus field was blank in the source data; no match attempted
FLAG_NO_MATCH = "no_match"       # fuzzy returned nothing (only if abbreviation table is empty)


# ── abbreviation table ─────────────────────────────────────────────────────────

def load_abbreviations() -> dict[str, str]:
    """Return {abbreviation_lower: full_term} from focus_abbreviations.csv."""
    if not ABBREV_CSV.exists():
        raise FileNotFoundError(f"Abbreviation table not found: {ABBREV_CSV}")
    table = {}
    with open(ABBREV_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            abbrev = row["abbreviation"].strip()
            if abbrev:
                table[abbrev.lower()] = row["full_term"].strip()
    return table


# ── fuzzy pass ─────────────────────────────────────────────────────────────────

def fuzzy_match(focus_raw: str, abbrev_table: dict[str, str], threshold: int):
    """
    Match one focus string against the abbreviation table.

    Returns (full_term, score, flag) where flag is one of the FLAG_* constants.
    score is 0-100; exact matches return 100.
    """
    if not focus_raw or not focus_raw.strip():
        return "", 0, FLAG_EMPTY

    focus = focus_raw.strip()
    focus_lower = focus.lower()

    # Exact match first (case-insensitive)
    if focus_lower in abbrev_table:
        return abbrev_table[focus_lower], 100, FLAG_EXACT

    # Character-level fuzzy match against all abbreviation keys
    keys = list(abbrev_table.keys())
    result = fuzz_process.extractOne(
        focus_lower, keys, scorer=fuzz.ratio
    )
    if result is None:
        return focus, 0, FLAG_NO_MATCH

    best_key, score, _ = result
    full_term = abbrev_table[best_key]

    if score >= threshold:
        return full_term, score, FLAG_FUZZY

    return full_term, score, FLAG_LOW_CONF


def top_candidates(focus_raw: str, abbrev_table: dict[str, str], n: int) -> list[dict]:
    """Return the top-n fuzzy candidates as a list of dicts for the LLM prompt."""
    if not focus_raw or not focus_raw.strip():
        return []
    keys = list(abbrev_table.keys())
    hits = fuzz_process.extract(
        focus_raw.strip().lower(), keys, scorer=fuzz.ratio, limit=n
    )
    return [
        {"abbreviation": k, "full_term": abbrev_table[k], "score": round(s)}
        for k, s, _ in hits
    ]


# ── LLM pass ───────────────────────────────────────────────────────────────────

def build_llm_prompt(batch: list[dict]) -> str:
    """
    batch is a list of dicts, each with keys:
        row_idx, id, name, focus_ocr, candidates
    """
    entries_json = json.dumps(
        [
            {
                "row_idx":    b["row_idx"],
                "id":         b["id"],
                "name":       b["name"],
                "focus_ocr":  b["focus_ocr"],
                "candidates": b["candidates"],
            }
            for b in batch
        ],
        ensure_ascii=False,
        indent=2,
    )
    return f"""\
You are correcting OCR errors in a database of trade associations.

Each entry has three fields to work with:
  - "focus_ocr"  : the abbreviated industry/sector term as read by OCR from a
                   scanned page. It should match one entry in the official
                   abbreviation table, but may have one or two characters wrong
                   due to OCR misreads.
  - "name"       : the full name of the trade association.
  - "candidates" : the top matches from the abbreviation table, ranked by
                   character similarity to focus_ocr. Each candidate has both
                   an "abbreviation" (the short form) and a "full_term" (its
                   expanded meaning).

Use TWO signals together to pick the best candidate:
  1. Character similarity — which candidate abbreviation is the most plausible
     OCR misread of focus_ocr? (e.g. "Clothg" → "Cloth", "Furnlt" → "Furnit")
  2. Semantic match — which candidate's full_term best describes the industry
     the association would belong to, given its name?

When both signals point to the same candidate, that is the answer. When they
conflict, prefer character similarity (OCR correction is the primary task) but
use the semantic match as a tiebreaker among equally plausible OCR corrections.

If none of the candidates is a plausible match on either signal, return
"NO_MATCH" for both fields.

Return ONLY a raw JSON array (no markdown, no explanation) with one object per
entry, in the same order:

  {{"row_idx": <integer>, "abbreviation": "<chosen abbreviation or NO_MATCH>", "full_term": "<chosen full_term or NO_MATCH>"}}

Entries to resolve:
{entries_json}
"""


def is_transient_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(t in msg for t in ("503", "500", "502", "504", "429",
                                   "unavailable", "overloaded", "quota"))


def parse_llm_response(text: str) -> list[dict]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text.strip(), flags=re.MULTILINE)
    return json.loads(text.strip())


def run_llm_pass(low_conf_rows: list[dict], abbrev_table: dict[str, str]) -> dict[int, dict]:
    """
    Send low-confidence rows to the LLM in batches.

    low_conf_rows: list of enriched row dicts that still have FLAG_LOW_CONF.
    Returns a dict keyed by row_idx: {"full_term": ..., "abbreviation": ...}
    """
    try:
        import google.genai as genai
        from google.genai import types
    except ImportError:
        print("  google-genai not installed — skipping LLM pass")
        return {}

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("  GOOGLE_API_KEY not set — skipping LLM pass")
        return {}

    safety_off = [
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",       threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]
    cfg = types.GenerateContentConfig(safety_settings=safety_off)

    client  = genai.Client(api_key=api_key)
    results = {}

    batches = [
        low_conf_rows[i : i + LLM_BATCH_SIZE]
        for i in range(0, len(low_conf_rows), LLM_BATCH_SIZE)
    ]
    total_batches = len(batches)

    for b_idx, batch in enumerate(batches, 1):
        # attach top-N candidates to each row
        payload = [
            {
                "row_idx":   row["_row_idx"],
                "id":        row.get("id", ""),
                "name":      row.get("name", ""),
                "focus_ocr": row.get("focus", ""),
                "candidates": top_candidates(row.get("focus", ""), abbrev_table, TOP_N_CANDIDATES),
            }
            for row in batch
        ]

        print(f"  LLM batch {b_idx}/{total_batches} ({len(batch)} rows) ...", end="  ", flush=True)
        prompt = build_llm_prompt(payload)

        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=LLM_MODEL, contents=[prompt], config=cfg
                )
                if response.text is None:
                    raise ValueError("response.text is None")
                parsed = parse_llm_response(response.text)
                for item in parsed:
                    idx = item.get("row_idx")
                    if idx is not None:
                        results[idx] = {
                            "abbreviation": item.get("abbreviation", ""),
                            "full_term":    item.get("full_term",    ""),
                        }
                print(f"{len(parsed)} resolved")
                break

            except json.JSONDecodeError as e:
                print(f"JSON parse error — {e}")
                break

            except Exception as e:
                if is_transient_error(e) and attempt < LLM_MAX_RETRIES:
                    wait = LLM_RETRY_BACKOFF * (2 ** (attempt - 1))
                    print(f"\n    error (attempt {attempt}): {e} — retry in {wait}s")
                    time.sleep(wait)
                    print(f"  LLM batch {b_idx}/{total_batches} ...", end="  ", flush=True)
                else:
                    print(f"ERROR: {e}")
                    break

        if b_idx < total_batches:
            time.sleep(LLM_DELAY)

    return results


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Match focus abbreviations to full terms in associations_raw.csv."
    )
    parser.add_argument(
        "--llm", action="store_true",
        help=f"Run {LLM_MODEL} on low-confidence rows after the fuzzy pass."
    )
    parser.add_argument(
        "--threshold", type=int, default=DEFAULT_THRESHOLD,
        help=f"Fuzzy score threshold 0-100 (default {DEFAULT_THRESHOLD}). "
             "Rows below this are sent to the LLM. Implies --llm."
    )
    args = parser.parse_args()

    # Specifying --threshold only makes sense with the LLM pass, so imply it.
    if args.threshold != DEFAULT_THRESHOLD:
        args.llm = True

    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run reqired scripts first.")
        return
    if not ABBREV_CSV.exists():
        print(f"ERROR: {ABBREV_CSV} not found. Run required scripts first.")
        return

    print(f"Loading abbreviation table from {ABBREV_CSV.name} ...")
    abbrev_table = load_abbreviations()
    print(f"  {len(abbrev_table)} abbreviations loaded")

    print(f"Loading associations from {INPUT_CSV.name} ...")
    with open(INPUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows):,} rows loaded")

    # ── fuzzy pass ─────────────────────────────────────────────────────────────
    print(f"\nFuzzy pass  (threshold={args.threshold}) ...")
    counts = {FLAG_EXACT: 0, FLAG_FUZZY: 0, FLAG_LOW_CONF: 0,
              FLAG_EMPTY: 0, FLAG_NO_MATCH: 0}

    for idx, row in enumerate(rows):
        full_term, score, flag = fuzzy_match(row.get("focus", ""), abbrev_table, args.threshold)
        row["focus_full"]  = full_term
        row["focus_score"] = score
        row["focus_flag"]  = flag
        row["_row_idx"]    = idx
        counts[flag] = counts.get(flag, 0) + 1

    print(f"  exact          : {counts[FLAG_EXACT]:>6,}")
    print(f"  fuzzy (≥{args.threshold})    : {counts[FLAG_FUZZY]:>6,}")
    print(f"  low confidence : {counts[FLAG_LOW_CONF]:>6,}")
    print(f"  empty focus    : {counts[FLAG_EMPTY]:>6,}")
    print(f"  no match       : {counts[FLAG_NO_MATCH]:>6,}")

    low_conf = [r for r in rows if r["focus_flag"] == FLAG_LOW_CONF]

    # ── LLM pass ───────────────────────────────────────────────────────────────
    if args.llm and low_conf:
        print(f"\nLLM pass  ({len(low_conf)} rows → {LLM_MODEL}) ...")
        llm_results = run_llm_pass(low_conf, abbrev_table)

        resolved = 0
        for row in low_conf:
            hit = llm_results.get(row["_row_idx"])
            if hit and hit.get("full_term", "").upper() != "NO_MATCH":
                row["focus_full"]  = hit["full_term"]
                row["focus_score"] = 0      # LLM doesn't produce a numeric score
                row["focus_flag"]  = FLAG_LLM
                resolved += 1

        still_low = sum(1 for r in rows if r["focus_flag"] == FLAG_LOW_CONF)
        print(f"  resolved by LLM : {resolved:>6,}")
        print(f"  still uncertain : {still_low:>6,}")
    elif args.llm and not low_conf:
        print("\nNo low-confidence rows — LLM pass skipped.")

    # ── write outputs ──────────────────────────────────────────────────────────
    # Drop the internal index before writing
    for row in rows:
        row.pop("_row_idx", None)

    original_fields = list(rows[0].keys()) if rows else []
    # Ensure the three new columns come right after focus, and no duplicates
    base_fields = [f for f in original_fields
                   if f not in ("focus_full", "focus_score", "focus_flag")]
    try:
        focus_pos = base_fields.index("focus") + 1
    except ValueError:
        focus_pos = len(base_fields)
    fieldnames = (base_fields[:focus_pos]
                  + ["focus_full", "focus_score", "focus_flag"]
                  + base_fields[focus_pos:])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows):,} rows → {OUTPUT_CSV}")

    still_low = [r for r in rows if r["focus_flag"] in (FLAG_LOW_CONF, FLAG_NO_MATCH)]
    if still_low:
        with open(LOW_CONF_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(still_low)
        print(f"Saved {len(still_low):,} uncertain rows → {LOW_CONF_CSV}")
    else:
        print("All rows matched — no low-confidence CSV written.")

    print("Done.")


if __name__ == "__main__":
    main()
