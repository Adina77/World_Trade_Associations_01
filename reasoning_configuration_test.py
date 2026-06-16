import time
import os
from google import genai
from google.genai import types
from PIL import Image

# Initialize the new client
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# We will test 3 configurations: 
# 1. 2.5 Flash with a capped budget (so it doesn't take 10 minutes)
# 2. 3.1 Flash-Lite with "High" reasoning (to fix the ID shift)
# 3. 3.1 Flash-Lite with "Medium" reasoning (for speed vs accuracy balance)

models_to_test = [
    {
        "name": "gemini-2.5-flash",
        "desc": "Capped Thinking Budget (1024 tokens)",
        "config": types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=1024)
        )
    },
    {
        "name": "gemini-3.1-flash-lite",
        "desc": "High Thinking Level",
        "config": types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="high")
        )
    },
    {
        "name": "gemini-3.1-flash-lite",
        "desc": "Medium Thinking Level",
        "config": types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="medium")
        )
    }
]

image_path = "image00050.jpg"

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
"""

try:
    img = Image.open(image_path)
except FileNotFoundError:
    print(f"Error: Could not find {image_path}.")
    exit()

print(f"Beginning reasoning configuration test...\n")

for test in models_to_test:
    print(f"--- Running {test['name']} ({test['desc']}) ---")
    start_time = time.time()
    
    try:
        response = client.models.generate_content(
            model=test['name'],
            contents=[PROMPT, img],
            config=test['config'] # This passes the custom reasoning rules!
        )
        end_time = time.time()
        
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print("Output snippet (first 1000 characters):")
        print(response.text[:1000]) 
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    print("\n" + "="*60 + "\n")