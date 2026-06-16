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

    # Initialize the new SDK client
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
    todo = [p for p in images if p.name not in done]

    print(f"Pages found:       {len(images):>5}")
    print(f"Already processed: {len(done):>5}")
    print(f"To process:        {len(todo):>5}")
    print()

    for i, img_path in enumerate(todo, 1):
        print(f"[{i:>4}/{len(todo)}]  {img_path.name}", end="  ...  ", flush=True)

        try:
            img = Image.open(img_path)
            
            # The updated generate_content syntax
            response = client.models.generate_content(
                model=MODEL,
                contents=[PROMPT, img]
            )
            
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