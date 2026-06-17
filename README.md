This project generates a dataset for trade associations from the book volume _World Guide to Trade Associations_, published in 2002. The book pages were scanned to generate a jpg image file for each page. These are the inputs for
the ocr_pipeline.py script. The script sends each scanned page image to the Gemini API and extracts structured association data (country, name, address, focus, ID number).

Progress is saved after every page, so the script can be safely interrupted and restarted. Pages already processed are skipped automatically.

The generated CSV file is cleaned with the script ocr_cleanup.py. Errors are then checked with ocr_error_check.py, and errors can be decreased by re-submitting to a superior Gemini model, using error_redo.py.

For now the repository is missing the scanned book pages (WorldGuideTrade_bookpages directory) due to the size of the image file collection.

Dependencies — install in your virtualenv:

```
pip install google-genai Pillow python-dotenv
```

Usage — run the scripts sequentially:

```
python ocr_pipeline.py
python ocr_cleanup.py
python ocr_error_check.py
python error_redo.py
```

You need an API key from Google AI Studio. Place the key in an .env file within your working directory. See the example file, .env.example. Be sure to add this file to .gitignore.

To compare models to choose for larger batches, the optional script model_comparison_test.py can be used.
