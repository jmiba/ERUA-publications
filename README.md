# Aurora SDG Publication Classifier

This Streamlit app helps you explore publications from the OpenAlex database for any ROR (Research Organization Registry) institution. It pulls the metadata, enriches missing abstracts (Semantic Scholar → Google Scholar), runs them through Aurora’s Sustainable Development Goals (SDG) classifiers, and gives you immediate visual and downloadable results that you can take into Excel or other tools.

## What you can do

- **Search institutions**: Enter a name or paste a ROR URL (e.g. `https://ror.org/02msan859`). The built-in search talks to the ROR registry and keeps results handy so you do not have to re-run the query every time.
- **Set filters**: Choose publication types, SDG classifier models, time windows (month-by-month slider), and optional record limits if you want a quick test run.
- **Fetch SDG predictions**: The app calls OpenAlex for metadata and Aurora for SDG scores. Missing abstracts are fetched from Semantic Scholar and, if needed, Google Scholar. Everything is cached locally in `cache.sqlite3` to avoid redundant network calls.
- **Inspect results instantly**: The “Preview” section shows 25 rows per page with friendly author/institution details. You can select a page and focus on a single row to drive the SDG chart.
- **Visualize SDG coverage**: The pie ring chart aggregates SDG scores across all rows or the selected publication.
- **Export data**: Download either a CSV or Excel file (with no external dependencies) for the entire result set.

## Demo
A live demo is available at [Streamlit Cloud - Aurora SDG Publicaton Classifier](https://aurora-sdg-publication-classifier.streamlit.app). Note that the demo instance may have usage limits and could be slower due to shared resources.

## High-level workflow

```mermaid
flowchart TB
    A[User picks ROR + options] --> B[Fetch OpenAlex works]
    B --> C{Abstract available?}
    C -->|Yes| D[Use OpenAlex abstract]
    C -->|No| E{Cached abstract?}
    E -->|Yes| D
    E -->|No| F{Semantic Scholar (via DOI)?}
    F -->|Found| D
    F -->|Missing| G{Google Scholar fallback?}
    G -->|Found| D
    G -->|Still missing| H[Use title for SDG]
    D --> I{SDG cached?}
    H --> I
    I -->|Cache valid| J[Reuse SDG results]
    I -->|Needs run| K[Call Aurora classifier]
    K --> L[Store SDG + abstract in cache]
    J --> L
    L --> M[Streamlit preview + charts]
    M --> N[Download CSV/XLSX]
```

## How it works in the background

1. **OpenAlex fetch**: We request the Works API using your selected ROR, date range, and publication type. The request uses a friendly User-Agent (set via `.streamlit/secrets.toml`) to comply with API guidelines.
2. **Caching**: Each returned record and SDG classification is stored in a local SQLite database (`cache.sqlite3`). When you rerun the query, previously fetched publications are loaded from the cache, so the Aurora SDG API is only contacted for new or uncached works. The cache runs in WAL mode for safe concurrent access.
3. **SDG classification**: Depending on the model you pick (“aurora-sdg”, “aurora-sdg-multi”, “elsevier-sdg-multi”, “osdg”, or “skip”), abstracts or titles are sent to the relevant Aurora endpoint. Short abstracts are skipped when the model requires a minimum length (e.g., the OSDG model).
4. **Abstract enrichment**: If OpenAlex provides no abstract we reuse any cached text, otherwise call the Semantic Scholar API (optional key via secrets). As a final fallback we search Google Scholar via `scholarly`. All retrieved abstracts are cleaned (HTML stripped) and cached.
5. **Preview**: The Streamlit UI converts the cached data into a small DataFrame. OpenAlex IDs and DOIs are shown as clickable links; the selection checkbox behaves like a radio button so only one item is highlighted at a time.
6. **Exports**: A custom lightweight Excel writer assembles `.xlsx` files without additional libraries, ensuring easy downloads even in bare environments.

## Getting started

1. **Install dependencies**: `pip install -r requirements.txt` (Streamlit, pandas, requests, Altair, scholarly).
2. **Configure secrets**: Create `.streamlit/secrets.toml` with at least an `http_user_agent`. Optional keys let you set a default start date and a Semantic Scholar API key.
3. **Run the app**: `streamlit run app.py` from the project directory.
4. **Use the interface**: Search or paste a ROR, choose options, and press “Fetch works and build CSV.” Progress bars show how many works have been processed.
5. **Download your data**: After the fetch completes you’ll see the chart, preview, and buttons for Excel/CSV downloads.

## Notes for non-technical users

- The app tries to minimize API calls by caching results. If you wonder why a second run finishes faster, it’s reusing the stored data (including fetched abstracts).
- CSV downloads are great for import into statistical packages. Excel downloads open directly in Microsoft Excel or LibreOffice.
- If you change parameters (date range, SDG model, etc.), remember to press the fetch button again to refresh the data.
- OpenAlex and Aurora APIs may throttle large requests. Start with a modest limit (e.g., 200 works) to validate your settings before removing the limit.

## Privacy and data location

All data is downloaded to your machine. The cache file `cache.sqlite3` lives beside the scripts and is ignored by git. You can delete it at any time to force a fresh fetch.

## Configuring secrets

The app relies on Streamlit’s secrets mechanism. Create a `.streamlit/secrets.toml` file with entries like:

```toml
http_user_agent = "OpenAlex+Aurora SDG fetcher (mailto:you@example.com)"
semantic_scholar_api_key = "YOUR-API-KEY"

[advanced_options]
default_from_date = "2020-01-01"
```

- `http_user_agent` is required and should include a contact email so OpenAlex can whitelist your requests.
- `semantic_scholar_api_key` is optional but allows the app to pull missing abstracts directly from Semantic Scholar.
- Under `[advanced_options]` you can set `default_from_date` to control the initial position of the publication date slider.

A sample file is included in `.streamlit/secrets.toml`; update it with your real values. If a key is set to `"None"`, the app treats it as missing.

---

Enjoy exploring how your institution’s publications map to the Sustainable Development Goals!
