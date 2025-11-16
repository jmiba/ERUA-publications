import csv
import io
import json
import re
import time
import unicodedata
import calendar
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import requests
import streamlit as st
import tomllib


BASE_WORKS = "https://api.openalex.org/works"
BASE_INSTITUTIONS = "https://api.openalex.org/institutions"
AURORA_BASE = "https://aurora-sdg.labs.vu.nl/classifier/classify"
SEMANTIC_SCHOLAR_API = (
    "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract"
)

PER_PAGE = 200
DEFAULT_USER_AGENT = (
    "OpenAlex+Aurora SDG fetcher (mailto:you@example.com)"
)

AURORA_MODELS = [
    ("aurora-sdg", "Aurora SDG mBERT (single-label, slower)"),
    ("aurora-sdg-multi", "Aurora SDG multi-label mBERT (fast)"),
    ("elsevier-sdg-multi", "Elsevier SDG multi-label mBERT (fast)"),
    ("osdg", "OSDG model (multi-label, 15 languages)"),
    ("skip", "Skip SDG classification (no Aurora API calls)"),
]

MIN_WORDS_BY_MODEL: Dict[str, int] = {"osdg": 50}
SECRET_HTTP_USER_AGENT = "http_user_agent"
SECRET_SEMANTIC_SCHOLAR_KEY = "semantic_scholar_api_key"
SECRET_DEFAULT_START = "advanced_options.default_from_date"
_SECRETS: Dict[str, Any] = {}
PREVIEW_COLUMNS = [
    "openalex_id",
    "title",
    "publication_date",
    "type",
    "doi",
    "authors",
    "institutions",
]
RESULT_SESSION_KEY = "fetch_result"
SDG_THRESHOLD_PERCENT = 3.0
FOCUS_ROW_KEY_PREFIX = "preview_focus_row_"
RADIO_CHECKBOX_CSS = """
<style>
div[data-testid="stCheckbox"] input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 1rem;
    height: 1rem;
    border: 2px solid var(--primary-color);
    border-radius: 50%;
    position: relative;
    cursor: pointer;
}
div[data-testid="stCheckbox"] input[type="checkbox"]:checked {
    background-color: var(--primary-color);
}
div[data-testid="stCheckbox"] input[type="checkbox"]:checked::after {
    content: "";
    position: absolute;
    top: 0.15rem;
    left: 0.15rem;
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background-color: white;
}
</style>
"""


def _load_secrets() -> Dict[str, Any]:
    if _SECRETS:
        return _SECRETS
    try:
        for key, value in st.secrets.items():
            _SECRETS[key] = value
    except Exception:
        pass
    if _SECRETS:
        return _SECRETS
    candidate_paths = [
        Path(".streamlit/secrets.toml"),
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
                if isinstance(data, dict):
                    _SECRETS.update(data)
        except Exception:
            continue
    return _SECRETS


def too_short_for_model(model: str, text: str) -> bool:
    need = MIN_WORDS_BY_MODEL.get(model, 0)
    return need > 0 and len((text or "").split()) < need


def is_ror_url(value: str) -> bool:
    return bool(re.match(r"^https?://ror\.org/[0-9a-z]{9}$", value.strip(), re.I))


def get_secret_text(name: str) -> Optional[str]:
    if "." in name:
        section, key = name.split(".", 1)
        raw_value = (_load_secrets().get(section) or {}).get(key)
    else:
        raw_value = _load_secrets().get(name)
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def resolve_user_agent() -> Tuple[str, bool]:
    secret_value = get_secret_text(SECRET_HTTP_USER_AGENT)
    if secret_value:
        return secret_value.strip(), True
    return DEFAULT_USER_AGENT, False


def resolve_semantic_scholar_key() -> Optional[str]:
    return get_secret_text(SECRET_SEMANTIC_SCHOLAR_KEY)


def subtract_months(base: date, months: int) -> date:
    year = base.year
    month = base.month - months
    day = base.day
    while month <= 0:
        month += 12
        year -= 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(day, last_day))


def search_institutions_by_name(
    name: str,
    user_agent: str,
    limit: int = 10,
    timeout: int = 30,
) -> List[dict]:
    params = {"search": name, "per-page": limit}
    headers = {"User-Agent": user_agent}
    response = requests.get(
        BASE_INSTITUTIONS, params=params, headers=headers, timeout=timeout
    )
    response.raise_for_status()
    return response.json().get("results", [])


def reconstruct_abstract(inv: Optional[Dict[str, List[int]]]) -> str:
    if not inv:
        return ""
    max_pos = -1
    for positions in inv.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return ""
    tokens_by_pos = [""] * (max_pos + 1)
    for token, positions in inv.items():
        for pos in positions:
            if pos < 0:
                continue
            if pos >= len(tokens_by_pos):
                tokens_by_pos.extend([""] * (pos - len(tokens_by_pos) + 1))
            tokens_by_pos[pos] = (
                f"{tokens_by_pos[pos]} {token}".strip() if tokens_by_pos[pos] else token
            )
    return " ".join(tokens_by_pos).strip()


def flatten_authors_and_institutions(authorships: List[dict]) -> Tuple[str, str]:
    if not authorships:
        return "", ""
    author_names: List[str] = []
    institution_names: List[str] = []
    for authorship in authorships:
        author = (authorship.get("author") or {}).get("display_name")
        if author:
            author_names.append(author)
        for inst in authorship.get("institutions") or []:
            name = inst.get("display_name")
            if name:
                institution_names.append(name)
    seen = set()
    deduped_insts: List[str] = []
    for name in institution_names:
        if name not in seen:
            seen.add(name)
            deduped_insts.append(name)
    return "; ".join(author_names), "; ".join(deduped_insts)


def make_filter(
    ror_url: str,
    from_date: Optional[str],
    to_date: Optional[str],
    wtype: Optional[str],
) -> str:
    parts = [
        f"institutions.ror:{ror_url}",
        "is_paratext:false",
    ]
    if from_date:
        parts.append(f"from_publication_date:{from_date}")
    if to_date:
        parts.append(f"to_publication_date:{to_date}")
    if wtype:
        parts.append(f"type:{wtype}")
    return ",".join(parts)


def classify_text_aurora(
    model: str,
    text: str,
    session: requests.Session,
    user_agent: str,
    retries: int = 3,
    pause: float = 0.4,
) -> Tuple[Optional[dict], str]:
    if not text:
        return None, "no text"
    url = f"{AURORA_BASE}/{model}"
    headers = {
        "User-Agent": user_agent,
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {"text": text}
    for attempt in range(1, retries + 1):
        try:
            resp = session.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            if resp.status_code == 429:
                time.sleep(pause * attempt + 0.5)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data or None, "" if data else "empty json"
        except requests.RequestException as exc:
            if attempt == retries:
                code = getattr(getattr(exc, "response", None), "status_code", None)
                return None, f"http_error:{code}"
            time.sleep(pause * attempt)
    return None, "unknown"


def get_abstract_from_semantic_scholar(
    doi: str,
    api_key: Optional[str] = None,
    retries: int = 3,
    pause: float = 0.5,
) -> Optional[str]:
    if not doi:
        return None
    cleaned_doi = doi.replace("https://doi.org/", "")
    url = SEMANTIC_SCHOLAR_API.format(doi=cleaned_doi)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code == 429:
                time.sleep(pause * attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data.get("abstract")
        except requests.RequestException:
            if attempt == retries:
                return None
            time.sleep(pause * attempt)
    return None


def format_sdg_predictions(sdg_json: Optional[dict]) -> str:
    if not sdg_json:
        return ""

    def fmt_line(score: float, code: str, name: Optional[str]) -> str:
        label = (name or f"SDG {code}".strip()).strip()
        if label.lower().startswith("sdg "):
            return f"{score * 100:.0f}% {label}"
        return f"{score * 100:.0f}% SDG {code} ({label})"

    items: List[Tuple[float, str, Optional[str]]] = []
    preds = sdg_json.get("predictions") if isinstance(sdg_json, dict) else None
    if isinstance(preds, list):
        for prediction in preds:
            sdg = prediction.get("sdg") or {}
            code = sdg.get("code")
            name = sdg.get("name")
            score = prediction.get("prediction")
            if code is None or score is None:
                continue
            try:
                items.append((float(score), str(code), name))
            except (TypeError, ValueError):
                continue

    if not items and isinstance(sdg_json, list):
        for entry in sdg_json:
            label = entry.get("label")
            score = entry.get("score")
            if label is None or score is None:
                continue
            match = re.search(r"\bSDG\s*(\d+)", str(label), re.I)
            code = match.group(1) if match else ""
            items.append((float(score), code, str(label)))

    if (
        not items
        and isinstance(sdg_json, dict)
        and "labels" in sdg_json
        and "scores" in sdg_json
    ):
        for label, score in zip(sdg_json.get("labels", []), sdg_json.get("scores", [])):
            match = re.search(r"\bSDG\s*(\d+)", str(label), re.I)
            code = match.group(1) if match else ""
            items.append((float(score), code, str(label)))

    if not items and isinstance(sdg_json, dict):
        numeric_keys = [key for key in sdg_json.keys() if str(key).isdigit()]
        for key in numeric_keys:
            try:
                items.append((float(sdg_json[key]), str(key), None))
            except (TypeError, ValueError):
                continue

    if (
        not items
        and isinstance(sdg_json, dict)
        and isinstance(sdg_json.get("results"), list)
    ):
        for result in sdg_json["results"]:
            code = result.get("sdg") or result.get("code")
            score = result.get("score") or result.get("prediction")
            name = result.get("name") or result.get("label")
            if code is None or score is None:
                continue
            items.append((float(score), str(code), name))

    if not items:
        return ""

    items.sort(key=lambda entry: entry[0], reverse=True)
    return "\n".join(fmt_line(score, code, name) for score, code, name in items)


def sanitize_filename(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    cleaned = re.sub(r"[^\w.\-]+", "_", normalized)
    return cleaned.strip("_")


def build_preview_rows(rows: List[Dict[str, str]], columns: List[str], limit: int = 20) -> List[Dict[str, str]]:
    preview: List[Dict[str, str]] = []
    for row in rows[:limit]:
        preview.append({col: row.get(col, "") for col in columns})
    return preview


def parse_sdg_formatted(value: str) -> List[Tuple[str, float, str]]:
    entries: List[Tuple[str, float, str]] = []
    if not value:
        return entries
    for line in value.splitlines():
        match = re.search(r"(\d+(?:\.\d+)?)%\s*(?:SDG\s*)?(\d+)(?:\s*\(([^)]+)\))?", line, re.I)
        if not match:
            continue
        pct = float(match.group(1))
        code = match.group(2)
        name = match.group(3) or ""
        entries.append((code, pct, name))
    return entries


def aggregate_sdg_counts(rows: List[Dict[str, str]]) -> List[Tuple[str, float]]:
    totals: Dict[str, float] = {}
    for row in rows:
        formatted = row.get("sdg_formatted") or ""
        for code, pct, name in parse_sdg_formatted(formatted):
            if pct < SDG_THRESHOLD_PERCENT:
                continue
            label = f"SDG {code}"
            if name:
                label = f"{label} ({name})"
            totals[label] = totals.get(label, 0.0) + pct
    return sorted(totals.items(), key=lambda pair: pair[1], reverse=True)


def render_sdg_pie_chart(data: List[Tuple[str, float]], title: str):
    if not data:
        st.info(f"No SDG predictions available for {title.lower()}.")
        return
    df = pd.DataFrame(data, columns=["SDG", "Value"])
    chart = (
        alt.Chart(df)
        .mark_arc()
        .encode(theta="Value", color="SDG", tooltip=["SDG", "Value"])
        .properties(width=400, height=400, title=title)
    )
    st.altair_chart(chart, use_container_width=True)


def build_output_filename(
    ror_url: str,
    wtype: Optional[str],
    model: str,
    from_date: str,
    to_date: Optional[str],
    limit_rows: Optional[int],
) -> str:
    ror_tail = ror_url.rstrip("/").split("/")[-1]
    type_part = wtype or "all"
    model_part = model if model != "skip" else "no-sdg"
    fname = f"openalex_{ror_tail}_{type_part}_{model_part}_{from_date}"
    if to_date and to_date != from_date:
        fname += f"_to{to_date}"
    if limit_rows:
        fname += f"_n{limit_rows}"
    return sanitize_filename(f"{fname}.csv")


def fetch_to_csv_with_sdg(
    ror_url: str,
    from_date_str: str,
    to_date_str: Optional[str],
    wtype: Optional[str],
    model: str,
    limit_rows: Optional[int],
    user_agent: str,
    semantic_scholar_key: Optional[str],
    progress_callback,
) -> Tuple[bytes, Dict[str, int]]:
    headers = {"User-Agent": user_agent}
    params = {
        "filter": make_filter(ror_url, from_date_str, to_date_str, wtype),
        "select": ",".join(
            [
                "id",
                "display_name",
                "title",
                "publication_date",
                "doi",
                "abstract_inverted_index",
                "type",
                "language",
                "open_access",
                "authorships",
            ]
        ),
        "per-page": PER_PAGE,
        "cursor": "*",
    }

    fieldnames = [
        "openalex_id",
        "title",
        "publication_date",
        "doi",
        "type",
        "language",
        "is_oa",
        "oa_status",
        "authors",
        "institutions",
        "abstract",
        "sdg_model",
        "sdg_response",
        "sdg_formatted",
        "sdg_note",
    ]

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()

    stats = {
        "rows": 0,
        "openalex_abstract_missing": 0,
        "semantic_scholar_abstracts": 0,
    }

    with requests.Session() as session:
        first_page = session.get(BASE_WORKS, params=params, headers=headers, timeout=60)
        first_page.raise_for_status()
        payload = first_page.json()
        total_expected = (payload.get("meta") or {}).get("count")
        results = payload.get("results", [])
        next_cursor = (payload.get("meta") or {}).get("next_cursor")

        def update_progress():
            progress_callback(stats["rows"], total_expected, limit_rows)

        def process_record(record: dict):
            authorships = record.get("authorships") or []
            authors, institutions = flatten_authors_and_institutions(authorships)
            abstract = reconstruct_abstract(
                record.get("abstract_inverted_index")
            )
            doi = record.get("doi") or ""
            if not abstract:
                stats["openalex_abstract_missing"] += 1
                if doi:
                    ss_abstract = get_abstract_from_semantic_scholar(doi, semantic_scholar_key)
                    if ss_abstract:
                        abstract = ss_abstract
                        stats["semantic_scholar_abstracts"] += 1

            text_for_sdg = abstract or record.get("display_name") or record.get("title") or ""
            sdg_json = None
            sdg_note = ""
            sdg_formatted = ""
            if model == "skip":
                sdg_note = "skipped: user selected 'skip'"
            elif model == "osdg" and too_short_for_model(model, text_for_sdg):
                sdg_note = "skipped: osdg requires >=50 words"
            else:
                sdg_json, sdg_note = classify_text_aurora(
                    model,
                    text_for_sdg,
                    session=session,
                    user_agent=user_agent,
                )
                sdg_formatted = format_sdg_predictions(sdg_json) if sdg_json else ""
                time.sleep(0.12)

            writer.writerow(
                {
                    "openalex_id": record.get("id", ""),
                    "title": record.get("display_name") or record.get("title") or "",
                    "publication_date": record.get("publication_date") or "",
                    "doi": doi,
                    "type": record.get("type") or "",
                    "language": record.get("language") or "",
                    "is_oa": (record.get("open_access") or {}).get("is_oa"),
                    "oa_status": (record.get("open_access") or {}).get("oa_status") or "",
                    "authors": authors,
                    "institutions": institutions,
                    "abstract": abstract,
                    "sdg_model": model,
                    "sdg_response": json.dumps(sdg_json, ensure_ascii=False) if sdg_json else "",
                    "sdg_formatted": sdg_formatted,
                    "sdg_note": sdg_note,
                }
            )
            stats["rows"] += 1
            update_progress()

        for record in results:
            if limit_rows is not None and stats["rows"] >= limit_rows:
                next_cursor = None
                break
            process_record(record)

        while next_cursor:
            if limit_rows is not None and stats["rows"] >= limit_rows:
                break
            params["cursor"] = next_cursor
            response = session.get(BASE_WORKS, params=params, headers=headers, timeout=60)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", [])
            next_cursor = (payload.get("meta") or {}).get("next_cursor")
            for record in results:
                if limit_rows is not None and stats["rows"] >= limit_rows:
                    next_cursor = None
                    break
                process_record(record)
            time.sleep(0.2)

    update_progress()
    buffer.seek(0)
    return buffer.getvalue().encode("utf-8"), stats


def render_institution_selector(user_agent: str) -> Optional[str]:
    st.header("Query setup", divider="violet")
    st.subheader("1. Institution", divider="blue")

    search_query = st.text_input(
        "Search by institution name first", placeholder="Europa-Universität Viadrina"
    )
    search_results: Optional[List[dict]] = None
    if st.button("Search ROR registry"):
        if not search_query.strip():
            st.warning("Please provide a search query.")
        else:
            with st.spinner("Searching institutions…"):
                try:
                    search_results = search_institutions_by_name(search_query.strip(), user_agent=user_agent)
                except requests.HTTPError as exc:
                    st.error(f"ROR search failed: {exc}")
                    search_results = []
                except requests.RequestException as exc:
                    st.error(f"ROR search error: {exc}")
                    search_results = []
            if search_results:
                options = {
                    f"{item.get('display_name', '—')} ({(item.get('country_code') or '').upper()}) — {item.get('ror', '—')}":
                    item.get("ror")
                    for item in search_results
                }
                choice = st.radio(
                    "Matches",
                    options=list(options.keys()),
                    key="institution_choice",
                )
                selected = options.get(choice)
                if selected:
                    st.session_state["selected_ror"] = selected
                    return selected
            else:
                st.info("No matches found.")

    ror_input = st.text_input(
        "…or enter a ROR URL directly (e.g., https://ror.org/02msan859)",
        value=st.session_state.get("selected_ror", ""),
    )

    if ror_input and is_ror_url(ror_input):
        st.session_state["selected_ror"] = ror_input.strip()
        return ror_input.strip()
    return st.session_state.get("selected_ror")


def render_publication_type_selector() -> Optional[str]:
    st.subheader("2. Publication type", divider="green")
    types = [
        "article",
        "book",
        "book-chapter",
        "proceedings-article",
        "proceedings",
        "reference-entry",
        "report",
        "dissertation",
        "dataset",
        "review",
        "editorial",
        "letter",
        "standard",
        "other",
    ]
    selected = st.selectbox("Filter by type", ["All"] + types)
    return None if selected == "All" else selected


def render_model_selector() -> str:
    st.subheader("3. SDG classifier", divider="yellow")
    labels = [f"{name} — {desc}" for name, desc in AURORA_MODELS]
    default_index = next((i for i, (name, _) in enumerate(AURORA_MODELS) if name == "aurora-sdg-multi"), 0)
    selection = st.selectbox("Choose a model", labels, index=default_index)
    return AURORA_MODELS[labels.index(selection)][0]


def render_advanced_options(
    semantic_key_from_secret: Optional[str],
    default_from_secret: Optional[str],
) -> Tuple[str, str, Optional[int]]:
    st.subheader("4. Advanced options", divider="orange")
    today = datetime.today().date().replace(day=1)
    start_str = default_from_secret or "2023-01-01"
    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date().replace(day=1)
    except ValueError:
        start_date = date(2023, 1, 1)
    months: List[date] = []
    year = start_date.year
    month = start_date.month
    while year < today.year or (year == today.year and month <= today.month):
        months.append(date(year, month, 1))
        month += 1
        if month > 12:
            month = 1
            year += 1
    if not months:
        months = [today]
    labels = [dt.strftime("%B %Y") for dt in months]
    label_to_date = dict(zip(labels, months))
    range_selection = st.select_slider(
        "Select publication period",
        options=labels,
        value=(labels[0], labels[-1]) if len(labels) > 1 else (labels[0], labels[0]),
        format_func=lambda label: label,
    )
    start_label, end_label = range_selection
    start_index = labels.index(start_label)
    end_index = labels.index(end_label)
    if start_index > end_index:
        start_label, end_label = end_label, start_label
    from_date = label_to_date[start_label]
    to_date = label_to_date[end_label]
    st.caption(f"Including works published from {from_date:%B %Y} through {to_date:%B %Y}.")
    limit_value = st.number_input(
        "Limit to first N records (0 = no limit)",
        min_value=0,
        value=0,
        step=50,
        help="Use to test the workflow without downloading everything.",
    )
    if semantic_key_from_secret:
        st.caption("Semantic Scholar API key loaded from secrets.toml.")
    else:
        st.info(
            "Add `semantic_scholar_api_key` to .streamlit/secrets.toml to fetch abstracts "
            "from Semantic Scholar when OpenAlex lacks them."
        )
    return from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"), (limit_value or None)


def main():
    st.set_page_config(page_title="Aurora SDG Publication Classifier", layout="wide")
    st.title("Aurora SDG Publication Classifier")
    st.caption(
        "Fetch publications for a ROR institution, classify them with Aurora SDG models, and export a CSV ready for analysis."
    )

    user_agent, has_user_agent_secret = resolve_user_agent()
    if not has_user_agent_secret:
        st.text_input(
            "HTTP User-Agent (set via secrets.toml)",
            value=user_agent,
            disabled=True,
        )
        st.warning(
            "Set `http_user_agent` in .streamlit/secrets.toml with your contact email "
            "for better API treatment."
        )

    ror_url = render_institution_selector(user_agent)
    publication_type = render_publication_type_selector()
    model = render_model_selector()
    semantic_scholar_key = resolve_semantic_scholar_key()
    default_from_date = get_secret_text(SECRET_DEFAULT_START)
    from_date_str, to_date_str, limit_rows = render_advanced_options(
        semantic_scholar_key,
        default_from_date,
    )

    if not ror_url:
        st.info("Provide a ROR URL or pick an institution to continue.")
        return

    if not is_ror_url(ror_url):
        st.error("The ROR value must look like https://ror.org/XXXXXXXXX.")
        return
    
    st.header("Run query and download results", divider="red")

    current_params = {
        "ror": ror_url,
        "type": publication_type,
        "model": model,
        "from": from_date_str,
        "to": to_date_str,
        "limit": limit_rows,
    }
    result_payload = st.session_state.get(RESULT_SESSION_KEY)

    run_button = st.button("Fetch works and build CSV", type="primary")
    if run_button:
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def progress_callback(done: int, expected: Optional[int], limit: Optional[int]):
            target = limit or expected
            fraction = 0.0
            if target:
                fraction = min(done / target, 1.0)
            progress_bar.progress(fraction)
            if expected:
                progress_text.text(f"Processed {done:,} of {expected:,} works")
            elif limit:
                progress_text.text(f"Processed {done:,} of {limit:,} requested works")
            else:
                progress_text.text(f"Processed {done:,} works")

        filename = build_output_filename(
            ror_url,
            publication_type,
            model,
            from_date_str,
            to_date_str,
            limit_rows,
        )

        with st.spinner("Contacting OpenAlex and Aurora APIs…"):
            try:
                csv_bytes, stats = fetch_to_csv_with_sdg(
                    ror_url=ror_url,
                    from_date_str=from_date_str,
                    to_date_str=to_date_str,
                    wtype=publication_type,
                    model=model,
                    limit_rows=limit_rows,
                    user_agent=user_agent.strip() or DEFAULT_USER_AGENT,
                    semantic_scholar_key=semantic_scholar_key,
                    progress_callback=progress_callback,
                )
            except requests.HTTPError as exc:
                progress_bar.empty()
                progress_text.empty()
                st.error(f"Request failed: {exc}")
                return
            except requests.RequestException as exc:
                progress_bar.empty()
                progress_text.empty()
                st.error(f"Network error: {exc}")
                return

        progress_bar.empty()
        progress_text.empty()
        result_payload = {
            "csv_bytes": csv_bytes,
            "stats": stats,
            "filename": filename,
            "params": current_params,
        }
        st.session_state[RESULT_SESSION_KEY] = result_payload
        st.session_state.pop("preview_focus_index", None)
        st.session_state.pop("preview_focus_mask", None)
    elif not result_payload:
        st.info("Click the button above to fetch publications.")
        return

    csv_bytes = result_payload["csv_bytes"]
    stats = result_payload["stats"]
    filename = result_payload["filename"]

    st.success(
        f"Wrote {stats['rows']:,} rows. "
        f"OpenAlex missing abstracts: {stats['openalex_abstract_missing']:,}; "
        f"retrieved from Semantic Scholar: {stats['semantic_scholar_abstracts']:,}."
    )
    try:
        csv_text = csv_bytes.decode("utf-8")
    except UnicodeDecodeError:
        csv_text = csv_bytes.decode("utf-8", errors="ignore")
    all_rows = list(csv.DictReader(io.StringIO(csv_text)))
    preview_rows = build_preview_rows(all_rows, PREVIEW_COLUMNS, limit=25)
    selected_index = st.session_state.get("preview_focus_index")
    focus_mask = st.session_state.get("preview_focus_mask")
    if not isinstance(focus_mask, list) or len(focus_mask) != len(preview_rows):
        focus_mask = [False] * len(preview_rows)
    if isinstance(selected_index, int) and 0 <= selected_index < len(focus_mask):
        focus_mask = [idx == selected_index for idx in range(len(focus_mask))]
    chart_rows: List[Dict[str, str]] = all_rows
    selected_title: Optional[str] = None
    if preview_rows:
        st.subheader("Preview")
        st.markdown(RADIO_CHECKBOX_CSS, unsafe_allow_html=True)
        focus_keys = [f"{FOCUS_ROW_KEY_PREFIX}{idx}" for idx in range(len(preview_rows))]
        for idx, key in enumerate(focus_keys):
            if key not in st.session_state:
                st.session_state[key] = focus_mask[idx]
        current_mask = [bool(st.session_state[key]) for key in focus_keys]
        if current_mask != focus_mask:
            selected_index = next((idx for idx, flag in enumerate(current_mask) if flag), None)
            if selected_index is not None:
                focus_mask = [idx == selected_index for idx in range(len(focus_keys))]
            else:
                focus_mask = [False] * len(focus_keys)

        for idx, key in enumerate(focus_keys):
            st.session_state[key] = focus_mask[idx]
        st.session_state["preview_focus_mask"] = focus_mask
        st.session_state["preview_focus_index"] = selected_index

        header_cols = st.columns(len(PREVIEW_COLUMNS) + 1)
        header_cols[0].markdown("**Focus**")
        for col_idx, col_name in enumerate(PREVIEW_COLUMNS):
            header_cols[col_idx + 1].markdown(f"**{col_name.replace('_', ' ').title()}**")

        for row_idx, row in enumerate(preview_rows):
            row_cols = st.columns(len(PREVIEW_COLUMNS) + 1)
            row_cols[0].checkbox("", key=focus_keys[row_idx])
            for col_idx, col_name in enumerate(PREVIEW_COLUMNS):
                value = row.get(col_name)
                row_cols[col_idx + 1].write(value if value not in (None, "") else "—")

        if selected_index is not None and 0 <= selected_index < len(all_rows):
            chart_rows = [all_rows[selected_index]]
            selected_title = all_rows[selected_index].get("title") or all_rows[selected_index].get("display_name")
        else:
            chart_rows = all_rows
            selected_title = None

        st.caption("Click the Focus circle to inspect SDGs for a single publication; click again to reset.")
    else:
        st.info("No preview rows available.")

    chart_data = aggregate_sdg_counts(chart_rows)
    st.subheader("SDG distribution")
    chart_title = "selected publication" if len(chart_rows) == 1 else "all publications"
    if chart_title == "selected publication" and selected_title:
        chart_title = f"selected publication '{selected_title}'"
    render_sdg_pie_chart(chart_data, f"SDGs in {chart_title}")

    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )

    st.caption(
        "Tip: rerun with model='skip' if you only need metadata, or use a smaller limit first to check the configuration."
    )


if __name__ == "__main__":
    main()
