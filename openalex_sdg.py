"""Utility functions for fetching OpenAlex works and running Aurora SDG classification."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import requests

# ------------------ CONFIG ------------------
BASE_WORKS = "https://api.openalex.org/works"
BASE_INSTITUTIONS = "https://api.openalex.org/institutions"
AURORA_BASE = "https://aurora-sdg.labs.vu.nl/classifier/classify"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract"

PER_PAGE = 200  # OpenAlex max
DEFAULT_FROM_DATE = "2023-01-01"
DEFAULT_USER_AGENT = "OpenAlex+Aurora SDG fetcher (mailto:you@example.com)"

AURORA_MODELS = [
    ("aurora-sdg", "Aurora SDG mBERT (single-label, slower)"),
    ("aurora-sdg-multi", "Aurora SDG multi-label mBERT (fast)"),
    ("elsevier-sdg-multi", "Elsevier SDG multi-label mBERT (fast)"),
    ("osdg", "OSDG model (multi-label, 15 languages)"),
    ("skip", "Skip SDG classification (no Aurora API calls)"),
]

MIN_WORDS_BY_MODEL = {"osdg": 50}
# --------------------------------------------

ProgressHook = Optional[Callable[[int, Optional[int], str], None]]


@dataclass
class FetchStats:
    total_expected: Optional[int]
    total_processed: int
    openalex_abstract_missing: int
    ss_abstract_retrieved: int


def too_short_for_model(model: str, text: str) -> bool:
    need = MIN_WORDS_BY_MODEL.get(model, 0)
    return need > 0 and len((text or "").split()) < need


def is_ror_url(value: str) -> bool:
    return bool(re.match(r"^https?://ror\.org/[0-9a-z]{9}$", value.strip(), flags=re.I))


def search_institutions_by_name(
    name: str, user_agent: str = DEFAULT_USER_AGENT, limit: int = 10
) -> List[dict]:
    params = {"search": name, "per-page": limit}
    headers = {"User-Agent": user_agent}
    response = requests.get(BASE_INSTITUTIONS, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json().get("results", [])


def reconstruct_abstract(inv: Optional[dict]) -> str:
    """Rebuild abstract text from OpenAlex 'abstract_inverted_index' or '_v3'."""
    if not inv or not isinstance(inv, dict):
        return ""
    max_pos = -1
    for positions in inv.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return ""
    tokens_by_pos = [""] * (max_pos + 1)
    for token, positions in inv.items():
        for p in positions:
            if p < 0:
                continue
            if p >= len(tokens_by_pos):
                tokens_by_pos.extend([""] * (p - len(tokens_by_pos) + 1))
            tokens_by_pos[p] = (
                (tokens_by_pos[p] + " " + token).strip() if tokens_by_pos[p] else token
            )
    return " ".join(tokens_by_pos).strip()


def flatten_authors_and_institutions(authorships: Sequence[dict]) -> Tuple[str, str]:
    if not authorships:
        return "", ""
    author_names: List[str] = []
    all_insts: List[str] = []
    for author_entry in authorships:
        author = (author_entry.get("author") or {}).get("display_name") or ""
        if author:
            author_names.append(author)
        for inst in author_entry.get("institutions") or []:
            name = inst.get("display_name") or ""
            if name:
                all_insts.append(name)
    seen = set()
    inst_names: List[str] = []
    for name in all_insts:
        if name not in seen:
            seen.add(name)
            inst_names.append(name)
    return "; ".join(author_names), "; ".join(inst_names)


def make_filter(ror_url: str, from_date: str, work_type: Optional[str]) -> str:
    parts = [
        f"institutions.ror:{ror_url}",
        "is_paratext:false",
        f"from_publication_date:{from_date}",
    ]
    if work_type:
        parts.append(f"type:{work_type}")
    return ",".join(parts)


def classify_text_aurora(
    model: str,
    text: str,
    session: requests.Session,
    user_agent: str = DEFAULT_USER_AGENT,
    retries: int = 3,
    pause: float = 0.4,
) -> Tuple[Optional[dict], str]:
    """
    Calls Aurora SDG classifier via POST, returns (json or None, note string).
    note is "" on success, or an explanation like "http_error:429" / "empty json".
    """
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
            resp = session.post(
                url, headers=headers, data=json.dumps(payload), timeout=60
            )
            if resp.status_code == 429:
                time.sleep(pause * attempt + 0.5)
                continue
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return None, "empty json"
            return data, ""
        except requests.RequestException as exc:
            if attempt == retries:
                code = getattr(getattr(exc, "response", None), "status_code", None)
                return None, f"http_error:{code}"
            time.sleep(pause * attempt)
    return None, "unknown"


def get_abstract_from_semantic_scholar(
    doi: str,
    session: Optional[requests.Session] = None,
    api_key: Optional[str] = None,
    retries: int = 3,
    pause: float = 0.5,
) -> Optional[str]:
    """
    Fetches abstract from Semantic Scholar using DOI.
    Returns abstract string or None on failure.
    """
    if not doi:
        return None
    cleaned_doi = doi.replace("https://doi.org/", "")
    url = SEMANTIC_SCHOLAR_API.format(doi=cleaned_doi)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    requester = session or requests
    for attempt in range(1, retries + 1):
        try:
            resp = requester.get(url, headers=headers, timeout=20)
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
    """
    Returns '\\n'-joined strings like "84% SDG 10 (Reduced inequalities)".
    Handles multiple API variants.
    """

    def fmt_line(score, code, name):
        code_str = str(code).strip()
        name_str = (name or f"SDG {code_str}").strip()
        if name_str.lower().startswith("sdg "):
            return "{pct:.0f}% {label}".format(pct=score * 100, label=name_str)
        return "{pct:.0f}% SDG {code} ({name})".format(
            pct=score * 100, code=code_str, name=name_str
        )

    if not sdg_json:
        return ""

    items: List[Tuple[float, str, str]] = []

    preds = sdg_json.get("predictions")
    if isinstance(preds, list) and preds:
        for entry in preds:
            sdg = entry.get("sdg") or {}
            code = sdg.get("code")
            name = sdg.get("name")
            score = entry.get("prediction")
            if code is None or score is None:
                continue
            try:
                items.append((float(score), code, name))
            except (TypeError, ValueError):
                continue

    if not items and isinstance(sdg_json, list):
        for entry in sdg_json:
            label = entry.get("label")
            score = entry.get("score")
            if label is None or score is None:
                continue
            match = re.search(r"\bSDG\s*(\d+)", str(label), flags=re.I)
            code = match.group(1) if match else ""
            items.append((float(score), code, str(label)))

    if (
        not items
        and isinstance(sdg_json, dict)
        and "labels" in sdg_json
        and "scores" in sdg_json
    ):
        labels = sdg_json.get("labels") or []
        scores = sdg_json.get("scores") or []
        for label, score in zip(labels, scores):
            match = re.search(r"\bSDG\s*(\d+)", str(label), flags=re.I)
            code = match.group(1) if match else ""
            items.append((float(score), code, str(label)))

    if not items and isinstance(sdg_json, dict):
        numeric_keys = [key for key in sdg_json.keys() if str(key).isdigit()]
        if numeric_keys:
            for key in numeric_keys:
                try:
                    items.append((float(sdg_json[key]), key, None))
                except (TypeError, ValueError):
                    continue

    if (
        not items
        and isinstance(sdg_json, dict)
        and isinstance(sdg_json.get("results"), list)
    ):
        for entry in sdg_json["results"]:
            code = entry.get("sdg") or entry.get("code")
            score = entry.get("score") or entry.get("prediction")
            name = entry.get("name") or entry.get("label")
            if code is None or score is None:
                continue
            items.append((float(score), code, name))

    if not items:
        return ""

    items.sort(key=lambda item: item[0], reverse=True)
    return "\n".join(fmt_line(score, code, name) for score, code, name in items)


def sanitize_filename(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = re.sub(r"[^\w\-\.]+", "_", value, flags=re.UNICODE)
    return value.strip("_")


def fetch_works_with_sdg(
    ror_url: str,
    from_date: str,
    work_type: Optional[str],
    model: str,
    limit_rows: Optional[int] = None,
    user_agent: str = DEFAULT_USER_AGENT,
    semantic_scholar_api_key: Optional[str] = None,
    progress_callback: ProgressHook = None,
) -> Tuple[List[Dict[str, object]], FetchStats]:
    params = {
        "filter": make_filter(ror_url, from_date, work_type),
        "select": "id,display_name,title,publication_date,doi,abstract_inverted_index,abstract_inverted_index_v3,type,language,open_access,authorships",
        "per-page": PER_PAGE,
        "cursor": "*",
    }
    headers = {"User-Agent": user_agent}

    stats = FetchStats(
        total_expected=None,
        total_processed=0,
        openalex_abstract_missing=0,
        ss_abstract_retrieved=0,
    )
    rows: List[Dict[str, object]] = []

    def emit_progress(message: str = "") -> None:
        if progress_callback:
            progress_callback(stats.total_processed, stats.total_expected, message)

    emit_progress("Starting fetch")

    with requests.Session() as session:
        # First page to establish total size
        response = session.get(BASE_WORKS, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        stats.total_expected = (
            (data.get("meta") or {}).get("count")
            if isinstance(data.get("meta"), dict)
            else None
        )
        results = data.get("results", []) or []
        next_cursor = (data.get("meta") or {}).get("next_cursor")

        def process_record(work: dict) -> None:
            nonlocal session
            openalex_id = work.get("id", "")
            title = work.get("display_name") or work.get("title") or ""
            publication_date = work.get("publication_date") or ""
            doi = work.get("doi") or ""
            work_type_val = work.get("type") or ""
            language = work.get("language") or ""
            open_access = work.get("open_access") or {}
            is_oa = open_access.get("is_oa")
            oa_status = open_access.get("oa_status") or ""
            authorships = work.get("authorships") or []
            authors_str, insts_str = flatten_authors_and_institutions(authorships)

            inv_v3 = work.get("abstract_inverted_index_v3")
            inv = work.get("abstract_inverted_index")
            abstract_text = reconstruct_abstract(inv_v3 if inv_v3 else inv)

            if not abstract_text:
                stats.openalex_abstract_missing += 1
                if doi:
                    ss_abstract = get_abstract_from_semantic_scholar(
                        doi, session=session, api_key=semantic_scholar_api_key
                    )
                    if ss_abstract:
                        abstract_text = ss_abstract
                        stats.ss_abstract_retrieved += 1

            text_for_sdg = abstract_text if abstract_text else title
            sdg_json: Optional[dict] = None
            sdg_note = ""
            sdg_formatted = ""

            if model == "skip":
                sdg_note = "skipped: user selected 'skip'"
            else:
                if model == "osdg" and too_short_for_model(model, text_for_sdg):
                    sdg_note = "skipped: osdg requires >=50 words"
                else:
                    sdg_json, sdg_note = classify_text_aurora(
                        model, text_for_sdg, session=session, user_agent=user_agent
                    )
                    sdg_formatted = (
                        format_sdg_predictions(sdg_json) if sdg_json is not None else ""
                    )
                    time.sleep(0.12)

            sdg_raw_str = (
                json.dumps(sdg_json, ensure_ascii=False) if sdg_json is not None else ""
            )

            rows.append(
                {
                    "openalex_id": openalex_id,
                    "title": title,
                    "publication_date": publication_date,
                    "doi": doi,
                    "type": work_type_val,
                    "language": language,
                    "is_oa": is_oa,
                    "oa_status": oa_status,
                    "authors": authors_str,
                    "institutions": insts_str,
                    "abstract": abstract_text,
                    "sdg_model": model,
                    "sdg_response": sdg_raw_str,
                    "sdg_formatted": sdg_formatted,
                    "sdg_note": sdg_note,
                }
            )
            stats.total_processed += 1
            emit_progress(f"Processed {title[:64] or openalex_id}")

        for work in results:
            if limit_rows is not None and stats.total_processed >= limit_rows:
                next_cursor = None
                break
            process_record(work)

        params["cursor"] = next_cursor

        while next_cursor:
            if limit_rows is not None and stats.total_processed >= limit_rows:
                break
            response = session.get(BASE_WORKS, params=params, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", []) or []
            next_cursor = (data.get("meta") or {}).get("next_cursor")
            for work in results:
                if limit_rows is not None and stats.total_processed >= limit_rows:
                    next_cursor = None
                    break
                process_record(work)
            params["cursor"] = next_cursor
            time.sleep(0.2)

    emit_progress("Completed")
    return rows, stats


__all__ = [
    "AURORA_MODELS",
    "DEFAULT_FROM_DATE",
    "DEFAULT_USER_AGENT",
    "FetchStats",
    "SEMANTIC_SCHOLAR_API",
    "fetch_works_with_sdg",
    "format_sdg_predictions",
    "is_ror_url",
    "sanitize_filename",
    "search_institutions_by_name",
]
