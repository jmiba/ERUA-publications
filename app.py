import csv
import io
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

import altair as alt
import pandas as pd
import requests
import streamlit as st
import tomllib

from openalex_sdg import (
    AURORA_MODELS,
    DEFAULT_USER_AGENT,
    FetchStats,
    fetch_works_with_sdg,
    is_ror_url,
    sanitize_filename,
    search_institutions_by_name,
)

SECRET_HTTP_USER_AGENT = "http_user_agent"
SECRET_SEMANTIC_SCHOLAR_KEY = "semantic_scholar_api_key"
SECRET_DEFAULT_START = "advanced_options.default_from_date"
_SECRETS: Dict[str, Any] = {}
PREVIEW_COLUMNS = [
    "openalex_id",
    "authors",
    "title",
    "publication_date",
    "type",
    "doi",
    "institutions",
]
PREVIEW_PAGE_SIZE = 25
CSV_FIELDNAMES = [
    "openalex_id",
    "authors",
    "title",
    "publication_date",
    "doi",
    "type",
    "language",
    "is_oa",
    "oa_status",
    "institutions",
    "abstract",
    "sdg_model",
    "sdg_response",
    "sdg_formatted",
    "sdg_note",
]
RESULT_SESSION_KEY = "fetch_result"
SDG_THRESHOLD_PERCENT = 3.0
RADIO_CHECKBOX_CSS = """
<style>
div[data-testid="stDataFrame"] div[role="checkbox"] input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 1rem;
    height: 1rem;
    border: 2px solid var(--primary-color);
    border-radius: 50%;
    position: relative;
    cursor: pointer;
}
div[data-testid="stDataFrame"] div[role="checkbox"] input[type="checkbox"]:checked {
    background-color: var(--primary-color);
}
div[data-testid="stDataFrame"] div[role="checkbox"] input[type="checkbox"]:checked::after {
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


def build_preview_rows(
    rows: List[Dict[str, Any]],
    columns: List[str],
    limit: int = 20,
    offset: int = 0,
) -> List[Dict[str, str]]:
    preview: List[Dict[str, str]] = []
    subset = rows[offset : offset + max(limit, 0)]
    for row in subset:
        preview.append({col: str(row.get(col, "") or "") for col in columns})
    return preview


def abbreviate_authors(value: str) -> str:
    if not value:
        return ""
    authors = [part.strip() for part in value.split(";") if part.strip()]
    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0]
    return f"{authors[0]} et al."


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


def aggregate_sdg_counts(rows: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
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
        .mark_arc(innerRadius=70)
        .encode(
            theta="Value",
            color=alt.Color("SDG")
            .legend(columns=1, labelLimit=300, titleLimit=300, title="Sustainable Development Goals"),
            tooltip=[
                alt.Tooltip("SDG", title="Sustainable Development Goal"),
                alt.Tooltip("Value", format=".1f", title="Concordance in %"),
            ],
        )
        .properties(width=1650, height=450, title=title)
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


def rows_to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})
    buffer.seek(0)
    return buffer.getvalue().encode("utf-8")


def _excel_col_name(idx: int) -> str:
    name = ""
    while idx >= 0:
        idx, remainder = divmod(idx, 26)
        name = chr(65 + remainder) + name
        idx -= 1
    return name


def rows_to_excel_bytes(rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> bytes:
    columns = columns or CSV_FIELDNAMES
    if not columns:
        columns = list({key for row in rows for key in row.keys()})
    sheet_rows: List[str] = []
    row_index = 1
    header_cells = []
    for col_idx, col_name in enumerate(columns):
        cell_ref = f"{_excel_col_name(col_idx)}{row_index}"
        header_cells.append(
            f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(str(col_name))}</t></is></c>'
        )
    sheet_rows.append(f'<row r="{row_index}">{"".join(header_cells)}</row>')
    for row in rows:
        row_index += 1
        cells = []
        for col_idx, col_name in enumerate(columns):
            value = row.get(col_name, "")
            text = "" if value is None else str(value)
            cell_ref = f"{_excel_col_name(col_idx)}{row_index}"
            cells.append(
                f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'
            )
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        "</worksheet>"
    )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )

    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>'
    )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        "</Types>"
    )

    core_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{escape('OpenAlex Results')}</dc:title>"
        "</cp:coreProperties>"
    )

    app_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Streamlit</Application>"
        "</Properties>"
    )

    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("docProps/app.xml", app_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        zf.writestr("xl/styles.xml", styles_xml)
    buffer.seek(0)
    return buffer.getvalue()


def render_institution_selector(user_agent: str) -> Optional[str]:
    st.header("Query setup", divider="rainbow")
    st.subheader("1. Institution", divider="violet")

    with st.form("institution_search_form", clear_on_submit=False):
        search_query = st.text_input(
            "Search by institution name first", placeholder="Europa-Universität Viadrina"
        )
        submitted = st.form_submit_button("Search ROR registry", type="primary")
    search_results: Optional[List[dict]] = st.session_state.get("institution_search_results")
    search_ran = st.session_state.get("institution_search_ran", False)
    if submitted:
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
            st.session_state["institution_search_results"] = search_results or []
            st.session_state["institution_search_ran"] = True
    search_results = st.session_state.get("institution_search_results")
    search_ran = st.session_state.get("institution_search_ran", False)
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
    elif search_ran:
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
    st.subheader("2. Publication type", divider="blue")
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
    st.subheader("3. SDG classifier", divider="green")
    desc_only = [desc for _, desc in AURORA_MODELS]
    default_index = next((i for i, (name, _) in enumerate(AURORA_MODELS) if name == "aurora-sdg-multi"), 0)
    selection = st.selectbox("Choose a model", desc_only, index=default_index)
    return AURORA_MODELS[desc_only.index(selection)][0]


def render_advanced_options(
    semantic_key_from_secret: Optional[str],
    default_from_secret: Optional[str],
) -> Tuple[str, str, Optional[int]]:
    st.subheader("4. Advanced options", divider="yellow")
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
    if not semantic_key_from_secret:
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
    
    st.header("Run query and download results", divider="rainbow")

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

        def progress_callback(done: int, expected: Optional[int], message: str):
            target = limit_rows or expected
            fraction = min(done / target, 1.0) if target else 0.0
            progress_bar.progress(fraction)
            if expected:
                status = f"Processed {done:,} of {expected:,} works"
            elif limit_rows:
                status = f"Processed {done:,} of {limit_rows:,} requested works"
            else:
                status = f"Processed {done:,} works"
            if message:
                status = f"{status} – {message}"
            progress_text.text(status)

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
                rows, stats = fetch_works_with_sdg(
                    ror_url=ror_url,
                    from_date=from_date_str,
                    work_type=publication_type,
                    model=model,
                    to_date=to_date_str,
                    limit_rows=limit_rows,
                    user_agent=user_agent.strip() or DEFAULT_USER_AGENT,
                    semantic_scholar_api_key=semantic_scholar_key,
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
        csv_bytes = rows_to_csv_bytes(rows)
        result_payload = {
            "csv_bytes": csv_bytes,
            "rows": rows,
            "stats": stats,
            "filename": filename,
            "params": current_params,
        }
        st.session_state[RESULT_SESSION_KEY] = result_payload
        st.session_state.pop("preview_focus_index", None)
        st.session_state["preview_page"] = 1
    elif not result_payload:
        st.info("Click the button above to fetch publications.")
        return

    csv_bytes: bytes = result_payload["csv_bytes"]
    stats: FetchStats = result_payload["stats"]
    filename: str = result_payload["filename"]
    rows: Optional[List[Dict[str, Any]]] = result_payload.get("rows")

    st.success(
        f"Wrote {stats.total_processed:,} rows. "
        f"OpenAlex missing abstracts: {stats.openalex_abstract_missing:,}; "
        f"retrieved from Semantic Scholar: {stats.ss_abstract_retrieved:,}."
    )
    if rows is None:
        try:
            csv_text = csv_bytes.decode("utf-8")
        except UnicodeDecodeError:
            csv_text = csv_bytes.decode("utf-8", errors="ignore")
        all_rows = list(csv.DictReader(io.StringIO(csv_text)))
    else:
        all_rows = rows
    total_rows = len(all_rows)
    selected_index = st.session_state.get("preview_focus_index")
    chart_rows: List[Dict[str, Any]] = all_rows
    selected_title: Optional[str] = None
    if total_rows > 0:
        st.write("")  # spacing
        st.subheader("Preview", divider="orange")
        st.markdown(RADIO_CHECKBOX_CSS, unsafe_allow_html=True)
        total_pages = max(1, math.ceil(total_rows / PREVIEW_PAGE_SIZE))
        st.session_state.setdefault("preview_page", 1)
        current_page = min(max(1, st.session_state["preview_page"]), total_pages)
        if total_pages > 1:
            first_col, prev_col, info_col, next_col, last_col = st.columns([1, 1, 2, 1, 1])

            def set_page(target: int):
                st.session_state["preview_page"] = max(1, min(total_pages, target))

            if first_col.button("⏮ First", disabled=current_page == 1):
                set_page(1)
            if prev_col.button("◀ Previous", disabled=current_page == 1):
                set_page(current_page - 1)
            if next_col.button("Next ▶", disabled=current_page == total_pages):
                set_page(current_page + 1)
            if last_col.button("Last ⏭", disabled=current_page == total_pages):
                set_page(total_pages)
            current_page = st.session_state["preview_page"]
            info_col.markdown(f"Page **{current_page} / {total_pages}**")
        else:
            current_page = 1
        start_index = (current_page - 1) * PREVIEW_PAGE_SIZE
        preview_rows = build_preview_rows(
            all_rows,
            PREVIEW_COLUMNS,
            limit=PREVIEW_PAGE_SIZE,
            offset=start_index,
        )
        visible_indices = list(range(start_index, start_index + len(preview_rows)))
        preview_df = pd.DataFrame(preview_rows)
        if "authors" in preview_df.columns:
            preview_df["authors"] = preview_df["authors"].apply(abbreviate_authors)
        preview_df.insert(0, "#", range(start_index + 1, start_index + 1 + len(preview_df)))
        rows_in_page = len(preview_df)
        table_height = 980 if rows_in_page >= PREVIEW_PAGE_SIZE else max(200, rows_in_page * 35 + 120)
        column_configs = {}
        for column in preview_df.columns:
            if column == "#":
                column_configs[column] = st.column_config.NumberColumn(
                    "#", help="Row number in this page", width="small"
                )
            elif column == "openalex_id":
                column_configs[column] = st.column_config.LinkColumn(
                    "OpenAlex ID",
                    help="Open the work in OpenAlex",
                    display_text=r"(?:https?://openalex\.org/)?(.+)",
                )
            elif column.lower() == "doi":
                column_configs[column] = st.column_config.LinkColumn(
                    "DOI",
                    help="Open this DOI in a new tab",
                    display_text=r"(?:https?://(?:dx\.)?doi\.org/)?(.+)",
                )
            else:
                column_configs[column] = st.column_config.TextColumn(column.replace("_", " ").title())
        st.data_editor(
            preview_df,
            hide_index=True,
            disabled=True,
            height=table_height,
            width="stretch",
            column_config=column_configs,
        )
        st.caption(f"Showing page {current_page} of {total_pages}.")
        dropdown_options = ["0 — All publications"]
        for idx, row in enumerate(all_rows):
            title_preview = (row.get("title") or row.get("display_name") or "(no title)")[:80]
            authors_preview = abbreviate_authors(row.get("authors") or "")
            if authors_preview:
                dropdown_options.append(f"{idx + 1} — {authors_preview}, {title_preview}")
            else:
                dropdown_options.append(f"{idx + 1} — {title_preview}")
        dropdown_default = (
            0 if selected_index is None else min(max(0, selected_index + 1), total_rows)
        )
        selected_option = st.selectbox(
            "Focus publication",
            options=list(range(len(dropdown_options))),
            format_func=lambda idx: dropdown_options[idx],
            index=dropdown_default,
        )
        selected_index = selected_option - 1 if selected_option > 0 else None
        st.session_state["preview_focus_index"] = selected_index

        if selected_index is not None and 0 <= selected_index < len(all_rows):
            chart_rows = [all_rows[selected_index]]
            row_info = all_rows[selected_index]
            author_info = abbreviate_authors(row_info.get("authors") or "")
            title_info = row_info.get("title") or row_info.get("display_name") or "(no title)"
            if author_info:
                selected_title = f"{author_info}, {title_info}"
            else:
                selected_title = title_info
        else:
            chart_rows = all_rows
            selected_index = None
            selected_title = None
        st.caption("Select a publication above (0 = All).")
    else:
        st.session_state["preview_page"] = 1
        st.info("No preview rows available.")

    chart_data = aggregate_sdg_counts(chart_rows)
    st.write("")
    st.subheader("SDG distribution", divider="red")
    chart_title = "selected publication" if len(chart_rows) == 1 else "all publications"
    if chart_title == "selected publication" and selected_title:
        chart_title = f"selected publication ({selected_title})"
    render_sdg_pie_chart(chart_data, f"SDGs in {chart_title}")

    st.write("")
    st.subheader("Download data sets", divider="gray")
    export_rows = rows or all_rows
    excel_bytes = rows_to_excel_bytes(export_rows, CSV_FIELDNAMES) if export_rows else None
    if excel_bytes:
        st.download_button(
            "Download Excel",
            data=excel_bytes,
            file_name=filename.replace(".csv", ".xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
