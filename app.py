import os, re, time, tempfile
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import streamlit as st

# =========================
# Streamlit page
# =========================
st.set_page_config(page_title="BSE — Company Update (Tabular Feed)", layout="wide")
st.title("📑 BSE — Company Update (Tabular Feed)")
st.caption("Lists all 'Company Update' announcements in a table (no summarization). Includes best-effort mapping to SEBI regulations.")

# =========================
# Small utilities
# =========================
_ILLEGAL_RX = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
def _clean(s: str) -> str:
    return _ILLEGAL_RX.sub('', s) if isinstance(s, str) else s

def _first_col(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _slug(s: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s or "")).strip("_")
    return (s[:maxlen] if len(s) > maxlen else s) or "file"

def _fmt(d: datetime.date) -> str:
    return d.strftime("%Y%m%d")

def _candidate_urls(row):
    cands = []
    att = str(row.get("ATTACHMENTNAME") or "").strip()
    if att:
        cands += [
            f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{att}",
            f"https://www.bseindia.com/xml-data/corpfiling/Attach/{att}",
            f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{att}",
        ]
    ns = str(row.get("NSURL") or "").strip()
    if ".pdf" in ns.lower():
        cands.append(ns if ns.lower().startswith("http") else "https://www.bseindia.com/" + ns.lstrip("/"))
    seen, out = set(), []
    for u in cands:
        if u and u not in seen:
            out.append(u); seen.add(u)
    return out

def _download_head_pdf_url(urls, timeout=20):
    """Return first PDF URL that responds with HTTP 200 and looks like a PDF (lightweight check)."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": "https://www.bseindia.com/corporates/ann.html",
        "Accept-Language": "en-US,en;q=0.9",
    })
    for u in urls:
        try:
            r = s.head(u, timeout=timeout, allow_redirects=True)
            if r.status_code == 200:
                # Some BSE servers don't send content-type reliably; accept .pdf extension or 200 OK
                if u.lower().endswith(".pdf") or "pdf" in (r.headers.get("content-type","").lower()):
                    return u
        except Exception:
            continue
    return ""

# =========================
# BSE fetch — STRICT base
# =========================
def fetch_bse_announcements(start_yyyymmdd: str,
                            end_yyyymmdd: str,
                            verbose: bool = False,
                            request_timeout: int = 25) -> pd.DataFrame:
    """
    Fetches raw announcements for the given date range.
    We will later filter to Category='Company Update' ONLY (no subcategory filter).
    """
    assert len(start_yyyymmdd) == 8 and len(end_yyyymmdd) == 8
    assert start_yyyymmdd <= end_yyyymmdd
    base_page = "https://www.bseindia.com/corporates/ann.html"
    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": base_page,
        "X-Requested-With": "XMLHttpRequest",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })

    try:
        s.get(base_page, timeout=15)
    except Exception:
        pass

    variants = [
        {"subcategory": "-1", "strSearch": "P"},
        {"subcategory": "-1", "strSearch": ""},
        {"subcategory": "",   "strSearch": "P"},
        {"subcategory": "",   "strSearch": ""},
    ]

    all_rows = []
    for v in variants:
        params = {
            "pageno": 1, "strCat": "-1", "subcategory": v["subcategory"],
            "strPrevDate": start_yyyymmdd, "strToDate": end_yyyymmdd,
            "strSearch": v["strSearch"], "strscrip": "", "strType": "C",
        }
        rows, total, page = [], None, 1
        while True:
            r = s.get(url, params=params, timeout=request_timeout)
            ct = r.headers.get("content-type","")
            if "application/json" not in ct:
                if verbose:
                    st.warning(f"[variant {v}] non-JSON on page {page} (ct={ct}).")
                break
            data = r.json()
            table = data.get("Table") or []
            rows.extend(table)
            if total is None:
                try:
                    total = int((data.get("Table1") or [{}])[0].get("ROWCNT") or 0)
                except Exception:
                    total = None
            if not table:
                break
            params["pageno"] += 1; page += 1; time.sleep(0.2)
            if total and len(rows) >= total:
                break
        if rows:
            all_rows = rows
            break

    if not all_rows:
        return pd.DataFrame()

    all_keys = set()
    for r in all_rows:
        all_keys.update(r.keys())
    return pd.DataFrame(all_rows, columns=list(all_keys))

def filter_company_update_only(df_in: pd.DataFrame, category_filter="Company Update") -> pd.DataFrame:
    if df_in.empty:
        return df_in.copy()
    cat_col = _first_col(df_in, ["CATEGORYNAME","CATEGORY","NEWS_CAT","NEWSCATEGORY","NEWS_CATEGORY"])
    if not cat_col:
        return df_in.copy()
    df2 = df_in.copy()
    df2["_cat_norm"] = df2[cat_col].map(lambda x: _norm(x).lower())
    return df2.loc[df2["_cat_norm"] == _norm(category_filter).lower()].drop(columns=["_cat_norm"])

# =========================
# Regulation inference
# =========================
# Best-effort mapping using headline/subcategory heuristics.
# Note: BSE text is noisy; we map the common cases conservatively.
_RULES = [
    (r"(record date|book closure)", "Reg 42 (Record Date/Book Closure)"),
    (r"(dividend|interim dividend|final dividend)", "Reg 30/42 (Dividend / Corporate Action)"),
    (r"(board meeting outcome|outcome of board meeting|bm outcome)", "Reg 30 (Outcome of Board Meeting)"),
    (r"(intimation of board meeting|board meeting intimation)", "Reg 29 (Board Meeting Intimation)"),
    (r"(analyst|investor)\s+(meet|call|presentation)", "Reg 30 & Reg 46 (Analyst/Investor Meet/Presentation)"),
    (r"(press release|media release|newspaper|publication)", "Reg 30 (Press/Media)"),
    (r"(credit rating|reaffirmed rating|rating update)", "Reg 30 (Credit Rating)"),
    (r"(agm|egm|postal ballot)", "Reg 30 & Reg 44 (Shareholder Meeting)"),
    (r"(appointment|resignation).*(director|cfo|cs|ceo|kmp)", "Reg 30 (KMP/Director changes; Sch III Part A)"),
    (r"(preferential issue|qip|private placement|allotment of shares|warrants)", "Reg 30 (Capital Raise)"),
    (r"(buyback)", "SEBI Buyback Regulations & Reg 30"),
    (r"(esop|stock option|rsu|grant)", "SEBI SBEB Regs & Reg 30"),
    (r"(pledge|encumbrance|release of pledge)", "Reg 31 & Reg 30 (Encumbrance)"),
    (r"(acquisition|amalgamation|merger|scheme of arrangement|demerger|slump sale|joint venture)", "Reg 30 (M&A / Scheme; Sch III)"),
    (r"(trading window closure|window closure)", "SEBI (PIT) Regulations"),
    (r"(related party|rpt)", "Reg 23 & Reg 30 (RPT)"),
    (r"(change in statutory auditor|auditor resignation|auditor appointment)", "Reg 30 (Auditors)"),
    (r"(intimation.*record date)", "Reg 42 (Record Date)"),
]

def infer_regulation(headline: str, subcategory: str) -> str:
    text = f"{headline or ''} {subcategory or ''}".lower()
    for rx, label in _RULES:
        if re.search(rx, text):
            return label
    # If the exchange itself says "Disclosure under Reg 30"
    if "reg 30" in text or "regulation 30" in text:
        return "Reg 30 (Disclosure)"
    return "Reg (best-effort not obvious)"

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("⚙️ Controls")
    today = datetime.now().date()
    start_date = st.date_input("Start date", value=today - timedelta(days=1), max_value=today)
    end_date   = st.date_input("End date", value=today, max_value=today, min_value=start_date)
    max_items  = st.slider("Max announcements to fetch", 20, 1000, 200, step=20)
    run        = st.button("🚀 Fetch Company Update")

# =========================
# Run
# =========================
if run:
    start_str, end_str = _fmt(start_date), _fmt(end_date)

    with st.status("Fetching announcements…", expanded=True):
        df_raw = fetch_bse_announcements(start_str, end_str, verbose=False)
        st.write(f"Raw rows fetched: **{len(df_raw)}**")
        df_hits = filter_company_update_only(df_raw, category_filter="Company Update")
        st.write(f"'Company Update' rows: **{len(df_hits)}**")

    if df_hits.empty:
        st.warning("No 'Company Update' announcements found in this window.")
        st.stop()

    if len(df_hits) > max_items:
        df_hits = df_hits.head(max_items)

    # Identify common columns
    nm_col   = _first_col(df_hits, ["SLONGNAME","SNAME","SC_NAME","COMPANYNAME"]) or "SLONGNAME"
    sub_col  = _first_col(df_hits, ["SUBCATEGORYNAME","SUBCATEGORY","SUB_CATEGORY","NEWS_SUBCATEGORY"]) or "SUBCATEGORYNAME"
    date_col = _first_col(df_hits, ["NEWS_DT","DT","NEWSDATE","NEWS_DATE"]) or "NEWS_DT"
    head_col = _first_col(df_hits, ["HEADLINE","NEWS_SUB","SUBJECT","HEADING"]) or "HEADLINE"

    # Build rows with PDF URL and inferred regulation
    out_rows = []
    with st.status("Building table…", expanded=False):
        for _, r in df_hits.iterrows():
            company   = _clean(str(r.get(nm_col) or "").strip())
            dt_raw    = str(r.get(date_col) or "").strip()
            headline  = _clean(str(r.get(head_col) or "").strip())
            subcat    = _clean(str(r.get(sub_col) or "").strip())

            # Date formatting (leave as-is if unknown)
            dt_show = dt_raw
            # Some feeds give "DD MMM YYYY HH:MM", others "YYYY-MM-DD", we'll not over-parse to avoid errors.

            urls = _candidate_urls(r)
            pdf_url = _download_head_pdf_url(urls) if urls else ""

            reg_guess = infer_regulation(headline, subcat)

            out_rows.append({
                "Company": company,
                "Date": dt_show,
                "Headline": headline,
                "Announcement Type": subcat or "NA",
                "Interpreted Regulation (best-effort)": reg_guess,
                "PDF Link": pdf_url
            })

    df_out = pd.DataFrame(out_rows, columns=[
        "Company","Date","Headline","Announcement Type","Interpreted Regulation (best-effort)","PDF Link"
    ])

    st.subheader("📋 Company Update — Tabular View")
    st.dataframe(df_out, use_container_width=True)

    # CSV download
    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name=f"bse_company_update_{start_str}_{end_str}.csv", mime="text/csv")

else:
    st.info("Pick your date range and click **Fetch Company Update** to see the table.")
