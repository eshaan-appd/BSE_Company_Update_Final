import os, re, io, time, tempfile, json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import streamlit as st
from openai import OpenAI

# =========================================
# Streamlit page
# =========================================
st.set_page_config(page_title="BSE — Company Update (PDF-aware tags)", layout="wide")
st.title("📑 BSE — Company Update (PDF-aware Announcement Type & Regulation)")
st.caption("Reads each PDF with OpenAI to extract Announcement Type (from PDF text) and cited Regulations. No summaries.")

# =========================================
# OpenAI client
# =========================================
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not api_key:
    st.error("Missing OPENAI_API_KEY (set env var or add to Streamlit secrets).")
    st.stop()

@st.cache_resource(show_spinner=False)
def _get_client(_api_key: str):
    return OpenAI(api_key=_api_key)

client = _get_client(api_key)

with st.expander("🔍 OpenAI connection diagnostics", expanded=False):
    key_src = "st.secrets" if "OPENAI_API_KEY" in st.secrets else "env"
    mask = lambda s: (s[:7] + "..." + s[-4:]) if s and len(s) > 12 else "unset"
    st.write("Key source:", key_src)
    st.write("API key (masked):", mask(st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")))
    try:
        _ = client.models.list()
        st.success("Models list ok — auth looks good.")
    except Exception as e:
        st.error(f"Models list failed: {e}")

# =========================================
# Small utilities
# =========================================
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

def _head_ok_pdf(url, timeout=20) -> bool:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        if r.status_code == 200:
            if url.lower().endswith(".pdf"):
                return True
            if "pdf" in (r.headers.get("content-type","").lower()):
                return True
    except Exception:
        pass
    return False

def _first_working_pdf(urls) -> str:
    for u in urls:
        if _head_ok_pdf(u):
            return u
    return ""

def _download_pdf(url: str, timeout=25) -> bytes:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": "https://www.bseindia.com/corporates/ann.html",
        "Accept-Language": "en-US,en;q=0.9",
    })
    r = s.get(url, timeout=timeout, allow_redirects=True, stream=False)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.content

# =========================================
# BSE fetch (no subcategory filter here)
# =========================================
@st.cache_data(show_spinner=False)
def fetch_bse_announcements(start_yyyymmdd: str,
                            end_yyyymmdd: str,
                            request_timeout: int = 25) -> pd.DataFrame:
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
            if "application/json" not in r.headers.get("content-type",""):
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

# =========================================
# OpenAI: extract announcement type & regulations from PDF
# =========================================
PDF_EXTRACTION_SYSTEM = """You are a meticulous compliance analyst for Indian listed company filings.
Read the attached BSE/SEBI filing PDF and return STRICT JSON with keys:
{
  "announcement_type_from_pdf": "<short type name from the filing or obvious from its contents>",
  "regulations_cited": ["<SEBI/LODR/PIT/etc citations exactly as written, minimal; if none, 'Not disclosed'>"]
}
Rules:
- Use concise names for announcement type (e.g., 'Outcome of Board Meeting', 'Intimation of Board Meeting', 'Record Date', 'Dividend Declaration',
  'Investor Presentation', 'Trading Window Closure', 'Credit Rating', 'Press Release', 'RPT Disclosure', 'Auditor Appointment', 'KMP change', 'Buyback', 'QIP/Preferential', etc.)
- If the PDF explicitly cites regulations (e.g., 'Regulation 30 of SEBI (LODR) Regulations, 2015'), include them in regulations_cited (exact text; avoid duplicates).
- If no clear regulation text is present, set regulations_cited to ['Not disclosed'].
- Output ONLY the JSON, no prose.
"""

def openai_extract_from_pdf(pdf_bytes: bytes, model: str = "gpt-4.1-mini", max_output_tokens: int = 200) -> dict:
    # Upload file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        uploaded = client.files.create(file=open(tmp.name, "rb"), purpose="assistants")

    # Ask the model to return STRICT JSON
    resp = client.responses.create(
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        input=[{
            "role": "system",
            "content": [{"type": "input_text", "text": PDF_EXTRACTION_SYSTEM}]
        },{
            "role": "user",
            "content": [{"type": "input_file", "file_id": uploaded.id}]
        }],
    )
    raw = (resp.output_text or "").strip()
    # Best-effort robust JSON parse
    try:
        data = json.loads(raw)
        if not isinstance(data, dict): raise ValueError("not dict")
        # Normalize
        t = data.get("announcement_type_from_pdf") or ""
        regs = data.get("regulations_cited") or []
        if isinstance(regs, str): regs = [regs]
        regs = [r.strip() for r in regs if str(r).strip()]
        if not regs: regs = ["Not disclosed"]
        return {
            "announcement_type_from_pdf": t.strip() or "Not disclosed",
            "regulations_cited": list(dict.fromkeys(regs))  # de-dup preserve order
        }
    except Exception:
        return {
            "announcement_type_from_pdf": "Not disclosed",
            "regulations_cited": ["Not disclosed"]
        }

def format_reg_list(regs: list[str]) -> str:
    if not regs: return "Not disclosed"
    # Collapse obvious duplicates/variants
    uniq = []
    seen = set()
    for r in regs:
        k = r.lower().strip()
        if k not in seen:
            seen.add(k); uniq.append(r)
    return "; ".join(uniq)

# =========================================
# Sidebar controls
# =========================================
with st.sidebar:
    st.header("⚙️ Controls")
    today = datetime.now().date()
    start_date = st.date_input("Start date", value=today - timedelta(days=1), max_value=today)
    end_date   = st.date_input("End date", value=today, max_value=today, min_value=start_date)

    model = st.selectbox(
        "OpenAI model",
        ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"],
        index=0,
        help="Models with file-reading capability. 4.1-mini/4o-mini are cost-efficient."
    )
    max_items  = st.slider("Max announcements to process", 10, 250, 60, step=10)
    max_workers = st.slider("Parallel PDF reads", 1, 6, 3, help="Lower if you hit 429s.")
    run = st.button("🚀 Fetch & Extract from PDFs", type="primary")

# =========================================
# Run
# =========================================
if run:
    start_str, end_str = _fmt(start_date), _fmt(end_date)

    with st.status("Fetching announcements…", expanded=True):
        df_raw = fetch_bse_announcements(start_str, end_str)
        st.write(f"Raw rows fetched: **{len(df_raw)}**")
        df_hits = filter_company_update_only(df_raw, category_filter="Company Update")
        st.write(f"'Company Update' rows: **{len(df_hits)}**")

    if df_hits.empty:
        st.warning("No 'Company Update' announcements found in this window.")
        st.stop()

    if len(df_hits) > max_items:
        df_hits = df_hits.head(max_items)

    # Identify columns
    nm_col   = _first_col(df_hits, ["SLONGNAME","SNAME","SC_NAME","COMPANYNAME"]) or "SLONGNAME"
    sub_col  = _first_col(df_hits, ["SUBCATEGORYNAME","SUBCATEGORY","SUB_CATEGORY","NEWS_SUBCATEGORY"]) or "SUBCATEGORYNAME"
    date_col = _first_col(df_hits, ["NEWS_DT","DT","NEWSDATE","NEWS_DATE"]) or "NEWS_DT"
    head_col = _first_col(df_hits, ["HEADLINE","NEWS_SUB","SUBJECT","HEADING"]) or "HEADLINE"

    rows = df_hits.to_dict(orient="records")

    st.subheader("📄 Processing PDFs with OpenAI")
    progress = st.progress(0)

    def worker(row):
        urls = _candidate_urls(row)
        pdf_url = _first_working_pdf(urls)
        if not pdf_url:
            return row, pdf_url, {"announcement_type_from_pdf":"Not disclosed","regulations_cited":["Not disclosed"]}

        try:
            pdf_bytes = _download_pdf(pdf_url, timeout=25)
            if not (pdf_bytes and len(pdf_bytes) > 500):
                raise RuntimeError("PDF too small")
            info = openai_extract_from_pdf(pdf_bytes, model=model, max_output_tokens=220)
            return row, pdf_url, info
        except Exception:
            return row, pdf_url, {"announcement_type_from_pdf":"Not disclosed","regulations_cited":["Not disclosed"]}

    out = []
    total = len(rows)
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(worker, r) for r in rows]
        for fut in as_completed(futs):
            row, pdf_url, info = fut.result()
            company   = _clean(str(row.get(nm_col) or "").strip())
            dt_show   = str(row.get(date_col) or "").strip()
            headline  = _clean(str(row.get(head_col) or "").strip())
            subcat    = _clean(str(row.get(sub_col) or "").strip())
            regs_str  = format_reg_list(info.get("regulations_cited") or [])
            # Build final table row
            out.append({
                "Company": company,
                "Date": dt_show,
                "Headline": headline,
                "Announcement Type (BSE subcategory)": subcat or "NA",
                "Interpreted Regulation (from PDF)": regs_str,
                "PDF Link": pdf_url if pdf_url else ""
            })
            done += 1
            progress.progress(min(1.0, done/total))

    # Sort by Date (fallback to as-is if parsing fails)
    df_out = pd.DataFrame(out, columns=[
        "Company","Date","Headline","Announcement Type (BSE subcategory)","Interpreted Regulation (from PDF)","PDF Link"
    ])

    st.subheader("📋 Company Update — PDF-aware Tabular View")
    st.dataframe(df_out.drop(columns=["PDF Link"]), use_container_width=True)

    # Provide PDF links separately (so it doesn't clutter the table)
    with st.expander("🔗 PDF Links (click to open/download)"):
        for i, r in df_out.iterrows():
            if r["PDF Link"]:
                st.markdown(f"- [{r['Company']} — {r['Date']}]({r['PDF Link']})  \n  `{r['Headline']}`")

    # CSV download (includes PDF Link)
    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV (with PDF links)",
        data=csv_bytes,
        file_name=f"bse_company_update_pdf_tags_{start_str}_{end_str}.csv",
        mime="text/csv"
    )
else:
    st.info("Pick your date range and click **Fetch & Extract from PDFs** to build the table.")
