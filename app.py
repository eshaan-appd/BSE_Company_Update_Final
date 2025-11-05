import os, io, re, time, tempfile
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np
from openai import OpenAI
import streamlit as st
import json
from io import BytesIO

# ---- OpenAI (Responses API) ----
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not api_key:
    st.error("Missing OPENAI_API_KEY (set env var or add to Streamlit secrets).")
    st.stop()
client = OpenAI(api_key=api_key)

# ---------------------------------------
# Streamlit Setup
# ---------------------------------------
st.set_page_config(page_title="BSE Company Update — OpenAI PDF Summarizer", layout="wide")
st.title("📈 BSE Company Update — M&A / Merger / Scheme / JV (OpenAI-only)")
st.caption("Fetch BSE announcements → upload PDFs → auto-tag Announcement Type + SEBI Regulation via OpenAI.")

# =========================================
# Utility functions
# =========================================
_ILLEGAL_RX = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
def _clean(s): return _ILLEGAL_RX.sub('', s) if isinstance(s, str) else s
def _first_col(df, names): return next((n for n in names if n in df.columns), None)
def _norm(s): return re.sub(r"\s+", " ", str(s or "")).strip()
def _slug(s, maxlen=60): return re.sub(r"[^A-Za-z0-9]+", "_", str(s or "")).strip("_")[:maxlen] or "file"

# =========================================
# Fetch BSE Announcements (filtered)
# =========================================
def fetch_bse_announcements_strict(start_yyyymmdd, end_yyyymmdd, verbose=True, request_timeout=25):
    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0", "Accept": "application/json", "X-Requested-With": "XMLHttpRequest"
    })

    variants = [
        {"subcategory": "-1", "strSearch": "P"},
        {"subcategory": "-1", "strSearch": ""},
        {"subcategory": "", "strSearch": "P"},
        {"subcategory": "", "strSearch": ""},
    ]
    all_rows = []
    for v in variants:
        params = {
            "pageno": 1, "strCat": "-1", "subcategory": v["subcategory"],
            "strPrevDate": start_yyyymmdd, "strToDate": end_yyyymmdd,
            "strSearch": v["strSearch"], "strType": "C",
        }
        rows = []
        while True:
            r = s.get(url, params=params, timeout=request_timeout)
            data = r.json()
            table = data.get("Table") or []
            if not table: break
            rows.extend(table)
            params["pageno"] += 1
            time.sleep(0.2)
        if rows:
            all_rows = rows; break
    if not all_rows: return pd.DataFrame(all_rows)

    df = pd.DataFrame(all_rows)
    df = df[df["CATEGORYNAME"].str.contains("Company Update", case=False, na=False)]
    df = df[df["SUBCATEGORYNAME"].str.contains(r"(Acquisition|Merger|Scheme|Joint)", case=False, na=False)]
    return df

# =========================================
# OpenAI Helpers
# =========================================
def _download_pdf(url, timeout=25):
    r = requests.get(url, timeout=timeout)
    if r.status_code == 200 and b"%PDF" in r.content[:10]:
        return r.content
    raise RuntimeError(f"Failed PDF download: {url}")

def _upload_to_openai(pdf_bytes, fname="document.pdf"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        return client.files.create(file=open(tmp.name, "rb"), purpose="assistants")

def summarize_pdf_with_openai(pdf_bytes, company, headline, subcat,
                              model="gpt-4.1-mini", max_output_tokens=800, temperature=0.2):
    fobj = _upload_to_openai(pdf_bytes, fname=f"{_slug(company)}.pdf")
    task = f"""
You are a meticulous compliance and regulatory analyst specializing in Indian listed company filings.
Read the attached BSE/SEBI filing PDF carefully and produce a table with exactly three columns:

1) Company
2) Announcement Type From PDF
3) Regulations

Guidelines:
- In "Company", copy the company name from the provided context below.
- In "Announcement Type From PDF", describe the specific type (Outcome of Board Meeting, Intimation, Dividend, etc.).
- In "Regulations":
  • Quote explicit SEBI / LODR / PIT / Companies Act citations if present.
  • If not, infer the most likely regulation using your knowledge of SEBI rules:
      - Board meeting → Regulation 30
      - Financial results → Regulation 33
      - Shareholding pattern → Regulation 31
      - Trading window closure → PIT Regs, Schedule B
      - Preferential issue / QIP → SEBI (ICDR) Regulations, Reg. 164
      - Press release → Regulation 30
      - KMP / Director change → Regulation 30
      - Dividend → Regulation 43
  • If none, write "Not disclosed".
- Output strictly in JSON with this structure (no prose):

{{
  "table": [
    {{
      "Company": "{company or 'NA'}",
      "Announcement Type From PDF": "<announcement type>",
      "Regulations": "<exact or inferred regulation(s)>"
    }}
  ]
}}

Context:
Company: {company or 'NA'}
Headline: {headline or 'NA'}
Subcategory: {subcat or 'NA'}
"""
    resp = client.responses.create(
        model=model, temperature=temperature, max_output_tokens=max_output_tokens,
        input=[{"role": "user", "content": [{"type": "input_text", "text": task},
                                            {"type": "input_file", "file_id": fobj.id}]}],
        response_format={"type": "json_object"}
    )
    try:
        return json.loads(resp.output_text)
    except Exception:
        return {"table": [{"Company": company, "Announcement Type From PDF": "Not disclosed", "Regulations": "Not disclosed"}]}

def safe_summarize(*args, **kwargs):
    for i in range(3):
        try: return summarize_pdf_with_openai(*args, **kwargs)
        except Exception as e:
            if "429" in str(e): time.sleep(2 * (i + 1)); continue
            raise
    return {"table": [{"Company": "NA", "Announcement Type From PDF": "Error", "Regulations": "Error"}]}

# =========================================
# Sidebar Controls
# =========================================
with st.sidebar:
    st.header("⚙️ Controls")
    today = datetime.now().date()
    start_date = st.date_input("Start date", value=today - timedelta(days=1))
    end_date = st.date_input("End date", value=today)
    model = st.selectbox("OpenAI model", ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"], index=0)
    max_workers = st.slider("Parallel summaries", 1, 8, 3)
    max_items = st.slider("Max announcements", 5, 200, 30)
    run = st.button("🚀 Fetch & Summarize")

# =========================================
# Pipeline Execution
# =========================================
if run:
    start_str, end_str = start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
    df_hits = fetch_bse_announcements_strict(start_str, end_str)
    if df_hits.empty:
        st.warning("No matching announcements found.")
        st.stop()

    nm_col = _first_col(df_hits, ["SLONGNAME", "SNAME", "COMPANYNAME"]) or "SLONGNAME"
    sub_col = _first_col(df_hits, ["SUBCATEGORYNAME", "SUBCATEGORY"]) or "SUBCATEGORYNAME"

    rows = []
    for _, row in df_hits.head(max_items).iterrows():
        urls = []
        att = str(row.get("ATTACHMENTNAME") or "")
        if att:
            urls += [
                f"https://www.bseindia.com/xml-data/corpfiling/Attach/{att}",
                f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{att}",
            ]
        rows.append((row, urls))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(
            lambda r=row, urls=urls: safe_summarize(
                _download_pdf(urls[0]), str(r.get(nm_col)), str(r.get("HEADLINE")), str(r.get(sub_col)),
                model=model
            )
        ) for row, urls in rows if urls]
        for i, fut in enumerate(as_completed(futs), start=1):
            data = fut.result()
            for entry in data.get("table", []):
                results.append(entry)

    if not results:
        st.warning("No summaries were generated.")
        st.stop()

    df_table = pd.DataFrame(results, columns=["Company", "Announcement Type From PDF", "Regulations"])

    # === Display + Downloads ===
    st.subheader("📑 Summaries (OpenAI)")
    st.dataframe(df_table, use_container_width=True)

    csv_bytes = df_table.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv_bytes,
                       file_name=f"bse_summary_{start_str}_{end_str}.csv", mime="text/csv")

    excel_buffer = BytesIO()
    df_table.to_excel(excel_buffer, index=False)
    st.download_button("⬇️ Download Excel", data=excel_buffer.getvalue(),
                       file_name=f"bse_summary_{start_str}_{end_str}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Pick a date range and click **Fetch & Summarize**.")
