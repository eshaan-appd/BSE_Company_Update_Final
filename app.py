# =========================================
# BSE fetch (robust)
# =========================================
import json as _json

def _parse_json_lenient(resp):
    """Return JSON even if server labels it text/plain; return {} on failure."""
    try:
        return resp.json()
    except Exception:
        try:
            txt = resp.text.strip()
            # Some days they wrap JSON with BOM or stray chars
            txt = txt.lstrip("\ufeff").strip()
            return _json.loads(txt)
        except Exception:
            return {}

@st.cache_data(show_spinner=False)
def fetch_bse_announcements(start_yyyymmdd: str,
                            end_yyyymmdd: str,
                            request_timeout: int = 25) -> pd.DataFrame:
    """
    Robust fetcher:
      - Accepts JSON even when content-type is text/plain
      - Tries multiple param variants
      - Tries both /AnnSubCategoryGetData/w and legacy /AnnGetData
      - Aggregates all pages across all variants; de-duplicates by NEWS_ID/ATTACHMENTNAME/HEADLINE
    """
    assert len(start_yyyymmdd) == 8 and len(end_yyyymmdd) == 8
    base_page = "https://www.bseindia.com/corporates/ann.html"
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": base_page,
        "Origin": "https://www.bseindia.com",
        "X-Requested-With": "XMLHttpRequest",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    try:
        s.get(base_page, timeout=15)
    except Exception:
        pass

    # Two endpoints we’ll try
    endpoints = [
        ("https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w", {
            "strCat": "-1",        # all cats
            "strType": "C",        # company updates list-type; still returns other cats too
        }),
        ("https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w", {   # legacy
            "strCat": "-1",
            "strType": "C",
        }),
    ]

    # Param variants that often change result sets
    variants = [
        {"subcategory": "-1", "strSearch": "P"},
        {"subcategory": "-1", "strSearch": ""},
        {"subcategory": "",   "strSearch": "P"},
        {"subcategory": "",   "strSearch": ""},
    ]

    all_rows = []
    for url, base_params in endpoints:
        for v in variants:
            params = dict(base_params)
            params.update({
                "pageno": 1,
                "subcategory": v.get("subcategory", ""),
                "strPrevDate": start_yyyymmdd,
                "strToDate": end_yyyymmdd,
                "strSearch": v.get("strSearch", ""),
                "strscrip": "",
            })
            page = 1
            while True:
                r = s.get(url, params=params, timeout=request_timeout)
                data = _parse_json_lenient(r)
                table = data.get("Table") or []
                if not isinstance(table, list):
                    break
                all_rows.extend(table)
                # pagination
                total = None
                try:
                    total = int((data.get("Table1") or [{}])[0].get("ROWCNT") or 0)
                except Exception:
                    total = None
                if not table:
                    break
                params["pageno"] += 1
                page += 1
                time.sleep(0.2)
                if total and len(table) == 0:
                    break  # safety

    if not all_rows:
        return pd.DataFrame()

    # De-dup
    def _row_key(r):
        return (
            str(r.get("NEWS_ID") or "").strip(),
            str(r.get("HEADLINE") or "").strip(),
            str(r.get("ATTACHMENTNAME") or "").strip(),
        )
    dedup = {}
    for r in all_rows:
        dedup[_row_key(r)] = r
    all_rows = list(dedup.values())

    # Normalize columns set
    all_keys = set()
    for r in all_rows:
        all_keys.update(r.keys())
    df = pd.DataFrame(all_rows, columns=list(all_keys))
    return df

def filter_company_update_only(df_in: pd.DataFrame, category_filter="Company Update") -> pd.DataFrame:
    if df_in.empty:
        return df_in.copy()
    cat_col = _first_col(df_in, ["CATEGORYNAME","CATEGORY","NEWS_CAT","NEWSCATEGORY","NEWS_CATEGORY"])
    if not cat_col:
        return df_in.copy()
    df2 = df_in.copy()
    df2["_cat_norm"] = df2[cat_col].astype(str).map(lambda x: _norm(x).lower())
    # Some days BSE uses tiny variants like "Company updates" or trailing spaces — be tolerant
    mask = df2["_cat_norm"].isin({
        _norm(category_filter).lower(),
        "company updates",
        "company  update",
        "companyup-date",  # rare glitch seen
    })
    return df2.loc[mask].drop(columns=["_cat_norm"])
