import io, json, re, tempfile, traceback, os
from typing import Optional
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ── Constants ──────────────────────────────────────────────────────────────
DOWNSAMPLE_N = 3000  
MAX_SIGNALS  = 2000  
LANE_H       = 40    
YAXIS_W      = 70
NAMES_W      = 400  

# Distinct colors for signals
COLORS = [
    "#e05252","#4caf7d","#4a90d9","#f0a030","#9b59b6", "#e67e22","#1abc9c","#e91e8c","#2196f3","#8bc34a",
    "#7c3aed","#00bcd4","#ff7043","#3f51b5","#388e3c", "#c62828","#0277bd","#37474f","#4527a0","#00695c",
    "#e53935","#1e88e5","#43a047","#fb8c00","#8e24aa", "#00acc1","#f4511e","#3949ab","#039be5","#00897b",
    "#7cb342","#c0ca33","#ffb300","#f06292","#4db6ac", "#7986cb","#a1887f","#90a4ae","#80cbc4","#ce93d8",
] * 50

# ── Parsers ────────────────────────────────────────────────────────────────
def _try_asammdf(raw_bytes):
    try:
        from asammdf import MDF
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
            tmp.write(raw_bytes); tmp_path = tmp.name
        with MDF(tmp_path) as mdf_file:
            df = mdf_file.to_dataframe(time_from_zero=False)
        df = df.reset_index()
        try: os.unlink(tmp_path)
        except: pass
        return df, "asammdf"
    except ImportError: return None, "asammdf_not_installed"
    except Exception as e: return None, str(e)

def _try_text(raw_bytes):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try: text = raw_bytes.decode(enc)
        except: continue
        for sep in ["\t", ",", ";", r"\s+"]:
            for skip in [0, 1, 2, 3]:
                try:
                    df = pd.read_csv(io.StringIO(text), sep=sep, skiprows=skip, engine="python", low_memory=False, on_bad_lines="skip")
                    if df.shape[1] >= 2 and df.shape[0] >= 5:
                        for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
                        df = df.dropna(axis=1, how="all")
                        if df.shape[1] >= 2: return df, "text"
                except: continue
    return None, None

def detect_time_col(df):
    for c in df.columns:
        if any(k in str(c).lower() for k in ("time", "timestamp", "ts", "zeit")):
            if pd.to_numeric(df[c], errors='coerce').notna().sum() > 2: return c
    num_cols = df.select_dtypes(include="number").columns.tolist()
    return num_cols[0] if num_cols else df.columns[0]

def is_numeric(s):
    if s.dtype == object:
        try: return pd.to_numeric(s, errors="coerce").notna().mean() > 0.1
        except: return False
    return np.issubdtype(s.dtype, np.number)

def downsample(x, y, n=DOWNSAMPLE_N):
    L = len(y)
    if L <= n: return x.tolist(), y.tolist()
    bkt = max(1, L // n)
    xi, yi = [], []
    for i in range(0, L, bkt):
        cy, cx = y[i:i+bkt], x[i:i+bkt]
        if not len(cy): continue
        for idx in sorted({int(np.argmin(cy)), int(np.argmax(cy))}):
            xi.append(float(cx[idx])); yi.append(float(cy[idx]))
    return xi, yi

def normalize_col_name(col: str) -> str:
    col = re.sub(r'\\.*$', '', str(col))
    col = re.sub(r':\d+$', '', col)
    return col.strip()

_COND_RE = re.compile(
    r'\b([A-Za-z_][A-Za-z0-9_]{2,})\s*'       
    r'(==|!=|>=|<=|>|<|=|≦|≧)\s*'             
    r'('                                      
    r'-?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?'     
    r'|True|False|TRUE|FALSE|true|false'      
    r'|aktiv|inaktiv|active|inactive|on|off|AN|AUS' 
    r'|[A-Za-z_][A-Za-z0-9_]{2,}'             
    r')'
    r'(?:\s*\[.*?\])?',                       
    re.UNICODE
)

_BOOL_MAP = {'true': 1.0, 'false': 0.0, 'aktiv': 1.0, 'inaktiv': 0.0, 'active': 1.0, 'inactive': 0.0, 'on': 1.0, 'off': 0.0, 'an': 1.0, 'aus': 0.0}

_STOPWORDS = {
    'SET','THE','AND','FOR','NOT','ARE','WITH','FROM','THIS','THAT','HAVE','WILL','WHEN','THEN','ALSO','INTO','ONLY','EACH','BOTH','TEST','CASE','MODE','TYPE','STATE','VALUE','TIME','DATA','FILE','SIGNAL','STATUS','ACTIVE','INACTIVE','FAULT','ERROR','ISSUED','INPUT','OUTPUT','PARAM','CHECK','STEP','NOTE','INFO','RANGE','NORMAL','VOLTAGE','SPEED','EVENT','CONDITION','OTHER','WITHOUT','INAKTIV','AKTIV','FEHLER','SELECTED','REPLACE','CONFIRMED','OCCURRING','TRUE','FALSE','NONE','NULL','ON','OFF','CAN','KL30','EM1','HV','LV','DTC','AE','AB','IN1','IN2','IN3','IN4','IN5','OUT1','OUT2','OUT3','LV2','NMH','ESP','HVK','LVDC','HVDC','SCSC','FME','RS','KM','KMH','MS','KPH','VTH','STH','VHYS','NO','NR','OPERATIONS','PREPARATION','IS','BY','DUE','TO','AN','A','NAN'
}

def _parse_cond_value(s: str):
    sl = s.lower().strip()
    if sl in _BOOL_MAP: return _BOOL_MAP[sl]
    try: return float(s.replace(',', '.'))
    except: 
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{2,}', s.strip()): return s.strip()
        return None

def _is_likely_signal(name: str) -> bool:
    if not name or len(name) < 2: return False
    if name.upper() in _STOPWORDS: return False
    if name.isnumeric(): return False
    has_underscore = '_' in name
    has_number = any(c.isdigit() for c in name)
    has_known_prefix = bool(re.match(r'^(?:vf|vx|cx|f[A-Z]|c[A-Z]|vd|vs|vn|Dem_|EM1_)', name, re.IGNORECASE))
    has_mixed = any(c.islower() for c in name) and any(c.isupper() for c in name)
    return bool(has_underscore or has_number or has_known_prefix or has_mixed)

def _col_category(col_name: str) -> Optional[str]:
    s = str(col_name).lower().strip()
    if any(k in s for k in ['input', 'param', 'eingang', 'ansteuer', 'vorgabe', 'stimulus', 'input parameter', 'input parameters', 'eingabe']): return 'input'
    if any(k in s for k in ['expected', 'expect', 'erwartet', 'soll', 'behavior', 'behaviour', 'verifi', 'reaction', 'response', 'expected behavior', 'expected behaviors', 'expected behaviour', 'expected behaviours', 'erwartetes', 'ergebnis']): return 'expected'
    return None

def _extract_conditions_from_rows(df_rows: pd.DataFrame, input_cols: list, expected_cols: list):
    in_blocks, expected_list, all_sigs_dict = [[]], [], {}
    scan_fallback = [c for c in df_rows.columns if df_rows[c].dtype == object] if (not input_cols and not expected_cols) else []
    
    def _parse_cell(text, cat):
        found = []
        text_str = str(text)
        for m in _COND_RE.finditer(text_str):
            sig, op_raw, val_str = m.group(1), m.group(2), m.group(3)
            if len(sig)>1 and sig.upper() not in _STOPWORDS and re.search(r'[a-zA-Z]', sig):
                op = '<=' if op_raw == '≦' else '>=' if op_raw == '≧' else '==' if op_raw == '=' else op_raw
                val = _parse_cond_value(val_str)
                if val is not None:
                    found.append({'name': sig, 'op': op, 'value': val, 'category': cat})
                    all_sigs_dict[sig] = None
                    if isinstance(val, str): all_sigs_dict[val] = None
                    
        for word in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]{2,})\b', text_str):
            if _is_likely_signal(word):
                all_sigs_dict[word] = None
                
        return found

    for _, row in df_rows.iterrows():
        is_or = False
        for c in input_cols:
            if c in row and pd.notna(row[c]) and str(row[c]).strip().upper() == 'OR':
                is_or = True; break
        if is_or:
            in_blocks.append([]); continue
        for c in input_cols:
            if c in row and pd.notna(row[c]): in_blocks[-1].extend(_parse_cell(row[c], 'input'))
        for c in expected_cols:
            if c in row and pd.notna(row[c]): expected_list.extend(_parse_cell(row[c], 'expected'))
        for c in scan_fallback:
            if c in row and pd.notna(row[c]): expected_list.extend(_parse_cell(row[c], 'expected'))
                
    return [b for b in in_blocks if b], expected_list, list(all_sigs_dict.keys())

_HEADER_KEYWORDS_NO   = {'no', 'no.', 'nr', 'nr.', '#', 'num', 'number'}
_HEADER_KEYWORDS_OPS  = {'operation', 'operations', 'description', 'step'}
_HEADER_KEYWORDS_IN   = {'input', 'input parameter', 'input parameters', 'eingang'}
_HEADER_KEYWORDS_EXP  = {'expected', 'expected behavior', 'expected behaviors', 'expected behaviour', 'erwartet', 'soll'}

def _find_header_row(df_raw: pd.DataFrame) -> int:
    best_row, best_score = 0, -1
    for i, row in df_raw.head(25).iterrows():
        cells = [str(v).strip().lower() for v in row.values if pd.notna(v) and str(v).strip()]
        if not cells: continue
        score = sum([3 if c in _HEADER_KEYWORDS_NO else 1 if c in _HEADER_KEYWORDS_OPS else 2 if c in _HEADER_KEYWORDS_IN else 2 if c in _HEADER_KEYWORDS_EXP else 2 if 'input param' in c else 2 if 'expected' in c else 0 for c in cells])
        if score > best_score: best_score, best_row = score, i
    return best_row if best_score > 0 else 0

def _find_no_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        cs = str(col).strip().lower().rstrip('.')
        if cs in {'no', 'no.', 'nr', 'nr.', '#', 'num', 'number', 'tc', 'case', 'test case', 'testcase', 'test no'}: return col
    for col in list(df.columns)[:5]:
        col_str = str(col).strip()
        if col_str.lower().startswith('unnamed'): continue
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(vals) > 0 and vals.max() < 500 and vals.min() >= 0:
            if len(vals[vals == vals.astype(int)]) / max(len(vals), 1) > 0.9: return col
    return None

def _parse_sheet_into_test_cases(sheet_name: str, df_sheet: pd.DataFrame) -> list:
    input_cols    = [c for c in df_sheet.columns if _col_category(c) == 'input']
    expected_cols = [c for c in df_sheet.columns if _col_category(c) == 'expected']
    no_col        = _find_no_col(df_sheet)
    test_cases = []

    if no_col is None:
        in_b, exp_l, sigs = _extract_conditions_from_rows(df_sheet, input_cols, expected_cols)
        return [{"tc_id": sheet_name + "-1", "sheet_name": sheet_name, "tc_num": 1, "title": sheet_name, "tab_label": sheet_name + " TC1", "signals": sigs, "in_blocks": in_b, "expected_list": exp_l, "df": df_sheet}]

    df_sheet = df_sheet.copy()
    df_sheet["_tc_num_"] = pd.to_numeric(df_sheet[no_col], errors="coerce")
    current_tc_num, current_rows = None, []

    def _flush(tc_num, rows):
        if not rows: return
        sub = pd.DataFrame(rows, columns=df_sheet.columns)
        in_b, exp_l, sigs = _extract_conditions_from_rows(sub, input_cols, expected_cols)
        label = sheet_name + " TC" + str(tc_num) 
        test_cases.append({"tc_id": sheet_name + "-" + str(tc_num), "sheet_name": sheet_name, "tc_num": tc_num, "title": label, "tab_label": label, "signals": sigs, "in_blocks": in_b, "expected_list": exp_l, "df": sub})

    for _, row in df_sheet.iterrows():
        tc_num = row["_tc_num_"]
        if pd.notna(tc_num):
            if int(tc_num) != current_tc_num:
                _flush(current_tc_num, current_rows)
                current_tc_num, current_rows = int(tc_num), [row.tolist()]
            else: current_rows.append(row.tolist())
        else:
            if current_tc_num is not None: current_rows.append(row.tolist())

    _flush(current_tc_num, current_rows)
    if not test_cases:
        in_b, exp_l, sigs = _extract_conditions_from_rows(df_sheet, input_cols, expected_cols)
        test_cases.append({"tc_id": sheet_name + "-1", "sheet_name": sheet_name, "tc_num": 1, "title": sheet_name, "tab_label": sheet_name + " TC1", "signals": sigs, "in_blocks": in_b, "expected_list": exp_l, "df": df_sheet})

    return test_cases

def parse_excel_checksheet(raw_bytes: bytes, dat_filename: str = "") -> dict:
    result = {"all_sheets": [], "signals": list(), "test_cases": [], "error": None}
    try: xl_file = pd.ExcelFile(io.BytesIO(raw_bytes))
    except Exception as e: result["error"] = str(e); return result

    result["all_sheets"] = xl_file.sheet_names
    if not result["all_sheets"]: return result

    test_cases, all_signals_d = [], {} 
    for sheet_name in result["all_sheets"]:
        try: df_raw = pd.read_excel(xl_file, sheet_name=sheet_name, header=None, dtype=str)
        except: continue
        header_row = _find_header_row(df_raw)
        try: df_sheet = pd.read_excel(xl_file, sheet_name=sheet_name, header=header_row, dtype=str)
        except: df_sheet = df_raw.copy()
        
        df_sheet.columns = [str(c).strip() for c in df_sheet.columns]
        # Drop RS column explicitly
        df_sheet = df_sheet.loc[:, ~df_sheet.columns.str.lower().isin(['rs'])]
        
        df_sheet = df_sheet.dropna(axis=1, how='all').reset_index(drop=True)
        df_sheet = df_sheet[~df_sheet.apply(lambda r: r.astype(str).str.lower().isin(['none', 'nan', '']).all(), axis=1)].reset_index(drop=True)
        if df_sheet.empty: continue
        for tc in _parse_sheet_into_test_cases(sheet_name, df_sheet):
            for sig in tc["signals"]: all_signals_d[sig] = None
            test_cases.append(tc)

    result["signals"] = list(all_signals_d.keys())
    result["test_cases"] = test_cases
    result["preview_df"] = df_sheet.head(100) 
    return result

def match_signals(excel_sigs: list, dat_cols: list) -> tuple:
    dat_set, dat_lower = set(dat_cols), {c.lower(): c for c in dat_cols}
    norm_to_orig, norm_lower_to_orig = {}, {}
    for c in dat_cols:
        norm = normalize_col_name(c)
        if norm not in norm_to_orig: norm_to_orig[norm] = c
        if norm.lower() not in norm_lower_to_orig: norm_lower_to_orig[norm.lower()] = c

    matched_set, matched, unmatched = set(), [], []
    for sig in excel_sigs:
        if sig in dat_set:
            if sig not in matched_set: matched.append(sig); matched_set.add(sig)
        elif sig.lower() in dat_lower:
            s = dat_lower[sig.lower()]
            if s not in matched_set: matched.append(s); matched_set.add(s)
        elif sig in norm_to_orig:
            s = norm_to_orig[sig]
            if s not in matched_set: matched.append(s); matched_set.add(s)
        elif sig.lower() in norm_lower_to_orig:
            s = norm_lower_to_orig[sig.lower()]
            if s not in matched_set: matched.append(s); matched_set.add(s)
        else: unmatched.append(sig)
    return matched, unmatched


# ── THE ULTIMATE V-GRAPH ENGINE + DYNAMIC EXPORT ─────────────────────────
VIEWER_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>MDA Signal Viewer</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#faf9f7;--surface:#ffffff;--surface2:#f1f5f9;
  --border:#e2e8f0;--border2:#cbd5e1;
  --accent:#0071e3;--accent2:#2563eb;
  --text:#1d1d1f;--text2:#334155;--text3:#86868b;
  --mono:'JetBrains Mono',monospace;--sans:-apple-system, BlinkMacSystemFont, "Inter", sans-serif;
  --green:#34c759;--orange:#f59e0b;--red:#ff3b30;
  --plot-bg:#faf9f7; --grid:#e2e8f0; --cursor:#1d4ed8;
}
[data-theme="dark"] {
  --bg:#000000;--surface:#161618;--surface2:#1c1c1e;
  --border:#272729;--border2:#38383a;
  --accent:#2997ff;--accent2:#60a5fa;
  --text:#f5f5f7;--text2:#cbd5e1;--text3:#94a3b8;
  --green:#30d158;--orange:#ff9f0a;--red:#ff453a;
  --plot-bg:#000000; --grid:#1c1c1e; --cursor:#fbbf24;
}
html,body{width:100%;height:100%;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:12px;overflow:hidden;transition:background-color 0.3s, color 0.3s;}
#app{display:flex;flex-direction:column;height:100vh;}

#toolbar {
    background:var(--surface); border-bottom:1px solid var(--border);
    display:flex; align-items:center; gap:8px; padding:8px 16px; 
    transition:0.3s; flex-wrap:wrap; min-height:44px; flex-shrink:0;
}
#toolbar button, #toolbar select, #toolbar input, .tlbl, .tsep { flex-shrink: 0; }
#toolbar button { background:var(--surface2); color:var(--text); border:1px solid var(--border2); padding:0 12px; height:28px; cursor:pointer; font-size:12px; font-family:var(--sans); font-weight:500; border-radius:8px; transition:0.2s;}
#toolbar button:hover { background:var(--border); border-color:var(--accent); }
#toolbar select, #toolbar input { background:var(--surface2); color:var(--text); border:1px solid var(--border2); height:28px; font-size:11px; font-family:var(--mono); padding:0 8px; border-radius:8px; outline:none; }
#search { width:160px; }
#search:focus { border-color:var(--accent); }
.tsep { width:1px; height:18px; background:var(--border); margin:0 6px; }

/* Toggle Button Styles */
.ix-toggle-off { background:var(--surface) !important; border-color:var(--border2) !important; color:var(--text3) !important; }
.ix-toggle-on { background:rgba(52,199,89,0.1) !important; border-color:var(--green) !important; color:var(--green) !important; font-weight:600 !important; }
.ix-toggle-disabled { opacity: 0.4; cursor: not-allowed !important; border-color: var(--border2) !important; color: var(--text3) !important; background: transparent !important; }

#headers-row { display:flex; height:32px; flex-shrink:0; border-bottom:1px solid var(--border); background:var(--surface2); transition:0.3s; }
.hdr-left { width:%%NAMES_W%%px; display:flex; flex-shrink:0; border-right:1px solid var(--border); }
.hdr-block { display:flex; align-items:center; flex-shrink:0; font-family:var(--mono); font-size:10px; letter-spacing:.1em; color:var(--text3); text-transform:uppercase; padding:0 14px;}

#workspace { display:flex; flex:1; flex-direction:column; overflow:hidden; position:relative; }
#body-row { display:flex; flex:1; overflow:hidden; background:var(--plot-bg); position:relative; }

#left-panel { width:%%NAMES_W%%px; flex-shrink:0; border-right:1px solid var(--border); overflow-x:auto; overflow-y:auto; scrollbar-width:none; background:var(--surface); display:flex; position:relative; z-index:20; }
#left-panel::-webkit-scrollbar { display:none; }
#names-col { width:100%; flex-shrink:0; transition:0.3s; }

#right-panel { flex:1; overflow-y:auto; overflow-x:hidden; position:relative; cursor:crosshair; }
#right-panel::-webkit-scrollbar{width:8px;height:10px;}
#right-panel::-webkit-scrollbar-track{background:var(--surface);}
#right-panel::-webkit-scrollbar-thumb{background:var(--border2);border-radius:5px;}
#right-inner { position:relative; width:100%; }

.nrow {
    display:flex; align-items:center; border-bottom:1px solid var(--border);
    padding:0 14px 0 16px; gap:10px; cursor:default; position:relative; transition:background .15s; box-sizing: border-box;
}
.nrow:hover { background:var(--surface2); }
.nrow-color-box { width:10px; height:10px; border-radius:3px; flex-shrink:0; }
.nrow-body { flex:1; min-width:0; display:flex; align-items:center; justify-content:space-between; padding-left:8px; }
.nrow-name { font-family:var(--mono); font-size:12px; font-weight:600; display:flex; align-items:center; white-space:nowrap; }
.nrow-val { font-family:var(--mono); font-size:12px; font-weight:700; color:var(--text); text-align:right; margin-left:auto; }

#plot-cv, #hover-cv, #cursor-cv { position:absolute; left:0; top:0; }
#hover-cv, #cursor-cv { pointer-events:none; }

#footer-row { height:32px; flex-shrink:0; border-top:1px solid var(--border); background:var(--surface2); display:flex; overflow:hidden; transition:0.3s; }
#xa-wrap { flex:1; position:relative; }
#xa-cv { position:absolute; left:0; top:0; }

/* Add Signal Modal */
#add-sig-modal { display:none; position:fixed; top:80px; left:50%; transform:translateX(-50%); background:var(--surface); border:1px solid var(--border); z-index:200; padding:16px; border-radius:12px; box-shadow:0 12px 40px rgba(0,0,0,0.2); width:400px; backdrop-filter:blur(20px); }
#add-sig-search { width:100%; margin-bottom:12px; padding:8px; border-radius:8px; border:1px solid var(--border2); background:var(--surface2); color:var(--text); outline:none; font-family:var(--sans); }
#add-sig-list { max-height:300px; overflow-y:auto; font-family:var(--mono); font-size:11px; }
.add-sig-item { padding:8px 12px; cursor:pointer; border-bottom:1px solid var(--border); transition:0.2s; color:var(--text); }
.add-sig-item:hover { background:var(--surface2); color:var(--accent); font-weight:600; }
</style>
</head><body>
<div id="app">

<div id="toolbar">
  <button onclick="resetView()">↺ Reset</button>
  <button onclick="zoomIn()">Zoom +</button>
  <button onclick="zoomOut()">Zoom −</button>
  
  <div class="tsep"></div>
  <span class="tlbl" style="font-size:11px;">Lane H:</span>
  <select id="lane-h-sel" onchange="changeLaneH()" style="width:65px;">
      <option value="40" selected>40px</option>
      <option value="60">60px</option>
      <option value="80">80px</option>
      <option value="120">120px</option>
  </select>

  <div class="tsep" id="scenario-sep" style="display:none;"></div>
  <span class="tlbl" id="scenario-lbl" style="display:none; font-size:11px;">Scenario:</span>
  <select id="scenario-sel" onchange="changeScenario()" style="display:none;"></select>

  <div class="tsep"></div>
  <input id="search" type="text" placeholder="🔍 Filter signals…" oninput="filterSigs(this.value)">
  <button id="add-sig-btn" onclick="openAddSignalModal()" style="border-color:var(--accent); color:var(--accent);">➕ Add Signal</button>
  <button id="modify-btn" onclick="toggleModifyMode()" class="ix-toggle-off">✏️ Modify</button>
  
  <div style="margin-left:auto; display:flex; gap:8px;">
      <button id="ix-btn" onclick="toggleAutoIntersect()" class="ix-toggle-off">⊕ Auto Intersect: OFF</button>
      <button id="fs-btn" onclick="toggleFullScreen()">⛶ Fullscreen</button>
  </div>
</div>

<div id="headers-row">
   <div class="hdr-left">
        <div class="hdr-block" style="width:100%; justify-content:space-between;">
            <span>SIGNALS</span>
        </div>
   </div>
   <div style="flex:1; display:flex; justify-content:flex-end; align-items:center; padding:0 16px;">
      <span id="st-view" style="font-family:var(--mono); font-size:10px; color:var(--text); white-space:nowrap; flex-shrink:0;">View: %%T_MIN_FMT%% — %%T_MAX_FMT%%</span>
   </div>
</div>

%%WARNING_HTML%%

<div id="workspace">
    <div id="body-row">
        <div id="left-panel">
            <div id="names-col"><div id="left-inner"></div></div>
        </div>
        <div id="right-panel">
            <div id="right-inner">
                <canvas id="plot-cv"></canvas>
                <canvas id="hover-cv"></canvas>
                <canvas id="cursor-cv"></canvas>
            </div>
        </div>
    </div>
    <div id="footer-row">
        <div style="width:%%NAMES_W%%px; flex-shrink:0; border-right:1px solid var(--border);"></div>
        <div id="xa-wrap"><canvas id="xa-cv"></canvas></div>
    </div>
</div>

<div id="add-sig-modal">
   <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
       <h3 style="font-family:var(--sans); margin:0; font-size:14px; color:var(--text);">Add Custom Signal</h3>
       <button onclick="document.getElementById('add-sig-modal').style.display='none'" style="background:none; border:none; font-size:20px; cursor:pointer; color:var(--text3);">&times;</button>
   </div>
   <input type="text" id="add-sig-search" placeholder="Type to search all .dat signals..." oninput="renderAddSigList(this.value)">
   <div id="add-sig-list"></div>
</div>

</div>
<script>
let lightMode = true;
function syncTheme() {
    try {
        if(window.parent && window.parent.document.documentElement.getAttribute('data-theme') === 'dark'){
            document.documentElement.setAttribute('data-theme', 'dark'); lightMode = false;
        } else {
            document.documentElement.removeAttribute('data-theme'); lightMode = true;
        }
    } catch(e) { lightMode = true; } 
    buildRows(); 
    redraw();
}
window.addEventListener('message', function(e) { 
    if (e.data === 'theme-toggled') syncTheme(); 
    if (e.data === 'tab-activated') { setTimeout(function(){ syncLayout(); redraw(); }, 50); }
});

const THEMES = {
    dark: { bg: '#000000', surface: 'rgba(28,28,30,0.6)', surface2: '#1c1c1e', border: '#38383a', border2: '#48484a', accent: '#2997ff', green: '#30d158', orange: '#ff9f0a', red: '#ff453a', text: '#f5f5f7', text3: '#86868b', grid: '#1c1c1e', cursor: '#fbbf24' },
    light: { bg: '#faf9f7', surface: 'rgba(255,255,255,0.75)', surface2: '#f1f5f9', border: '#e2e8f0', border2: '#cbd5e1', accent: '#0071e3', green: '#34c759', orange: '#f59e0b', red: '#ef4444', text: '#1d1d1f', text3: '#86868b', grid: '#e2e8f0', cursor: '#1d4ed8' }
};

const T_MIN=%%T_MIN%%, T_MAX=%%T_MAX%%;
const ALL_SIGS=%%SIG_DATA%%;
const INITIAL_VIS_NAMES=%%INITIAL_VIS_NAMES%%;
const IN_BLOCKS=%%IN_BLOCKS_JS%%; 
const EXP_LIST=%%EXP_LIST_JS%%;
const PREV_IN_BLOCKS=%%PREV_IN_BLOCKS_JS%%;
const PREV_EXP_LIST=%%PREV_EXP_LIST_JS%%;
const IS_TC1=%%IS_TC1%%;
const IS_ALL_SIGNALS=%%IS_ALL_SIGNALS%%;
window.TC_ROWS = %%TC_ROWS_JS%%;

let LANE_H = %%LANE_H%%;
const NAMES_W=%%NAMES_W%%;
let PLOT_W = 800; 

let currentScenario = 0;
const scenarioSel = document.getElementById('scenario-sel');
if (IN_BLOCKS.length > 1) {
    document.getElementById('scenario-sep').style.display = 'block';
    document.getElementById('scenario-lbl').style.display = 'block';
    scenarioSel.style.display = 'block';
    scenarioSel.innerHTML = IN_BLOCKS.map((_, i) => `<option value="${i}">Scenario ${i+1}</option>`).join('');
}

if (IS_ALL_SIGNALS) {
    let addBtn = document.getElementById('add-sig-btn');
    if (addBtn) addBtn.style.display = 'none';
    let ixBtn = document.getElementById('ix-btn');
    if (ixBtn) ixBtn.style.display = 'none';
    let modBtn = document.getElementById('modify-btn');
    if (modBtn) modBtn.style.display = 'none';
}

function getActiveConds() {
    var conds = [];
    if (IN_BLOCKS.length > 0 && IN_BLOCKS[currentScenario]) { 
        IN_BLOCKS[currentScenario].forEach(function(c){ conds.push(c); }); 
    }
    EXP_LIST.forEach(function(c){ conds.push(c); });
    return conds;
}

// Global state for Auto Intersect Toggle
let isAutoIntersectOn = false;
let isModifyMode = false;

function toggleModifyMode() {
    isModifyMode = !isModifyMode;
    var btn = document.getElementById('modify-btn');
    btn.className = isModifyMode ? 'ix-toggle-on' : 'ix-toggle-off';
    buildRows();
    syncLayout();
    redraw();
}

function updateAutoIntersectBtn() {
    if(IS_ALL_SIGNALS) return;
    var btn = document.getElementById('ix-btn');
    var hasConds = getActiveConds().length > 0;
    
    if (!hasConds) {
        btn.className = 'ix-toggle-disabled';
        btn.disabled = true;
        btn.innerHTML = '⊕ Auto Intersect: N/A';
        isAutoIntersectOn = false;
        unlockCursor();
    } else {
        btn.disabled = false;
        if (isAutoIntersectOn) {
            btn.className = 'ix-toggle-on';
            btn.innerHTML = '⊕ Auto Intersect: ON';
        } else {
            btn.className = 'ix-toggle-off';
            btn.innerHTML = '⊕ Auto Intersect: OFF';
        }
    }
}

function toggleAutoIntersect() {
    isAutoIntersectOn = !isAutoIntersectOn;
    updateAutoIntersectBtn();
    if(isAutoIntersectOn) {
        autoIntersect();
    } else {
        unlockCursor();
    }
}

function getScenarioSigNames() {
    var activeConds = getActiveConds();
    var neededNames = new Set(activeConds.map(function(c){ return c.name.toLowerCase(); }));
    activeConds.forEach(function(c){
        if(typeof c.value === 'string') neededNames.add(c.value.toLowerCase());
    });
    return neededNames;
}

function getCondsFor(name){ 
    if(!name) return []; 
    return getActiveConds().filter(function(c){ return c.name.toLowerCase() === name.toLowerCase(); }); 
}

function filterSigs(q){
  q=(q||'').toLowerCase().trim();
  if(q) {
      visSigs=ALL_SIGS.filter(function(s){return s.name.toLowerCase().indexOf(q)>=0;});
  } else {
      if(INITIAL_VIS_NAMES.length > 0) {
          visSigs=ALL_SIGS.filter(function(s){return INITIAL_VIS_NAMES.includes(s.name.toLowerCase());});
      } else {
          visSigs=ALL_SIGS.slice(); 
      }
  }
  buildRows();
  syncLayout();
  redraw();
}

function changeScenario() {
    currentScenario = parseInt(scenarioSel.value);
    document.getElementById('search').value = '';
    filterSigs(''); 
    updateAutoIntersectBtn();
    if(isAutoIntersectOn) autoIntersect();
}

// ── ADD SIGNAL LOGIC ───────────────────────────────────────────────────
function openAddSignalModal() {
    document.getElementById('add-sig-modal').style.display = 'block';
    document.getElementById('add-sig-search').value = '';
    renderAddSigList('');
}
function renderAddSigList(q) {
    q = q.toLowerCase();
    var list = document.getElementById('add-sig-list');
    list.innerHTML = '';
    var avail = ALL_SIGS.filter(s => s.name.toLowerCase().includes(q) && !visSigs.includes(s));
    
    var frag = document.createDocumentFragment();
    avail.slice(0, 500).forEach(s => {
        var d = document.createElement('div');
        d.className = 'add-sig-item';
        d.textContent = s.name;
        d.onclick = function() { 
            visSigs.unshift(s); // APPEND TO TOP
            buildRows(); syncLayout(); redraw(); 
            document.getElementById('add-sig-modal').style.display='none'; 
        };
        frag.appendChild(d);
    });
    list.appendChild(frag);
}
function removeVisSig(idx) {
    var target = ALL_SIGS[idx];
    visSigs = visSigs.filter(s => s !== target);
    buildRows(); syncLayout(); redraw();
}

// ── DYNAMIC EVALUATION TABLE GENERATOR ──────────────────────────────────
function generateSubstitutedTable(t) {
    if (!window.TC_ROWS || window.TC_ROWS.length === 0) return "<p style='color:var(--text3); font-family:var(--sans); font-size:13px;'>No checksheet data available for this Test Case.</p>";
    var cols = Object.keys(window.TC_ROWS[0]);
    
    var html = '<table class="report-table" style="width:100%; border-collapse:collapse; margin-bottom:30px;"><thead><tr>';
    cols.forEach(c => { html += '<th style="background:#f1f5f9; padding:10px; border:1px solid #cbd5e1; text-align:left;">' + c + '</th>'; });
    html += '</tr></thead><tbody>';
    
    window.TC_ROWS.forEach(row => {
        html += '<tr>';
        cols.forEach(c => {
            var cellVal = row[c];
            if (cellVal === null || cellVal === undefined) cellVal = "";
            var origText = String(cellVal);
            var subText = origText;
            
            var words = origText.match(/\b([A-Za-z_][A-Za-z0-9_]{2,})\b/g) || [];
            var replaced = new Set();
            words.forEach(w => {
                if (replaced.has(w)) return;
                var s = ALL_SIGS.find(x => x.name.toLowerCase() === w.toLowerCase());
                if (s) {
                    var val = getValAt(s.name, t);
                    var valStr = fmtS(val);
                    subText = subText.replace(new RegExp('\\b'+w+'\\b', 'g'), '<span style="color:#0071e3;font-weight:bold;">'+valStr+'</span>');
                    replaced.add(w);
                }
            });
            
            if (origText !== subText && replaced.size > 0) {
                html += '<td style="padding:10px; border:1px solid #cbd5e1; vertical-align:top;"><div style="margin-bottom:6px;">'+origText+'</div><div style="font-family:monospace; background:rgba(0,0,0,0.03); padding:6px; border-radius:4px; font-size:11px; border:1px solid rgba(0,0,0,0.05);"><b>Calculation Check:</b><br>'+subText+'</div></td>';
            } else {
                html += '<td style="padding:10px; border:1px solid #cbd5e1; vertical-align:top;">'+origText+'</td>';
            }
        });
        html += '</tr>';
    });
    html += '</tbody></table>';
    return html;
}

// ── EXPOSE REPORT GENERATOR FOR TOP WINDOW ──────────────────────────────
window.getReportPart = function(tcLabel) {
    var isHidden = (PLOT_W === 0 || rightPanel.clientWidth === 0);
    var tempW = isHidden ? 1000 : PLOT_W;
    
    var oldScrollTop = rightPanel.scrollTop;
    var oldCvTop = _cvTop;
    var oldCvH = _cvH;
    var oldLaneH = LANE_H;

    // Force strict 40px rendering for standardized report layout
    LANE_H = 40;
    PLOT_W = tempW;
    _cvTop = 0;
    
    buildRows(); 
    _cvH = totalH();
    
    plotCV.width = PLOT_W; hoverCV.width = PLOT_W; cursorCV.width = PLOT_W;
    plotCV.height = _cvH; hoverCV.height = _cvH; cursorCV.height = _cvH;
    plotCV.style.top = '0px'; hoverCV.style.top = '0px'; cursorCV.style.top = '0px';

    drawPlot(); 
    if(lockedX !== null) drawLockedCursor(lockedX);
    
    // CRITICAL: Refresh values exactly at locked point before capturing HTML
    var evalT = cursorLocked ? xToT(lockedX) : vStart;
    updateVP(evalT);

    var compCV = document.createElement('canvas');
    compCV.width = PLOT_W; compCV.height = _cvH;
    var ctx = compCV.getContext('2d');
    ctx.fillStyle = lightMode ? '#faf9f7' : '#000000';
    ctx.fillRect(0, 0, PLOT_W, _cvH);
    ctx.drawImage(plotCV, 0, 0);
    ctx.drawImage(cursorCV, 0, 0);
    var graphB64 = compCV.toDataURL('image/png');

    var oldXaW = xaCV.width;
    var oldXaStyleW = xaCV.style.width;
    xaCV.width = PLOT_W;
    xaCV.style.width = PLOT_W + 'px';
    drawXAxis(true); 
    var xaB64 = xaCV.toDataURL('image/png');
    xaCV.width = oldXaW;
    xaCV.style.width = oldXaStyleW;

    var leftHtml = leftInner.innerHTML;
    // Strip Remove buttons for export
    leftHtml = leftHtml.replace(/<button[^>]*>[\s\S]*?<\/button>/gi, ''); 

    // Restore viewport state
    LANE_H = oldLaneH;
    _cvTop = oldCvTop; _cvH = oldCvH;
    buildRows(); 
    syncLayout(); 
    redraw();
    rightPanel.scrollTop = oldScrollTop;
    
    var evalTableHtml = generateSubstitutedTable(evalT);
    var missingHtml = "%%MISSING_SIGS_TEXT%%";
    
    var pageBg = lightMode ? '#ffffff' : '#000000';
    var borderClr = lightMode ? '#cbd5e1' : '#38383a';

    var html = `
      <div style="margin-bottom: 60px; background:${pageBg}; padding:24px; border-radius:12px; border:1px solid ${borderClr}; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
          <h2 style="margin-top:0; color:#0071e3; border-bottom:2px solid #0071e3; padding-bottom:10px; font-size:22px;">${tcLabel}</h2>
          
          <h3 style="margin-top:24px; font-size:16px;">Checksheet Data & Calculations</h3>
          ${evalTableHtml}
          
          <h3 style="margin-top:30px; font-size:16px;">Signal Intersection Graph</h3>
          <div style="display: flex; border: 1px solid ${borderClr}; border-radius: 8px; overflow-x: auto; box-shadow: 0 4px 12px rgba(0,0,0,0.1); width: 100%; box-sizing: border-box; background: ${pageBg};">
             <div style="width: ${NAMES_W}px; flex-shrink: 0; border-right: 1px solid ${borderClr}; background: ${pageBg}; box-sizing: border-box; overflow-x: auto;">${leftHtml}</div>
             <div style="flex: 0 0 ${PLOT_W}px; width: ${PLOT_W}px; display: flex; flex-direction: column; box-sizing: border-box; overflow: hidden;">
                <img src="${graphB64}" style="width: ${PLOT_W}px; height: ${_cvH}px; display: block; max-width: none;">
                <img src="${xaB64}" style="width: ${PLOT_W}px; height:32px; display: block; max-width: none;">
             </div>
          </div>
      </div>`;
    return html;
}


let vStart=T_MIN,vEnd=T_MAX,lockedX=null,cursorLocked=false,hoverX=null,drag=null;
let visSigs=[],_cvTop=0,_cvH=600;
let prevIntersectT=null; 

const rightPanel = document.getElementById('right-panel');
const leftPanel = document.getElementById('left-panel');
const leftInner = document.getElementById('left-inner');
const rightInner = document.getElementById('right-inner');

const plotCV = document.getElementById('plot-cv');
const hoverCV = document.getElementById('hover-cv');
const cursorCV = document.getElementById('cursor-cv');
const xaWrap = document.getElementById('xa-wrap');
const xaCV = document.getElementById('xa-cv');

function lob(arr,val){var lo=0,hi=arr.length-1;while(lo<hi){var m=(lo+hi+1)>>1;arr[m]<=val?lo=m:hi=m-1;}return Math.max(0,lo);}
function fmt(v){if(v==null||isNaN(v))return'—';var a=Math.abs(v);if(a===0)return'0';if(a>=1e6)return v.toExponential(3);if(a>=1000)return v.toFixed(2);if(a>=1)return v.toFixed(4);return v.toFixed(5);}
function fmtS(v){if(v==null||isNaN(v))return'—';var a=Math.abs(v);if(a===0)return'0';if(a>=1e5)return v.toExponential(1);if(a>=10000)return v.toFixed(0);if(a>=100)return v.toFixed(1);if(a>=1)return v.toFixed(3);return v.toFixed(3);}
function fmtT(t){var a=Math.abs(t);if(a>=1000)return t.toFixed(1);if(a>=100)return t.toFixed(2);if(a>=10)return t.toFixed(3);return t.toFixed(4);}

function changeLaneH() {
    LANE_H = parseInt(document.getElementById('lane-h-sel').value);
    buildRows(); 
    syncLayout();
    redraw();
}

function toggleFullScreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
        document.getElementById('fs-btn').textContent = '🗗 Exit Fullscreen';
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
            document.getElementById('fs-btn').textContent = '⛶ Fullscreen';
        }
    }
}

function getValAt(name, t) {
  var s = visSigs.find(function(x){ return x.name.toLowerCase() === name.toLowerCase(); });
  if(!s) s = ALL_SIGS.find(function(x){ return x.name.toLowerCase() === name.toLowerCase(); });
  if(!s) return null;
  var i = lob(s.x, t);
  return (i>=0 && i<s.y.length) ? s.y[i] : null;
}

function evalCond(val, ev, t){
  if(val === null || ev === null) return null;
  var tgt = typeof ev.value === 'string' ? getValAt(ev.value, t) : ev.value;
  if(tgt === null) return null;
  var eps = Math.max(Math.abs(tgt)*0.01, 0.02); 
  switch(ev.op){
    case'=':case'==':return Math.abs(val-tgt)<=eps;
    case'>': return val>tgt;
    case'>=':return val>=tgt-eps;
    case'<': return val<tgt;
    case'<=':return val<=tgt+eps;
    case'!=':return Math.abs(val-tgt)>eps;
  }
  return null;
}

function buildRows(){
  leftInner.innerHTML='';
  
  visSigs.forEach(function(s){
    var i=ALL_SIGS.indexOf(s);
    var sigColor = s.color; 

    var nr=document.createElement('div');
    nr.className='nrow';nr.id='nr-'+i;nr.title=s.name;
    nr.style.height = LANE_H + 'px';
    nr.style.boxSizing = 'border-box';
    
    // Clean text-based Modify Delete button
    var isAdded = (!IS_ALL_SIGNALS && INITIAL_VIS_NAMES.length > 0 && !INITIAL_VIS_NAMES.includes(s.name.toLowerCase()));
    var delBtn = '';
    if (isAdded && isModifyMode) {
        delBtn = '<button onclick="removeVisSig('+i+')" style="background:var(--red);border:none;cursor:pointer;color:#fff;padding:4px 8px;margin-left:8px;border-radius:4px;font-size:10px;font-weight:600;" title="Remove Signal">Remove</button>';
    }

    var tickSpan = '<span id="vs-'+i+'" style="margin-right:6px; font-weight:bold; font-size:12px;"></span>';

    nr.innerHTML=
      '<div style="display:flex; align-items:center; width:100%; height:100%;">' +
        '<div class="nrow-color-box" style="background:'+sigColor+'; margin-right:8px;"></div>' +
        '<div class="nrow-body" style="flex:1; min-width:0; display:flex; align-items:center;">' +
            '<span class="nrow-name" style="color:'+sigColor+';">'+tickSpan+s.name+'</span>' +
            '<span class="nrow-val" id="vv-'+i+'" style="color:'+sigColor+';">—</span>' +
            delBtn +
        '</div>' +
      '</div>';
    leftInner.appendChild(nr);
  });
  leftInner.style.height=(visSigs.length*LANE_H)+'px';
}

function totalH(){return Math.max(400,visSigs.length*LANE_H);}
function tToX(t){return((t-vStart)/(vEnd-vStart))*PLOT_W;}
function xToT(px){return vStart+(px/PLOT_W)*(vEnd-vStart);}

function syncLayout(){
  var vh = rightPanel.clientHeight || 600;
  PLOT_W = rightPanel.clientWidth || 800; 

  var sTop = rightPanel.scrollTop;
  var buf = LANE_H * 4;
  
  _cvTop = Math.floor(Math.max(0, sTop - buf) / LANE_H) * LANE_H;
  _cvH = Math.max(vh, Math.min(totalH(), sTop + vh + buf) - _cvTop);
  
  [plotCV, hoverCV, cursorCV].forEach(function(cv){
    if(cv.width !== PLOT_W) cv.width = PLOT_W;
    if(cv.height !== Math.ceil(_cvH)) cv.height = Math.ceil(_cvH);
    cv.style.width = PLOT_W + 'px';
    cv.style.height = Math.ceil(_cvH) + 'px';
    cv.style.top = _cvTop + 'px';
  });

  var totHeight = totalH() + 'px';
  leftInner.style.height = totHeight;
  rightInner.style.height = totHeight;
}

function laneY(i){return i*LANE_H-_cvTop;}
function laneInView(i){var t=laneY(i);return t+LANE_H>0&&t<_cvH;}

function getGridTicks(overrideW) {
    var vpW = overrideW || PLOT_W;
    if(vpW <= 0) return null;
    var tL = vStart;
    var tR = vEnd;
    var range = tR - tL;
    
    var maxTicks = Math.max(10, Math.floor(vpW / 60)); 
    var roughStep = range / maxTicks;
    if(roughStep <= 0) return null;
    
    var p = Math.floor(Math.log10(roughStep));
    var stepPower = Math.pow(10, p);
    var norm = roughStep / stepPower;
    var niceStep = (norm < 1.5) ? 1 : (norm < 3) ? 2 : (norm < 7) ? 5 : 10;
    niceStep *= stepPower;
    
    return { niceStep: niceStep, tL: tL, tR: tR, range: range, vpW: vpW };
}

function drawPlot(){
  var thm = lightMode ? THEMES.light : THEMES.dark;
  var W=PLOT_W,H=Math.ceil(_cvH),ctx=plotCV.getContext('2d');
  ctx.fillStyle=thm.bg; ctx.fillRect(0,0,W,H);
  
  var g = getGridTicks();
  if (g) {
      ctx.strokeStyle=thm.grid; ctx.lineWidth=1; ctx.beginPath();
      var firstAll = Math.ceil(vStart / g.niceStep) * g.niceStep;
      for(var t = firstAll; t <= vEnd; t += g.niceStep){
          var px = tToX(t);
          ctx.moveTo(px, 0); ctx.lineTo(px, H);
      }
      ctx.stroke();
  }

  visSigs.forEach(function(s,i){
    if(!laneInView(i))return;
    var lT=laneY(i),lB=lT+LANE_H;
    var pad = LANE_H <= 50 ? 4 : 10; 
    var dT=lT+pad,dB=lB-pad,dH=dB-dT,rng=s.mx-s.mn;
    
    ctx.strokeStyle=thm.border; ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(0,lB-0.5);ctx.lineTo(W,lB-0.5);ctx.stroke();
    ctx.save();ctx.beginPath();ctx.rect(0,lT,W,LANE_H);ctx.clip();

    ctx.strokeStyle=thm.grid; ctx.lineWidth=1; ctx.beginPath();
    for(let g=1; g<4; g++){ let gy = dT + (dH * (g/4)); ctx.moveTo(0, gy); ctx.lineTo(W, gy); }
    ctx.stroke();

    if(s.raw_mn<0&&s.raw_mx>0&&rng>0){
      var zy=dT+(1-(0-s.mn)/rng)*dH;
      ctx.strokeStyle=lightMode?'rgba(0,0,0,0.1)':'rgba(255,255,255,0.15)';
      ctx.lineWidth=1;ctx.setLineDash([4,4]);
      ctx.beginPath();ctx.moveTo(0,zy);ctx.lineTo(W,zy);ctx.stroke();ctx.setLineDash([]);
    }

    var xs=s.x,ys=s.y,len=xs.length;
    if(!len){ctx.restore();return;}
    var i0=0,i1=len-1;
    while(i0<len-1&&xs[i0+1]<vStart)i0++;
    while(i1>0&&xs[i1-1]>vEnd)i1--;
    i0=Math.max(0,i0-1);i1=Math.min(len-1,i1+1);
    var step=Math.max(1,Math.floor((i1-i0)/(W*2)));
    
    ctx.strokeStyle = s.color;
    ctx.lineWidth=1.5;ctx.lineJoin='round';ctx.beginPath();var go=false;
    for(var j=i0;j<=i1;j+=step){
        var px=tToX(xs[j]),norm=rng>0?Math.max(0,Math.min(1,(ys[j]-s.mn)/rng)):0.5,py=dT+(1-norm)*dH;
        if(!go){ctx.moveTo(px,py);go=true;}else ctx.lineTo(px,py);
    }
    ctx.stroke();ctx.restore();
  });
}

function drawXAxis(exportMode = false){
  var thm = lightMode ? THEMES.light : THEMES.dark;
  var W = exportMode ? PLOT_W : (xaWrap.clientWidth || 800);
  if(xaCV.width !== W) xaCV.width = W;
  xaCV.height = 32; xaCV.style.width = W + 'px';
  
  var ctx = xaCV.getContext('2d');
  ctx.fillStyle = thm.surface2; ctx.fillRect(0,0,W,32);
  
  var g = getGridTicks(W); 
  if (!g) return;

  ctx.font = '10px JetBrains Mono,monospace';
  ctx.fillStyle = thm.text3; ctx.textAlign = 'center';
  ctx.strokeStyle = thm.border2;

  var startT = exportMode ? vStart : g.tL;
  var endT = exportMode ? vEnd : g.tR;
  var first = Math.ceil(startT / g.niceStep) * g.niceStep;

  for(var t = first; t <= endT; t += g.niceStep){
    var px = exportMode ? tToX(t) : ((t - g.tL) / g.range) * W;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,8); ctx.stroke();
    ctx.fillText(fmtT(t), px, 22);

    ctx.lineWidth = 1;
    var subTicks = (g.niceStep === 2 * Math.pow(10, Math.floor(Math.log10(g.niceStep)))) ? 4 : 5;
    var subStep = g.niceStep / subTicks;
    for(var j=1; j<subTicks; j++) {
        var st = t - g.niceStep + j*subStep;
        if (st >= startT && st <= endT) {
            var spx = exportMode ? tToX(st) : ((st - g.tL) / g.range) * W;
            ctx.beginPath(); ctx.moveTo(spx,0); ctx.lineTo(spx,4); ctx.stroke();
        }
    }
  }
}

function drawHoverLine(canvX){
  var thm = lightMode ? THEMES.light : THEMES.dark;
  var H=Math.ceil(_cvH),ctx=hoverCV.getContext('2d');
  ctx.clearRect(0,0,PLOT_W,H);if(canvX===null)return;
  var t=xToT(canvX);
  ctx.strokeStyle=thm.text3; ctx.lineWidth=1;ctx.setLineDash([4,4]);
  ctx.beginPath();ctx.moveTo(canvX,0);ctx.lineTo(canvX,H);ctx.stroke();ctx.setLineDash([]);
  
  visSigs.forEach(function(s,i){
    if(!laneInView(i))return;
    var lT=laneY(i),pad=LANE_H<=50?4:10,dT=lT+pad,dH=LANE_H-pad*2,rng=s.mx-s.mn;
    var idx=lob(s.x,t);if(idx<0||idx>=s.y.length)return;
    var v=s.y[idx],norm=rng>0?Math.max(0,Math.min(1,(v-s.mn)/rng)):0.5,py=dT+(1-norm)*dH;
    
    ctx.fillStyle=s.color;ctx.globalAlpha=0.9;
    ctx.beginPath();ctx.arc(canvX,py,4,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;
    
    var vStr = fmt(v);
    ctx.font = '600 9px var(--mono)';
    var vTw = ctx.measureText(vStr).width;
    var textX = canvX + 11;
    var boxX = canvX + 8;
    if (canvX > PLOT_W - 60) { textX = canvX - vTw - 11; boxX = canvX - vTw - 14; }
    ctx.fillStyle = lightMode ? 'rgba(255,255,255,0.9)' : 'rgba(28,28,30,0.9)';
    ctx.fillRect(boxX, py - 7, vTw + 6, 14);
    ctx.fillStyle = s.color;
    ctx.textAlign = 'left';
    ctx.fillText(vStr, textX, py + 3);
  });
}

function drawLockedCursor(canvX){
  var thm = lightMode ? THEMES.light : THEMES.dark;
  var H=Math.ceil(_cvH),ctx=cursorCV.getContext('2d');
  ctx.clearRect(0,0,PLOT_W,H);if(canvX===null)return;
  var t=xToT(canvX);
  var cCursor = thm.cursor;
  
  // Draw PREVIOUS intersect line & values
  if (prevIntersectT !== null) {
      var pX = tToX(prevIntersectT);
      if (pX >= 0 && pX <= PLOT_W) {
          ctx.strokeStyle = thm.text3; ctx.lineWidth = 1.5; ctx.setLineDash([5,5]);
          ctx.beginPath(); ctx.moveTo(pX,0); ctx.lineTo(pX,H); ctx.stroke(); ctx.setLineDash([]);
          
          var pTag = IS_TC1 ? ' ATTEMPT ' : ' PREV TC ';
          ctx.font='bold 9px JetBrains Mono,monospace';
          var pTw=ctx.measureText(pTag).width+8;
          ctx.fillStyle=thm.surface2; ctx.fillRect(pX-pTw/2, H-20, pTw, 14);
          ctx.fillStyle=thm.text3; ctx.textAlign='center'; ctx.fillText(pTag, pX, H-10);
          
          visSigs.forEach(function(s,i){
              if(!laneInView(i))return;
              var lT=laneY(i),pad=LANE_H<=50?4:10,dT=lT+pad,dH=LANE_H-pad*2,rng=s.mx-s.mn;
              var idx=lob(s.x,prevIntersectT);if(idx<0||idx>=s.y.length)return;
              var v=s.y[idx],norm=rng>0?Math.max(0,Math.min(1,(v-s.mn)/rng)):0.5,py=dT+(1-norm)*dH;
              
              ctx.fillStyle=thm.bg; ctx.beginPath();ctx.arc(pX,py,4,0,Math.PI*2);ctx.fill();
              ctx.strokeStyle=thm.text3; ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(pX,py,3,0,Math.PI*2);ctx.stroke();
              
              var vStr = fmt(v);
              ctx.font = '500 8.5px var(--mono)';
              var vTw2 = ctx.measureText(vStr).width;
              var textX = pX + 8;
              var boxX = pX + 5;
              if (pX > PLOT_W - 50) { textX = pX - vTw2 - 8; boxX = pX - vTw2 - 11; }
              ctx.fillStyle = thm.surface;
              ctx.fillRect(boxX, py - 6, vTw2 + 6, 12);
              ctx.fillStyle = thm.text3;
              ctx.textAlign = 'left';
              ctx.fillText(vStr, textX, py + 3);
          });
      }
  }

  // Draw MAIN intersect line
  ctx.strokeStyle=cCursor; ctx.lineWidth=1.5;
  ctx.beginPath();ctx.moveTo(canvX,0);ctx.lineTo(canvX,H);ctx.stroke();
  var tag=' C1: '+fmtT(t)+' s ';
  ctx.font='bold 10px JetBrains Mono,monospace';
  var tw=ctx.measureText(tag).width+8;
  ctx.fillStyle=cCursor; ctx.fillRect(canvX-tw/2,2,tw,16);
  ctx.fillStyle='#ffffff';ctx.textAlign='center';ctx.fillText(tag,canvX,13);
  
  visSigs.forEach(function(s,i){
    if(!laneInView(i))return;
    var lT=laneY(i),pad=LANE_H<=50?4:10,dT=lT+pad,dH=LANE_H-pad*2,rng=s.mx-s.mn;
    var idx=lob(s.x,t);if(idx<0||idx>=s.y.length)return;
    var v=s.y[idx],norm=rng>0?Math.max(0,Math.min(1,(v-s.mn)/rng)):0.5,py=dT+(1-norm)*dH;
    
    ctx.fillStyle=thm.bg; ctx.beginPath();ctx.arc(canvX,py,6,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle=s.color; ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(canvX,py,4.5,0,Math.PI*2);ctx.stroke();
    ctx.fillStyle=s.color; ctx.beginPath();ctx.arc(canvX,py,2,0,Math.PI*2);ctx.fill();
    
    var vStr = fmt(v);
    ctx.font = '600 9px var(--mono)';
    var vTw = ctx.measureText(vStr).width;
    var textX = canvX + 11;
    var boxX = canvX + 8;
    if (canvX > PLOT_W - 60) { textX = canvX - vTw - 11; boxX = canvX - vTw - 14; }
    ctx.fillStyle = thm.surface;
    ctx.fillRect(boxX, py - 7, vTw + 6, 14);
    ctx.fillStyle = s.color;
    ctx.textAlign = 'left';
    ctx.fillText(vStr, textX, py + 3);
  });
}

function updateVP(t){
  if(t === null) return;
  var thm = lightMode ? THEMES.light : THEMES.dark;
  
  visSigs.forEach(function(s){
    var ai = ALL_SIGS.indexOf(s);
    var v = getValAt(s.name, t);
    
    var valSpan = document.getElementById('vv-'+ai);
    if(valSpan) valSpan.textContent = fmt(v);

    var conds = getCondsFor(s.name);
    var vs=document.getElementById('vs-'+ai);
    if(vs){
        if(conds.length > 0){
          var metAll = true;
          for(var j=0; j<conds.length; j++) {
              if(!evalCond(v, conds[j], t)) { metAll = false; break; }
          }
          vs.textContent = metAll ? '✓ ' : '✗ ';
          vs.style.color = metAll ? thm.green : thm.red;
        } else {
          var isValid = (v !== null);
          vs.textContent = isValid ? '✓ ' : '✗ ';
          vs.style.color = isValid ? thm.green : thm.red;
        }
    }
  });
}

function clearVP(){
  ALL_SIGS.forEach(function(_,i){
    var e=document.getElementById('vv-'+i);if(e)e.textContent='—';
    var s=document.getElementById('vs-'+i);if(s)s.textContent='';
  });
}

function findGenericIntersection(conds) {
    if(!conds || !conds.length) return null;
    var tSet = new Set();
    if(ALL_SIGS[0] && ALL_SIGS[0].x) {
        for(var i=0; i<ALL_SIGS[0].x.length; i++) tSet.add(ALL_SIGS[0].x[i]);
    }
    var tArr = Array.from(tSet).sort((a,b)=>a-b);
    var bestT = null, bestScore = Infinity;
    for(var i=0; i<tArr.length; i++){
        var t = tArr[i];
        var allMet = true, totalScore = 0;
        for(var j=0; j<conds.length; j++){
           var c = conds[j];
           var val = getValAt(c.name, t);
           var isMet = evalCond(val, c, t);
           if (!isMet) allMet = false;
           var tgt = typeof c.value==='string' ? getValAt(c.value, t) : c.value;
           if(val !== null && tgt !== null) {
               var diff = Math.abs(val - tgt);
               if (isMet) diff = 0; 
               totalScore += diff / (Math.abs(tgt)+0.1);
           } else { totalScore += 1000; }
        }
        if(allMet){ return {t: t, exact: true}; }
        if(totalScore < bestScore){ bestScore = totalScore; bestT = t; }
    }
    return {t: bestT, exact: false};
}

function computePrevIntersect(mainT) {
    if (IS_TC1) {
        var activeConds = getActiveConds();
        if(!activeConds.length) return null;
        var tSet = new Set();
        if(ALL_SIGS[0] && ALL_SIGS[0].x) ALL_SIGS[0].x.forEach(x => tSet.add(x));
        var tArr = Array.from(tSet).sort((a,b)=>a-b);
        for(var i=0; i<tArr.length; i++) {
            var t = tArr[i];
            if (mainT && t >= mainT) break;
            var metCount = 0;
            activeConds.forEach(c => { if(evalCond(getValAt(c.name, t), c, t)) metCount++; });
            if (metCount > 0) return t;
        }
        return mainT ? Math.max(T_MIN, mainT - (T_MAX-T_MIN)*0.05) : null;
    } else {
        var bestPrevT = null;
        if (PREV_IN_BLOCKS.length > 0) {
            for(var s=0; s<PREV_IN_BLOCKS.length; s++) {
                var prevConds = [];
                PREV_IN_BLOCKS[s].forEach(c => prevConds.push(c));
                PREV_EXP_LIST.forEach(c => prevConds.push(c));
                var res = findGenericIntersection(prevConds);
                if (res && res.t !== null && res.exact) {
                    bestPrevT = res.t;
                    break;
                } else if (res && res.t !== null) {
                    bestPrevT = res.t; 
                }
            }
        } else {
            var prevConds = [];
            PREV_EXP_LIST.forEach(c => prevConds.push(c));
            var res = findGenericIntersection(prevConds);
            if (res && res.t !== null) bestPrevT = res.t;
        }
        return bestPrevT;
    }
}

function autoIntersect(){
  var activeConds = getActiveConds();
  var result = findGenericIntersection(activeConds);
  if(!result || result.t === null){return;}
  
  prevIntersectT = computePrevIntersect(result.t);
  
  if (result.t < vStart || result.t > vEnd) {
      var span = vEnd - vStart;
      vStart = Math.max(T_MIN, result.t - span / 2);
      vEnd = Math.min(T_MAX, vStart + span);
      if (vEnd === T_MAX) vStart = Math.max(T_MIN, vEnd - span);
  }
  
  var canvX = tToX(result.t);
  setCursorLocked(true, canvX);
  
  hoverCV.getContext('2d').clearRect(0,0,PLOT_W,Math.ceil(_cvH));
  drawLockedCursor(canvX); 
  updateVP(result.t);
  drawXAxis(); 
}

function setCursorLocked(locked,x){
  cursorLocked=locked;lockedX=locked?x:null;
  if(!locked){
      cursorCV.getContext('2d').clearRect(0,0,PLOT_W,Math.ceil(_cvH));
      clearVP();
  }
}

function unlockCursor() {
    setCursorLocked(false, null);
    if(hoverX !== null) updateVP(xToT(hoverX));
}

let _raf=false;
function redraw(){
  if(_raf)return;_raf=true;
  requestAnimationFrame(function(){
    _raf=false;syncLayout();drawPlot();drawXAxis();
    if(lockedX!==null) { lockedX = tToX(xToT(lockedX)); drawLockedCursor(lockedX); }
    if(hoverX!==null)drawHoverLine(hoverX);
    if(cursorLocked) updateVP(xToT(lockedX));
  });
}

// ── Two Pane Synced Scroll Logic ──
let isSyncLeft = false;
let isSyncRight = false;

leftPanel.addEventListener('scroll', function() {
    if (!isSyncLeft) {
        isSyncRight = true;
        rightPanel.scrollTop = this.scrollTop;
    }
    isSyncLeft = false;
});

rightPanel.addEventListener('scroll', function() {
    if (!isSyncRight) {
        isSyncLeft = true;
        leftPanel.scrollTop = this.scrollTop;
    }
    isSyncRight = false;
    
    if(rightPanel.scrollTop<_cvTop+LANE_H||rightPanel.scrollTop+rightPanel.clientHeight>_cvTop+_cvH-LANE_H)redraw();
    else if(lockedX!==null)drawLockedCursor(lockedX);
});

rightPanel.addEventListener('mousedown',function(ev){
  drag={sx:ev.clientX, vStart: vStart, vEnd: vEnd, span: vEnd - vStart}; 
  rightPanel.style.cursor='grabbing'; 
});

window.addEventListener('mouseup',function(){ drag=null; rightPanel.style.cursor='crosshair'; });

rightPanel.addEventListener('mousemove',function(ev){
  var rect=rightInner.getBoundingClientRect();
  var cx=ev.clientX-rect.left;
  
  if(drag){
      var dx = ev.clientX - drag.sx;
      var dt = (dx / PLOT_W) * drag.span;
      vStart = drag.vStart - dt;
      vEnd = drag.vEnd - dt;
      
      if(vStart < T_MIN) { vStart = T_MIN; vEnd = vStart + drag.span; }
      if(vEnd > T_MAX) { vEnd = T_MAX; vStart = Math.max(T_MIN, vEnd - drag.span); }
      redraw();
      return;
  }
  
  if (cx < 0 || cx > PLOT_W) return;
  hoverX=cx; 
  drawHoverLine(hoverX);
  
  if(!cursorLocked) { updateVP(xToT(hoverX)); }
});

rightPanel.addEventListener('mouseleave',function(){
  hoverX=null; 
  hoverCV.getContext('2d').clearRect(0,0,PLOT_W,Math.ceil(_cvH));
  if(cursorLocked) { updateVP(xToT(lockedX)); } 
  else { clearVP(); }
});

rightPanel.addEventListener('click',function(ev){
  var rect=rightInner.getBoundingClientRect();
  if(drag&&Math.abs(ev.clientX-drag.sx)>4)return;
  var cx=ev.clientX-rect.left;
  
  if (isAutoIntersectOn) {
      toggleAutoIntersect();
  }

  if(!cursorLocked) {
      setCursorLocked(true,cx); 
  } else { 
      unlockCursor();
      return;
  }
  hoverCV.getContext('2d').clearRect(0,0,PLOT_W,Math.ceil(_cvH));
  drawLockedCursor(lockedX);
  updateVP(xToT(lockedX));
});

rightPanel.addEventListener('dblclick',function(){
  fitAll();
  if(isAutoIntersectOn) toggleAutoIntersect();
  else unlockCursor();
});

rightPanel.addEventListener('wheel',function(ev){
  if(!ev.ctrlKey){return;}
  ev.preventDefault();
  var rect=rightInner.getBoundingClientRect();
  var vpx=ev.clientX-rect.left;
  if(vpx < 0) return;
  var pivot=xToT(vpx);
  var factor=ev.deltaY>0?1.25:0.8,minS=(T_MAX-T_MIN)*0.0002;
  var span=Math.max(minS,Math.min(T_MAX-T_MIN,(vEnd-vStart)*factor));
  vStart=pivot-(vpx/PLOT_W)*span;vEnd=vStart+span;
  if(vStart<T_MIN){vStart=T_MIN;vEnd=T_MIN+span;}
  if(vEnd>T_MAX){vEnd=T_MAX;vStart=Math.max(T_MIN,T_MAX-span);}
  redraw();
},{passive:false});

function zoomIn(){var c=(vStart+vEnd)/2,s=Math.max((T_MAX-T_MIN)*2e-4,(vEnd-vStart)*0.6);vStart=Math.max(T_MIN,c-s/2);vEnd=Math.min(T_MAX,c+s/2);redraw();}
function zoomOut(){var c=(vStart+vEnd)/2,s=Math.min(T_MAX-T_MIN,(vEnd-vStart)/0.6);vStart=Math.max(T_MIN,c-s/2);vEnd=Math.min(T_MAX,vStart+s);redraw();}

function resetView(){
    fitAll();
    if(isAutoIntersectOn) toggleAutoIntersect();
    else unlockCursor();
    redraw();
}
function fitAll(){vStart=T_MIN;vEnd=T_MAX;redraw();}

window.addEventListener('resize',redraw);

setTimeout(function(){
    document.getElementById('search').value = '';
    filterSigs(''); 
    syncTheme(); 
    fitAll(); 
    redraw();
    updateAutoIntersectBtn();
    
    if(getActiveConds().length > 0) {
        toggleAutoIntersect(); 
    }
}, 150);
</script>
</body></html>
"""

def build_viewer_html(df, time_col, filename="data", signal_cols=None, in_blocks=None, exp_list=None, prev_in_blocks=None, prev_exp_list=None, is_tc1=False, unmatched_sigs=None, tc_df=None, is_all_signals=False):
    unmatched_sigs = unmatched_sigs or []
    t_vals = pd.to_numeric(df[time_col], errors="coerce").values
    valid  = ~np.isnan(t_vals)
    
    if valid.sum() < 2:
        t_vals = np.arange(len(df), dtype=float)
        valid = np.ones(len(df), dtype=bool)
        time_col = "Row Index (Auto)"
        
    t_np   = t_vals[valid]
    df_v   = df.iloc[np.where(valid)[0]].reset_index(drop=True)
    t_min, t_max = float(t_np.min()), float(t_np.max())

    if t_min == t_max: t_max = t_min + 1.0

    skip = {time_col}
    all_numeric_cols = [c for c in df.columns if c not in skip and is_numeric(df[c])]
    
    initial_vis_names = [normalize_col_name(c).lower() for c in signal_cols] if signal_cols else []

    js_sigs = []
    for i, col in enumerate(all_numeric_cols[:MAX_SIGNALS]):
        raw = pd.to_numeric(df_v[col], errors="coerce").values
        mask = ~np.isnan(raw)
        if mask.sum() < 2: raw = np.zeros(len(t_np))
        elif not mask.all(): raw = pd.Series(raw).interpolate(method='linear', limit_direction='both').fillna(0).values
            
        raw_mn, raw_mx = float(raw.min()), float(raw.max())
        if not np.isfinite(raw_mn): raw_mn = 0.0
        if not np.isfinite(raw_mx): raw_mx = 0.0
        
        span = raw_mx - raw_mn
        pad = span * 0.06 if span > 0 else max(abs(raw_mn) * 0.06, 0.5)
        xs, ys = downsample(t_np, raw, DOWNSAMPLE_N)
        
        xs = [x if np.isfinite(x) else None for x in xs]
        ys = [y if np.isfinite(y) else None for y in ys]
        
        js_sigs.append({
            "name": normalize_col_name(col), "color": COLORS[i % len(COLORS)],
            "x": xs, "y": ys, "mn": raw_mn - pad, "mx": raw_mx + pad, "raw_mn": raw_mn, "raw_mx": raw_mx,
        })

    tc_rows_json = json.dumps(tc_df.fillna("").to_dict(orient="records")) if tc_df is not None else "[]"

    html = VIEWER_TEMPLATE
    
    warning_html = ""
    if unmatched_sigs:
        warning_html = f'<div style="background:var(--red, #ff3b30); color:#fff; padding:6px 16px; font-size:12px; font-family:var(--sans); font-weight:600; text-align:center; z-index:50; flex-shrink:0;">⚠ Warning: The following expected signals were not found in the DAT file: {", ".join(unmatched_sigs)}</div>'

    missing_text = ", ".join(unmatched_sigs) if unmatched_sigs else "None"

    html = html.replace("%%NAMES_W%%", str(NAMES_W))
    html = html.replace("%%LANE_H%%", str(LANE_H))
    html = html.replace("%%IX_BTN_DISABLED%%", "" if (in_blocks or exp_list) else "disabled")
    html = html.replace("%%T_MIN%%", str(t_min))
    html = html.replace("%%T_MAX%%", str(t_max))
    html = html.replace("%%T_MIN_FMT%%", f"{t_min:.4f}")
    html = html.replace("%%T_MAX_FMT%%", f"{t_max:.4f}")
    html = html.replace("%%WARNING_HTML%%", warning_html)
    html = html.replace("%%MISSING_SIGS_TEXT%%", missing_text)
    html = html.replace("%%TIME_COL_JS%%", json.dumps(time_col))
    html = html.replace("%%SIG_DATA%%", json.dumps(js_sigs))
    html = html.replace("%%INITIAL_VIS_NAMES%%", json.dumps(initial_vis_names))
    html = html.replace("%%IN_BLOCKS_JS%%", json.dumps(in_blocks or []))
    html = html.replace("%%EXP_LIST_JS%%", json.dumps(exp_list or []))
    html = html.replace("%%PREV_IN_BLOCKS_JS%%", json.dumps(prev_in_blocks or []))
    html = html.replace("%%PREV_EXP_LIST_JS%%", json.dumps(prev_exp_list or []))
    html = html.replace("%%IS_TC1%%", "true" if is_tc1 else "false")
    html = html.replace("%%IS_ALL_SIGNALS%%", "true" if is_all_signals else "false")
    html = html.replace("%%TC_ROWS_JS%%", tc_rows_json)

    return html

# ── FASTAPI ROUTES ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, dat_file: UploadFile = File(...), xlsx_file: UploadFile = File(None)):
    raw_dat = await dat_file.read()
    df, method = _try_asammdf(raw_dat)
    if df is None: df, method = _try_text(raw_dat)
    if df is None: return HTMLResponse(f"<h3>Error parsing DAT file</h3><p>{method}</p>")

    test_cases = []
    checksheet_html = ""
    if xlsx_file and xlsx_file.filename:
        raw_xlsx = await xlsx_file.read()
        xl_dict = parse_excel_checksheet(raw_xlsx, dat_file.filename)
        test_cases = xl_dict.get("test_cases", [])
        if xl_dict.get("preview_df") is not None:
            checksheet_html = xl_dict["preview_df"].head(100).fillna("").to_html(classes="df-table", index=False)

    time_col = detect_time_col(df)
    dat_signal_cols = [c for c in df.columns if c != time_col and is_numeric(df[c])]

    # Build "All Signals" Viewer
    all_signals_html = build_viewer_html(df, time_col, dat_file.filename, is_all_signals=True)
    dat_html = df.head(100).fillna("").to_html(classes="df-table", index=False)

    tc_render_data = []
    prev_in_blocks = []
    prev_exp_list = []
    
    for idx, tc in enumerate(test_cases):
        tc_m, tc_u = match_signals(tc["signals"], dat_signal_cols)

        tc_html = build_viewer_html(
            df, time_col, dat_file.filename,
            signal_cols=tc_m,
            in_blocks=tc.get("in_blocks", []),
            exp_list=tc.get("expected_list", []),
            prev_in_blocks=prev_in_blocks,
            prev_exp_list=prev_exp_list,
            is_tc1=(idx == 0),
            unmatched_sigs=tc_u,
            tc_df=tc.get("df")
        )
        tc_render_data.append({
            "tab_label": tc["tab_label"],
            "html": tc_html,
            "has_missing": len(tc_u) > 0 
        })
        
        prev_in_blocks = tc.get("in_blocks", [])
        prev_exp_list = tc.get("expected_list", [])

    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "filename": dat_file.filename,
            "all_signals_html": all_signals_html,
            "dat_html": dat_html,
            "checksheet_html": checksheet_html,
            "test_cases": tc_render_data
        }
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting FastAPI Server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)