#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemoNoveltyAgent -- Streamlit Frontend
Run:  streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import json
import time
import re
import io
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

# ---------------------------------------------------------------------------
#  Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MemoNoveltyAgent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
#  Session state defaults
# ---------------------------------------------------------------------------
if "browser_fs" not in st.session_state:
    st.session_state.browser_fs = False
if "_was_browser_fs" not in st.session_state:
    st.session_state._was_browser_fs = False

# ---------------------------------------------------------------------------
#  CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.hero-title {
    font-size:2.4rem; font-weight:800;
    background:linear-gradient(135deg,#1e3a5f 0%,#3b82f6 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:0;
}
.hero-sub {color:#64748b; font-size:1.1rem; margin-top:0;}
.stProgress > div > div > div > div {
    background:linear-gradient(90deg,#3b82f6,#10b981);
}
div[data-testid="stStatusWidget"] {font-size:0.92rem;}
div[data-testid="stStatusWidget"] > button,
div[data-testid="stStatusWidget"] button[kind="header"],
button[data-testid="baseButton-header"] {
    background-color: #ef4444 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 4px 12px !important;
}
div[data-testid="stStatusWidget"] > button:hover,
button[data-testid="baseButton-header"]:hover {
    background-color: #dc2626 !important;
}
div[data-testid="stStatusWidget"] svg[data-testid="stLogoSpin"],
div[data-testid="stStatusWidget"] > div > svg {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ===================================================================
#  PDF / HTML report generation helpers
# ===================================================================

def _safe_latin1(text: str) -> str:
    """Encode text to latin-1 safely for PDF rendering."""
    return text.encode('latin-1', errors='replace').decode('latin-1')


def generate_pdf_report(report_text: str, paper_name: str) -> Optional[bytes]:
    """Generate a formatted PDF from the report text using fpdf2. Returns None if unavailable."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    def _force_break(text: str, max_chars: int = 80) -> str:
        words = text.split(' ')
        out = []
        for w in words:
            while len(w) > max_chars:
                out.append(w[:max_chars])
                w = w[max_chars:]
            out.append(w)
        return ' '.join(out)

    class ReportPDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, 'Insight Agent - Innovation Analysis Report', 0, 1, 'C')
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

        def safe_multi_cell(self, w, h, txt, **kwargs):
            try:
                self.multi_cell(w, h, txt, **kwargs)
            except Exception:
                try:
                    truncated = txt[:500] + '...' if len(txt) > 500 else txt
                    self.multi_cell(w, h, truncated, **kwargs)
                except Exception:
                    self.ln(h)

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    # Title page
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(30, 58, 95)
    pdf.cell(0, 14, 'Innovation Analysis Report', ln=True, align='C')
    pdf.ln(8)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(80, 80, 80)
    safe_paper = _safe_latin1(_force_break(paper_name))
    pdf.safe_multi_cell(0, 8, safe_paper, align='C')
    pdf.ln(6)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.cell(0, 8, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
    pdf.ln(10)
    pdf.set_draw_color(59, 130, 246)
    pdf.line(40, pdf.get_y(), 170, pdf.get_y())

    # Content pages
    pdf.add_page()
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin

    for line in report_text.split('\n'):
        stripped = line.strip()

        if stripped.startswith('=' * 10):
            pdf.set_draw_color(59, 130, 246)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
            continue
        if stripped.startswith('---'):
            pdf.set_draw_color(210, 210, 210)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
            continue
        if stripped.startswith('## '):
            pdf.ln(5)
            pdf.set_font('Helvetica', 'B', 15)
            pdf.set_text_color(30, 58, 95)
            pdf.set_x(pdf.l_margin)
            pdf.safe_multi_cell(usable_w, 8, _safe_latin1(_force_break(stripped[3:])))
            pdf.ln(3)
            continue
        if stripped.startswith('### '):
            pdf.ln(3)
            pdf.set_font('Helvetica', 'B', 12)
            pdf.set_text_color(50, 80, 120)
            pdf.set_x(pdf.l_margin)
            pdf.safe_multi_cell(usable_w, 7, _safe_latin1(_force_break(stripped[4:])))
            pdf.ln(2)
            continue
        if stripped.startswith('**') and stripped.endswith('**'):
            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_text_color(20, 20, 20)
            pdf.set_x(pdf.l_margin)
            pdf.safe_multi_cell(usable_w, 6, _safe_latin1(_force_break(stripped.strip('* '))))
            pdf.ln(1)
            continue
        if stripped == '':
            pdf.ln(3)
            continue

        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(40, 40, 40)
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', stripped)
        clean = _safe_latin1(_force_break(clean))
        pdf.set_x(pdf.l_margin)
        pdf.safe_multi_cell(usable_w, 5, clean)

    output = pdf.output()
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    elif isinstance(output, str):
        return output.encode('latin-1', errors='replace')
    else:
        return bytes(output)


def generate_html_report(report_text: str, paper_name: str) -> str:
    """Convert markdown report to styled HTML suitable for printing to PDF."""

    def _esc(t):
        return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    parts = []
    for line in report_text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('=' * 10):
            parts.append('<hr class="thick">')
        elif stripped.startswith('---'):
            parts.append('<hr>')
        elif stripped.startswith('### '):
            parts.append(f'<h3>{_esc(stripped[4:])}</h3>')
        elif stripped.startswith('## '):
            parts.append(f'<h2>{_esc(stripped[3:])}</h2>')
        elif stripped.startswith('**') and stripped.endswith('**'):
            parts.append(f'<p><strong>{_esc(stripped.strip("* "))}</strong></p>')
        elif stripped == '':
            parts.append('<br>')
        else:
            processed = _esc(stripped)
            processed = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', processed)
            parts.append(f'<p>{processed}</p>')

    body = '\n'.join(parts)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Innovation Report - {_esc(paper_name)}</title>
<style>
  body {{ font-family: 'Georgia', 'Times New Roman', serif; max-width: 820px; margin: 0 auto;
         padding: 40px 30px; color: #1a1a2e; line-height: 1.7; }}
  h1 {{ color: #1e3a5f; border-bottom: 3px solid #3b82f6; padding-bottom: 12px; margin-top: 0; }}
  h2 {{ color: #1e3a5f; margin-top: 32px; border-bottom: 1px solid #cbd5e1; padding-bottom: 6px; }}
  h3 {{ color: #2c5282; margin-top: 24px; }}
  p {{ margin: 6px 0; }}
  hr {{ border: none; border-top: 1px solid #e2e8f0; margin: 24px 0; }}
  hr.thick {{ border-top: 3px solid #3b82f6; margin: 30px 0; }}
  strong {{ color: #1a202c; }}
  .meta {{ color: #64748b; font-size: 0.9rem; }}
  @media print {{ body {{ max-width: none; padding: 20px; font-size: 11pt; }} }}
</style>
</head>
<body>
<h1>Innovation Analysis Report</h1>
<p class="meta"><em>{_esc(paper_name)}</em></p>
<p class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<hr class="thick">
{body}
</body>
</html>"""


def extract_novelty_summary(report_text: str) -> Optional[str]:
    """Extract Section 3 (Novelty Summary) from the report text."""
    match = re.search(
        r'(## 3\.?\s*Novelty Summary.*?)(?=\n---\n|\n## References|\n={5,}|\Z)',
        report_text, re.S
    )
    return match.group(1).strip() if match else None


# ===================================================================
#  Config helpers
# ===================================================================

def read_config_file() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}
    api = cfg.setdefault("api", {})
    env_defaults = {
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "openai_base_url": os.environ.get("OPENAI_BASE_URL"),
        "openai_model": os.environ.get("OPENAI_MODEL"),
        "api_key": os.environ.get("RAGFLOW_API_KEY"),
        "base_url": os.environ.get("RAGFLOW_BASE_URL"),
        "dataset_name": os.environ.get("RAGFLOW_DATASET_NAME"),
        "chat_name": os.environ.get("RAGFLOW_CHAT_NAME"),
    }
    for key, value in env_defaults.items():
        if value and not api.get(key):
            api[key] = value
    memory_model = os.environ.get("MEMORY_EMBEDDING_MODEL")
    if memory_model:
        cfg.setdefault("expert_memory", {})["embedding_model_path"] = memory_model
    return cfg


def _deep_merge(base: dict, over: dict):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def write_config_file(updates: dict):
    cfg = read_config_file()
    _deep_merge(cfg, updates)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _collect_ui_updates() -> dict:
    """Collect current sidebar widget values into a config-shaped dict."""
    return {
        "paper_name": st.session_state.get("input_paper_name", ""),
        "use_full_text_in_comparison": st.session_state.get("cfg_full_text", True),
        "max_total_papers": st.session_state.get("cfg_max_papers", 199),
        "api": {
            "openai_base_url": st.session_state.get("cfg_openai_base_url", ""),
            "openai_api_key": st.session_state.get("cfg_openai_api_key", ""),
            "openai_timeout": float(st.session_state.get("cfg_timeout", 600)),
            "base_url": st.session_state.get("cfg_ragflow_base_url", ""),
            "api_key": st.session_state.get("cfg_ragflow_api_key", ""),
        },
        "llm_config": {
            "model": st.session_state.get("cfg_model", "gpt-4o"),
            "temperature": st.session_state.get("cfg_temp", 0.7),
        },
        "rag": {
            "page_size": st.session_state.get("cfg_page_size", 7),
        },
        "paths": {
            "database_dir": st.session_state.get("cfg_db_dir", "./database"),
            "result_dir": st.session_state.get("cfg_result_dir", "./results/pipeline_state"),
            "innovation_points_dir": st.session_state.get("cfg_innovation_dir", "./results/innovation_points"),
            "paper_summary_dir": st.session_state.get("cfg_summary_dir", "./results/paper_summaries"),
            "compare_result_dir": st.session_state.get("cfg_compare_dir", "./results/comparisons"),
            "reports_dir": st.session_state.get("cfg_reports_dir", "./results/reports"),
            "draft_reports_dir": st.session_state.get("cfg_draft_dir", "./results/draft_reports"),
            "validated_reports_dir": st.session_state.get("cfg_validated_dir", "./results/validated_reports"),
            "polished_reports_dir": st.session_state.get("cfg_polished_dir", "./results/polished_reports"),
        },
        "expert_memory": {
            "enabled": st.session_state.get("cfg_memory_enabled", True),
            "construct_memory_dir": st.session_state.get("cfg_construct_memory_dir", "../Construct_memory"),
            "embedding_model_path": st.session_state.get("cfg_memory_model_path", "Qwen/Qwen3-Embedding-8B"),
            "top_k": st.session_state.get("cfg_memory_top_k", 10),
            "gpu": st.session_state.get("cfg_memory_gpu", 0),
            "max_length": st.session_state.get("cfg_memory_max_length", 512),
            "selection_attempts": st.session_state.get("cfg_memory_selection_attempts", 3),
        },
    }


def _save_settings_callback():
    write_config_file(_collect_ui_updates())
    st.toast("Settings saved to config.json", icon="💾")


def build_full_config(defaults: dict) -> dict:
    """Merge UI values into the full config (including prompts etc.)."""
    cfg = json.loads(json.dumps(defaults))
    _deep_merge(cfg, _collect_ui_updates())
    return cfg


# ===================================================================
#  Stream capture
# ===================================================================

class StreamCapture:
    def __init__(self, container, original=None):
        self.container = container
        self.original = original or sys.__stdout__
        self.buf = ""
        self._ts = time.time()

    def write(self, data):
        if not data:
            return
        self.original.write(data)
        self.buf += str(data)
        if time.time() - self._ts > 0.4:
            self._render()
            self._ts = time.time()

    def flush(self):
        self.original.flush()
        self._render()

    def _render(self):
        if self.buf:
            tail = self.buf[-50000:] if len(self.buf) > 50000 else self.buf
            try:
                self.container.code(tail, language=None)
            except Exception:
                pass


@contextmanager
def capture_output(container):
    old_out, old_err = sys.stdout, sys.stderr
    cap = StreamCapture(container, old_out)
    sys.stdout = cap
    sys.stderr = cap
    try:
        yield cap
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        cap.flush()


# ===================================================================
#  Misc helpers
# ===================================================================

def sanitize_filename(name: str) -> str:
    name = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_", ".")).rstrip()
    return name[:200]


def find_main_pdf(folder: str):
    for fn in os.listdir(folder):
        if fn.lower().startswith("main_") and fn.lower().endswith(".pdf"):
            return os.path.join(folder, fn)
    return None


def ckpt_save(d, name, content):
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, name)
    with open(fp, "w", encoding="utf-8") as f:
        if isinstance(content, (dict, list)):
            json.dump(content, f, ensure_ascii=False, indent=2)
        else:
            f.write(str(content))


def ckpt_load(d, name, as_json=False):
    fp = os.path.join(d, name)
    if not os.path.exists(fp):
        return None
    try:
        with open(fp, "r", encoding="utf-8") as f:
            if as_json:
                data = json.load(f)
                return data if data else None
            txt = f.read()
            return txt if txt and txt.strip() else None
    except Exception:
        return None


CHECKPOINT_NAMES = [
    ("summary.txt", False, "Summary"),
    ("innovation_points.txt", False, "Novelty Points"),
    ("comparison_data.json", True, "Comparison Data"),
    ("initial_report.txt", False, "Initial Report"),
    ("validated_report.txt", False, "Validated Report"),
    ("polished_report.txt", False, "Polished Report"),
]


def get_checkpoint_status(ckpt_dir: str) -> list:
    out = []
    for fname, _, label in CHECKPOINT_NAMES:
        exists = os.path.exists(os.path.join(ckpt_dir, fname))
        out.append((label, exists))
    return out


# ===================================================================
#  SIDEBAR  (always rendered)
# ===================================================================
defaults = read_config_file()

st.sidebar.markdown("## ⚙️ Settings")

with st.sidebar.expander("🔑 API Keys", expanded=True):
    st.text_input(
        "OpenAI-Compatible Base URL",
        value=defaults.get("api", {}).get("openai_base_url", "https://api.openai.com/v1"),
        type="password", key="cfg_openai_base_url",
    )
    st.text_input(
        "OpenAI API Key",
        value=defaults.get("api", {}).get("openai_api_key", ""),
        type="password", key="cfg_openai_api_key",
    )
    st.text_input(
        "RAGFlow Base URL",
        value=defaults.get("api", {}).get("base_url", "http://localhost:9380"),
        type="password", key="cfg_ragflow_base_url",
    )
    st.text_input(
        "RAGFlow API Key",
        value=defaults.get("api", {}).get("api_key", ""),
        type="password", key="cfg_ragflow_api_key",
    )

with st.sidebar.expander("🤖 Model"):
    st.text_input("Model Name",
        value=defaults.get("llm_config", {}).get("model", "gpt-4o"),
        key="cfg_model")
    st.slider("Temperature", 0.0, 1.0,
        value=float(defaults.get("llm_config", {}).get("temperature", 0.7)),
        step=0.05, key="cfg_temp")
    st.number_input("Max Reference Papers", 10, 500,
        value=int(defaults.get("max_total_papers", 199)),
        key="cfg_max_papers")
    st.slider(
        "🔍 Retrieval Intensity (chunks per query)",
        min_value=1, max_value=30,
        value=int(defaults.get("rag", {}).get("page_size", 7)),
        help="Number of chunks retrieved per RAG query. Higher = more context but slower.",
        key="cfg_page_size",
    )

with st.sidebar.expander("📁 Paths"):
    st.text_input("Database Directory",
        value=defaults.get("paths", {}).get("database_dir", "./database"),
        key="cfg_db_dir")
    st.text_input("Pipeline State Directory",
        value=defaults.get("paths", {}).get("result_dir", "./results/pipeline_state"),
        key="cfg_result_dir")
    st.text_input("Final Reports Directory",
        value=defaults.get("paths", {}).get("reports_dir", "./results/reports"),
        key="cfg_reports_dir")

with st.sidebar.expander("🧠 Expert Memory"):
    st.checkbox("Enable expert memory retrieval",
        value=defaults.get("expert_memory", {}).get("enabled", True),
        key="cfg_memory_enabled")
    st.text_input("Construct Memory Directory",
        value=defaults.get("expert_memory", {}).get("construct_memory_dir", "../Construct_memory"),
        key="cfg_construct_memory_dir")
    st.text_input("Embedding Model Path or HF ID",
        value=defaults.get("expert_memory", {}).get("embedding_model_path", "Qwen/Qwen3-Embedding-8B"),
        key="cfg_memory_model_path")
    st.number_input("Memory Top-K", 1, 50,
        value=int(defaults.get("expert_memory", {}).get("top_k", 10)),
        key="cfg_memory_top_k")
    st.number_input("GPU ID (-1 for CPU)", -1, 16,
        value=int(defaults.get("expert_memory", {}).get("gpu", 0)),
        key="cfg_memory_gpu")

with st.sidebar.expander("🔧 Advanced"):
    st.checkbox("Use full paper text in comparison",
        value=defaults.get("use_full_text_in_comparison", True),
        key="cfg_full_text")
    st.number_input("API Timeout (s)", 60, 3600,
        value=int(defaults.get("api", {}).get("openai_timeout", 600)),
        key="cfg_timeout")

st.sidebar.button("💾 Save Settings to config.json",
                   on_click=_save_settings_callback,
                   use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("MemoNoveltyAgent · Open-Source Paper Innovation Analyzer")


# ===================================================================
#  MAIN AREA -- header with browser fullscreen toggle
# ===================================================================
_col_hero, _col_fs = st.columns([11, 1])
with _col_hero:
    st.markdown('<p class="hero-title">🔬 MemoNoveltyAgent</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Automated Paper Novelty Analysis &mdash; '
        "crawl, extract, compare, and generate a full report.</p>",
        unsafe_allow_html=True,
    )

with _col_fs:
    def _toggle_browser_fs():
        st.session_state.browser_fs = not st.session_state.browser_fs

    _fs_label = "🔲 Exit" if st.session_state.browser_fs else "⛶ Full"
    st.button(
        _fs_label,
        on_click=_toggle_browser_fs,
        use_container_width=True,
        help="Toggle browser fullscreen (F11). Sidebar stays available.",
    )

# ===================================================================
#  Browser Fullscreen API injection (pure display, no sidebar hiding)
# ===================================================================
if st.session_state.browser_fs:
    st.session_state._was_browser_fs = True
    components.html("""
    <script>
    (function() {
        var pd = parent.document;
        var old = pd.getElementById('fs-overlay');
        if (old) old.remove();
        if (pd.fullscreenElement || pd.webkitFullscreenElement) return;
        var ov = pd.createElement('div');
        ov.id = 'fs-overlay';
        ov.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;'
            + 'background:rgba(0,0,0,0.82);z-index:999999;display:flex;'
            + 'align-items:center;justify-content:center;cursor:pointer;'
            + 'transition:opacity 0.4s ease;';
        ov.innerHTML = '<div style="text-align:center;user-select:none;">'
            + '<div style="font-size:4rem;margin-bottom:16px;">🖥️</div>'
            + '<div style="font-size:1.5rem;color:#fff;font-weight:600;margin-bottom:8px;">'
            + 'Click anywhere to enter fullscreen</div>'
            + '<div style="font-size:0.85rem;color:#999;">'
            + 'Press ESC to exit browser fullscreen</div>'
            + '</div>';
        function dismiss() {
            ov.style.opacity = '0';
            setTimeout(function() { if (ov.parentNode) ov.remove(); }, 400);
        }
        ov.addEventListener('click', function() {
            var el = pd.documentElement;
            var rfs = el.requestFullscreen || el.webkitRequestFullscreen
                   || el.mozRequestFullScreen || el.msRequestFullscreen;
            if (rfs) {
                try {
                    var p = rfs.call(el);
                    if (p && p.then) p.then(function(){}).catch(function(){});
                } catch(e) {}
            }
            dismiss();
        });
        pd.body.appendChild(ov);
        setTimeout(function() {
            if (ov && ov.parentNode && ov.style.opacity !== '0') dismiss();
        }, 6000);
    })();
    </script>
    """, height=0)

elif st.session_state._was_browser_fs:
    st.session_state._was_browser_fs = False
    components.html("""
    <script>
    (function() {
        var pd = parent.document;
        var old = pd.getElementById('fs-overlay');
        if (old) old.remove();
        if (pd.fullscreenElement || pd.webkitFullscreenElement) {
            var efs = pd.exitFullscreen || pd.webkitExitFullscreen
                   || pd.mozCancelFullScreen || pd.msExitFullscreen;
            if (efs) {
                try { efs.call(pd); } catch(e) {}
            }
        }
    })();
    </script>
    """, height=0)

st.markdown("---")

# Paper name input (synced with config)
st.text_input(
    "📄 Paper Title (exact title as it appears on the paper / arXiv)",
    value=defaults.get("paper_name", ""),
    placeholder="e.g.  AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments",
    key="input_paper_name",
)

# ===================================================================
#  Database Explorer
# ===================================================================
with st.expander("📁 Database & Checkpoint Explorer", expanded=False):
    db_path = Path(st.session_state.get("cfg_db_dir", "./database"))
    res_path = Path(st.session_state.get("cfg_result_dir", "./result"))

    col_db, col_res = st.columns(2)

    with col_db:
        st.markdown("#### 📂 Paper Databases")
        if db_path.exists() and any(db_path.iterdir()):
            for folder in sorted(db_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not folder.is_dir():
                    continue
                mains = len(list(folder.glob("MAIN_*.pdf")))
                refs = len(list(folder.glob("REF_*.pdf")))
                total = mains + refs
                target = int(st.session_state.get("cfg_max_papers", 199))
                icon = "✅" if mains >= 1 and total >= target else "⏳"
                display_name = folder.name[:65] + ("…" if len(folder.name) > 65 else "")
                st.markdown(
                    f"{icon} **{display_name}**  \n"
                    f"&nbsp;&nbsp;&nbsp;`MAIN: {mains}` &nbsp;|&nbsp; `REF: {refs}` &nbsp;|&nbsp; `Total: {total}`"
                )
        else:
            st.info("No paper databases found yet. They will appear here after your first analysis.")

    with col_res:
        st.markdown("#### 📋 Results & Checkpoints")
        if res_path.exists() and any(res_path.iterdir()):
            for folder in sorted(res_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not folder.is_dir():
                    continue
                display_name = folder.name[:55] + ("…" if len(folder.name) > 55 else "")
                reports = list(folder.glob("*.txt"))
                report_icon = "✅" if reports else "❌"
                st.markdown(f"**{display_name}**  \n&nbsp;&nbsp;&nbsp;Final report: {report_icon}")

                ckpt_d = folder / "checkpoints"
                if ckpt_d.exists():
                    statuses = get_checkpoint_status(str(ckpt_d))
                    chips = "  ".join(
                        f"{'🟢' if ok else '⚪'} {lbl}" for lbl, ok in statuses
                    )
                    st.markdown(f"&nbsp;&nbsp;&nbsp;{chips}")
        else:
            st.info("No results yet.")

# ===================================================================
#  Action buttons
# ===================================================================
st.markdown("---")
col_start, col_info, col_clear = st.columns([2, 2, 1])

# Session state defaults
for k, v in [("pipeline_results", None), ("pipeline_complete", False), ("report_path", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

with col_start:
    start_btn = st.button("🚀 Start / Resume Analysis", type="primary", use_container_width=True)

with col_info:
    st.markdown(
        '<div style="padding-top:0.35rem;color:#94a3b8;font-size:0.85rem;text-align:center;">'
        'To <b>pause</b>, click the red <b>Stop</b> button in the top-right corner.<br>'
        'All completed steps are auto-saved — just click <b>Start / Resume</b> to continue.'
        '</div>',
        unsafe_allow_html=True,
    )

with col_clear:
    if st.button("🗑️ Clear", use_container_width=True):
        for k in ("pipeline_results", "pipeline_complete", "report_path"):
            st.session_state[k] = None
        st.rerun()

# ===================================================================
#  PIPELINE
# ===================================================================
if start_btn:
    paper_name = st.session_state.get("input_paper_name", "").strip()
    openai_api_key = st.session_state.get("cfg_openai_api_key", "").strip()
    ragflow_api_key = st.session_state.get("cfg_ragflow_api_key", "").strip()

    if not paper_name:
        st.error("Please enter a paper title."); st.stop()
    if not openai_api_key:
        st.error("Please provide your OpenAI API key in the sidebar."); st.stop()
    if not ragflow_api_key:
        st.error("Please provide your RAGFlow API key in the sidebar."); st.stop()

    # Build & save config
    config = build_full_config(defaults)
    write_config_file(_collect_ui_updates())

    # Dirs
    safe_name = sanitize_filename(paper_name)
    paper_result_dir = os.path.join(config["paths"]["result_dir"], safe_name)
    os.makedirs(paper_result_dir, exist_ok=True)
    ckpt_dir = os.path.join(paper_result_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    R = {}
    failed = False
    progress = st.progress(0, text="Initializing…")

    # ----------------------------------------------------------------
    #  STEP 1 -- Crawl
    # ----------------------------------------------------------------
    with st.status("📥  Step 1/7 — Downloading papers…", expanded=True) as s1:
        log1 = st.empty()
        with capture_output(log1):
            try:
                from clawer_papers import download_paper_if_needed
                R["folder"] = download_paper_if_needed(config)
                if R["folder"]:
                    R["main_pdf"] = find_main_pdf(R["folder"])
            except Exception as e:
                st.error(f"Step 1 error: {e}"); failed = True
        if failed or not R.get("folder") or not R.get("main_pdf"):
            s1.update(label="📥  Step 1 — FAILED ❌", state="error")
            if not R.get("folder"):
                st.error("Could not download / locate paper folder.")
            elif not R.get("main_pdf"):
                st.error(f"MAIN_*.pdf not found in `{R.get('folder')}`")
            st.stop()
        s1.update(label="📥  Step 1 — Papers ready ✅", state="complete", expanded=False)
    progress.progress(1 / 7, "Step 1 ✅ Papers ready")

    # ----------------------------------------------------------------
    #  STEP 2 -- RAGFlow
    # ----------------------------------------------------------------
    with st.status("📤  Step 2/7 — Uploading & parsing in RAGFlow…", expanded=True) as s2:
        log2 = st.empty()
        with capture_output(log2):
            try:
                from Create_database_and_parse import upload_pdfs_to_ragflow, wait_for_parsing_completion
                main_ds, all_ds = upload_pdfs_to_ragflow(config, R["folder"])
                if not all_ds:
                    raise RuntimeError("Failed to create RAGFlow dataset.")
                if not wait_for_parsing_completion(all_ds, config):
                    raise RuntimeError("all_dataset parsing timed out / failed.")
                if main_ds:
                    wait_for_parsing_completion(main_ds, config)
                R["all_ds"] = all_ds
            except Exception as e:
                st.error(f"Step 2 error: {e}"); failed = True
        if failed:
            s2.update(label="📤  Step 2 — FAILED ❌", state="error"); st.stop()
        s2.update(label="📤  Step 2 — Database ready ✅", state="complete", expanded=False)
    progress.progress(2 / 7, "Step 2 ✅ Database ready")

    # ----------------------------------------------------------------
    #  STEP 3 -- Summary + Novelty Points
    # ----------------------------------------------------------------
    with st.status("🔍  Step 3/7 — Extracting summary & novelty points…", expanded=True) as s3:
        log3 = st.empty()
        with capture_output(log3):
            try:
                summary = ckpt_load(ckpt_dir, "summary.txt")
                if summary:
                    print("[OK] Summary loaded from checkpoint.")
                else:
                    from Generate_Mainpaper_summary import get_paper_summary
                    summary = get_paper_summary(config, paper_name, R["main_pdf"])
                    if summary:
                        ckpt_save(ckpt_dir, "summary.txt", summary)

                innovation = ckpt_load(ckpt_dir, "innovation_points.txt")
                if innovation:
                    print("[OK] Novelty points loaded from checkpoint.")
                else:
                    from Generate_innovation_points import get_paper_innovation
                    innovation = get_paper_innovation(config, paper_name, R["main_pdf"])
                    if innovation:
                        ckpt_save(ckpt_dir, "innovation_points.txt", innovation)

                if not summary or not innovation:
                    raise RuntimeError("Summary or novelty point extraction returned empty.")
                R["summary"] = summary
                R["innovation"] = innovation
            except Exception as e:
                st.error(f"Step 3 error: {e}"); failed = True
        if failed:
            s3.update(label="🔍  Step 3 — FAILED ❌", state="error"); st.stop()
        s3.update(label="🔍  Step 3 — Extracted ✅", state="complete", expanded=False)
    progress.progress(3 / 7, "Step 3 ✅ Summary & novelty points extracted")

    # ----------------------------------------------------------------
    #  STEP 4 -- Compare
    # ----------------------------------------------------------------
    with st.status("⚖️  Step 4/7 — Comparing novelty points…", expanded=True) as s4:
        log4 = st.empty()
        with capture_output(log4):
            try:
                comparison = ckpt_load(ckpt_dir, "comparison_data.json", as_json=True)
                if comparison:
                    print("[OK] Comparison data loaded from checkpoint.")
                else:
                    from Compare_innovation_points import compare_paper_innovations
                    comparison = compare_paper_innovations(
                        config, paper_name, R["all_ds"].name, R["innovation"], R["main_pdf"]
                    )
                    if comparison:
                        ckpt_save(ckpt_dir, "comparison_data.json", comparison)
                if not comparison:
                    raise RuntimeError("Comparison returned empty.")
                R["comparison"] = comparison
            except Exception as e:
                st.error(f"Step 4 error: {e}"); failed = True
        if failed:
            s4.update(label="⚖️  Step 4 — FAILED ❌", state="error"); st.stop()
        s4.update(label="⚖️  Step 4 — Compared ✅", state="complete", expanded=False)
    progress.progress(4 / 7, "Step 4 ✅ Comparisons complete")

    # ----------------------------------------------------------------
    #  STEP 5 -- Generate report
    # ----------------------------------------------------------------
    with st.status("📝  Step 5/7 — Generating report…", expanded=True) as s5:
        log5 = st.empty()
        with capture_output(log5):
            try:
                initial_report = ckpt_load(ckpt_dir, "initial_report.txt")
                point_count = len(R["comparison"])
                if initial_report:
                    print("[OK] Initial report loaded from checkpoint.")
                else:
                    from Write_reports import InnovationReportGenerator
                    gen = InnovationReportGenerator(config)
                    body, point_count = gen.generate_comprehensive_report(
                        paper_name, R["summary"], R["innovation"], R["comparison"]
                    )
                    if body:
                        initial_report = body
                        ckpt_save(ckpt_dir, "initial_report.txt", initial_report)
                if not initial_report:
                    raise RuntimeError("Report generation returned empty.")
                R["report"] = initial_report
            except Exception as e:
                st.error(f"Step 5 error: {e}"); failed = True
        if failed:
            s5.update(label="📝  Step 5 — FAILED ❌", state="error"); st.stop()
        s5.update(label="📝  Step 5 — Report generated ✅", state="complete", expanded=False)
    progress.progress(5 / 7, "Step 5 ✅ Report generated")

    # ----------------------------------------------------------------
    #  STEP 6 -- Validate citations (with sub-steps)
    # ----------------------------------------------------------------
    with st.status("🔎  Step 6/7 — Validating citations…", expanded=True) as s6:
        validated = ckpt_load(ckpt_dir, "validated_report.txt")
        if validated:
            st.write("✅ Validated report loaded from checkpoint.")
        else:
            try:
                from Validate_and_correct_citations import CitationValidator
                validator = CitationValidator(config)
                report_for_validation = R["report"]

                # Sub-step 6.1: Extract citations
                with st.status("📎 6.1 — Extracting citations…", expanded=False) as s6_1:
                    log6_1 = st.empty()
                    with capture_output(log6_1):
                        citations = validator.extract_citations(report_for_validation)
                    n_cit = len(citations) if citations else 0
                    s6_1.update(label=f"📎 6.1 — Extracted {n_cit} citations ✅", state="complete", expanded=False)

                if not citations:
                    st.write("No citations found. Using original report.")
                    validated = report_for_validation
                else:
                    # Sub-step 6.2: Deduplicate
                    with st.status("🔄 6.2 — Deduplicating citations…", expanded=False) as s6_2:
                        log6_2 = st.empty()
                        with capture_output(log6_2):
                            deduped_groups, _ = validator.deduplicate_citations(citations)
                        total_deduped = sum(len(v) for v in deduped_groups.values())
                        s6_2.update(
                            label=f"🔄 6.2 — {total_deduped} unique citations across {len(deduped_groups)} refs ✅",
                            state="complete", expanded=False
                        )

                    # Sub-step 6.3: Validate against PDFs
                    with st.status("📄 6.3 — Validating against PDFs…", expanded=False) as s6_3:
                        log6_3 = st.empty()
                        with capture_output(log6_3):
                            validation_results = {}
                            for ref_name, citation_list in deduped_groups.items():
                                pdf_path = validator.find_pdf_by_name(ref_name, R["folder"])
                                if not pdf_path:
                                    print(f"  [validate] {ref_name}: PDF not found, skipping")
                                    continue
                                print(f"  [validate] {ref_name}: found PDF")
                                validation = validator.validate_citations(ref_name, citation_list, pdf_path)
                                validation_results[ref_name] = {
                                    'pdf_path': pdf_path,
                                    'citations': citation_list,
                                    'validation': validation
                                }
                        s6_3.update(
                            label=f"📄 6.3 — Validated {len(validation_results)} references ✅",
                            state="complete", expanded=False
                        )

                    # Sub-step 6.4: Correct report
                    with st.status("✏️ 6.4 — Correcting report…", expanded=False) as s6_4:
                        log6_4 = st.empty()
                        with capture_output(log6_4):
                            validated = validator.correct_report(report_for_validation, validation_results)
                        s6_4.update(label="✏️ 6.4 — Report corrected ✅", state="complete", expanded=False)

                if validated:
                    ckpt_save(ckpt_dir, "validated_report.txt", validated)

            except Exception as e:
                st.warning(f"Citation validation error: {e} — using unvalidated report.")
                validated = R["report"]

        R["validated"] = validated or R["report"]
        s6.update(label="🔎  Step 6 — Citations validated ✅", state="complete", expanded=False)
    progress.progress(6 / 7, "Step 6 ✅ Citations validated")

    # ----------------------------------------------------------------
    #  STEP 7 -- Polish
    # ----------------------------------------------------------------
    with st.status("✨  Step 7/7 — Polishing final report…", expanded=True) as s7:
        log7 = st.empty()
        with capture_output(log7):
            try:
                polished = ckpt_load(ckpt_dir, "polished_report.txt")
                if polished:
                    print("[OK] Polished report loaded from checkpoint.")
                    R["final"] = polished
                else:
                    from Final_polish import ReportPolisher
                    polisher = ReportPolisher(config)
                    R["final"] = polisher.polish_single_report(
                        R["validated"], paper_name, point_count
                    )
                    if R["final"]:
                        ckpt_save(ckpt_dir, "polished_report.txt", R["final"])
            except Exception as e:
                st.warning(f"Polish error: {e} — using validated report.")
                R["final"] = R.get("validated", R.get("report", ""))
        s7.update(label="✨  Step 7 — Polished ✅", state="complete", expanded=False)
    progress.progress(1.0, "🎉 Pipeline complete!")

    # Save final report (txt for backward compat)
    report_path = os.path.join(paper_result_dir, f"{safe_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(R.get("final", ""))

    st.session_state["pipeline_results"] = R
    st.session_state["pipeline_complete"] = True
    st.session_state["report_path"] = report_path

    st.balloons()
    st.success(f"🎉 Pipeline complete! Report saved to `{report_path}`")


# ===================================================================
#  RESULTS DISPLAY (persists via session_state)
# ===================================================================
if st.session_state.get("pipeline_complete") and st.session_state.get("pipeline_results"):
    R = st.session_state["pipeline_results"]
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    tab_summary, tab_comp, tab_novelty_summary, tab_report = st.tabs(
        ["📋 Paper Summary", "⚖️ Comparisons", "💡 Novelty Summary", "📄 Final Report & Download"]
    )

    with tab_summary:
        st.markdown(R.get("summary", "*Not available.*"))

    with tab_comp:
        comp = R.get("comparison")
        if comp:
            for item in comp:
                with st.expander(f"Novelty Point {item['point_number']}", expanded=False):
                    st.markdown(item["content"])
        else:
            st.info("No comparison data available.")

    with tab_novelty_summary:
        final_text = R.get("final", "")
        novelty_summary = extract_novelty_summary(final_text) if final_text else None
        if novelty_summary:
            st.markdown(novelty_summary)
        else:
            st.info("Novelty Summary (Section 3) not available. Check the Final Report tab for the complete report.")

    with tab_report:
        final_text = R.get("final", "")
        if final_text:
            st.text_area("Final Report", final_text, height=500, disabled=True)

            st.markdown("#### 📥 Download Report")

            col_pdf, col_html, col_txt = st.columns(3)

            # PDF download
            with col_pdf:
                pdf_bytes = generate_pdf_report(final_text, st.session_state.get("input_paper_name", "Report"))
                if pdf_bytes:
                    st.download_button(
                        "📥 Download PDF",
                        data=pdf_bytes,
                        file_name=f"innovation_report_{datetime.now():%Y%m%d_%H%M%S}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.caption("PDF unavailable (install `fpdf2`)")

            # HTML download (always available)
            with col_html:
                html_report = generate_html_report(
                    final_text,
                    st.session_state.get("input_paper_name", "Report")
                )
                st.download_button(
                    "📥 Download HTML",
                    data=html_report,
                    file_name=f"innovation_report_{datetime.now():%Y%m%d_%H%M%S}.html",
                    mime="text/html",
                    use_container_width=True,
                )

            # TXT download (fallback)
            with col_txt:
                st.download_button(
                    "📥 Download TXT",
                    data=final_text,
                    file_name=f"innovation_report_{datetime.now():%Y%m%d_%H%M%S}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        else:
            st.info("No final report available.")