
# app_virtual_pitch.py
# ------------------------------------------------------------
# Virtual Pitch Position Selector — NO GROUPING
# Each unique position token from the DataFrame is its own clickable label.
# Background: mplsoccer Pitch; Foreground: Plotly scatter with click-to-toggle.
# ------------------------------------------------------------
import io
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Optional: click capture
try:
    from streamlit_plotly_events import plotly_events
    _HAS_SPE = True
except Exception:
    _HAS_SPE = False

from mplsoccer import Pitch
from PIL import Image

st.set_page_config(page_title="Virtual Pitch Position Selector", layout="wide")

# ---------------------
# Helpers
# ---------------------
def tokens_from_series(series: pd.Series) -> List[str]:
    toks = []
    for cell in series.dropna().astype(str):
        for tok in str(cell).split(","):
            tok = tok.strip()
            if tok:
                toks.append(tok)  # DO NOT normalize; exact label
    # unique preserving order
    seen = set()
    uniq = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def default_coordinates_for_labels(labels: List[str], pitch_length: int = 120, pitch_width: int = 80) -> Dict[str, Tuple[float, float]]:
    """Assign coordinates PER EXACT LABEL. Known exact labels get canonical coords.
    Unknown labels are placed on a shelf at the bottom, spaced evenly.
    """
    # Known exact labels (no aliasing). Only if the label matches exactly, it gets the coord.
    # 4-3-3 base, but labels are distinct.
    known: Dict[str, Tuple[float, float]] = {
        # GK
        "GK": (6, pitch_width/2),
        # Back four
        "RB": (20, pitch_width*0.25),
        "RCB": (18, pitch_width*0.45),
        "CB": (18, pitch_width/2),
        "LCB": (18, pitch_width*0.55),
        "LB": (20, pitch_width*0.75),
        # Wing-backs
        "RWB": (28, pitch_width*0.28),
        "LWB": (28, pitch_width*0.72),
        # Midfield (single pivot / double / triple)
        "DM": (38, pitch_width/2),
        "CDM": (38, pitch_width/2),
        "RDM": (38, pitch_width*0.44),
        "LDM": (38, pitch_width*0.56),
        "CM": (52, pitch_width/2),
        "RCM": (52, pitch_width*0.44),
        "LCM": (52, pitch_width*0.56),
        "AM": (66, pitch_width/2),
        "CAM": (66, pitch_width/2),
        "RAM": (66, pitch_width*0.44),
        "LAM": (66, pitch_width*0.56),
        # Wingers / wide mids
        "RW": (78, pitch_width*0.30),
        "LW": (78, pitch_width*0.70),
        "RM": (70, pitch_width*0.33),
        "LM": (70, pitch_width*0.67),
        # Forwards
        "RF": (92, pitch_width*0.45),
        "LF": (92, pitch_width*0.55),
        "SS": (88, pitch_width/2),
        "CF": (100, pitch_width/2),
        "ST": (100, pitch_width/2),
        # Full list can be extended. Only exact matches are used.
    }

    coords: Dict[str, Tuple[float, float]] = {}
    shelf = [lbl for lbl in labels if lbl not in known]
    # Space unknowns across the bottom stripe (y ~ 6)
    if shelf:
        xs = np.linspace(8, pitch_length-8, num=len(shelf))
        y = np.full(len(shelf), 6.0)
        for i, lbl in enumerate(shelf):
            coords[lbl] = (float(xs[i]), float(y[i]))

    # Add known exact ones that are present
    for lbl in labels:
        if lbl in known:
            coords[lbl] = known[lbl]

    return coords

def render_pitch_bg(pitch_type: str = "statsbomb", line_zorder: int = 2, pitch_length: int = 120, pitch_width: int = 80) -> Image.Image:
    """Draw an mplsoccer Pitch to a PIL image buffer to use as Plotly background."""
    pitch = Pitch(pitch_type=pitch_type, pitch_color="white", line_color="black",
                  line_zorder=line_zorder, goal_type="box", pitch_length=pitch_length, pitch_width=pitch_width)
    fig, ax = pitch.draw(figsize=(10, 6.666), tight_layout=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    return img

def ensure_session_state(keys_defaults: Dict[str, object]):
    for k, v in keys_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def toggle_selection_by_label(label: str):
    sel = set(st.session_state["selected_labels"])
    if label in sel:
        sel.remove(label)
    else:
        sel.add(label)
    st.session_state["selected_labels"] = list(sel)

# ---------------------
# Sidebar — data + options
# ---------------------
st.sidebar.header("Dados")
uploaded = st.sidebar.file_uploader("Envie seu arquivo (Excel/CSV)", type=["xlsx", "xls", "csv"])
pos_col = st.sidebar.text_input("Nome da coluna de posições (exato)", value="Position")

pitch_type = st.sidebar.selectbox("Pitch (mplsoccer)", ["statsbomb", "opta", "tracab", "wyscout", "uefa", "metricasports"], index=0)
pitch_length = st.sidebar.number_input("Comprimento", value=120, step=1, min_value=90, max_value=130)
pitch_width = st.sidebar.number_input("Largura", value=80, step=1, min_value=60, max_value=90)

ensure_session_state({
    "selected_labels": [],
})

# ---------------------
# Load data
# ---------------------
df: pd.DataFrame | None = None
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")

if df is None:
    st.info("Envie um arquivo e indique a coluna de posições para começar.")
    st.stop()

if pos_col not in df.columns:
    st.error(f"Coluna '{pos_col}' não encontrada. Colunas disponíveis: {list(df.columns)}")
    st.stop()

labels = tokens_from_series(df[pos_col])
if not labels:
    st.warning("Não encontrei posições na coluna informada.")
    st.stop()

# ---------------------
# Coordinates per EXACT label (no grouping)
# ---------------------
coords = default_coordinates_for_labels(labels, pitch_length=pitch_length, pitch_width=pitch_width)

# ---------------------
# Build Plotly figure over mplsoccer background
# ---------------------
bg = render_pitch_bg(pitch_type=pitch_type, pitch_length=pitch_length, pitch_width=pitch_width)
w, h = bg.size

fig = go.Figure()

# Set background image
fig.add_layout_image(
    dict(
        source=bg,
        xref="x",
        yref="y",
        x=0,
        y=pitch_width,
        sizex=pitch_length,
        sizey=pitch_width,
        sizing="stretch",
        layer="below",
        opacity=1.0,
    )
)

# Scatter labels
xs = [coords[l][0] for l in labels]
ys = [coords[l][1] for l in labels]
texts = labels
customdata = labels

fig.add_trace(go.Scatter(
    x=xs, y=ys, mode="markers+text",
    text=texts, textposition="top center",
    marker=dict(size=14, line=dict(width=1), opacity=0.9),
    customdata=customdata,
    hovertemplate="%{customdata}<extra></extra>",
))

# Layout matching pitch coordinates
fig.update_xaxes(range=[0, pitch_length], visible=False, constrain="domain")
fig.update_yaxes(range=[0, pitch_width], visible=False, scaleanchor="x", scaleratio=1)

fig.update_layout(
    margin=dict(l=10, r=10, t=20, b=10),
    dragmode=False,
    height=650,
)

st.subheader("Selecione posições clicando no campo (cada rótulo é independente)")
col_plot, col_sel = st.columns([2.2, 1.0])

with col_plot:
    if _HAS_SPE:
        clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="pitch")
        st.plotly_chart(fig, use_container_width=True)
        if clicked:
            # Each click returns pointNumber, curveNumber, etc.
            # Retrieve the label using that index.
            try:
                idx = int(clicked[0].get("pointIndex", clicked[0].get("pointNumber", -1)))
            except Exception:
                idx = -1
            if 0 <= idx < len(labels):
                toggle_selection_by_label(labels[idx])
    else:
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Dica: instale `streamlit-plotly-events` para habilitar clique. Fallback ao multiselect ao lado.")

with col_sel:
    st.markdown("**Selecionadas**")
    # Fallback multiselect also shown so user can adjust
    current = st.session_state["selected_labels"]
    new_sel = st.multiselect("",
                             options=labels,
                             default=current,
                             key="msel_positions")
    # Keep both in sync
    st.session_state["selected_labels"] = new_sel
    st.write(f"{len(new_sel)} posição(ões) selecionada(s).")

# Expose selection for downstream use
st.divider()
st.markdown("### Resultado")
st.json({"selected_positions": st.session_state["selected_labels"]})
