# app.py
# ------------------------------------------------------------
# PCA Analysis — Football Metrics (checkbox position selector)
# Positions: ONLY tokens from your DataFrame (comma-separated, case-insensitive)
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="PCA Analysis — Football Metrics", page_icon="⚽", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; padding-bottom: .75rem; }
      .stButton>button, .stDownloadButton>button { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_excel_files(files) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_excel(f)
        # keep file name as a "League" hint if you like
        df["League"] = getattr(f, "name", "Uploaded")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def tokens_from_cell(cell) -> list[str]:
    if not isinstance(cell, str):
        cell = str(cell)
    return [t.strip() for t in cell.split(",") if t.strip()]

def norm_token(t: str) -> str:
    return t.upper().strip()

def build_display_map(series: pd.Series) -> dict:
    """normalized token -> original display text (preserve first seen)."""
    disp = {}
    for cell in series.dropna().astype(str):
        for tok in tokens_from_cell(cell):
            n = norm_token(tok)
            disp.setdefault(n, tok)
    return disp

def checkbox_selector_positions(tokens_norm_sorted: list[str], display_map: dict, key_prefix: str) -> list[str]:
    """Checkbox grid with Select all / Clear / Invert actions."""
    st.markdown("**Positions**")
    sel_key = f"{key_prefix}_selected"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = set()

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Select all"):
            st.session_state[sel_key] = set(tokens_norm_sorted)
    with c2:
        if st.button("Clear"):
            st.session_state[sel_key] = set()
    with c3:
        if st.button("Invert"):
            st.session_state[sel_key] = set(set(tokens_norm_sorted) - st.session_state[sel_key])

    # grid of checkboxes
    per_row = 6
    rows = (len(tokens_norm_sorted) + per_row - 1) // per_row
    i = 0
    for _ in range(rows):
        cols = st.columns(per_row)
        for col in cols:
            if i >= len(tokens_norm_sorted):
                break
            tn = tokens_norm_sorted[i]; i += 1
            label = display_map.get(tn, tn)
            checked = tn in st.session_state[sel_key]
            new_val = col.checkbox(label, value=checked, key=f"{key_prefix}_{tn}")
            # sync back to set
            if new_val:
                st.session_state[sel_key].add(tn)
            else:
                st.session_state[sel_key].discard(tn)

    # return display labels (exactly as in data) for selected
    return [display_map.get(tn, tn) for tn in sorted(st.session_state[sel_key])]

# -------------------------
# App
# -------------------------
st.title("⚽ PCA Analysis — Physical & Technical Metrics")
st.markdown("Upload your Excel file(s), pick **positions using checkboxes**, choose metrics, and explore a **2D PCA**.")

# Sidebar — data & options
with st.sidebar:
    st.header("1) Data upload")
    files = st.file_uploader(
        "Select Excel file(s)",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
        help="Your data should include at least 'Player', 'Position' and numeric columns."
    )
    if not files:
        st.info("Upload at least one file to continue.")
        st.stop()

    data = load_excel_files(files)

    # Position column
    candidate_cols = [c for c in data.columns if c.lower() in ("pos", "position", "positions")]
    position_col = st.selectbox("Position column", options=candidate_cols or ["Position"])
    if position_col not in data.columns:
        st.error(f"Column '{position_col}' not found.")
        st.stop()

    st.divider()
    st.header("2) Filters")

    # Minutes
    minute_cols = [c for c in data.columns if "min" in c.lower() or "minutes" in c.lower()]
    if minute_cols:
        minute_col = st.selectbox("Minutes column", minute_cols, index=0)
        max_min = int(max(1, float(pd.to_numeric(data[minute_col], errors="coerce").fillna(0).max())))
        min_minutes = st.slider("Minimum minutes", 0, max_min, 0, step=50)
    else:
        minute_col, min_minutes = None, 0
        st.caption("No minutes column detected — minute filter disabled.")

    # Age
    if "Age" in data.columns:
        ages = pd.to_numeric(data["Age"], errors="coerce")
        if ages.notna().any():
            a_min, a_max = int(ages.min()), int(ages.max())
            age_range = st.slider("Age range", a_min, a_max, (a_min, a_max))
        else:
            age_range = None
            st.caption("Age column is not numeric — age filter disabled.")
    else:
        age_range = None
        st.caption("No 'Age' column — age filter disabled.")

    st.divider()
    st.header("3) Plot options")
    color_by = st.selectbox("Color points by",
                            options=[c for c in ["League", "Team", position_col] if c in data.columns] or ["League"])
    point_size = st.slider("Point size", 4, 16, 8)
    point_opacity = st.slider("Point opacity", 0.2, 1.0, 0.85)
    show_loadings = st.toggle("Show loadings (metric vectors)", value=True)
    loadings_scale = st.slider("Loadings length", 1.0, 6.0, 3.0, 0.5)

    st.divider()
    st.header("4) Export")
    export_format = st.radio("Format", ["HTML", "PNG (via kaleido)"], horizontal=True)
    export_btn = st.button("Export plot")

# Build position tokens from data (ONLY what's in the file)
pos_series = data[position_col].dropna().astype(str)
display_map = build_display_map(pos_series)       # normalized -> original text
tokens_norm_sorted = sorted(display_map.keys())
if not tokens_norm_sorted:
    st.error(f"No positions found in column '{position_col}'.")
    st.stop()

# Position selector — checkboxes
st.header("Select positions")
# (replaced) selected_display = checkbox_selector_positions(tokens_norm_sorted, display_map, key_prefix="poschk")
# Highlight players (optional)
st.subheader("Highlight players (optional)")
player_col = "Player" if "Player" in data.columns else None
if player_col:
    player_opts = sorted(pd.Series(data[player_col].astype(str)).unique().tolist())
    highlighted_players = st.multiselect("Select up to 5 players to label",
                                         options=player_opts, max_selections=5, placeholder="Type a name…")
else:
    highlighted_players = []
    st.caption("No 'Player' column — highlight disabled.")

# Metrics selection
st.header("Metrics selection")
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found — please upload data with numeric metrics.")
    st.stop()

non_metric_hints = {"age", "height", "weight", "minutes", "min", "games"}
default_metrics = [c for c in numeric_cols if c.lower() not in non_metric_hints] or numeric_cols
selected_metrics = st.multiselect("Pick at least two numeric columns for PCA",
                                  options=numeric_cols,
                                  default=default_metrics[: min(6, len(default_metrics))])
if len(selected_metrics) < 2:
    st.warning("Select at least two numeric columns to run PCA.")
    st.stop()

# Filtering (positions/minutes/age)
df = data.copy()

selected_norm = {norm_token(s) for s in selected_display}
if selected_norm:
    df = df[df[position_col].astype(str).apply(
        lambda s: any(norm_token(t) in selected_norm for t in tokens_from_cell(s))
    )]

if minute_col:
    df[minute_col] = pd.to_numeric(df[minute_col], errors="coerce")
    df = df[df[minute_col] >= min_minutes]

if age_range is not None and "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

df_numeric = df.dropna(subset=selected_metrics).copy()
if df_numeric.empty:
    st.warning("No rows left after filters. Try relaxing filters or pick different metrics.")
    st.stop()

# PCA
X = df_numeric[selected_metrics].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df_numeric["PCA1"] = coords[:, 0]
df_numeric["PCA2"] = coords[:, 1]
exp1, exp2 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]

# Plot (Plotly)
st.header("PCA plot")
fig = go.Figure()

group_col = color_by if color_by in df_numeric.columns else "League"
groups = sorted(df_numeric[group_col].astype(str).unique().tolist())
palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
pal = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

def hover_text(row):
    parts = []
    if "Player" in row and pd.notna(row["Player"]): parts.append(f"<b>{row['Player']}</b>")
    if "Team" in row and pd.notna(row["Team"]):     parts.append(f"Club: {row['Team']}")
    if group_col in row and pd.notna(row[group_col]): parts.append(f"{group_col}: {row[group_col]}")
    if "Age" in row and pd.notna(row["Age"]):
        try: parts.append(f"Age: {int(row['Age'])}")
        except Exception: parts.append(f"Age: {row['Age']}")
    parts.append(f"PC1: {row['PCA1']:.2f}")
    parts.append(f"PC2: {row['PCA2']:.2f}")
    return "<br>".join(parts)

# Optional highlight
if player_col:
    df_numeric["_is_high"] = df_numeric[player_col].astype(str).isin(set(highlighted_players))
else:
    df_numeric["_is_high"] = False

for g in groups:
    dfg = df_numeric[df_numeric[group_col].astype(str) == g]
    base = dfg[~dfg["_is_high"]]
    if not base.empty:
        fig.add_trace(go.Scatter(
            x=base["PCA1"], y=base["PCA2"], mode="markers", name=str(g),
            marker=dict(size=8, opacity=0.85, color=pal[g]),
            text=[hover_text(r) for _, r in base.iterrows()],
            hoverinfo="text", hovertemplate="%{text}<extra></extra>",
        ))
    hi = dfg[dfg["_is_high"]]
    if not hi.empty and player_col:
        for _, r in hi.iterrows():
            fig.add_trace(go.Scatter(
                x=[r["PCA1"]], y=[r["PCA2"]], mode="markers+text", name=str(r[player_col]),
                marker=dict(size=12, opacity=1.0, color=pal[g], symbol="diamond", line=dict(width=2, color="black")),
                text=[str(r[player_col])], textposition="bottom center",
                hovertext=[hover_text(r)], hoverinfo="text", hovertemplate="%{hovertext}<extra></extra>",
                legendgroup="highlighted", showlegend=True,
            ))

if show_loadings:
    comps = pca.components_
    for i, metric in enumerate(selected_metrics):
        fig.add_trace(go.Scatter(
            x=[0, comps[0, i] * loadings_scale], y=[0, comps[1, i] * loadings_scale],
            mode="lines+text", line=dict(width=2), text=[None, metric],
            textposition="top center", showlegend=False, hoverinfo="skip",
        ))

fig.update_layout(
    title=f"PCA — PC1 ({exp1:.1%}) vs PC2 ({exp2:.1%})",
    xaxis_title="PC1", yaxis_title="PC2",
    template="plotly_white", height=780, margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# Export
if export_btn:
    if export_format.startswith("HTML"):
        html = fig.to_html(full_html=True, include_plotlyjs="cdn")
        st.download_button("Download HTML", data=html, file_name="pca_analysis.html", mime="text/html")
    else:
        try:
            img = fig.to_image(format="png", width=1400, height=900, scale=3)  # needs kaleido
            st.download_button("Download PNG", data=img, file_name="pca_analysis.png", mime="image/png")
        except Exception:
            st.warning("PNG export requires the 'kaleido' package. Install with: `pip install kaleido`.")

# Data preview
with st.expander("Data preview"):
    meta_cols = [c for c in ["Player", position_col, "League", "Team", "Age"] if c in df_numeric.columns]
    st.dataframe(df_numeric[[*meta_cols, *selected_metrics, "PCA1", "PCA2"]].head(200),
                 use_container_width=True, height=320)

st.caption("Positions are taken exactly from your file. Comma-separated tokens count individually for filtering.")
# app.py
# ------------------------------------------------------------
# PCA Analysis — Football Metrics (Horizontal 16:9 mplsoccer template)
# Positions shown = EXACT tokens from your DataFrame (comma-separated)
# Deterministic field coordinates inferred PER TOKEN (no aliasing / no grouping)
# ------------------------------------------------------------
import io
import re
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Click capture
try:
    from streamlit_plotly_events import plotly_events
    _PLOTLY_EVENTS = True
except Exception:
    _PLOTLY_EVENTS = False

# Pitch rendering
try:
    from mplsoccer import Pitch
    _MPLSOCCER = True
except Exception:
    _MPLSOCCER = False

# -------------------------
# Page & CSS
# -------------------------
st.set_page_config(page_title="PCA Analysis — Football Metrics", page_icon="⚽", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
      .stButton>button, .stDownloadButton>button { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Domain (StatsBomb full, horizontal)
# -------------------------
X_MIN, X_MAX = 0.0, 120.0
Y_MIN, Y_MAX = 0.0, 80.0

# -------------------------
# EXACT template: horizontal 16:9 pitch with pads
# -------------------------
def make_mplsoccer_horizontal_png() -> bytes:
    if not _MPLSOCCER:
        return b""
    FIGWIDTH = 16
    FIGHEIGHT = 9
    NROWS = 1
    NCOLS = 1
    MAX_GRID = 1

    PAD_TOP = 2
    PAD_BOTTOM = 2
    PAD_SIDES = (((80 + PAD_BOTTOM + PAD_TOP) * FIGWIDTH / FIGHEIGHT) - 120) / 2

    pitch = Pitch(
        pad_top=PAD_TOP, pad_bottom=PAD_BOTTOM,
        pad_left=PAD_SIDES, pad_right=PAD_SIDES,
        pitch_color="grass", stripe=True, line_color="white",
        pitch_type="statsbomb", linewidth=2, goal_type="box"
    )
    GRID_WIDTH, GRID_HEIGHT = pitch.grid_dimensions(
        figwidth=FIGWIDTH, figheight=FIGHEIGHT,
        nrows=NROWS, ncols=NCOLS, max_grid=MAX_GRID, space=0
    )
    fig, ax = pitch.grid(
        figheight=FIGHEIGHT,
        grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT,
        title_height=0, endnote_height=0
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return buf.getvalue()

# -------------------------
# Data helpers
# -------------------------
def tokens_from_cell(cell) -> list[str]:
    if not isinstance(cell, str):
        cell = str(cell)
    return [t.strip() for t in cell.split(",") if t.strip()]

def norm_token(t: str) -> str:
    return t.upper().strip()

def build_display_map(series: pd.Series) -> dict:
    """Map normalized token -> original display text (keeps first appearance)."""
    disp = {}
    for cell in series.dropna().astype(str):
        for tok in tokens_from_cell(cell):
            n = norm_token(tok)
            disp.setdefault(n, tok)  # preserve human text exactly as in file
    return disp

# -------------------------
# Coordinate inference — NO aliasing, only rules per token shape
# We don't rename tokens; we just compute x,y from their pattern.
# -------------------------
SIDE_LEFT_Y   = 14.0
SIDE_RIGHT_Y  = 66.0
SIDE_CENTER_Y = 40.0

# base x by “band” (defense → attack)
X_GK   =  8.0
X_DEF  = 22.0        # CB/LB/RB tier
X_WB   = 34.0        # wing-backs a bit higher than def
X_DMF  = 42.0
X_CMF  = 55.0
X_AMF  = 70.0
X_WF   = 78.0
X_IF   = 86.0        # inside forwards (LF/RF)
X_CF   =  95.0       # CF/SS
X_ST   = 102.0       # out-and-out striker
X_FALLBACK = 60.0

_side_re = re.compile(r'^(L|R)')

def infer_xy_for_token(token_raw: str) -> tuple[float, float]:
    """
    Rules:
      - Use the token EXACTLY as given (after uppercasing for parsing).
      - Side: startswith 'L' -> y=L, 'R' -> y=R, else y=C.
      - Band by suffix keywords (checked in priority order, without altering the token itself).
    """
    t = norm_token(token_raw)

    # side
    m = _side_re.match(t)
    if m:
        y = SIDE_LEFT_Y if m.group(1) == 'L' else SIDE_RIGHT_Y
    else:
        y = SIDE_CENTER_Y

    # bands (priority)
    # goalkeeper
    if t.endswith("GK"):
        return (X_GK, y)

    # very common defenders
    if t.endswith("CB") or t in {"RB","LB"}:
        return (X_DEF, y)

    # wing-back
    if t.endswith("WB"):
        return (X_WB, y)

    # strictly midfield families found in your data: DMF/CMF/AMF
    if t.endswith("DMF"):
        return (X_DMF, y)
    if t.endswith("CMF"):
        return (X_CMF, y)
    if t.endswith("AMF"):
        return (X_AMF, y)

    # wide forwards
    if t.endswith("WF"):
        return (X_WF, y)

    # inside forwards (LF/RF), second striker
    if t in {"LF","RF"} or t == "SS":
        return (X_IF, y)

    # centre-forward / striker
    if t in {"CF"}:
        return (X_CF, y)
    if t in {"ST"}:
        return (X_ST, y)

    # fallback: put unfamiliar tokens in a neat shelf just below touchline
    return (X_FALLBACK, 6.0)

# -------------------------
# Plotly overlay aligned to mplsoccer PNG
# -------------------------
def build_pitch_overlay(selected_norm: set[str],
                        tokens_norm: list[str],
                        display_map: dict,
                        height: int = 640) -> go.Figure:
    fig = go.Figure()
    fig.update_xaxes(range=[X_MIN, X_MAX], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[Y_MIN, Y_MAX], showgrid=False, zeroline=False,
                     visible=False, scaleanchor="x", scaleratio=1)

    # background
    if _MPLSOCCER:
        png = make_mplsoccer_horizontal_png()
        if png:
            img = Image.open(io.BytesIO(png))
            fig.add_layout_image(
                dict(
                    source=img,
                    xref="x", yref="y",
                    x=X_MIN, y=Y_MAX, sizex=X_MAX - X_MIN, sizey=Y_MAX - Y_MIN,
                    sizing="stretch", layer="below", opacity=1.0
                )
            )
    else:
        fig.add_shape(type="rect", x0=X_MIN, y0=Y_MIN, x1=X_MAX, y1=Y_MAX,
                      fillcolor="#3a5f39", line=dict(color="#3a5f39"))

    # points (ONLY tokens from df)
    xs, ys, texts, order_norm = [], [], [], []
    for tn in tokens_norm:
        x, y = infer_xy_for_token(display_map[tn])  # compute from the exact token string
        xs.append(x); ys.append(y)
        texts.append(display_map[tn])    # show exactly as in file
        order_norm.append(tn)

    colors = ["#2563EB" if tn in selected_norm else "#6B7280" for tn in order_norm]

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(size=22, line=dict(width=2, color="white"), color=colors, symbol="circle"),
            text=texts, textposition="middle center", textfont=dict(color="white", size=12),
            customdata=order_norm,
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="Positions", showlegend=False,
        )
    )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#3a5f39",
        paper_bgcolor="#3a5f39",
    )
    return fig

# -------------------------
# Selector UI
# -------------------------
def ensure_state(key: str):
    if key not in st.session_state:
        st.session_state[key] = set()

def toggle_selected(key: str, tok_norm: str):
    ensure_state(key)
    if tok_norm in st.session_state[key]:
        st.session_state[key].remove(tok_norm)
    else:
        st.session_state[key].add(tok_norm)

def pitch_selector_horizontal(label: str,
                              key_prefix: str,
                              tokens_norm: list[str],
                              display_map: dict) -> list[str]:
    st.markdown(f"**{label}**")
    sel_key = f"{key_prefix}_sel"
    ensure_state(sel_key)

    c1, c2, c3, _ = st.columns([1,1,1,6])
    with c1:
        if st.button("Select all", key=f"{key_prefix}_all"):
            st.session_state[sel_key] = set(tokens_norm)
    with c2:
        if st.button("Clear", key=f"{key_prefix}_clear"):
            st.session_state[sel_key] = set()
    with c3:
        if st.button("Invert", key=f"{key_prefix}_invert"):
            st.session_state[sel_key] = set(set(tokens_norm) - st.session_state[sel_key])

    if _PLOTLY_EVENTS:
        fig = build_pitch_overlay(st.session_state[sel_key], tokens_norm, display_map)
        clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                               override_height=640, override_width="100%")
        if clicks:
            tn = clicks[0].get("customdata")
            if tn:
                toggle_selected(sel_key, tn)
                st.rerun()
            else:
                idx = clicks[0].get("pointIndex")
                if idx is not None and 0 <= idx < len(tokens_norm):
                    toggle_selected(sel_key, tokens_norm[idx])
                    st.rerun()
        st.caption("Tip: Click a label on the pitch to toggle selection.")
    else:
        st.warning("Interactive clicks require `streamlit-plotly-events`. Using a compact button grid.")
        per_row = 11
        i = 0
        rows = (len(tokens_norm) + per_row - 1) // per_row
        for _ in range(rows):
            cols = st.columns(per_row, gap="small")
            for col in cols:
                if i >= len(tokens_norm): break
                tn = tokens_norm[i]; i += 1
                label_btn = display_map.get(tn, tn)
                sel = tn in st.session_state[sel_key]
                label_btn = f"● {label_btn}" if sel else label_btn
                if col.button(label_btn, key=f"{key_prefix}_{tn}"):
                    toggle_selected(sel_key, tn)

    return [display_map.get(tn, tn) for tn in sorted(st.session_state[sel_key])]

# -------------------------
# Load data
# -------------------------
@st.cache_data(show_spinner=False)
def load_excel_files(files) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_excel(f)
        df["League"] = getattr(f, "name", "Uploaded")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# -------------------------
# App
# -------------------------
st.title("⚽ PCA Analysis — Physical & Technical Metrics")
st.markdown("Upload your Excel file(s), select positions **exactly as in your dataset** on a horizontal mplsoccer pitch (16:9), then run a 2D PCA.")

with st.sidebar:
    st.header("1) Data upload")
    files = st.file_uploader(
        "Select Excel file(s)",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
        help="Must include at least 'Player', 'Position' and numeric columns."
    )
    if not files:
        st.info("Upload at least one file to continue.")
        st.stop()

    data = load_excel_files(files)

    # Position column
    cand_cols = [c for c in data.columns if c.lower() in ("pos", "position", "positions")]
    position_col = st.selectbox("Position column", options=cand_cols or ["Position"])
    if position_col not in data.columns:
        st.error(f"Column '{position_col}' not found.")
        st.stop()

    st.divider()
    st.header("2) Filters")

    minute_cols = [c for c in data.columns if "min" in c.lower() or "minutes" in c.lower()]
    if minute_cols:
        minute_col = st.selectbox("Minutes column", minute_cols, index=0)
        max_min = int(max(1, float(pd.to_numeric(data[minute_col], errors="coerce").fillna(0).max())))
        min_minutes = st.slider("Minimum minutes", 0, max_min, 0, step=50)
    else:
        minute_col, min_minutes = None, 0
        st.caption("No minutes column detected — minute filter disabled.")

    if "Age" in data.columns:
        ages = pd.to_numeric(data["Age"], errors="coerce")
        if ages.notna().any():
            a_min, a_max = int(ages.min()), int(ages.max())
            age_range = st.slider("Age range", a_min, a_max, (a_min, a_max))
        else:
            age_range = None
            st.caption("Age column is not numeric — age filter disabled.")
    else:
        age_range = None
        st.caption("No 'Age' column — age filter disabled.")

    st.divider()
    st.header("3) Plot options")
    color_by = st.selectbox("Color points by", options=[c for c in ["League", "Team", position_col] if c in data.columns] or ["League"])
    point_size = st.slider("Point size", 4, 16, 8)
    point_opacity = st.slider("Point opacity", 0.2, 1.0, 0.85)
    show_loadings = st.toggle("Show loadings (metric vectors)", value=True)
    loadings_scale = st.slider("Loadings length", 1.0, 6.0, 3.0, 0.5)

    st.divider()
    st.header("4) Export")
    export_format = st.radio("Format", ["HTML", "PNG (300 DPI)"], horizontal=True)
    export_btn = st.button("Export plot")

# Build tokens ONLY from your data
pos_series = data[position_col].dropna().astype(str)
display_map = build_display_map(pos_series)
tokens_norm = sorted(display_map.keys())
if not tokens_norm:
    st.error(f"No positions found in column '{position_col}'.")
    st.stop()

# Pitch selector
st.header("Select positions on the pitch")
selected_display = pitch_selector_horizontal(
    label="Click labels to (de)select. Only positions that exist in your data are shown.",
    key_prefix="pitch",
    tokens_norm=tokens_norm,
    display_map=display_map,
)

# Highlight players (optional)
st.subheader("Highlight players (optional)")
player_col = "Player" if "Player" in data.columns else None
if player_col:
    player_opts = sorted(pd.Series(data[player_col].astype(str)).unique().tolist())
    highlighted_players = st.multiselect("Select up to 5 players to label in the plot", options=player_opts, max_selections=5)
else:
    highlighted_players = []
    st.caption("No 'Player' column — highlight disabled.")

# Metrics
st.header("Metrics selection")
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found — please upload data with numeric metrics.")
    st.stop()
default_metrics = [c for c in numeric_cols if c.lower() not in {"age","height","weight","minutes","min","games"}] or numeric_cols
selected_metrics = st.multiselect("Pick at least two numeric columns for PCA", options=numeric_cols, default=default_metrics[: min(6, len(default_metrics))])
if len(selected_metrics) < 2:
    st.warning("Select at least two numeric columns to run PCA.")
    st.stop()

# Filtering
df = data.copy()
selected_norm = {norm_token(s) for s in selected_display}
if selected_norm:
    df = df[df[position_col].astype(str).apply(lambda s: any(norm_token(t) in selected_norm for t in tokens_from_cell(s)))]

if minute_col:
    df[minute_col] = pd.to_numeric(df[minute_col], errors="coerce")
    df = df[df[minute_col] >= min_minutes]

if age_range is not None and "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

df_numeric = df.dropna(subset=selected_metrics).copy()
if df_numeric.empty:
    st.warning("No rows left after filters. Try relaxing filters or choosing different metrics.")
    st.stop()

# PCA
X = df_numeric[selected_metrics].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df_numeric["PCA1"] = coords[:, 0]
df_numeric["PCA2"] = coords[:, 1]
exp1, exp2 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]

# PCA plot
st.header("PCA plot")
fig = go.Figure()
group_col = color_by if color_by in df_numeric.columns else "League"
groups = sorted(df_numeric[group_col].astype(str).unique().tolist())
palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
pal = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

def hover_text(row):
    parts = []
    if "Player" in row and pd.notna(row["Player"]): parts.append(f"<b>{row['Player']}</b>")
    if "Team" in row and pd.notna(row["Team"]):     parts.append(f"Club: {row['Team']}")
    if group_col != "League" and group_col in row and pd.notna(row[group_col]): parts.append(f"{group_col}: {row[group_col]}")
    if "Age" in row and pd.notna(row["Age"]):
        try: parts.append(f"Age: {int(row['Age'])}")
        except Exception: parts.append(f"Age: {row['Age']}")
    parts.append(f"PC1: {row['PCA1']:.2f}")
    parts.append(f"PC2: {row['PCA2']:.2f}")
    return "<br>".join(parts)

if player_col:
    df_numeric["_is_high"] = df_numeric[player_col].astype(str).isin(set(highlighted_players))
else:
    df_numeric["_is_high"] = False

for g in groups:
    dfg = df_numeric[df_numeric[group_col].astype(str) == g]
    normal = dfg[~dfg["_is_high"]]
    if not normal.empty:
        fig.add_trace(go.Scatter(
            x=normal["PCA1"], y=normal["PCA2"], mode="markers", name=str(g),
            marker=dict(size=point_size, opacity=point_opacity, color=pal[g]),
            text=[hover_text(r) for _, r in normal.iterrows()],
            hoverinfo="text", hovertemplate="%{text}<extra></extra>",
        ))
    high = dfg[dfg["_is_high"]]
    if not high.empty and player_col:
        for _, r in high.iterrows():
            fig.add_trace(go.Scatter(
                x=[r["PCA1"]], y=[r["PCA2"]], mode="markers+text", name=str(r[player_col]),
                marker=dict(size=point_size+4, opacity=1.0, color=pal[g], symbol="diamond", line=dict(width=2, color="black")),
                text=[str(r[player_col])], textposition="bottom center",
                hovertext=[hover_text(r)], hoverinfo="text", hovertemplate="%{hovertext}<extra></extra>",
                legendgroup="highlighted", showlegend=True,
            ))

if show_loadings:
    comps = pca.components_
    for i, metric in enumerate(selected_metrics):
        fig.add_trace(go.Scatter(
            x=[0, comps[0, i] * loadings_scale], y=[0, comps[1, i] * loadings_scale],
            mode="lines+text", line=dict(width=2), text=[None, metric],
            textposition="top center", showlegend=False, hoverinfo="skip",
        ))

fig.update_layout(
    title=f"PCA — PC1 ({exp1:.1%}) vs PC2 ({exp2:.1%})",
    xaxis_title="PC1", yaxis_title="PC2",
    template="plotly_white", height=820, margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# Export
with st.sidebar:
    export_btn = st.button("Export now") if 'export_btn' not in locals() else export_btn
    if export_btn:
        html = fig.to_html(full_html=True, include_plotlyjs="cdn")
        st.download_button("Download HTML", data=html, file_name="pca_analysis.html", mime="text/html")

with st.expander("Data preview"):
    meta_cols = [c for c in ["Player", position_col, "League", "Team", "Age"] if c in df_numeric.columns]
    st.dataframe(df_numeric[[*meta_cols, *selected_metrics, "PCA1", "PCA2"]].head(200), use_container_width=True, height=320)

st.caption("Only positions present in your file are shown on the pitch (labels exactly as in the dataset). Comma-separated tokens count individually.")
# app.py
# ------------------------------------------------------------
# PCA Analysis — Football Metrics
# Horizontal mplsoccer pitch (16:9, pads conforme template) + clickable overlay
# Positions used = ONLY tokens from your DataFrame (comma-separated)
# ------------------------------------------------------------
import io
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Click capture (recommended)
try:
    from streamlit_plotly_events import plotly_events
    _PLOTLY_EVENTS = True
except Exception:
    _PLOTLY_EVENTS = False

# Pretty pitch via mplsoccer (Matplotlib)
try:
    from mplsoccer import Pitch
    _MPLSOCCER = True
except Exception:
    _MPLSOCCER = False

# -------------------------
# Streamlit page setup
# -------------------------
st.set_page_config(
    page_title="PCA Analysis — Football Metrics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 0.8rem; }
      .stButton>button, .stDownloadButton>button { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Domain (StatsBomb horizontal): x ∈ [0..120], y ∈ [0..80]
# -------------------------
X_MIN, X_MAX = 0.0, 120.0
Y_MIN, Y_MAX = 0.0, 80.0

# =========================
# EXACTLY your template (horizontal 16:9)
# =========================
def make_mplsoccer_horizontal_png() -> bytes:
    """
    Draw a beautiful StatsBomb full horizontal pitch with 16:9 figure,
    pads computed exactly as in your snippet, and return PNG bytes.
    """
    if not _MPLSOCCER:
        return b""
    FIGWIDTH = 16
    FIGHEIGHT = 9
    NROWS = 1
    NCOLS = 1
    MAX_GRID = 1

    # here we setup the padding to get a 16:9 aspect ratio for the axis
    # note 80 is the StatsBomb width and 120 is the StatsBomb length
    # this will extend the (axis) grassy effect to the figure edges
    PAD_TOP = 2
    PAD_BOTTOM = 2
    PAD_SIDES = (((80 + PAD_BOTTOM + PAD_TOP) * FIGWIDTH / FIGHEIGHT) - 120) / 2

    pitch = Pitch(
        pad_top=PAD_TOP, pad_bottom=PAD_BOTTOM,
        pad_left=PAD_SIDES, pad_right=PAD_SIDES,
        pitch_color="grass", stripe=True, line_color="white",
        pitch_type="statsbomb", linewidth=2, goal_type="box"
    )

    # calculate the maximum grid_height/width
    GRID_WIDTH, GRID_HEIGHT = pitch.grid_dimensions(
        figwidth=FIGWIDTH, figheight=FIGHEIGHT,
        nrows=NROWS, ncols=NCOLS, max_grid=MAX_GRID, space=0
    )

    # plot
    fig, ax = pitch.grid(
        figheight=FIGHEIGHT,
        grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT,
        title_height=0, endnote_height=0
    )

    # Save to PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return buf.getvalue()

# =========================
# Position helpers
# =========================
def tokens_from_cell(cell) -> list[str]:
    """Split by comma and normalize for comparison (UPPER/strip)."""
    if not isinstance(cell, str):
        cell = str(cell)
    return [t.strip().upper() for t in cell.split(",") if t.strip()]

def build_display_map(series: pd.Series) -> dict:
    """Map normalized token -> original display text (preserve first occurrence)."""
    disp = {}
    for cell in series.dropna().astype(str):
        for tok in cell.split(","):
            t = tok.strip()
            if not t:
                continue
            n = t.upper()
            disp.setdefault(n, t)
    return disp

# Horizontal anchors (only to pick COORDS; we always DISPLAY the dataset text)
# Known roles are placed intuitively; unknown tokens are arranged along a neat shelf.
ANCHORS = {
    # GK & back line
    "GK":  (6, 40),
    "RB":  (25, 66), "RCB": (22, 48), "CB":  (22, 40), "LCB": (22, 32), "LB":  (25, 14),
    "RWB": (40, 70), "LWB": (40, 10),

    # Midfield
    "RDM": (40, 48), "DM":  (40, 40), "LDM": (40, 32),
    "RCM": (55, 52), "CM":  (55, 40), "LCM": (55, 28),
    "RAM": (68, 52), "AM":  (70, 40), "LAM": (72, 28),

    # Wingers / forwards
    "RW":  (78, 66), "LW":  (78, 14),
    "RF":  (88, 55), "CF":  (90, 40), "LF":  (88, 25),
    "SS":  (84, 40), "ST":  (102, 40),
}

COORD_ALIAS = {
    "CDM": "DM", "DMF": "DM",
    "CMF": "CM",
    "CAM": "AM", "AMF": "AM",
    "RM": "RW", "LM": "LW",
    "RWF": "RW", "LWF": "LW",
    "RST": "ST", "LST": "ST",
}

def coord_for_token_norm(tn: str):
    base = COORD_ALIAS.get(tn, tn)
    return ANCHORS.get(base)

def layout_unknowns(tokens_norm: list[str]) -> dict[str, tuple[float, float]]:
    """Lay out unknown labels along a tidy baseline near y≈6, spread in x."""
    if not tokens_norm:
        return {}
    xs = np.linspace(12, 108, len(tokens_norm)) if len(tokens_norm) > 1 else np.array([60.0])
    y = 6.0
    return {t: (float(xs[i]), y) for i, t in enumerate(tokens_norm)}

# =========================
# Plotly overlay builder (horizontal)
# =========================
def build_horizontal_pitch_overlay(selected_norm: set[str],
                                   tokens_norm_in_data: list[str],
                                   display_map: dict,
                                   height: int = 640) -> go.Figure:
    """
    Create a Plotly figure with axes x:[0..120], y:[0..80], put the mplsoccer PNG
    as background (stretched to the domain), and overlay clickable markers.
    """
    fig = go.Figure()
    fig.update_xaxes(range=[X_MIN, X_MAX], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[Y_MIN, Y_MAX], showgrid=False, zeroline=False,
                     visible=False, scaleanchor="x", scaleratio=1)

    # Background (PNG from EXACT template)
    if _MPLSOCCER:
        png = make_mplsoccer_horizontal_png()
        if png:
            img = Image.open(io.BytesIO(png))
            fig.add_layout_image(
                dict(
                    source=img,
                    xref="x", yref="y",
                    x=X_MIN, y=Y_MAX,              # top-left corner in data coords
                    sizex=X_MAX - X_MIN, sizey=Y_MAX - Y_MIN,
                    sizing="stretch",
                    layer="below",
                    opacity=1.0,
                )
            )
    else:
        # fallback rectangle in grass color
        fig.add_shape(type="rect", x0=X_MIN, y0=Y_MIN, x1=X_MAX, y1=Y_MAX,
                      fillcolor="#3a5f39", line=dict(color="#3a5f39"))

    # Coordinates for dataset tokens only
    unknown = []
    xs, ys, labels, order_norm = [], [], [], []
    for tn in tokens_norm_in_data:
        xy = coord_for_token_norm(tn)
        if xy is None:
            unknown.append(tn)
        else:
            x, y = xy
            xs.append(x); ys.append(y)
            labels.append(display_map.get(tn, tn))
            order_norm.append(tn)
    if unknown:
        placed = layout_unknowns(unknown)
        for tn in unknown:
            x, y = placed[tn]
            xs.append(x); ys.append(y)
            labels.append(display_map.get(tn, tn))
            order_norm.append(tn)

    colors = ["#2563EB" if tn in selected_norm else "#6B7280" for tn in order_norm]

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(size=22, line=dict(width=2, color="white"), color=colors, symbol="circle"),
            text=labels, textposition="middle center", textfont=dict(color="white", size=12),
            customdata=order_norm,
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="Positions", showlegend=False,
        )
    )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#3a5f39",
        paper_bgcolor="#3a5f39",
    )
    return fig

# =========================
# Selector UI
# =========================
def ensure_state(key: str):
    if key not in st.session_state:
        st.session_state[key] = set()

def toggle_selected(key: str, tok_norm: str):
    ensure_state(key)
    if tok_norm in st.session_state[key]:
        st.session_state[key].remove(tok_norm)
    else:
        st.session_state[key].add(tok_norm)

def pitch_selector_horizontal(label: str,
                              key_prefix: str,
                              tokens_norm_in_data: list[str],
                              display_map: dict) -> list[str]:
    """Horizontal selector: ONLY tokens from your dataset, clickable over the pitch."""
    st.markdown(f"**{label}**")
    sel_key = f"{key_prefix}_sel"
    ensure_state(sel_key)

    c1, c2, c3, _ = st.columns([1,1,1,6])
    with c1:
        if st.button("Select all", key=f"{key_prefix}_all"):
            st.session_state[sel_key] = set(tokens_norm_in_data)
    with c2:
        if st.button("Clear", key=f"{key_prefix}_clear"):
            st.session_state[sel_key] = set()
    with c3:
        if st.button("Invert", key=f"{key_prefix}_invert"):
            st.session_state[sel_key] = set(set(tokens_norm_in_data) - st.session_state[sel_key])

    if _PLOTLY_EVENTS:
        fig = build_horizontal_pitch_overlay(st.session_state[sel_key], tokens_norm_in_data, display_map)
        clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                               override_height=640, override_width="100%")
        if clicks:
            tok_norm = clicks[0].get("customdata")
            if tok_norm:
                toggle_selected(sel_key, tok_norm)
                st.rerun()
            else:
                idx = clicks[0].get("pointIndex")
                if idx is not None and 0 <= idx < len(tokens_norm_in_data):
                    toggle_selected(sel_key, tokens_norm_in_data[idx])
                    st.rerun()
        st.caption("Tip: Click a label on the pitch to toggle selection.")
    else:
        st.warning("Interactive clicks require `streamlit-plotly-events`. Using a compact button grid instead.")
        per_row = 11
        i = 0
        rows = (len(tokens_norm_in_data) + per_row - 1) // per_row
        for _ in range(rows):
            cols = st.columns(per_row, gap="small")
            for col in cols:
                if i >= len(tokens_norm_in_data): break
                tn = tokens_norm_in_data[i]; i += 1
                label_btn = display_map.get(tn, tn)
                sel = tn in st.session_state[sel_key]
                label_btn = f"● {label_btn}" if sel else label_btn
                if col.button(label_btn, key=f"{key_prefix}_{tn}"):
                    toggle_selected(sel_key, tn)

    # Return display labels (exactly as in data) for selected tokens
    return [display_map.get(tn, tn) for tn in sorted(st.session_state[sel_key])]

# =========================
# Data loading
# =========================
@st.cache_data(show_spinner=False)
def load_excel_files(files) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_excel(f)
        df["League"] = getattr(f, "name", "Uploaded")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# =========================
# App
# =========================
st.title("⚽ PCA Analysis — Football Metrics")
st.markdown(
    "Upload your Excel file(s), select positions **straight from your dataset** on a "
    "**horizontal** mplsoccer pitch (16:9 template), pick numeric metrics, and explore a **2D PCA**."
)

# Sidebar — upload & options
with st.sidebar:
    st.header("1) Data upload")
    files = st.file_uploader(
        "Select Excel file(s)",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
        help="Must include at least 'Player', 'Position' and numeric columns."
    )
    if not files:
        st.info("Upload at least one file to continue.")
        st.stop()

    data = load_excel_files(files)

    # Position column
    cand_cols = [c for c in data.columns if c.lower() in ("pos", "position", "positions")]
    position_col = st.selectbox("Position column", options=cand_cols or ["Position"])
    if position_col not in data.columns:
        st.error(f"Column '{position_col}' not found. Please select a valid column.")
        st.stop()

    st.divider()
    st.header("2) Filters")

    # Minutes
    minute_cols = [c for c in data.columns if "min" in c.lower() or "minutes" in c.lower()]
    if minute_cols:
        minute_col = st.selectbox("Minutes column", minute_cols, index=0)
        max_min = int(max(1, float(pd.to_numeric(data[minute_col], errors="coerce").fillna(0).max())))
        min_minutes = st.slider("Minimum minutes", 0, max_min, 0, step=50)
    else:
        minute_col, min_minutes = None, 0
        st.caption("No minutes column detected — minute filter disabled.")

    # Age
    if "Age" in data.columns:
        ages = pd.to_numeric(data["Age"], errors="coerce")
        if ages.notna().any():
            a_min, a_max = int(ages.min()), int(ages.max())
            age_range = st.slider("Age range", a_min, a_max, (a_min, a_max))
        else:
            age_range = None
            st.caption("Age column is not numeric — age filter disabled.")
    else:
        age_range = None
        st.caption("No 'Age' column — age filter disabled.")

    st.divider()
    st.header("3) Plot options")
    color_by = st.selectbox(
        "Color points by",
        options=[c for c in ["League", "Team", position_col] if c in data.columns] or ["League"]
    )
    point_size = st.slider("Point size", 4, 16, 8)
    point_opacity = st.slider("Point opacity", 0.2, 1.0, 0.85)
    show_loadings = st.toggle("Show loadings (metric vectors)", value=True)
    loadings_scale = st.slider("Loadings length", 1.0, 6.0, 3.0, 0.5)

    st.divider()
    st.header("4) Export")
    export_format = st.radio("Format", ["HTML", "PNG (300 DPI)"], horizontal=True)
    export_btn = st.button("Export plot")

# Build token universe ONLY from your file
pos_series = data[position_col].dropna().astype(str)
display_map = build_display_map(pos_series)     # normalized -> original display text
tokens_norm = sorted(display_map.keys())
if not tokens_norm:
    st.error(f"No positions found in column '{position_col}'. Please check your file.")
    st.stop()

# Pitch selector (horizontal, template-true)
st.header("Select positions on the pitch")
selected_display = pitch_selector_horizontal(
    label="Click labels to (de)select. Only positions that exist in your data are shown.",
    key_prefix="pitch",
    tokens_norm_in_data=tokens_norm,
    display_map=display_map,
)

# Highlight players (optional)
st.subheader("Highlight players (optional)")
player_col = "Player" if "Player" in data.columns else None
if player_col:
    player_opts = sorted(pd.Series(data[player_col].astype(str)).unique().tolist())
    highlighted_players = st.multiselect(
        "Select up to 5 players to label in the plot",
        options=player_opts, max_selections=5, placeholder="Type a name…",
    )
else:
    highlighted_players = []
    st.caption("No 'Player' column — highlight disabled.")

# Metrics
st.header("Metrics selection")
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found — please upload data with numeric metrics.")
    st.stop()
non_metric_hints = {"Age", "Height", "Weight", "Minutes", "Min", "Games"}
default_metrics = [c for c in numeric_cols if c not in non_metric_hints] or numeric_cols
selected_metrics = st.multiselect(
    "Pick at least two numeric columns for PCA",
    options=numeric_cols,
    default=default_metrics[: min(6, len(default_metrics))],
)
if len(selected_metrics) < 2:
    st.warning("Select at least two numeric columns to run PCA.")
    st.stop()

# Filtering (positions/minutes/age)
df = data.copy()

# Position filter (each comma-separated token counts individually)
selected_norm = {s.upper().strip() for s in selected_display}
if selected_norm and position_col in df.columns:
    df = df[df[position_col].astype(str).apply(
        lambda s: any(tok in selected_norm for tok in tokens_from_cell(s))
    )]

if minute_col:
    df[minute_col] = pd.to_numeric(df[minute_col], errors="coerce")
    df = df[df[minute_col] >= min_minutes]

if age_range is not None and "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

df_numeric = df.dropna(subset=selected_metrics).copy()
if df_numeric.empty:
    st.warning("No rows left after filters. Try relaxing filters or choosing different metrics.")
    st.stop()

# PCA
X = df_numeric[selected_metrics].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df_numeric["PCA1"] = coords[:, 0]
df_numeric["PCA2"] = coords[:, 1]
exp1, exp2 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]

# PCA plot
st.header("PCA plot")
fig = go.Figure()
group_col = color_by if color_by in df_numeric.columns else "League"
groups = sorted(df_numeric[group_col].astype(str).unique().tolist())

base_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]
palette = {g: base_colors[i % len(base_colors)] for i, g in enumerate(groups)}

def hover_text(row):
    parts = []
    if "Player" in row and pd.notna(row["Player"]):
        parts.append(f"<b>{row['Player']}</b>")
    if "Team" in row and pd.notna(row["Team"]):
        parts.append(f"Club: {row['Team']}")
    if group_col != "League" and group_col in row and pd.notna(row[group_col]):
        parts.append(f"{group_col}: {row[group_col]}")
    if "Age" in row and pd.notna(row["Age"]):
        try:
            parts.append(f"Age: {int(row['Age'])}")
        except Exception:
            parts.append(f"Age: {row['Age']}")
    parts.append(f"PC1: {row['PCA1']:.2f}")
    parts.append(f"PC2: {row['PCA2']:.2f}")
    return "<br>".join(parts)

# mark highlights
if player_col:
    df_numeric["_is_high"] = df_numeric[player_col].astype(str).isin(set(highlighted_players))
else:
    df_numeric["_is_high"] = False

for g in groups:
    dfg = df_numeric[df_numeric[group_col].astype(str) == g]
    normal = dfg[~dfg["_is_high"]]
    if not normal.empty:
        fig.add_trace(
            go.Scatter(
                x=normal["PCA1"], y=normal["PCA2"],
                mode="markers",
                name=str(g),
                marker=dict(size=point_size, opacity=point_opacity, color=palette[g]),
                text=[hover_text(r) for _, r in normal.iterrows()],
                hoverinfo="text",
                hovertemplate="%{text}<extra></extra>",
            )
        )
    high = dfg[dfg["_is_high"]]
    if not high.empty and player_col:
        for _, r in high.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[r["PCA1"]], y=[r["PCA2"]],
                    mode="markers+text",
                    name=str(r[player_col]),
                    marker=dict(size=point_size + 4, opacity=1.0,
                                color=palette[g], symbol="diamond", line=dict(width=2, color="black")),
                    text=[str(r[player_col])],
                    textposition="bottom center",
                    hovertext=[hover_text(r)],
                    hoverinfo="text",
                    hovertemplate="%{hovertext}<extra></extra>",
                    legendgroup="highlighted",
                    showlegend=True,
                )
            )

if show_loadings:
    comps = pca.components_
    for i, metric in enumerate(selected_metrics):
        fig.add_trace(
            go.Scatter(
                x=[0, comps[0, i] * loadings_scale],
                y=[0, comps[1, i] * loadings_scale],
                mode="lines+text",
                line=dict(width=2),
                text=[None, metric],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

fig.update_layout(
    title=f"PCA — PC1 ({exp1:.1%}) vs PC2 ({exp2:.1%})",
    xaxis_title="PC1",
    yaxis_title="PC2",
    template="plotly_white",
    height=820,
    margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# Export
with st.sidebar:
    if export_btn:
        if export_format.startswith("HTML"):
            html = fig.to_html(full_html=True, include_plotlyjs="cdn")
            st.download_button("Download HTML", data=html, file_name="pca_analysis.html", mime="text/html")
        else:
            try:
                img = fig.to_image(format="png", width=1400, height=900, scale=3)  # requires kaleido
                st.download_button("Download PNG", data=img, file_name="pca_analysis.png", mime="image/png")
            except Exception:
                st.warning("PNG export requires the 'kaleido' package. Install with: `pip install kaleido`.")

# Data preview
with st.expander("Data preview"):
    meta_cols = [c for c in ["Player", position_col, "League", "Team", "Age"] if c in df_numeric.columns]
    st.dataframe(
        df_numeric[[*meta_cols, *selected_metrics, "PCA1", "PCA2"]].head(200),
        use_container_width=True,
        height=320,
    )

st.caption(
    "Only positions present in your file are shown on the pitch (labels displayed exactly as in the dataset). "
    "Comma-separated tokens count individually for filtering."
)


# app.py
# ------------------------------------------------------------
# PCA Analysis — Football Metrics (checkbox position selector)
# Positions: ONLY tokens from your DataFrame (comma-separated, case-insensitive)
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# ---------- Virtual Pitch Selector (NO GROUPING) ----------
# Each position label from the dataframe (comma-separated tokens) is its own clickable item.
from typing import Dict, List, Tuple
import io
from PIL import Image
from mplsoccer import Pitch

# Optional click capture; fall back to multiselect if not available.
try:
    from streamlit_plotly_events import plotly_events
    _HAS_SPE = True
except Exception:
    _HAS_SPE = False

def _tokens_from_series_exact(series: pd.Series) -> List[str]:
    toks = []
    for cell in series.dropna().astype(str):
        for tok in str(cell).split(","):
            tok = tok.strip()
            if tok:
                toks.append(tok)  # exact, no normalization
    # unique preserving order
    seen = set()
    uniq = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def _coords_for_labels_exact(labels: List[str], pitch_length: int = 120, pitch_width: int = 80) -> Dict[str, Tuple[float, float]]:
    # Canonical exact matches only (no aliasing)
    known: Dict[str, Tuple[float, float]] = {
        "GK": (6, pitch_width/2),
        "RB": (20, pitch_width*0.25),
        "RCB": (18, pitch_width*0.45),
        "CB": (18, pitch_width/2),
        "LCB": (18, pitch_width*0.55),
        "LB": (20, pitch_width*0.75),
        "RWB": (28, pitch_width*0.28),
        "LWB": (28, pitch_width*0.72),
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
        "RW": (78, pitch_width*0.30),
        "LW": (78, pitch_width*0.70),
        "RM": (70, pitch_width*0.33),
        "LM": (70, pitch_width*0.67),
        "RF": (92, pitch_width*0.45),
        "LF": (92, pitch_width*0.55),
        "SS": (88, pitch_width/2),
        "CF": (100, pitch_width/2),
        "ST": (100, pitch_width/2),
    }
    coords: Dict[str, Tuple[float, float]] = {}
    shelf = [lbl for lbl in labels if lbl not in known]
    if shelf:
        import numpy as _np
        xs = _np.linspace(8, pitch_length-8, num=len(shelf))
        y = _np.full(len(shelf), 6.0)
        for i, lbl in enumerate(shelf):
            coords[lbl] = (float(xs[i]), float(y[i]))
    for lbl in labels:
        if lbl in known:
            coords[lbl] = known[lbl]
    return coords

def _render_mpl_pitch_image(pitch_type: str = "statsbomb", pitch_length: int = 120, pitch_width: int = 80) -> str:
    pitch = Pitch(pitch_type=pitch_type, pitch_color="white", line_color="black",
                  goal_type="box", pitch_length=pitch_length, pitch_width=pitch_width)
    fig, ax = pitch.draw(figsize=(10, 6.666), tight_layout=True)
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return "data:image/png;base64," + b64

def virtual_pitch_selector_positions(series: pd.Series, *, key_prefix: str = "pos", pitch_type: str = "statsbomb",
                                     pitch_length: int = 120, pitch_width: int = 80) -> list[str]:
    import plotly.graph_objects as go
    labels = _tokens_from_series_exact(series)
    if not labels:
        st.warning("Não encontrei posições na coluna informada.")
        return []

    # keep state
    sel_key = f"{key_prefix}_selected_labels"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = []

    coords = _coords_for_labels_exact(labels, pitch_length=pitch_length, pitch_width=pitch_width)
    bg = _render_mpl_pitch_image(pitch_type=pitch_type, pitch_length=pitch_length, pitch_width=pitch_width)
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=bg, xref="x", yref="y", x=0, y=pitch_width,
            sizex=pitch_length, sizey=pitch_width, sizing="stretch",
            layer="below", opacity=1.0,
        )
    )
                except Exception:
                    idx = -1
                if 0 <= idx < len(labels):
                    lab = labels[idx]
                    cur = set(st.session_state[sel_key])
                    if lab in cur:
                        cur.remove(lab)
                    else:
                        cur.add(lab)
                    st.session_state[sel_key] = list(cur)
        else:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Dica: instale `streamlit-plotly-events` para habilitar cliques no campo.")

    with col_sel:
        current = st.session_state[sel_key]
        new_sel = st.multiselect("Selecionadas", options=labels, default=current, key=f"{key_prefix}_msel")
        st.session_state[sel_key] = new_sel
        st.write(f"{len(new_sel)} posição(ões) selecionada(s).")

    return st.session_state[sel_key]
# ----------------------------------------------------------

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="PCA Analysis — Football Metrics", page_icon="⚽", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; padding-bottom: .75rem; }
      .stButton>button, .stDownloadButton>button { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_excel_files(files) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_excel(f)
        # keep file name as a "League" hint if you like
        df["League"] = getattr(f, "name", "Uploaded")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def tokens_from_cell(cell) -> list[str]:
    if not isinstance(cell, str):
        cell = str(cell)
    return [t.strip() for t in cell.split(",") if t.strip()]

def norm_token(t: str) -> str:
    return t.upper().strip()

def build_display_map(series: pd.Series) -> dict:
    """normalized token -> original display text (preserve first seen)."""
    disp = {}
    for cell in series.dropna().astype(str):
        for tok in tokens_from_cell(cell):
            n = norm_token(tok)
            disp.setdefault(n, tok)
    return disp

def virtual_pitch_selector_positions(df[pos_col], key_prefix='pos', pitch_type='statsbomb', pitch_length=120, pitch_width=80) -> list[str]:
    """Checkbox grid with Select all / Clear / Invert actions."""
    st.markdown("**Positions**")
    sel_key = f"{key_prefix}_selected"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = set()

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Select all"):
            st.session_state[sel_key] = set(tokens_norm_sorted)
    with c2:
        if st.button("Clear"):
            st.session_state[sel_key] = set()
    with c3:
        if st.button("Invert"):
            st.session_state[sel_key] = set(set(tokens_norm_sorted) - st.session_state[sel_key])

    # grid of checkboxes
    per_row = 6
    rows = (len(tokens_norm_sorted) + per_row - 1) // per_row
    i = 0
    for _ in range(rows):
        cols = st.columns(per_row)
        for col in cols:
            if i >= len(tokens_norm_sorted):
                break
            tn = tokens_norm_sorted[i]; i += 1
            label = display_map.get(tn, tn)
            checked = tn in st.session_state[sel_key]
            new_val = col.checkbox(label, value=checked, key=f"{key_prefix}_{tn}")
            # sync back to set
            if new_val:
                st.session_state[sel_key].add(tn)
            else:
                st.session_state[sel_key].discard(tn)

    # return display labels (exactly as in data) for selected
    return [display_map.get(tn, tn) for tn in sorted(st.session_state[sel_key])]

# -------------------------
# App
# -------------------------
st.title("⚽ PCA Analysis — Physical & Technical Metrics")
st.markdown("Upload your Excel file(s), pick **positions using checkboxes**, choose metrics, and explore a **2D PCA**.")

# Sidebar — data & options
with st.sidebar:
    st.header("1) Data upload")
    files = st.file_uploader(
        "Select Excel file(s)",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
        help="Your data should include at least 'Player', 'Position' and numeric columns."
    )
    if not files:
        st.info("Upload at least one file to continue.")
        st.stop()

    data = load_excel_files(files)

    # Position column
    candidate_cols = [c for c in data.columns if c.lower() in ("pos", "position", "positions")]
    position_col = st.selectbox("Position column", options=candidate_cols or ["Position"])
    if position_col not in data.columns:
        st.error(f"Column '{position_col}' not found.")
        st.stop()

    st.divider()
    st.header("2) Filters")

    # Minutes
    minute_cols = [c for c in data.columns if "min" in c.lower() or "minutes" in c.lower()]
    if minute_cols:
        minute_col = st.selectbox("Minutes column", minute_cols, index=0)
        max_min = int(max(1, float(pd.to_numeric(data[minute_col], errors="coerce").fillna(0).max())))
        min_minutes = st.slider("Minimum minutes", 0, max_min, 0, step=50)
    else:
        minute_col, min_minutes = None, 0
        st.caption("No minutes column detected — minute filter disabled.")

    # Age
    if "Age" in data.columns:
        ages = pd.to_numeric(data["Age"], errors="coerce")
        if ages.notna().any():
            a_min, a_max = int(ages.min()), int(ages.max())
            age_range = st.slider("Age range", a_min, a_max, (a_min, a_max))
        else:
            age_range = None
            st.caption("Age column is not numeric — age filter disabled.")
    else:
        age_range = None
        st.caption("No 'Age' column — age filter disabled.")

    st.divider()
    st.header("3) Plot options")
    color_by = st.selectbox("Color points by",
                            options=[c for c in ["League", "Team", position_col] if c in data.columns] or ["League"])
    point_size = st.slider("Point size", 4, 16, 8)
    point_opacity = st.slider("Point opacity", 0.2, 1.0, 0.85)
    show_loadings = st.toggle("Show loadings (metric vectors)", value=True)
    loadings_scale = st.slider("Loadings length", 1.0, 6.0, 3.0, 0.5)

    st.divider()
    st.header("4) Export")
    export_format = st.radio("Format", ["HTML", "PNG (via kaleido)"], horizontal=True)
    export_btn = st.button("Export plot")

# Build position tokens from data (ONLY what's in the file)
pos_series = data[position_col].dropna().astype(str)
display_map = build_display_map(pos_series)       # normalized -> original text
tokens_norm_sorted = sorted(display_map.keys())
if not tokens_norm_sorted:
    st.error(f"No positions found in column '{position_col}'.")
    st.stop()

# Position selector — checkboxes
st.header("Select positions")
selected_display = virtual_pitch_selector_positions(df[pos_col], key_prefix='pos', pitch_type='statsbomb', pitch_length=120, pitch_width=80)

# Highlight players (optional)
st.subheader("Highlight players (optional)")
player_col = "Player" if "Player" in data.columns else None
if player_col:
    player_opts = sorted(pd.Series(data[player_col].astype(str)).unique().tolist())
    highlighted_players = st.multiselect("Select up to 5 players to label",
                                         options=player_opts, max_selections=5, placeholder="Type a name…")
else:
    highlighted_players = []
    st.caption("No 'Player' column — highlight disabled.")

# Metrics selection
st.header("Metrics selection")
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found — please upload data with numeric metrics.")
    st.stop()

non_metric_hints = {"age", "height", "weight", "minutes", "min", "games"}
default_metrics = [c for c in numeric_cols if c.lower() not in non_metric_hints] or numeric_cols
selected_metrics = st.multiselect("Pick at least two numeric columns for PCA",
                                  options=numeric_cols,
                                  default=default_metrics[: min(6, len(default_metrics))])
if len(selected_metrics) < 2:
    st.warning("Select at least two numeric columns to run PCA.")
    st.stop()

# Filtering (positions/minutes/age)
df = data.copy()

selected_norm = {norm_token(s) for s in selected_display}
if selected_norm:
    df = df[df[position_col].astype(str).apply(
        lambda s: any(norm_token(t) in selected_norm for t in tokens_from_cell(s))
    )]

if minute_col:
    df[minute_col] = pd.to_numeric(df[minute_col], errors="coerce")
    df = df[df[minute_col] >= min_minutes]

if age_range is not None and "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

df_numeric = df.dropna(subset=selected_metrics).copy()
if df_numeric.empty:
    st.warning("No rows left after filters. Try relaxing filters or pick different metrics.")
    st.stop()

# PCA
X = df_numeric[selected_metrics].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df_numeric["PCA1"] = coords[:, 0]
df_numeric["PCA2"] = coords[:, 1]
exp1, exp2 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]

# Plot (Plotly)
st.header("PCA plot")
fig = go.Figure()

group_col = color_by if color_by in df_numeric.columns else "League"
groups = sorted(df_numeric[group_col].astype(str).unique().tolist())
palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
pal = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

def hover_text(row):
    parts = []
    if "Player" in row and pd.notna(row["Player"]): parts.append(f"<b>{row['Player']}</b>")
    if "Team" in row and pd.notna(row["Team"]):     parts.append(f"Club: {row['Team']}")
    if group_col in row and pd.notna(row[group_col]): parts.append(f"{group_col}: {row[group_col]}")
    if "Age" in row and pd.notna(row["Age"]):
        try: parts.append(f"Age: {int(row['Age'])}")
        except Exception: parts.append(f"Age: {row['Age']}")
    parts.append(f"PC1: {row['PCA1']:.2f}")
    parts.append(f"PC2: {row['PCA2']:.2f}")
    return "<br>".join(parts)

# Optional highlight
if player_col:
    df_numeric["_is_high"] = df_numeric[player_col].astype(str).isin(set(highlighted_players))
else:
    df_numeric["_is_high"] = False

for g in groups:
    dfg = df_numeric[df_numeric[group_col].astype(str) == g]
    base = dfg[~dfg["_is_high"]]
    if not base.empty:
        fig.add_trace(go.Scatter(
            x=base["PCA1"], y=base["PCA2"], mode="markers", name=str(g),
            marker=dict(size=8, opacity=0.85, color=pal[g]),
            text=[hover_text(r) for _, r in base.iterrows()],
            hoverinfo="text", hovertemplate="%{text}<extra></extra>",
        ))
    hi = dfg[dfg["_is_high"]]
    if not hi.empty and player_col:
        for _, r in hi.iterrows():
            fig.add_trace(go.Scatter(
                x=[r["PCA1"]], y=[r["PCA2"]], mode="markers+text", name=str(r[player_col]),
                marker=dict(size=12, opacity=1.0, color=pal[g], symbol="diamond", line=dict(width=2, color="black")),
                text=[str(r[player_col])], textposition="bottom center",
                hovertext=[hover_text(r)], hoverinfo="text", hovertemplate="%{hovertext}<extra></extra>",
                legendgroup="highlighted", showlegend=True,
            ))

if show_loadings:
    comps = pca.components_
    for i, metric in enumerate(selected_metrics):
        fig.add_trace(go.Scatter(
            x=[0, comps[0, i] * loadings_scale], y=[0, comps[1, i] * loadings_scale],
            mode="lines+text", line=dict(width=2), text=[None, metric],
            textposition="top center", showlegend=False, hoverinfo="skip",
        ))

fig.update_layout(
    title=f"PCA — PC1 ({exp1:.1%}) vs PC2 ({exp2:.1%})",
    xaxis_title="PC1", yaxis_title="PC2",
    template="plotly_white", height=780, margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# Export
if export_btn:
    if export_format.startswith("HTML"):
        html = fig.to_html(full_html=True, include_plotlyjs="cdn")
        st.download_button("Download HTML", data=html, file_name="pca_analysis.html", mime="text/html")
    else:
        try:
            img = fig.to_image(format="png", width=1400, height=900, scale=3)  # needs kaleido
            st.download_button("Download PNG", data=img, file_name="pca_analysis.png", mime="image/png")
        except Exception:
            st.warning("PNG export requires the 'kaleido' package. Install with: `pip install kaleido`.")

# Data preview
with st.expander("Data preview"):
    meta_cols = [c for c in ["Player", position_col, "League", "Team", "Age"] if c in df_numeric.columns]
    st.dataframe(df_numeric[[*meta_cols, *selected_metrics, "PCA1", "PCA2"]].head(200),
                 use_container_width=True, height=320)

st.caption("Positions are taken exactly from your file. Comma-separated tokens count individually for filtering.")
# app.py
# ------------------------------------------------------------
# PCA Analysis — Football Metrics (Horizontal 16:9 mplsoccer template)
# Positions shown = EXACT tokens from your DataFrame (comma-separated)
# Deterministic field coordinates inferred PER TOKEN (no aliasing / no grouping)
# ------------------------------------------------------------
import io
import re
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Click capture
try:
    from streamlit_plotly_events import plotly_events
    _PLOTLY_EVENTS = True
except Exception:
    _PLOTLY_EVENTS = False

# Pitch rendering
try:
    from mplsoccer import Pitch
    _MPLSOCCER = True
except Exception:
    _MPLSOCCER = False

# -------------------------
# Page & CSS
# -------------------------
st.set_page_config(page_title="PCA Analysis — Football Metrics", page_icon="⚽", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
      .stButton>button, .stDownloadButton>button { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Domain (StatsBomb full, horizontal)
# -------------------------
X_MIN, X_MAX = 0.0, 120.0
Y_MIN, Y_MAX = 0.0, 80.0

# -------------------------
# EXACT template: horizontal 16:9 pitch with pads
# -------------------------
def make_mplsoccer_horizontal_png() -> bytes:
    if not _MPLSOCCER:
        return b""
    FIGWIDTH = 16
    FIGHEIGHT = 9
    NROWS = 1
    NCOLS = 1
    MAX_GRID = 1

    PAD_TOP = 2
    PAD_BOTTOM = 2
    PAD_SIDES = (((80 + PAD_BOTTOM + PAD_TOP) * FIGWIDTH / FIGHEIGHT) - 120) / 2

    pitch = Pitch(
        pad_top=PAD_TOP, pad_bottom=PAD_BOTTOM,
        pad_left=PAD_SIDES, pad_right=PAD_SIDES,
        pitch_color="grass", stripe=True, line_color="white",
        pitch_type="statsbomb", linewidth=2, goal_type="box"
    )
    GRID_WIDTH, GRID_HEIGHT = pitch.grid_dimensions(
        figwidth=FIGWIDTH, figheight=FIGHEIGHT,
        nrows=NROWS, ncols=NCOLS, max_grid=MAX_GRID, space=0
    )
    fig, ax = pitch.grid(
        figheight=FIGHEIGHT,
        grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT,
        title_height=0, endnote_height=0
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return buf.getvalue()

# -------------------------
# Data helpers
# -------------------------
def tokens_from_cell(cell) -> list[str]:
    if not isinstance(cell, str):
        cell = str(cell)
    return [t.strip() for t in cell.split(",") if t.strip()]

def norm_token(t: str) -> str:
    return t.upper().strip()

def build_display_map(series: pd.Series) -> dict:
    """Map normalized token -> original display text (keeps first appearance)."""
    disp = {}
    for cell in series.dropna().astype(str):
        for tok in tokens_from_cell(cell):
            n = norm_token(tok)
            disp.setdefault(n, tok)  # preserve human text exactly as in file
    return disp

# -------------------------
# Coordinate inference — NO aliasing, only rules per token shape
# We don't rename tokens; we just compute x,y from their pattern.
# -------------------------
SIDE_LEFT_Y   = 14.0
SIDE_RIGHT_Y  = 66.0
SIDE_CENTER_Y = 40.0

# base x by “band” (defense → attack)
X_GK   =  8.0
X_DEF  = 22.0        # CB/LB/RB tier
X_WB   = 34.0        # wing-backs a bit higher than def
X_DMF  = 42.0
X_CMF  = 55.0
X_AMF  = 70.0
X_WF   = 78.0
X_IF   = 86.0        # inside forwards (LF/RF)
X_CF   =  95.0       # CF/SS
X_ST   = 102.0       # out-and-out striker
X_FALLBACK = 60.0

_side_re = re.compile(r'^(L|R)')

def infer_xy_for_token(token_raw: str) -> tuple[float, float]:
    """
    Rules:
      - Use the token EXACTLY as given (after uppercasing for parsing).
      - Side: startswith 'L' -> y=L, 'R' -> y=R, else y=C.
      - Band by suffix keywords (checked in priority order, without altering the token itself).
    """
    t = norm_token(token_raw)

    # side
    m = _side_re.match(t)
    if m:
        y = SIDE_LEFT_Y if m.group(1) == 'L' else SIDE_RIGHT_Y
    else:
        y = SIDE_CENTER_Y

    # bands (priority)
    # goalkeeper
    if t.endswith("GK"):
        return (X_GK, y)

    # very common defenders
    if t.endswith("CB") or t in {"RB","LB"}:
        return (X_DEF, y)

    # wing-back
    if t.endswith("WB"):
        return (X_WB, y)

    # strictly midfield families found in your data: DMF/CMF/AMF
    if t.endswith("DMF"):
        return (X_DMF, y)
    if t.endswith("CMF"):
        return (X_CMF, y)
    if t.endswith("AMF"):
        return (X_AMF, y)

    # wide forwards
    if t.endswith("WF"):
        return (X_WF, y)

    # inside forwards (LF/RF), second striker
    if t in {"LF","RF"} or t == "SS":
        return (X_IF, y)

    # centre-forward / striker
    if t in {"CF"}:
        return (X_CF, y)
    if t in {"ST"}:
        return (X_ST, y)

    # fallback: put unfamiliar tokens in a neat shelf just below touchline
    return (X_FALLBACK, 6.0)

# -------------------------
# Plotly overlay aligned to mplsoccer PNG
# -------------------------
def build_pitch_overlay(selected_norm: set[str],
                        tokens_norm: list[str],
                        display_map: dict,
                        height: int = 640) -> go.Figure:
    fig = go.Figure()
    fig.update_xaxes(range=[X_MIN, X_MAX], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[Y_MIN, Y_MAX], showgrid=False, zeroline=False,
                     visible=False, scaleanchor="x", scaleratio=1)

    # background
    if _MPLSOCCER:
        png = make_mplsoccer_horizontal_png()
        if png:
            img = Image.open(io.BytesIO(png))
            fig.add_layout_image(
                dict(
                    source=img,
                    xref="x", yref="y",
                    x=X_MIN, y=Y_MAX, sizex=X_MAX - X_MIN, sizey=Y_MAX - Y_MIN,
                    sizing="stretch", layer="below", opacity=1.0
                )
            )
    else:
        fig.add_shape(type="rect", x0=X_MIN, y0=Y_MIN, x1=X_MAX, y1=Y_MAX,
                      fillcolor="#3a5f39", line=dict(color="#3a5f39"))

    # points (ONLY tokens from df)
    xs, ys, texts, order_norm = [], [], [], []
    for tn in tokens_norm:
        x, y = infer_xy_for_token(display_map[tn])  # compute from the exact token string
        xs.append(x); ys.append(y)
        texts.append(display_map[tn])    # show exactly as in file
        order_norm.append(tn)

    colors = ["#2563EB" if tn in selected_norm else "#6B7280" for tn in order_norm]

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(size=22, line=dict(width=2, color="white"), color=colors, symbol="circle"),
            text=texts, textposition="middle center", textfont=dict(color="white", size=12),
            customdata=order_norm,
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="Positions", showlegend=False,
        )
    )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#3a5f39",
        paper_bgcolor="#3a5f39",
    )
    return fig

# -------------------------
# Selector UI
# -------------------------
def ensure_state(key: str):
    if key not in st.session_state:
        st.session_state[key] = set()

def toggle_selected(key: str, tok_norm: str):
    ensure_state(key)
    if tok_norm in st.session_state[key]:
        st.session_state[key].remove(tok_norm)
    else:
        st.session_state[key].add(tok_norm)

def pitch_selector_horizontal(label: str,
                              key_prefix: str,
                              tokens_norm: list[str],
                              display_map: dict) -> list[str]:
    st.markdown(f"**{label}**")
    sel_key = f"{key_prefix}_sel"
    ensure_state(sel_key)

    c1, c2, c3, _ = st.columns([1,1,1,6])
    with c1:
        if st.button("Select all", key=f"{key_prefix}_all"):
            st.session_state[sel_key] = set(tokens_norm)
    with c2:
        if st.button("Clear", key=f"{key_prefix}_clear"):
            st.session_state[sel_key] = set()
    with c3:
        if st.button("Invert", key=f"{key_prefix}_invert"):
            st.session_state[sel_key] = set(set(tokens_norm) - st.session_state[sel_key])

    if _PLOTLY_EVENTS:
        fig = build_pitch_overlay(st.session_state[sel_key], tokens_norm, display_map)
        clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                               override_height=640, override_width="100%")
        if clicks:
            tn = clicks[0].get("customdata")
            if tn:
                toggle_selected(sel_key, tn)
                st.rerun()
            else:
                idx = clicks[0].get("pointIndex")
                if idx is not None and 0 <= idx < len(tokens_norm):
                    toggle_selected(sel_key, tokens_norm[idx])
                    st.rerun()
        st.caption("Tip: Click a label on the pitch to toggle selection.")
    else:
        st.warning("Interactive clicks require `streamlit-plotly-events`. Using a compact button grid.")
        per_row = 11
        i = 0
        rows = (len(tokens_norm) + per_row - 1) // per_row
        for _ in range(rows):
            cols = st.columns(per_row, gap="small")
            for col in cols:
                if i >= len(tokens_norm): break
                tn = tokens_norm[i]; i += 1
                label_btn = display_map.get(tn, tn)
                sel = tn in st.session_state[sel_key]
                label_btn = f"● {label_btn}" if sel else label_btn
                if col.button(label_btn, key=f"{key_prefix}_{tn}"):
                    toggle_selected(sel_key, tn)

    return [display_map.get(tn, tn) for tn in sorted(st.session_state[sel_key])]

# -------------------------
# Load data
# -------------------------
@st.cache_data(show_spinner=False)
def load_excel_files(files) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_excel(f)
        df["League"] = getattr(f, "name", "Uploaded")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# -------------------------
# App
# -------------------------
st.title("⚽ PCA Analysis — Physical & Technical Metrics")
st.markdown("Upload your Excel file(s), select positions **exactly as in your dataset** on a horizontal mplsoccer pitch (16:9), then run a 2D PCA.")

with st.sidebar:
    st.header("1) Data upload")
    files = st.file_uploader(
        "Select Excel file(s)",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
        help="Must include at least 'Player', 'Position' and numeric columns."
    )
    if not files:
        st.info("Upload at least one file to continue.")
        st.stop()

    data = load_excel_files(files)

    # Position column
    cand_cols = [c for c in data.columns if c.lower() in ("pos", "position", "positions")]
    position_col = st.selectbox("Position column", options=cand_cols or ["Position"])
    if position_col not in data.columns:
        st.error(f"Column '{position_col}' not found.")
        st.stop()

    st.divider()
    st.header("2) Filters")

    minute_cols = [c for c in data.columns if "min" in c.lower() or "minutes" in c.lower()]
    if minute_cols:
        minute_col = st.selectbox("Minutes column", minute_cols, index=0)
        max_min = int(max(1, float(pd.to_numeric(data[minute_col], errors="coerce").fillna(0).max())))
        min_minutes = st.slider("Minimum minutes", 0, max_min, 0, step=50)
    else:
        minute_col, min_minutes = None, 0
        st.caption("No minutes column detected — minute filter disabled.")

    if "Age" in data.columns:
        ages = pd.to_numeric(data["Age"], errors="coerce")
        if ages.notna().any():
            a_min, a_max = int(ages.min()), int(ages.max())
            age_range = st.slider("Age range", a_min, a_max, (a_min, a_max))
        else:
            age_range = None
            st.caption("Age column is not numeric — age filter disabled.")
    else:
        age_range = None
        st.caption("No 'Age' column — age filter disabled.")

    st.divider()
    st.header("3) Plot options")
    color_by = st.selectbox("Color points by", options=[c for c in ["League", "Team", position_col] if c in data.columns] or ["League"])
    point_size = st.slider("Point size", 4, 16, 8)
    point_opacity = st.slider("Point opacity", 0.2, 1.0, 0.85)
    show_loadings = st.toggle("Show loadings (metric vectors)", value=True)
    loadings_scale = st.slider("Loadings length", 1.0, 6.0, 3.0, 0.5)

    st.divider()
    st.header("4) Export")
    export_format = st.radio("Format", ["HTML", "PNG (300 DPI)"], horizontal=True)
    export_btn = st.button("Export plot")

# Build tokens ONLY from your data
pos_series = data[position_col].dropna().astype(str)
display_map = build_display_map(pos_series)
tokens_norm = sorted(display_map.keys())
if not tokens_norm:
    st.error(f"No positions found in column '{position_col}'.")
    st.stop()

# Pitch selector
st.header("Select positions on the pitch")
selected_display = pitch_selector_horizontal(
    label="Click labels to (de)select. Only positions that exist in your data are shown.",
    key_prefix="pitch",
    tokens_norm=tokens_norm,
    display_map=display_map,
)

# Highlight players (optional)
st.subheader("Highlight players (optional)")
player_col = "Player" if "Player" in data.columns else None
if player_col:
    player_opts = sorted(pd.Series(data[player_col].astype(str)).unique().tolist())
    highlighted_players = st.multiselect("Select up to 5 players to label in the plot", options=player_opts, max_selections=5)
else:
    highlighted_players = []
    st.caption("No 'Player' column — highlight disabled.")

# Metrics
st.header("Metrics selection")
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found — please upload data with numeric metrics.")
    st.stop()
default_metrics = [c for c in numeric_cols if c.lower() not in {"age","height","weight","minutes","min","games"}] or numeric_cols
selected_metrics = st.multiselect("Pick at least two numeric columns for PCA", options=numeric_cols, default=default_metrics[: min(6, len(default_metrics))])
if len(selected_metrics) < 2:
    st.warning("Select at least two numeric columns to run PCA.")
    st.stop()

# Filtering
df = data.copy()
selected_norm = {norm_token(s) for s in selected_display}
if selected_norm:
    df = df[df[position_col].astype(str).apply(lambda s: any(norm_token(t) in selected_norm for t in tokens_from_cell(s)))]

if minute_col:
    df[minute_col] = pd.to_numeric(df[minute_col], errors="coerce")
    df = df[df[minute_col] >= min_minutes]

if age_range is not None and "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

df_numeric = df.dropna(subset=selected_metrics).copy()
if df_numeric.empty:
    st.warning("No rows left after filters. Try relaxing filters or choosing different metrics.")
    st.stop()

# PCA
X = df_numeric[selected_metrics].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df_numeric["PCA1"] = coords[:, 0]
df_numeric["PCA2"] = coords[:, 1]
exp1, exp2 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]

# PCA plot
st.header("PCA plot")
fig = go.Figure()
group_col = color_by if color_by in df_numeric.columns else "League"
groups = sorted(df_numeric[group_col].astype(str).unique().tolist())
palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
pal = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

def hover_text(row):
    parts = []
    if "Player" in row and pd.notna(row["Player"]): parts.append(f"<b>{row['Player']}</b>")
    if "Team" in row and pd.notna(row["Team"]):     parts.append(f"Club: {row['Team']}")
    if group_col != "League" and group_col in row and pd.notna(row[group_col]): parts.append(f"{group_col}: {row[group_col]}")
    if "Age" in row and pd.notna(row["Age"]):
        try: parts.append(f"Age: {int(row['Age'])}")
        except Exception: parts.append(f"Age: {row['Age']}")
    parts.append(f"PC1: {row['PCA1']:.2f}")
    parts.append(f"PC2: {row['PCA2']:.2f}")
    return "<br>".join(parts)

if player_col:
    df_numeric["_is_high"] = df_numeric[player_col].astype(str).isin(set(highlighted_players))
else:
    df_numeric["_is_high"] = False

for g in groups:
    dfg = df_numeric[df_numeric[group_col].astype(str) == g]
    normal = dfg[~dfg["_is_high"]]
    if not normal.empty:
        fig.add_trace(go.Scatter(
            x=normal["PCA1"], y=normal["PCA2"], mode="markers", name=str(g),
            marker=dict(size=point_size, opacity=point_opacity, color=pal[g]),
            text=[hover_text(r) for _, r in normal.iterrows()],
            hoverinfo="text", hovertemplate="%{text}<extra></extra>",
        ))
    high = dfg[dfg["_is_high"]]
    if not high.empty and player_col:
        for _, r in high.iterrows():
            fig.add_trace(go.Scatter(
                x=[r["PCA1"]], y=[r["PCA2"]], mode="markers+text", name=str(r[player_col]),
                marker=dict(size=point_size+4, opacity=1.0, color=pal[g], symbol="diamond", line=dict(width=2, color="black")),
                text=[str(r[player_col])], textposition="bottom center",
                hovertext=[hover_text(r)], hoverinfo="text", hovertemplate="%{hovertext}<extra></extra>",
                legendgroup="highlighted", showlegend=True,
            ))

if show_loadings:
    comps = pca.components_
    for i, metric in enumerate(selected_metrics):
        fig.add_trace(go.Scatter(
            x=[0, comps[0, i] * loadings_scale], y=[0, comps[1, i] * loadings_scale],
            mode="lines+text", line=dict(width=2), text=[None, metric],
            textposition="top center", showlegend=False, hoverinfo="skip",
        ))

fig.update_layout(
    title=f"PCA — PC1 ({exp1:.1%}) vs PC2 ({exp2:.1%})",
    xaxis_title="PC1", yaxis_title="PC2",
    template="plotly_white", height=820, margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# Export
with st.sidebar:
    export_btn = st.button("Export now") if 'export_btn' not in locals() else export_btn
    if export_btn:
        html = fig.to_html(full_html=True, include_plotlyjs="cdn")
        st.download_button("Download HTML", data=html, file_name="pca_analysis.html", mime="text/html")

with st.expander("Data preview"):
    meta_cols = [c for c in ["Player", position_col, "League", "Team", "Age"] if c in df_numeric.columns]
    st.dataframe(df_numeric[[*meta_cols, *selected_metrics, "PCA1", "PCA2"]].head(200), use_container_width=True, height=320)

st.caption("Only positions present in your file are shown on the pitch (labels exactly as in the dataset). Comma-separated tokens count individually.")
# app.py
# ------------------------------------------------------------
# PCA Analysis — Football Metrics
# Horizontal mplsoccer pitch (16:9, pads conforme template) + clickable overlay
# Positions used = ONLY tokens from your DataFrame (comma-separated)
# ------------------------------------------------------------
import io
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Click capture (recommended)
try:
    from streamlit_plotly_events import plotly_events
    _PLOTLY_EVENTS = True
except Exception:
    _PLOTLY_EVENTS = False

# Pretty pitch via mplsoccer (Matplotlib)
try:
    from mplsoccer import Pitch
    _MPLSOCCER = True
except Exception:
    _MPLSOCCER = False

# -------------------------
# Streamlit page setup
# -------------------------
st.set_page_config(
    page_title="PCA Analysis — Football Metrics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 0.8rem; }
      .stButton>button, .stDownloadButton>button { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Domain (StatsBomb horizontal): x ∈ [0..120], y ∈ [0..80]
# -------------------------
X_MIN, X_MAX = 0.0, 120.0
Y_MIN, Y_MAX = 0.0, 80.0

# =========================
# EXACTLY your template (horizontal 16:9)
# =========================
def make_mplsoccer_horizontal_png() -> bytes:
    """
    Draw a beautiful StatsBomb full horizontal pitch with 16:9 figure,
    pads computed exactly as in your snippet, and return PNG bytes.
    """
    if not _MPLSOCCER:
        return b""
    FIGWIDTH = 16
    FIGHEIGHT = 9
    NROWS = 1
    NCOLS = 1
    MAX_GRID = 1

    # here we setup the padding to get a 16:9 aspect ratio for the axis
    # note 80 is the StatsBomb width and 120 is the StatsBomb length
    # this will extend the (axis) grassy effect to the figure edges
    PAD_TOP = 2
    PAD_BOTTOM = 2
    PAD_SIDES = (((80 + PAD_BOTTOM + PAD_TOP) * FIGWIDTH / FIGHEIGHT) - 120) / 2

    pitch = Pitch(
        pad_top=PAD_TOP, pad_bottom=PAD_BOTTOM,
        pad_left=PAD_SIDES, pad_right=PAD_SIDES,
        pitch_color="grass", stripe=True, line_color="white",
        pitch_type="statsbomb", linewidth=2, goal_type="box"
    )

    # calculate the maximum grid_height/width
    GRID_WIDTH, GRID_HEIGHT = pitch.grid_dimensions(
        figwidth=FIGWIDTH, figheight=FIGHEIGHT,
        nrows=NROWS, ncols=NCOLS, max_grid=MAX_GRID, space=0
    )

    # plot
    fig, ax = pitch.grid(
        figheight=FIGHEIGHT,
        grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT,
        title_height=0, endnote_height=0
    )

    # Save to PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return buf.getvalue()

# =========================
# Position helpers
# =========================
def tokens_from_cell(cell) -> list[str]:
    """Split by comma and normalize for comparison (UPPER/strip)."""
    if not isinstance(cell, str):
        cell = str(cell)
    return [t.strip().upper() for t in cell.split(",") if t.strip()]

def build_display_map(series: pd.Series) -> dict:
    """Map normalized token -> original display text (preserve first occurrence)."""
    disp = {}
    for cell in series.dropna().astype(str):
        for tok in cell.split(","):
            t = tok.strip()
            if not t:
                continue
            n = t.upper()
            disp.setdefault(n, t)
    return disp

# Horizontal anchors (only to pick COORDS; we always DISPLAY the dataset text)
# Known roles are placed intuitively; unknown tokens are arranged along a neat shelf.
ANCHORS = {
    # GK & back line
    "GK":  (6, 40),
    "RB":  (25, 66), "RCB": (22, 48), "CB":  (22, 40), "LCB": (22, 32), "LB":  (25, 14),
    "RWB": (40, 70), "LWB": (40, 10),

    # Midfield
    "RDM": (40, 48), "DM":  (40, 40), "LDM": (40, 32),
    "RCM": (55, 52), "CM":  (55, 40), "LCM": (55, 28),
    "RAM": (68, 52), "AM":  (70, 40), "LAM": (72, 28),

    # Wingers / forwards
    "RW":  (78, 66), "LW":  (78, 14),
    "RF":  (88, 55), "CF":  (90, 40), "LF":  (88, 25),
    "SS":  (84, 40), "ST":  (102, 40),
}

COORD_ALIAS = {
    "CDM": "DM", "DMF": "DM",
    "CMF": "CM",
    "CAM": "AM", "AMF": "AM",
    "RM": "RW", "LM": "LW",
    "RWF": "RW", "LWF": "LW",
    "RST": "ST", "LST": "ST",
}

def coord_for_token_norm(tn: str):
    base = COORD_ALIAS.get(tn, tn)
    return ANCHORS.get(base)

def layout_unknowns(tokens_norm: list[str]) -> dict[str, tuple[float, float]]:
    """Lay out unknown labels along a tidy baseline near y≈6, spread in x."""
    if not tokens_norm:
        return {}
    xs = np.linspace(12, 108, len(tokens_norm)) if len(tokens_norm) > 1 else np.array([60.0])
    y = 6.0
    return {t: (float(xs[i]), y) for i, t in enumerate(tokens_norm)}

# =========================
# Plotly overlay builder (horizontal)
# =========================
def build_horizontal_pitch_overlay(selected_norm: set[str],
                                   tokens_norm_in_data: list[str],
                                   display_map: dict,
                                   height: int = 640) -> go.Figure:
    """
    Create a Plotly figure with axes x:[0..120], y:[0..80], put the mplsoccer PNG
    as background (stretched to the domain), and overlay clickable markers.
    """
    fig = go.Figure()
    fig.update_xaxes(range=[X_MIN, X_MAX], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[Y_MIN, Y_MAX], showgrid=False, zeroline=False,
                     visible=False, scaleanchor="x", scaleratio=1)

    # Background (PNG from EXACT template)
    if _MPLSOCCER:
        png = make_mplsoccer_horizontal_png()
        if png:
            img = Image.open(io.BytesIO(png))
            fig.add_layout_image(
                dict(
                    source=img,
                    xref="x", yref="y",
                    x=X_MIN, y=Y_MAX,              # top-left corner in data coords
                    sizex=X_MAX - X_MIN, sizey=Y_MAX - Y_MIN,
                    sizing="stretch",
                    layer="below",
                    opacity=1.0,
                )
            )
    else:
        # fallback rectangle in grass color
        fig.add_shape(type="rect", x0=X_MIN, y0=Y_MIN, x1=X_MAX, y1=Y_MAX,
                      fillcolor="#3a5f39", line=dict(color="#3a5f39"))

    # Coordinates for dataset tokens only
    unknown = []
    xs, ys, labels, order_norm = [], [], [], []
    for tn in tokens_norm_in_data:
        xy = coord_for_token_norm(tn)
        if xy is None:
            unknown.append(tn)
        else:
            x, y = xy
            xs.append(x); ys.append(y)
            labels.append(display_map.get(tn, tn))
            order_norm.append(tn)
    if unknown:
        placed = layout_unknowns(unknown)
        for tn in unknown:
            x, y = placed[tn]
            xs.append(x); ys.append(y)
            labels.append(display_map.get(tn, tn))
            order_norm.append(tn)

    colors = ["#2563EB" if tn in selected_norm else "#6B7280" for tn in order_norm]

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(size=22, line=dict(width=2, color="white"), color=colors, symbol="circle"),
            text=labels, textposition="middle center", textfont=dict(color="white", size=12),
            customdata=order_norm,
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="Positions", showlegend=False,
        )
    )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#3a5f39",
        paper_bgcolor="#3a5f39",
    )
    return fig

# =========================
# Selector UI
# =========================
def ensure_state(key: str):
    if key not in st.session_state:
        st.session_state[key] = set()

def toggle_selected(key: str, tok_norm: str):
    ensure_state(key)
    if tok_norm in st.session_state[key]:
        st.session_state[key].remove(tok_norm)
    else:
        st.session_state[key].add(tok_norm)

def pitch_selector_horizontal(label: str,
                              key_prefix: str,
                              tokens_norm_in_data: list[str],
                              display_map: dict) -> list[str]:
    """Horizontal selector: ONLY tokens from your dataset, clickable over the pitch."""
    st.markdown(f"**{label}**")
    sel_key = f"{key_prefix}_sel"
    ensure_state(sel_key)

    c1, c2, c3, _ = st.columns([1,1,1,6])
    with c1:
        if st.button("Select all", key=f"{key_prefix}_all"):
            st.session_state[sel_key] = set(tokens_norm_in_data)
    with c2:
        if st.button("Clear", key=f"{key_prefix}_clear"):
            st.session_state[sel_key] = set()
    with c3:
        if st.button("Invert", key=f"{key_prefix}_invert"):
            st.session_state[sel_key] = set(set(tokens_norm_in_data) - st.session_state[sel_key])

    if _PLOTLY_EVENTS:
        fig = build_horizontal_pitch_overlay(st.session_state[sel_key], tokens_norm_in_data, display_map)
        clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                               override_height=640, override_width="100%")
        if clicks:
            tok_norm = clicks[0].get("customdata")
            if tok_norm:
                toggle_selected(sel_key, tok_norm)
                st.rerun()
            else:
                idx = clicks[0].get("pointIndex")
                if idx is not None and 0 <= idx < len(tokens_norm_in_data):
                    toggle_selected(sel_key, tokens_norm_in_data[idx])
                    st.rerun()
        st.caption("Tip: Click a label on the pitch to toggle selection.")
    else:
        st.warning("Interactive clicks require `streamlit-plotly-events`. Using a compact button grid instead.")
        per_row = 11
        i = 0
        rows = (len(tokens_norm_in_data) + per_row - 1) // per_row
        for _ in range(rows):
            cols = st.columns(per_row, gap="small")
            for col in cols:
                if i >= len(tokens_norm_in_data): break
                tn = tokens_norm_in_data[i]; i += 1
                label_btn = display_map.get(tn, tn)
                sel = tn in st.session_state[sel_key]
                label_btn = f"● {label_btn}" if sel else label_btn
                if col.button(label_btn, key=f"{key_prefix}_{tn}"):
                    toggle_selected(sel_key, tn)

    # Return display labels (exactly as in data) for selected tokens
    return [display_map.get(tn, tn) for tn in sorted(st.session_state[sel_key])]

# =========================
# Data loading
# =========================
@st.cache_data(show_spinner=False)
def load_excel_files(files) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_excel(f)
        df["League"] = getattr(f, "name", "Uploaded")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# =========================
# App
# =========================
st.title("⚽ PCA Analysis — Football Metrics")
st.markdown(
    "Upload your Excel file(s), select positions **straight from your dataset** on a "
    "**horizontal** mplsoccer pitch (16:9 template), pick numeric metrics, and explore a **2D PCA**."
)

# Sidebar — upload & options
with st.sidebar:
    st.header("1) Data upload")
    files = st.file_uploader(
        "Select Excel file(s)",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
        help="Must include at least 'Player', 'Position' and numeric columns."
    )
    if not files:
        st.info("Upload at least one file to continue.")
        st.stop()

    data = load_excel_files(files)

    # Position column
    cand_cols = [c for c in data.columns if c.lower() in ("pos", "position", "positions")]
    position_col = st.selectbox("Position column", options=cand_cols or ["Position"])
    if position_col not in data.columns:
        st.error(f"Column '{position_col}' not found. Please select a valid column.")
        st.stop()

    st.divider()
    st.header("2) Filters")

    # Minutes
    minute_cols = [c for c in data.columns if "min" in c.lower() or "minutes" in c.lower()]
    if minute_cols:
        minute_col = st.selectbox("Minutes column", minute_cols, index=0)
        max_min = int(max(1, float(pd.to_numeric(data[minute_col], errors="coerce").fillna(0).max())))
        min_minutes = st.slider("Minimum minutes", 0, max_min, 0, step=50)
    else:
        minute_col, min_minutes = None, 0
        st.caption("No minutes column detected — minute filter disabled.")

    # Age
    if "Age" in data.columns:
        ages = pd.to_numeric(data["Age"], errors="coerce")
        if ages.notna().any():
            a_min, a_max = int(ages.min()), int(ages.max())
            age_range = st.slider("Age range", a_min, a_max, (a_min, a_max))
        else:
            age_range = None
            st.caption("Age column is not numeric — age filter disabled.")
    else:
        age_range = None
        st.caption("No 'Age' column — age filter disabled.")

    st.divider()
    st.header("3) Plot options")
    color_by = st.selectbox(
        "Color points by",
        options=[c for c in ["League", "Team", position_col] if c in data.columns] or ["League"]
    )
    point_size = st.slider("Point size", 4, 16, 8)
    point_opacity = st.slider("Point opacity", 0.2, 1.0, 0.85)
    show_loadings = st.toggle("Show loadings (metric vectors)", value=True)
    loadings_scale = st.slider("Loadings length", 1.0, 6.0, 3.0, 0.5)

    st.divider()
    st.header("4) Export")
    export_format = st.radio("Format", ["HTML", "PNG (300 DPI)"], horizontal=True)
    export_btn = st.button("Export plot")

# Build token universe ONLY from your file
pos_series = data[position_col].dropna().astype(str)
display_map = build_display_map(pos_series)     # normalized -> original display text
tokens_norm = sorted(display_map.keys())
if not tokens_norm:
    st.error(f"No positions found in column '{position_col}'. Please check your file.")
    st.stop()

# Pitch selector (horizontal, template-true)
st.header("Select positions on the pitch")
selected_display = pitch_selector_horizontal(
    label="Click labels to (de)select. Only positions that exist in your data are shown.",
    key_prefix="pitch",
    tokens_norm_in_data=tokens_norm,
    display_map=display_map,
)

# Highlight players (optional)
st.subheader("Highlight players (optional)")
player_col = "Player" if "Player" in data.columns else None
if player_col:
    player_opts = sorted(pd.Series(data[player_col].astype(str)).unique().tolist())
    highlighted_players = st.multiselect(
        "Select up to 5 players to label in the plot",
        options=player_opts, max_selections=5, placeholder="Type a name…",
    )
else:
    highlighted_players = []
    st.caption("No 'Player' column — highlight disabled.")

# Metrics
st.header("Metrics selection")
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found — please upload data with numeric metrics.")
    st.stop()
non_metric_hints = {"Age", "Height", "Weight", "Minutes", "Min", "Games"}
default_metrics = [c for c in numeric_cols if c not in non_metric_hints] or numeric_cols
selected_metrics = st.multiselect(
    "Pick at least two numeric columns for PCA",
    options=numeric_cols,
    default=default_metrics[: min(6, len(default_metrics))],
)
if len(selected_metrics) < 2:
    st.warning("Select at least two numeric columns to run PCA.")
    st.stop()

# Filtering (positions/minutes/age)
df = data.copy()

# Position filter (each comma-separated token counts individually)
selected_norm = {s.upper().strip() for s in selected_display}
if selected_norm and position_col in df.columns:
    df = df[df[position_col].astype(str).apply(
        lambda s: any(tok in selected_norm for tok in tokens_from_cell(s))
    )]

if minute_col:
    df[minute_col] = pd.to_numeric(df[minute_col], errors="coerce")
    df = df[df[minute_col] >= min_minutes]

if age_range is not None and "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

df_numeric = df.dropna(subset=selected_metrics).copy()
if df_numeric.empty:
    st.warning("No rows left after filters. Try relaxing filters or choosing different metrics.")
    st.stop()

# PCA
X = df_numeric[selected_metrics].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df_numeric["PCA1"] = coords[:, 0]
df_numeric["PCA2"] = coords[:, 1]
exp1, exp2 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]

# PCA plot
st.header("PCA plot")
fig = go.Figure()
group_col = color_by if color_by in df_numeric.columns else "League"
groups = sorted(df_numeric[group_col].astype(str).unique().tolist())

base_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]
palette = {g: base_colors[i % len(base_colors)] for i, g in enumerate(groups)}

def hover_text(row):
    parts = []
    if "Player" in row and pd.notna(row["Player"]):
        parts.append(f"<b>{row['Player']}</b>")
    if "Team" in row and pd.notna(row["Team"]):
        parts.append(f"Club: {row['Team']}")
    if group_col != "League" and group_col in row and pd.notna(row[group_col]):
        parts.append(f"{group_col}: {row[group_col]}")
    if "Age" in row and pd.notna(row["Age"]):
        try:
            parts.append(f"Age: {int(row['Age'])}")
        except Exception:
            parts.append(f"Age: {row['Age']}")
    parts.append(f"PC1: {row['PCA1']:.2f}")
    parts.append(f"PC2: {row['PCA2']:.2f}")
    return "<br>".join(parts)

# mark highlights
if player_col:
    df_numeric["_is_high"] = df_numeric[player_col].astype(str).isin(set(highlighted_players))
else:
    df_numeric["_is_high"] = False

for g in groups:
    dfg = df_numeric[df_numeric[group_col].astype(str) == g]
    normal = dfg[~dfg["_is_high"]]
    if not normal.empty:
        fig.add_trace(
            go.Scatter(
                x=normal["PCA1"], y=normal["PCA2"],
                mode="markers",
                name=str(g),
                marker=dict(size=point_size, opacity=point_opacity, color=palette[g]),
                text=[hover_text(r) for _, r in normal.iterrows()],
                hoverinfo="text",
                hovertemplate="%{text}<extra></extra>",
            )
        )
    high = dfg[dfg["_is_high"]]
    if not high.empty and player_col:
        for _, r in high.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[r["PCA1"]], y=[r["PCA2"]],
                    mode="markers+text",
                    name=str(r[player_col]),
                    marker=dict(size=point_size + 4, opacity=1.0,
                                color=palette[g], symbol="diamond", line=dict(width=2, color="black")),
                    text=[str(r[player_col])],
                    textposition="bottom center",
                    hovertext=[hover_text(r)],
                    hoverinfo="text",
                    hovertemplate="%{hovertext}<extra></extra>",
                    legendgroup="highlighted",
                    showlegend=True,
                )
            )

if show_loadings:
    comps = pca.components_
    for i, metric in enumerate(selected_metrics):
        fig.add_trace(
            go.Scatter(
                x=[0, comps[0, i] * loadings_scale],
                y=[0, comps[1, i] * loadings_scale],
                mode="lines+text",
                line=dict(width=2),
                text=[None, metric],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

fig.update_layout(
    title=f"PCA — PC1 ({exp1:.1%}) vs PC2 ({exp2:.1%})",
    xaxis_title="PC1",
    yaxis_title="PC2",
    template="plotly_white",
    height=820,
    margin=dict(l=20, r=20, t=60, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# Export
with st.sidebar:
    if export_btn:
        if export_format.startswith("HTML"):
            html = fig.to_html(full_html=True, include_plotlyjs="cdn")
            st.download_button("Download HTML", data=html, file_name="pca_analysis.html", mime="text/html")
        else:
            try:
                img = fig.to_image(format="png", width=1400, height=900, scale=3)  # requires kaleido
                st.download_button("Download PNG", data=img, file_name="pca_analysis.png", mime="image/png")
            except Exception:
                st.warning("PNG export requires the 'kaleido' package. Install with: `pip install kaleido`.")

# Data preview
with st.expander("Data preview"):
    meta_cols = [c for c in ["Player", position_col, "League", "Team", "Age"] if c in df_numeric.columns]
    st.dataframe(
        df_numeric[[*meta_cols, *selected_metrics, "PCA1", "PCA2"]].head(200),
        use_container_width=True,
        height=320,
    )

st.caption(
    "Only positions present in your file are shown on the pitch (labels displayed exactly as in the dataset). "
    "Comma-separated tokens count individually for filtering."
)



