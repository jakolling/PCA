# app.py
# ------------------------------------------------------------
# PCA Analysis — Football Metrics
# Half-vertical mplsoccer pitch (true vertical) + precise clickable overlay
# Positions used = ONLY tokens from your DataFrame (comma-separated)
# ------------------------------------------------------------
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Click capture on Plotly
try:
    from streamlit_plotly_events import plotly_events
    _PLOTLY_EVENTS = True
except Exception:
    _PLOTLY_EVENTS = False

# Pretty pitch via mplsoccer
try:
    from mplsoccer import Pitch
    _MPLSOCCER = True
except Exception:
    _MPLSOCCER = False

# ===============================
# Streamlit page setup & minimal CSS
# ===============================
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

# ============================================================
# TRUE Half-Vertical geometry & anchors
# We adopt a vertical half: width = 80 (x: 0..80), length half = 60 (y: 0..60)
# Goal line at y=60 (top), halfway at y=0 (bottom).
# ============================================================
X_MIN, X_MAX = 0.0, 80.0
Y_MIN, Y_MAX = 0.0, 60.0   # vertical half

# Carefully tuned anchors for an ATTACKING HALF (vertical):
# - Defenders near y≈8–12
# - Midfield y≈20–38
# - AM line y≈44–50
# - Forwards y≈54–58
ANCHORS = {
    # last line (defensive side of this half)
    "GK":  (40, 4),

    # back line
    "RB":  (16, 10),
    "RCB": (28, 12),
    "CB":  (40, 12),
    "LCB": (52, 12),
    "LB":  (64, 10),

    # wing-backs slightly higher/wider
    "RWB": (12, 16),
    "LWB": (68, 16),

    # midfield
    "RDM": (30, 22),
    "DM":  (40, 22),
    "LDM": (50, 22),

    "RCM": (30, 30),
    "CM":  (40, 30),
    "LCM": (50, 30),

    "RAM": (32, 38),
    "AM":  (40, 38),
    "LAM": (48, 38),

    # wingers / forwards
    "RW":  (16, 46),
    "LW":  (64, 46),

    "RF":  (32, 52),
    "CF":  (40, 54),
    "LF":  (48, 52),
    "SS":  (40, 48),
    "ST":  (40, 58),
}

# Coordinate aliases (ONLY to choose coordinates; display keeps dataset text)
COORD_ALIAS = {
    "CDM": "DM", "DMF": "DM",
    "CMF": "CM",
    "CAM": "AM", "AMF": "AM",
    "RM": "RW", "LM": "LW",
    "RWF": "RW", "LWF": "LW",
    "RST": "ST", "LST": "ST",
    # Common shorthands
    "RCB": "RCB", "LCB": "LCB",
    "RDM": "RDM", "LDM": "LDM",
    "RWB": "RWB", "LWB": "LWB",
}

# ===============================
# Helpers
# ===============================
def ensure_state(key: str):
    if key not in st.session_state:
        st.session_state[key] = set()

def toggle_selected(key: str, tok_norm: str):
    ensure_state(key)
    if tok_norm in st.session_state[key]:
        st.session_state[key].remove(tok_norm)
    else:
        st.session_state[key].add(tok_norm)

def tokens_from_cell(cell) -> list[str]:
    """Split by comma and normalize for comparison (UPPER/strip)."""
    if not isinstance(cell, str):
        cell = str(cell)
    return [t.strip().upper() for t in cell.split(",") if t.strip()]

def build_display_map(series: pd.Series) -> dict:
    """
    Map normalized token -> original display text (preserve first occurrence).
    Only tokens present in the dataset appear here.
    """
    disp = {}
    for cell in series.dropna().astype(str):
        for tok in cell.split(","):
            t = tok.strip()
            if not t:
                continue
            n = t.upper()
            if n not in disp:
                disp[n] = t
    return disp

@st.cache_data(show_spinner=False)
def draw_mplsoccer_half_vertical_png(
    pitch_color="#1e3a1a",  # deep but vibrant grass
    line_color="#f4f4f4",
    stripe=True,
    dpi=300,
) -> bytes:
    """
    Render a VERTICAL HALF pitch using mplsoccer and return PNG bytes.
    We explicitly set the figure ratio to match 80:60 (width:height).
    """
    if not _MPLSOCCER:
        return b""
    # Aspect ratio: width:height = 80:60 => 4:3. Choose a nice vertical canvas.
    figsize = (6, 9)  # wider than tall in inches is fine; mplsoccer handles coords
    pitch = Pitch(
        pitch_type="statsbomb",
        half=True,                # <-- ensures vertical half
        pitch_color=pitch_color,
        line_color=line_color,
        stripe=stripe,
        linewidth=2,
        goal_type="box",
        # tight layout handled by savefig below
    )
    buf = io.BytesIO()
    fig, ax = pitch.draw(figsize=figsize)
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return buf.getvalue()

def coord_for_token_norm(tok_norm: str) -> tuple[float, float] | None:
    """
    Choose coordinates for a token (normalized) using anchors; unknown -> None (laid out later).
    """
    base = COORD_ALIAS.get(tok_norm, tok_norm)
    return ANCHORS.get(base)

def layout_unknowns(tokens_norm: list[str]) -> dict[str, tuple[float, float]]:
    """
    Place unknown tokens along a neat baseline near the halfway line (y≈6–10),
    spread horizontally to stay readable on a VERTICAL half pitch.
    """
    if not tokens_norm:
        return {}
    n = len(tokens_norm)
    xs = np.linspace(10, 70, n) if n > 1 else np.array([40.0])
    ys = np.linspace(6, 10, n) if n > 2 else np.array([8.0] * n)
    return {tokens_norm[i]: (float(xs[i]), float(ys[i])) for i in range(n)}

def build_pitch_with_overlay(selected_norm: set[str],
                             tokens_norm_in_data: list[str],
                             display_map: dict,
                             height: int = 640) -> go.Figure:
    """
    Build a Plotly figure aligned to the mplsoccer HALF-VERTICAL pitch:
    x: 0..80, y: 0..60 (goal line at y=60). The mplsoccer PNG is the background.
    We overlay clickable markers at perfectly matched data coordinates.
    """
    # Base figure & axes
    fig = go.Figure()
    fig.update_xaxes(range=[X_MIN, X_MAX], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[Y_MIN, Y_MAX], showgrid=False, zeroline=False, visible=False,
                     scaleanchor="x", scaleratio=1)

    # Background image (mplsoccer)
    if _MPLSOCCER:
        png = draw_mplsoccer_half_vertical_png()
        if png:
            img = Image.open(io.BytesIO(png))
            fig.add_layout_image(
                dict(
                    source=img,
                    xref="x", yref="y",
                    x=X_MIN, y=Y_MAX,                # top-left corner in data coords
                    sizex=X_MAX - X_MIN, sizey=Y_MAX - Y_MIN,
                    sizing="stretch",
                    layer="below",
                    opacity=1.0,
                )
            )
    else:
        # graceful fallback: colored rectangle
        fig.add_shape(type="rect", x0=X_MIN, y0=Y_MIN, x1=X_MAX, y1=Y_MAX,
                      fillcolor="#1e3a1a", line=dict(color="#1e3a1a"))

    # Build coordinates for dataset tokens (ONLY tokens from data)
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

    # Lay out unknowns
    if unknown:
        unk_map = layout_unknowns(unknown)
        for tn in unknown:
            x, y = unk_map[tn]
            xs.append(x); ys.append(y)
            labels.append(display_map.get(tn, tn))
            order_norm.append(tn)

    # Colors (selected vs not)
    colors = ["#2563EB" if tn in selected_norm else "#6B7280" for tn in order_norm]

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(size=22, line=dict(width=2, color="white"), color=colors, symbol="circle"),
            text=labels, textposition="middle center", textfont=dict(color="white", size=12),
            customdata=order_norm,
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False,
            name="Positions",
        )
    )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#1e3a1a",
        paper_bgcolor="#1e3a1a",
    )
    return fig

def pitch_selector_half_vertical(label: str,
                                 key_prefix: str,
                                 tokens_norm_in_data: list[str],
                                 display_map: dict) -> list[str]:
    """
    HALF-VERTICAL selector: shows ONLY tokens from your dataset (no extras).
    Display text = exactly as in your file; selection is stored in normalized form.
    """
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
        fig = build_pitch_with_overlay(st.session_state[sel_key], tokens_norm_in_data, display_map)
        clicks = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                               override_height=640, override_width="100%")
        if clicks:
            # Prefer robust customdata
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

@st.cache_data(show_spinner=False)
def load_excel_files(files) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_excel(f)
        df["League"] = getattr(f, "name", "Uploaded")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# ===============================
# App
# ===============================
st.title("⚽ PCA Analysis — Physical & Technical Metrics")
st.markdown(
    "Upload your Excel file(s), select positions **straight from your dataset** on a "
    "**true vertical half-pitch** (mplsoccer), pick numeric metrics, and explore a **2D PCA**."
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

# Build dataset-driven token universe (ONLY what you have)
pos_series = data[position_col].dropna().astype(str)
display_map = build_display_map(pos_series)     # normalized -> original display text
tokens_norm = sorted(display_map.keys())
if not tokens_norm:
    st.error(f"No positions found in column '{position_col}'. Please check your file.")
    st.stop()

# Pitch selector (TRUE vertical half)
st.header("Select positions on the pitch")
selected_display = pitch_selector_half_vertical(
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
    "Only positions present in your file are shown. Half-pitch is vertical (y: 0→60, goal line at the top). "
    "Comma-separated tokens count individually for filtering."
)
# app.py
# ------------------------------------------------------------
# PCA Analysis — Football Metrics
# Beautiful mplsoccer-styled "half vertical pitch" + clickable selector
# Positions are derived from the uploaded dataset (comma-separated tokens)
# ------------------------------------------------------------
import io
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Optional: clicks on Plotly
try:
    from streamlit_plotly_events import plotly_events
    _PLOTLY_EVENTS = True
except Exception:
    _PLOTLY_EVENTS = False

# Pretty pitch background
try:
    from mplsoccer import Pitch
    _MPLSOCCER = True
except Exception:
    _MPLSOCCER = False

# -------------------------
# Page config & light theming
# -------------------------
st.set_page_config(
    page_title="PCA Analysis — Football Metrics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://docs.streamlit.io/",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "Explore football data with a clean UI, mplsoccer-styled half-vertical pitch selector, and 2D PCA."
    },
)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1.0rem; }
      .stButton>button, .stDownloadButton>button { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Geometry — Half vertical pitch coordinates (StatsBomb units)
# mplsoccer half-vertical uses x in [0..80] (width), y in [60..120] (half length)
# We'll place standard roles intuitively in the attacking half.
# ============================================================
Y_MIN, Y_MAX = 60, 120  # half vertical
X_MIN, X_MAX = 0, 80

# Hand-crafted layout for common roles (beautiful & readable)
POS_COORDS_HALF_VERTICAL = {
    # Keeper just below the halfway area (so it still shows up)
    "GK":  (40, 62),

    # Defensive line (closer to halfway)
    "RB":  (16, 72),
    "RCB": (28, 76),
    "CB":  (40, 78),
    "LCB": (52, 76),
    "LB":  (64, 72),

    # Wing-backs slightly higher/wider
    "RWB": (12, 78),
    "LWB": (68, 78),

    # Midfield block
    "RDM": (30, 88),
    "DM":  (40, 88),
    "LDM": (50, 88),

    "RCM": (30, 96),
    "CM":  (40, 96),
    "LCM": (50, 96),

    "RAM": (32, 104),
    "AM":  (40, 104),
    "LAM": (48, 104),

    # Wingers / forwards
    "RW":  (16, 110),
    "LW":  (64, 110),
    "RF":  (32, 112),
    "CF":  (40, 114),
    "LF":  (48, 112),
    "SS":  (40, 110),  # second striker
    "ST":  (40, 118),
}

# Alias map to gently normalize less-common labels into our anchor set
POS_ALIASES = {
    "RCB": "RCB", "LCB": "LCB",
    "RDM": "RDM", "LDM": "LDM",
    "RM": "RW", "LM": "LW",
    "CAM": "AM", "CDM": "DM", "CMF": "CM", "DMF": "DM", "AMF": "AM",
    "RWB": "RWB", "LWB": "LWB",
    "RWF": "RW", "LWF": "LW",
    "RST": "ST", "LST": "ST", "CF": "CF",
    "SS": "SS",
}

# -------------------------
# Helpers
# -------------------------
def ensure_session_set(key: str):
    if key not in st.session_state:
        st.session_state[key] = set()

def toggle_selection(sel_key: str, pos: str):
    ensure_session_set(sel_key)
    pos = pos.upper().strip()
    if pos in st.session_state[sel_key]:
        st.session_state[sel_key].remove(pos)
    else:
        st.session_state[sel_key].add(pos)

def normalize_positions_cell(cell) -> list[str]:
    """Split comma-separated positions and uppercase/trim each token."""
    if not isinstance(cell, str):
        cell = str(cell)
    tokens = [t.strip().upper() for t in cell.split(",") if t.strip()]
    return tokens

def canonical_label(label: str) -> str:
    """Map label via aliases when available; else return original."""
    l = label.upper().strip()
    return POS_ALIASES.get(l, l)

def place_unknown_positions(labels: list[str]) -> dict[str, tuple[float, float]]:
    """
    Place any unknown labels along a neat baseline strip at y ≈ 66..74,
    spreading horizontally. Returns {label: (x, y)} for those unknowns.
    """
    unknowns = [l for l in labels if canonical_label(l) not in POS_COORDS_HALF_VERTICAL]
    coords = {}
    if not unknowns:
        return coords
    # Spread across [10..70] to avoid edges
    n = len(unknowns)
    xs = np.linspace(12, 68, n) if n > 1 else np.array([40.0])
    y = 66.0
    for i, lab in enumerate(unknowns):
        coords[lab] = (float(xs[i]), y)
    return coords

@st.cache_data(show_spinner=False)
def make_mplsoccer_half_vertical_png(
    pitch_color="#22311d",
    line_color="#f5f5f5",
    figsize=(6, 10),   # vertical
    dpi=300,
    stripe=True,
) -> bytes:
    """
    Render a vertical half-pitch (StatsBomb) with a pretty mplsoccer style.
    """
    if not _MPLSOCCER:
        return b""
    pitch = Pitch(
        pitch_type="statsbomb",
        half=True,               # <— half vertical
        pitch_color=pitch_color,
        line_color=line_color,
        stripe=stripe,
        linewidth=2,
        goal_type="box",
    )
    fig, ax = pitch.draw(figsize=figsize)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return buf.getvalue()

def draw_half_vertical_plotly_bg(selected: set[str],
                                 available_labels: list[str],
                                 height: int = 600) -> go.Figure:
    """
    Build a Plotly figure in half-vertical coordinates (x: 0..80, y: 60..120)
    with an mplsoccer PNG background and clickable position markers.
    Only labels present in the dataset are shown by default.
    """
    fig = go.Figure()
    fig.update_xaxes(range=[X_MIN, X_MAX], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[Y_MIN, Y_MAX], showgrid=False, zeroline=False,
                     visible=False, scaleanchor="x", scaleratio=1)

    # Background
    if _MPLSOCCER:
        png_bytes = make_mplsoccer_half_vertical_png()
        if png_bytes:
            img = Image.open(io.BytesIO(png_bytes))
            fig.add_layout_image(
                dict(
                    source=img,
                    xref="x", yref="y",
                    x=X_MIN, y=Y_MAX,
                    sizex=X_MAX - X_MIN, sizey=Y_MAX - Y_MIN,
                    sizing="stretch",
                    layer="below",
                    opacity=1.0,
                )
            )
    else:
        # graceful fallback
        fig.add_shape(type="rect", x0=X_MIN, y0=Y_MIN, x1=X_MAX, y1=Y_MAX,
                      fillcolor="#22311d", line=dict(color="#22311d"))

    # Build coordinate map for labels
    unknown_map = place_unknown_positions(available_labels)
    def pos_xy(label: str) -> tuple[float, float]:
        c = canonical_label(label)
        if c in POS_COORDS_HALF_VERTICAL:
            return POS_COORDS_HALF_VERTICAL[c]
        return unknown_map.get(label, (40.0, 66.0))

    # Plot markers for all unique labels from data
    xs, ys, texts = [], [], []
    for lab in available_labels:
        x, y = pos_xy(lab)
        xs.append(x); ys.append(y); texts.append(lab)

    # Color by selection
    colors = ["#2563EB" if lab in selected else "#6B7280" for lab in texts]
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=22, line=dict(width=2, color="white"), color=colors, symbol="circle"),
            text=texts, textposition="middle center", textfont=dict(color="white", size=12),
            customdata=texts,
            hovertemplate="<b>%{customdata}</b><extra></extra>",
            name="Positions", showlegend=False,
        )
    )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#22311d",
        paper_bgcolor="#22311d",
    )
    return fig

def position_selector_half_pitch(label: str,
                                 key_prefix: str,
                                 dataset_labels: list[str]) -> list[str]:
    """
    Half-vertical pitch selector with mplsoccer background + Plotly clickable markers.
    dataset_labels: unique tokens from the dataset (already uppercased).
    """
    st.markdown(f"**{label}**")
    sel_key = f"{key_prefix}_selected"
    ensure_session_set(sel_key)

    # Initialize: preselect nothing by default
    if not st.session_state[sel_key]:
        st.session_state[sel_key] = set()

    # Controls
    c1, c2, c3, c4 = st.columns([1,1,1,5])
    with c1:
        if st.button("Select all", key=f"{key_prefix}_all"):
            st.session_state[sel_key] = set(dataset_labels)
    with c2:
        if st.button("Clear", key=f"{key_prefix}_clear"):
            st.session_state[sel_key] = set()
    with c3:
        if st.button("Invert", key=f"{key_prefix}_invert"):
            st.session_state[sel_key] = set(set(dataset_labels) - st.session_state[sel_key])

    # Interactive canvas
    if _PLOTLY_EVENTS:
        fig = draw_half_vertical_plotly_bg(st.session_state[sel_key], dataset_labels)
        clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                                override_height=600, override_width="100%")
        if clicked:
            idx = clicked[0].get("pointIndex", None)
            if idx is not None and 0 <= idx < len(dataset_labels):
                pos = dataset_labels[idx]
                toggle_selection(sel_key, pos)
                st.rerun()
        st.caption(
            "Tip: Click a label on the pitch to toggle selection. "
            "Unknown/rare roles are arranged along the baseline for clarity."
        )
    else:
        st.warning("Interactive clicks require `streamlit-plotly-events`. A button grid is shown below.")
        # Fallback chip grid
        per_row = 10
        i = 0
        rows = (len(dataset_labels) + per_row - 1) // per_row
        for _ in range(rows):
            cols = st.columns(per_row, gap="small")
            for col in cols:
                if i >= len(dataset_labels): break
                opt = dataset_labels[i]; i += 1
                sel = opt in st.session_state[sel_key]
                label_btn = f"● {opt}" if sel else opt
                if col.button(label_btn, key=f"{key_prefix}_{opt}"):
                    toggle_selection(sel_key, opt)

    return sorted(list(st.session_state[sel_key]))

@st.cache_data(show_spinner=False)
def load_excel_files(files) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_excel(f)
        df["League"] = getattr(f, "name", "Uploaded")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def infer_position_tokens(df: pd.DataFrame, pos_col: str = "Position") -> list[str]:
    """Extract unique comma-separated position tokens from df[pos_col], uppercase & sorted."""
    if pos_col not in df.columns:
        return []
    s = df[pos_col].dropna().astype(str)
    tokens = set()
    for cell in s:
        for tok in normalize_positions_cell(cell):
            tokens.add(tok)
    return sorted(tokens)

# -------------------------
# App header
# -------------------------
st.title("⚽ PCA Analysis — Physical & Technical Metrics")
st.markdown(
    "Upload your Excel file(s), select positions **on a beautiful mplsoccer half-vertical pitch**, "
    "pick numeric metrics, and explore a **2D PCA** plot with loadings."
)

# -------------------------
# Sidebar (Upload & Filters)
# -------------------------
with st.sidebar:
    st.header("1) Data upload")
    uploaded_files = st.file_uploader(
        "Select one or more Excel files",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
        help="Your file should include at least 'Player', 'Position', and numeric columns."
    )
    if not uploaded_files:
        st.info("Upload at least one file to continue.")
        st.stop()

    data = load_excel_files(uploaded_files)

    # Position column choice (in case dataset uses another header)
    cand_cols = [c for c in data.columns if c.lower() in ("pos", "position", "positions")]
    position_column = st.selectbox("Position column", options=cand_cols or ["Position"])
    if position_column not in data.columns:
        st.error(f"Column '{position_column}' not found. Please select a valid column.")
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
        options=[c for c in ["League", "Team", position_column] if c in data.columns] or ["League"],
        help="Color grouping used in the PCA scatter."
    )
    point_size = st.slider("Point size", 4, 16, 8)
    point_opacity = st.slider("Point opacity", 0.2, 1.0, 0.85)
    show_loadings = st.toggle("Show loadings (metric vectors)", value=True)
    loadings_scale = st.slider("Loadings length", 1.0, 6.0, 3.0, 0.5)

    st.divider()
    st.header("4) Export")
    export_format = st.radio("Format", ["HTML", "PNG (300 DPI)"], horizontal=True)
    export_btn = st.button("Export plot")

# -------------------------
# Build dataset-driven position list
# -------------------------
dataset_positions = infer_position_tokens(data, pos_col=position_column)
if not dataset_positions:
    st.warning(f"No positions found in column '{position_column}'. The pitch will show common anchors only.")
    # Fallback to a basic set so UI stays usable
    dataset_positions = sorted(set(list(POS_COORDS_HALF_VERTICAL.keys())))

# -------------------------
# Pitch selector (half vertical, dataset-driven)
# -------------------------
st.header("Select positions on the pitch")
selected_positions = position_selector_half_pitch(
    label="Click labels on the pitch to (de)select. Each comma-separated position in your data counts individually.",
    key_prefix="pitchpos",
    dataset_labels=dataset_positions,
)

# -------------------------
# Highlight players (optional)
# -------------------------
st.subheader("Highlight players (optional)")
player_col = "Player" if "Player" in data.columns else None
if player_col:
    player_options = sorted(pd.Series(data[player_col].astype(str)).unique().tolist())
    highlighted_players = st.multiselect(
        "Select up to 5 players to label in the plot",
        options=player_options,
        max_selections=5,
        placeholder="Type a name…",
    )
else:
    highlighted_players = []
    st.caption("No 'Player' column — highlight disabled.")

# -------------------------
# Metrics selection
# -------------------------
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
    help="These variables will determine the PCA components."
)
if len(selected_metrics) < 2:
    st.warning("Select at least two numeric columns to run PCA.")
    st.stop()

# -------------------------
# Data filtering
# -------------------------
df = data.copy()

# Position filter — treat comma-separated tokens individually; case-insensitive
if selected_positions and position_column in df.columns:
    selected_upper = {p.upper().strip() for p in selected_positions}
    df = df[df[position_column].astype(str).apply(
        lambda s: any(tok in selected_upper for tok in normalize_positions_cell(s))
    )]

# Minutes filter
if minute_col:
    df[minute_col] = pd.to_numeric(df[minute_col], errors="coerce")
    df = df[df[minute_col] >= min_minutes]

# Age filter
if age_range is not None and "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

# Keep rows with all selected metrics
df_numeric = df.dropna(subset=selected_metrics).copy()
if df_numeric.empty:
    st.warning("No rows left after filters. Try relaxing filters or choosing different metrics.")
    st.stop()

# -------------------------
# PCA
# -------------------------
X = df_numeric[selected_metrics].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
df_numeric["PCA1"] = coords[:, 0]
df_numeric["PCA2"] = coords[:, 1]
exp1, exp2 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]

# -------------------------
# Plot (Plotly)
# -------------------------
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
    if position_column in row and pd.notna(row[position_column]):
        parts.append(f"Position: {row[position_column]}")
    if "Age" in row and pd.notna(row["Age"]):
        try:
            parts.append(f"Age: {int(row['Age'])}")
        except Exception:
            parts.append(f"Age: {row['Age']}")
    parts.append(f"PC1: {row['PCA1']:.2f}")
    parts.append(f"PC2: {row['PCA2']:.2f}")
    return "<br>".join(parts)

# Highlight flags
if player_col:
    df_numeric["_is_high"] = df_numeric[player_col].astype(str).isin(set(highlighted_players))
else:
    df_numeric["_is_high"] = False

# Draw groups
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
                    marker=dict(
                        size=point_size + 4, opacity=1.0, color=palette[g],
                        symbol="diamond", line=dict(width=2, color="black")
                    ),
                    text=[str(r[player_col])],
                    textposition="bottom center",
                    hovertext=[hover_text(r)],
                    hoverinfo="text",
                    hovertemplate="%{hovertext}<extra></extra>",
                    legendgroup="highlighted",
                    showlegend=True,
                )
            )

# Loadings
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
st.success("PCA computed and plotted successfully.")

# -------------------------
# Export
# -------------------------
if export_btn:
    if export_format.startswith("HTML"):
        html = fig.to_html(full_html=True, include_plotlyjs="cdn")
        st.download_button("Download HTML", data=html, file_name="pca_analysis.html", mime="text/html")
    else:
        try:
            img_bytes = fig.to_image(format="png", width=1400, height=900, scale=3)  # needs kaleido
            st.download_button("Download PNG", data=img_bytes, file_name="pca_analysis.png", mime="image/png")
        except Exception:
            st.warning("PNG export requires the 'kaleido' package. Install with: `pip install kaleido`.")

# -------------------------
# Data preview
# -------------------------
with st.expander("Data preview"):
    meta_cols = [c for c in ["Player", position_column, "League", "Team", "Age"] if c in df_numeric.columns]
    st.dataframe(
        df_numeric[[*meta_cols, *selected_metrics, "PCA1", "PCA2"]].head(200),
        use_container_width=True,
        height=320,
    )

st.caption(
    "Positions are matched individually when separated by commas (e.g., 'CB, RB' matches both CB and RB). "
    "The pitch uses mplsoccer half-vertical styling; clicks are handled by a transparent Plotly layer."
)

