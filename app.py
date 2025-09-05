# app.py
# ------------------------------------------------------------
# PCA Analysis — Football Metrics (with football-pitch selector)
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Optional click support on Plotly (recommended)
try:
    from streamlit_plotly_events import plotly_events
    _PLOTLY_EVENTS = True
except Exception:
    _PLOTLY_EVENTS = False

# -------------------------
# Page config & top styling
# -------------------------
st.set_page_config(
    page_title="PCA Analysis — Football Metrics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://docs.streamlit.io/",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "Explore football data with a clean UI, pitch position selector, and 2D PCA."
    },
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stButton>button {border-radius: 999px;}
    .stDownloadButton>button {border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Pitch position coordinates
# (120 x 80 reference space)
# -------------------------
POSITION_POINTS = {
    # GK
    "GK":  (6, 40),

    # Back line
    "RB":  (25, 66),
    "RCB": (22, 48),
    "CB":  (22, 40),
    "LCB": (22, 32),
    "LB":  (25, 14),

    # Wing-backs
    "RWB": (40, 70),
    "LWB": (40, 10),

    # Midfield
    "DM":  (40, 40),
    "RCM": (55, 52),
    "CM":  (55, 40),
    "LCM": (55, 28),
    "AM":  (70, 40),

    # Wingers / forwards
    "RW":  (75, 68),
    "LW":  (75, 12),
    "RF":  (85, 55),
    "CF":  (88, 40),
    "LF":  (85, 25),
    "ST":  (95, 40),
}
DEFAULT_POSITION_ORDER = [
    "GK",
    "RB", "RCB", "CB", "LCB", "LB",
    "RWB", "LWB",
    "DM", "RCM", "CM", "LCM", "AM",
    "RW", "LW", "RF", "CF", "LF", "ST",
]

# -------------------------
# Utils
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

def draw_pitch_plotly(selected: set[str], height: int = 560) -> go.Figure:
    """Plotly pitch with clickable markers for positions."""
    L, W = 120, 80
    fig = go.Figure()

    # Outer pitch & features
    fig.add_shape(type="rect", x0=0, y0=0, x1=L, y1=W, line=dict(width=2))
    fig.add_shape(type="line", x0=L/2, y0=0, x1=L/2, y1=W, line=dict(width=1))

    theta = np.linspace(0, 2*np.pi, 200)
    cx, cy, r = L/2, W/2, 10
    fig.add_trace(go.Scatter(x=cx + r*np.cos(theta), y=cy + r*np.sin(theta),
                             mode="lines", line=dict(width=1), hoverinfo="skip", showlegend=False))
    # Boxes
    fig.add_shape(type="rect", x0=0,   y0=18, x1=18,  y1=62, line=dict(width=1))
    fig.add_shape(type="rect", x0=0,   y0=30, x1=6,   y1=50, line=dict(width=1))
    fig.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(width=1))
    fig.add_shape(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(width=1))
    # Spots
    fig.add_trace(go.Scatter(x=[12, 108, 60], y=[40, 40, 40], mode="markers",
                             marker=dict(size=4), hoverinfo="skip", showlegend=False))

    # Positions (markers+labels)
    xs, ys, labels = [], [], []
    for pos in DEFAULT_POSITION_ORDER:
        x, y = POSITION_POINTS[pos]
        xs.append(x); ys.append(y); labels.append(pos)

    colors = ["#2563EB" if pos in selected else "#6B7280" for pos in labels]
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=20, line=dict(width=2, color="black"), color=colors, symbol="circle"),
            text=labels, textposition="middle center", textfont=dict(color="white", size=12),
            customdata=labels,
            hovertemplate="<b>%{customdata}</b><extra></extra>",
            name="Positions", showlegend=False,
        )
    )

    fig.update_xaxes(range=[-2, L+2], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-2, W+2], showgrid=False, zeroline=False, visible=False)
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#0C4A1E",  # grass tone
        paper_bgcolor="#0C4A1E",
    )
    return fig

def position_selector(label: str, key_prefix: str) -> list[str]:
    """Pitch selector (clickable if streamlit-plotly-events is available) with chip fallback."""
    st.markdown(f"**{label}**")
    sel_key = f"{key_prefix}_selected"
    ensure_session_set(sel_key)

    c1, c2, c3, c4 = st.columns([1,1,1,3])
    with c1:
        if st.button("Select all", key=f"{key_prefix}_all"):
            st.session_state[sel_key] = set(DEFAULT_POSITION_ORDER)
    with c2:
        if st.button("Clear", key=f"{key_prefix}_clear"):
            st.session_state[sel_key] = set()
    with c3:
        if st.button("Invert", key=f"{key_prefix}_invert"):
            st.session_state[sel_key] = set(set(DEFAULT_POSITION_ORDER) - st.session_state[sel_key])

    if _PLOTLY_EVENTS:
        fig = draw_pitch_plotly(st.session_state[sel_key])
        clicked = plotly_events(
            fig, click_event=True, hover_event=False, select_event=False,
            override_height=560, override_width="100%"
        )
        if clicked:
            idx = clicked[0].get("pointIndex", None)
            if idx is not None and 0 <= idx < len(DEFAULT_POSITION_ORDER):
                pos = DEFAULT_POSITION_ORDER[idx]
                toggle_selection(sel_key, pos)
                st.rerun()
        st.caption("Tip: Click any label on the pitch to toggle selection.")
    else:
        st.warning("Interactive clicks require `streamlit-plotly-events`. Falling back to chip selector.")
        chips_per_row = 10
        rows = (len(DEFAULT_POSITION_ORDER) + chips_per_row - 1) // chips_per_row
        i = 0
        for _ in range(rows):
            cols = st.columns(chips_per_row, gap="small")
            for col in cols:
                if i >= len(DEFAULT_POSITION_ORDER): break
                opt = DEFAULT_POSITION_ORDER[i]; i += 1
                sel = opt in st.session_state[sel_key]
                label = f"● {opt}" if sel else opt
                if col.button(label, key=f"{key_prefix}_{opt}"):
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

def normalize_pos_cell(cell: str) -> list[str]:
    """Split comma-separated positions and uppercase/trim each item."""
    if not isinstance(cell, str):
        cell = str(cell)
    parts = [p.strip().upper() for p in cell.split(",") if p.strip()]
    return parts

# -------------------------
# App header
# -------------------------
st.title("⚽ PCA Analysis — Physical & Technical Metrics")
st.markdown(
    "Upload your Excel files, select positions **on the football pitch**, pick numeric metrics, "
    "and explore a **2D PCA** plot with loadings. Everything is optimized for a smooth workflow."
)

# -------------------------
# Sidebar (Upload & Filters)
# -------------------------
with st.sidebar:
    st.header("1) Data upload")
    uploaded_files = st.file_uploader(
        "Select up to 10 Excel files",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
        help="Each file should contain at least 'Player', 'Position', and some numeric columns."
    )
    if not uploaded_files:
        st.info("Upload at least one file to continue.")
        st.stop()

    data = load_excel_files(uploaded_files)

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
        options=[c for c in ["League", "Team", "Position"] if c in data.columns] or ["League"],
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
# Pitch selector
# -------------------------
st.header("Select positions on the pitch")
selected_positions = position_selector(
    label="Click labels on the pitch to (de)select. Each comma-separated position in your data counts individually.",
    key_prefix="pitchpos",
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

# Position filter (treat comma-separated positions individually; case-insensitive)
if selected_positions and "Position" in df.columns:
    selected_upper = {p.upper().strip() for p in selected_positions}
    df = df[df["Position"].astype(str).apply(
        lambda s: any(p in selected_upper for p in normalize_pos_cell(s))
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
    if "Position" in row and pd.notna(row["Position"]):
        parts.append(f"Position: {row['Position']}")
    if "Age" in row and pd.notna(row["Age"]):
        try:
            parts.append(f"Age: {int(row['Age'])}")
        except Exception:
            parts.append(f"Age: {row['Age']}")
    parts.append(f"PC1: {row['PCA1']:.2f}")
    parts.append(f"PC2: {row['PCA2']:.2f}")
    return "<br>".join(parts)

# Mark highlighted (if any)
if player_col:
    df_numeric["_is_high"] = df_numeric[player_col].astype(str).isin(set(highlighted_players))
else:
    df_numeric["_is_high"] = False

# Normal groups
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
    # Highlighted points
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

# Loadings (vectors)
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
        # PNG export requires kaleido
        try:
            img_bytes = fig.to_image(format="png", width=1400, height=900, scale=3)
            st.download_button("Download PNG", data=img_bytes, file_name="pca_analysis.png", mime="image/png")
        except Exception:
            st.warning("PNG export requires the 'kaleido' package. Install with: `pip install kaleido`.")

# -------------------------
# Data preview
# -------------------------
with st.expander("Data preview"):
    meta_cols = [c for c in ["Player", "Position", "League", "Team", "Age"] if c in df_numeric.columns]
    st.dataframe(
        df_numeric[[*meta_cols, *selected_metrics, "PCA1", "PCA2"]].head(200),
        use_container_width=True,
        height=320,
    )

st.caption(
    "Positions are matched individually when separated by commas (e.g., 'CB, RB' matches both CB and RB). "
    "Use the pitch above for quick include/exclude."
)
