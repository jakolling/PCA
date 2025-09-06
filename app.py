# app_refined_streamlit_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================
# Page config & theming
# =========================
st.set_page_config(
    page_title="PCA Analysis — Football Metrics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://docs.streamlit.io/",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "PCA explorer for football data."
    },
)

# =========================
# Helper: chip-like selector (com ✅)
# =========================
def chip_selector(
    label: str,
    options: list[str],
    key_prefix: str,
    default_selected: list[str] | None = None,
    chips_per_row: int = 8,
) -> list[str]:
    """
    Renders a grid of toggle buttons (chips). Clicking a chip toggles selection.
    Mostra '✅' nos itens selecionados.
    """
    sel_key = f"{key_prefix}_selected"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = set(default_selected or [])

    # Cabeçalho + ações
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.markdown(f"**{label}**  \nSelecionadas: **{len(st.session_state[sel_key])}**")
    with c2:
        if st.button("Selecionar tudo", key=f"{key_prefix}_all"):
            st.session_state[sel_key] = set(options)
    with c3:
        if st.button("Limpar", key=f"{key_prefix}_clear"):
            st.session_state[sel_key] = set()

    # Render chips em linhas
    rows = (len(options) + chips_per_row - 1) // chips_per_row
    idx = 0
    for _ in range(rows):
        cols = st.columns(chips_per_row, gap="small")
        for col in cols:
            if idx >= len(options):
                break
            opt = options[idx]
            idx += 1
            selected = opt in st.session_state[sel_key]

            btn_label = f"✅ {opt}" if selected else opt
            if col.button(btn_label, key=f"{key_prefix}_{opt}"):
                if selected:
                    st.session_state[sel_key].discard(opt)
                else:
                    st.session_state[sel_key].add(opt)

    return sorted(list(st.session_state[sel_key]))

# =========================
# Sidebar — Data & Filters
# =========================
st.title("⚽ PCA Analysis — Physical & Technical Metrics")

st.markdown(
    """
This app helps you upload **up to 10 Excel files**, filter players, pick numeric metrics, and visualize a **2D PCA** with loadings vectors.  
Use the sidebar to upload files and set filters, then see the plot and export options on the right.
"""
)

with st.sidebar:
    st.header("1) Data upload")
    uploaded_files = st.file_uploader(
        "Select up to 10 Excel files",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
        help="Each file should contain at least 'Player', 'Position', numeric columns, etc.",
    )

    if not uploaded_files:
        st.info("Upload at least one file to continue.")
        st.stop()

    dfs = []
    for f in uploaded_files:
        df = pd.read_excel(f)
        df["League"] = f.name
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # Normalizar posições
    if "Position" in data.columns:
        pos_series = (
            data["Position"]
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
            .replace({"nan": np.nan})
            .dropna()
        )
        all_positions = sorted(pos_series.unique().tolist())
    else:
        all_positions = []

    st.divider()
    st.header("2) Filters")

    # Filtro de minutos
    minute_cols = [c for c in data.columns if "min" in c.lower() or "minutes" in c.lower()]
    if minute_cols:
        minute_col = st.selectbox("Minutes column", minute_cols)
        min_minutes = st.slider(
            "Minimum minutes",
            min_value=0,
            max_value=int(max(1, float(pd.to_numeric(data[minute_col], errors="coerce").fillna(0).max()))),
            value=0,
            step=50,
        )
    else:
        minute_col = None
        min_minutes = 0
        st.caption("No minutes column detected — minute filter disabled.")

    # Filtro de idade
    if "Age" in data.columns:
        age_col = "Age"
        ages = pd.to_numeric(data[age_col], errors="coerce")
        if ages.notna().any():
            a_min, a_max = int(ages.min()), int(ages.max())
            age_range = st.slider("Age range", min_value=a_min, max_value=a_max, value=(a_min, a_max))
        else:
            age_col = None
            age_range = None
            st.caption("Age column is not numeric — age filter disabled.")
    else:
        age_col = None
        age_range = None
        st.caption("No 'Age' column — age filter disabled.")

    st.divider()
    st.header("3) Plot options")
    color_by = st.selectbox(
        "Color points by",
        options=[c for c in ["League", "Team", "Position"] if c in data.columns] or ["League"],
    )
    point_size = st.slider("Point size", 4, 16, 8)
    point_opacity = st.slider("Point opacity", 0.2, 1.0, 0.8)
    show_loadings = st.toggle("Show loadings (metric vectors)", value=True)
    loadings_scale = st.slider("Loadings length", 1.0, 5.0, 3.0, 0.5)

    st.divider()
    st.header("4) Export")
    export_format = st.radio("Format", ["HTML", "PNG (300 DPI)"], horizontal=True)
    export_btn = st.button("Export plot")

# =========================
# Main — Positions & Metrics
# =========================
st.header("Positions (click to select)")

if all_positions:
    selected_positions = chip_selector(
        label="Click the positions you want to include",
        options=all_positions,
        key_prefix="poschips",
        default_selected=[],
        chips_per_row=10,
    )
else:
    selected_positions = []
    st.warning("No 'Position' column found. Position filter not available.")

# Highlight players
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

# Numeric metric selection
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

# =========================
# Data filtering
# =========================
df = data.copy()

if selected_positions and "Position" in df.columns:
    df = df[df["Position"].astype(str).apply(
        lambda s: any(p.strip() in s.split(",") for p in selected_positions)
    )]

if minute_col:
    df[minute_col] = pd.to_numeric(df[minute_col], errors="coerce")
    df = df[df[minute_col] >= min_minutes]

if age_col and age_range:
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df = df[(df[age_col] >= age_range[0]) & (df[age_col] <= age_range[1])]

df_numeric = df.dropna(subset=selected_metrics).copy()
meta_cols = [c for c in ["Player", "Position", "League", "Team", "Age"] if c in df_numeric.columns]
if df_numeric.empty:
    st.warning("No rows left after filters. Try relaxing filters or selecting different metrics.")
    st.stop()

# =========================
# PCA
# =========================
X = df_numeric[selected_metrics].astype(float).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)

df_numeric["PCA1"] = coords[:, 0]
df_numeric["PCA2"] = coords[:, 1]

exp1, exp2 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]

# =========================
# Plot
# =========================
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
    parts.append(f"PCA1: {row['PCA1']:.2f}")
    parts.append(f"PCA2: {row['PCA2']:.2f}")
    return "<br>".join(parts)

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
                    marker=dict(
                        size=point_size + 4,
                        opacity=1.0,
                        color=palette[g],
                        symbol="diamond",
                        line=dict(width=2, color="black"),
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

# =========================
# Export
# =========================
if export_btn:
    if export_format.startswith("HTML"):
        html = fig.to_html(full_html=True, include_plotlyjs="cdn")
        st.download_button(
            label="Download HTML",
            data=html,
            file_name="pca_analysis.html",
            mime="text/html",
        )
    else:
        try:
            img_bytes = fig.to_image(format="png", width=1400, height=900, scale=3)
            st.download_button(
                label="Download PNG",
                data=img_bytes,
                file_name="pca_analysis.png",
                mime="image/png",
            )
        except Exception:
            st.warning("PNG export requires the 'kaleido' package. Install with: `pip install kaleido`")

# =========================
# Footer
# =========================
with st.expander("Data preview"):
    st.dataframe(
        df_numeric[[*(meta_cols or []), *selected_metrics, "PCA1", "PCA2"]].head(200),
        use_container_width=True,
        height=320,
    )

st.caption("Tip: Use the chips above to quickly include/exclude positions, and the sidebar for filters and plot styling.")
