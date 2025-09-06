# app_refined_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_plotly_events import plotly_events

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
# Helper: Football pitch position selector (clickable)
# =========================
def pitch_selector(
    label: str,
    options: list[str],
    key_prefix: str,
    default_selected: list[str] | None = None,
) -> list[str]:
    """
    Renderiza um campo de futebol com marcadores clicáveis nas posições.
    Ao clicar, alterna seleção. Retorna lista de posições selecionadas.
    Usa st.session_state para persistir.
    """
    sel_key = f"{key_prefix}_selected"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = set(default_selected or [])

    st.markdown(f"**{label}**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Selecionar tudo", key=f"{key_prefix}_all"):
            st.session_state[sel_key] = set(options)
    with c2:
        if st.button("Limpar", key=f"{key_prefix}_clear"):
            st.session_state[sel_key] = set()

    # Dimensões do campo (escala 105 x 68)
    WIDTH, HEIGHT = 105, 68

    # Coordenadas sugeridas para posições "clássicas"
    base_coords = {
        "GK": (5, HEIGHT/2),
        "RB": (25, HEIGHT*0.2),
        "RWB": (30, HEIGHT*0.2),
        "CB": (20, HEIGHT/2),
        "RCB": (18, HEIGHT*0.35),
        "LCB": (18, HEIGHT*0.65),
        "LB": (25, HEIGHT*0.8),
        "LWB": (30, HEIGHT*0.8),
        "DM": (40, HEIGHT/2),
        "RDM": (40, HEIGHT*0.38),
        "LDM": (40, HEIGHT*0.62),
        "CM": (52, HEIGHT/2),
        "RCM": (52, HEIGHT*0.38),
        "LCM": (52, HEIGHT*0.62),
        "AM": (65, HEIGHT/2),
        "RAM": (65, HEIGHT*0.38),
        "LAM": (65, HEIGHT*0.62),
        "RW": (75, HEIGHT*0.2),
        "LW": (75, HEIGHT*0.8),
        "RM": (60, HEIGHT*0.25),
        "LM": (60, HEIGHT*0.75),
        "CF": (88, HEIGHT/2),
        "ST": (88, HEIGHT/2),
        "RS": (85, HEIGHT*0.4),
        "LS": (85, HEIGHT*0.6),
    }

    # Para opções que não estejam no dicionário, posiciona-as no meio-campo
    unknowns = [p for p in options if p not in base_coords]
    if unknowns:
        y_vals = np.linspace(HEIGHT*0.25, HEIGHT*0.75, len(unknowns))
        for i, pos in enumerate(unknowns):
            base_coords[pos] = (52, float(y_vals[i]))

    # Filtra só as posições presentes nas opções
    coords = {p: base_coords[p] for p in options if p in base_coords}

    # Cria o campo
    fig = go.Figure()

    # Gramado (retângulo)
    fig.add_shape(type="rect", x0=0, y0=0, x1=WIDTH, y1=HEIGHT,
                  line=dict(color="#2e7d32"), fillcolor="#3fa34d")
    # Linha de meio-campo
    fig.add_shape(type="line", x0=WIDTH/2, y0=0, x1=WIDTH/2, y1=HEIGHT,
                  line=dict(color="white", width=2))
    # Círculo central
    fig.add_shape(type="circle",
                  x0=WIDTH/2-9.15, y0=HEIGHT/2-9.15,
                  x1=WIDTH/2+9.15, y1=HEIGHT/2+9.15,
                  line=dict(color="white", width=2))
    # Grandes áreas
    fig.add_shape(type="rect", x0=0, y0=HEIGHT*0.2, x1=16.5, y1=HEIGHT*0.8,
                  line=dict(color="white", width=2))
    fig.add_shape(type="rect", x0=WIDTH-16.5, y0=HEIGHT*0.2, x1=WIDTH, y1=HEIGHT*0.8,
                  line=dict(color="white", width=2))
    # Pequenas áreas
    fig.add_shape(type="rect", x0=0, y0=HEIGHT*0.35, x1=5.5, y1=HEIGHT*0.65,
                  line=dict(color="white", width=2))
    fig.add_shape(type="rect", x0=WIDTH-5.5, y0=HEIGHT*0.35, x1=WIDTH, y1=HEIGHT*0.65,
                  line=dict(color="white", width=2))
    # Pontos de pênalti
    fig.add_trace(go.Scatter(x=[11], y=[HEIGHT/2], mode="markers",
                             marker=dict(size=6, symbol="x"),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[WIDTH-11], y=[HEIGHT/2], mode="markers",
                             marker=dict(size=6, symbol="x"),
                             showlegend=False, hoverinfo="skip"))

    # Marcadores das posições
    xs, ys, texts, colors = [], [], [], []
    for pos, (x, y) in coords.items():
        xs.append(x); ys.append(y); texts.append(pos)
        colors.append("#2563EB" if pos in st.session_state[sel_key] else "#0E1117")

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(size=20, color=colors, line=dict(color="white", width=1.5)),
            text=texts, textposition="middle center",
            hoverinfo="text", hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )

    fig.update_xaxes(visible=False, range=[-2, WIDTH+2])
    fig.update_yaxes(visible=False, range=[-2, HEIGHT+2], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#3fa34d",
        paper_bgcolor=st.get_option("theme.backgroundColor") or "white",
    )

    # Captura clique
    clicked = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        key=f"{key_prefix}_events"
    )
    if clicked:
        point = clicked[0]
        idx = point.get("pointNumber")
        pos_clicked = texts[idx] if idx is not None and 0 <= idx < len(texts) else None
        if pos_clicked:
            if pos_clicked in st.session_state[sel_key]:
                st.session_state[sel_key].discard(pos_clicked)
            else:
                st.session_state[sel_key].add(pos_clicked)

    st.caption(
        "Selecionadas: " +
        (", ".join(sorted(st.session_state[sel_key])) if st.session_state[sel_key] else "nenhuma")
    )

    return sorted(list(st.session_state[sel_key]))

# =========================
# Sidebar — Data & Filters
# =========================
st.title("⚽ PCA Analysis — Physical & Technical Metrics")

st.markdown(
    """
Este app permite enviar **até 10 arquivos Excel**, filtrar jogadores, escolher métricas numéricas
e visualizar um **PCA 2D** com vetores de loadings.
Use a barra lateral para carregar dados e ajustar filtros; o gráfico e exportações ficam à direita.
"""
)

with st.sidebar:
    st.header("1) Data upload")
    uploaded_files = st.file_uploader(
        "Selecione até 10 arquivos Excel",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
        help="Cada arquivo deve conter pelo menos 'Player', 'Position' e colunas numéricas.",
    )

    if not uploaded_files:
        st.info("Envie ao menos um arquivo para continuar.")
        st.stop()

    dfs = []
    for f in uploaded_files:
        df = pd.read_excel(f)
        # mantém o nome do arquivo como tag 'League'
        df["League"] = f.name
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # Normaliza 'Position' para lista
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
        minute_col = st.selectbox("Coluna de minutos", minute_cols, help="Filtra por minutos jogados.")
        min_minutes = st.slider(
            "Mínimo de minutos",
            min_value=0,
            max_value=int(max(1, float(pd.to_numeric(data[minute_col], errors="coerce").fillna(0).max()))),
            value=0,
            step=50,
        )
    else:
        minute_col = None
        min_minutes = 0
        st.caption("Nenhuma coluna de minutos detectada — filtro desativado.")

    # Filtro de idade
    if "Age" in data.columns:
        age_col = "Age"
        ages = pd.to_numeric(data[age_col], errors="coerce")
        if ages.notna().any():
            a_min, a_max = int(ages.min()), int(ages.max())
            age_range = st.slider("Faixa etária", min_value=a_min, max_value=a_max, value=(a_min, a_max))
        else:
            age_col = None
            st.caption("Coluna de idade não numérica — filtro desativado.")
            age_range = None
    else:
        age_col = None
        age_range = None
        st.caption("Sem coluna 'Age' — filtro desativado.")

    st.divider()
    st.header("3) Plot options")
    color_by = st.selectbox(
        "Colorir pontos por",
        options=[c for c in ["League", "Team", "Position"] if c in data.columns] or ["League"],
        help="Agrupamento usado para cores dos pontos.",
    )
    point_size = st.slider("Tamanho do ponto", 4, 16, 8)
    point_opacity = st.slider("Opacidade do ponto", 0.2, 1.0, 0.8)
    show_loadings = st.toggle("Mostrar loadings (vetores de métricas)", value=True)
    loadings_scale = st.slider("Comprimento dos loadings", 1.0, 5.0, 3.0, 0.5)

    st.divider()
    st.header("4) Export")
    export_format = st.radio("Formato", ["HTML", "PNG (300 DPI)"], horizontal=True)
    export_btn = st.button("Exportar gráfico")

# =========================
# Main — Positions & Metrics
# =========================
st.header("Posições (clique no campo)")

if all_positions:
    selected_positions = pitch_selector(
        label="Clique no campo para (de)selecionar posições",
        options=all_positions,
        key_prefix="pitch",
        default_selected=[],
    )
else:
    selected_positions = []
    st.warning("Nenhuma coluna 'Position' encontrada. Filtro por posição indisponível.")

# Destaque de jogadores
st.subheader("Destacar jogadores (opcional)")
player_col = "Player" if "Player" in data.columns else None
if player_col:
    player_options = sorted(pd.Series(data[player_col].astype(str)).unique().tolist())
    highlighted_players = st.multiselect(
        "Selecione até 5 jogadores para rotular no gráfico",
        options=player_options,
        max_selections=5,
        placeholder="Digite um nome…",
    )
else:
    highlighted_players = []
    st.caption("Sem coluna 'Player' — destaque desativado.")

# Seleção de métricas numéricas
st.header("Seleção de métricas")
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("Nenhuma coluna numérica encontrada — envie dados com métricas numéricas.")
    st.stop()

# Exclui colunas possivelmente não-métricas do default
non_metric_hints = {"Age", "Height", "Weight", "Minutes", "Min", "Games"}
default_metrics = [c for c in numeric_cols if c not in non_metric_hints] or numeric_cols

selected_metrics = st.multiselect(
    "Escolha pelo menos duas colunas numéricas para o PCA",
    options=numeric_cols,
    default=default_metrics[: min(6, len(default_metrics))],
    help="Variáveis usadas para calcular os componentes do PCA.",
)
if len(selected_metrics) < 2:
    st.warning("Selecione pelo menos duas colunas para executar o PCA.")
    st.stop()

# =========================
# Data filtering
# =========================
df = data.copy()

# Filtro por posição
if selected_positions and "Position" in df.columns:
    df = df[df["Position"].astype(str).apply(
        lambda s: any(p.strip() in s.split(",") for p in selected_positions)
    )]

# Filtro por minutos
if minute_col:
    df[minute_col] = pd.to_numeric(df[minute_col], errors="coerce")
    df = df[df[minute_col] >= min_minutes]

# Filtro por idade
if age_col and age_range:
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df = df[(df[age_col] >= age_range[0]) & (df[age_col] <= age_range[1])]

# Mantém apenas linhas com todas as métricas selecionadas
df_numeric = df.dropna(subset=selected_metrics).copy()

# Colunas de metadados opcionais
meta_cols = [c for c in ["Player", "Position", "League", "Team", "Age"] if c in df_numeric.columns]
if df_numeric.empty:
    st.warning("Nenhuma linha restante após os filtros. Afrouxe os filtros ou escolha outras métricas.")
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
# Plotly figure
# =========================
st.header("PCA plot")
fig = go.Figure()

group_col = color_by if color_by in df_numeric.columns else "League"
groups = sorted(df_numeric[group_col].astype(str).unique().tolist())

# Paleta de 10 cores base
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

# Normal + destacados
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

# Loadings (vetores)
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
st.success("PCA calculado e plotado com sucesso.")

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
        # PNG exige Kaleido
        try:
            img_bytes = fig.to_image(format="png", width=1400, height=900, scale=3)
            st.download_button(
                label="Download PNG",
                data=img_bytes,
                file_name="pca_analysis.png",
                mime="image/png",
            )
        except Exception:
            st.warning(
                "A exportação em PNG requer o pacote 'kaleido'. Instale com: `pip install kaleido`."
            )

# =========================
# Footer
# =========================
with st.expander("Data preview"):
    st.dataframe(
        df_numeric[[*(meta_cols or []), *selected_metrics, "PCA1", "PCA2"]].head(200),
        use_container_width=True,
        height=320,
    )

st.caption(
    "Dica: use o mini-campo para filtrar posições e a barra lateral para filtros e estilo do gráfico."
)
