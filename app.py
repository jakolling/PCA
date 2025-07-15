import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PCA Analysis App", layout="wide")

def main():
    st.title("⚽ PCA Analysis - Physical/Technical Metrics")

    st.markdown("""
    Upload up to **10 XLS files**, filter by position, minimum minutes played,
    choose numeric columns for PCA, highlight players, and visualize the results.
    """)

    # 1️⃣ Upload XLS files
    st.header("1️⃣ Upload XLS Files")
    uploaded_files = st.file_uploader(
        "Select up to 10 XLS/XLSX files",
        type=["xls", "xlsx"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("👉 Please upload at least one XLS file to continue.")
        st.stop()

    # Read files
    dfs = []
    for file in uploaded_files:
        df = pd.read_excel(file)
        # Auto-generate League name from file name
        df["League"] = file.name
        dfs.append(df)
        st.success(f"✅ File added: {file.name}")

    combined_df = pd.concat(dfs, ignore_index=True)

    # 2️⃣ Filter by Position
    st.header("2️⃣ Filter by Position")
    if "Position" in combined_df.columns:
        all_positions = combined_df["Position"].dropna().unique().tolist()
        positions = st.multiselect(
            "Select one or more positions to include:",
            options=all_positions
        )
        if positions:
            combined_df = combined_df[combined_df["Position"].isin(positions)]
            st.success(f"{len(combined_df)} players found for selected positions.")

    # 3️⃣ Filter by Minutes
    st.header("3️⃣ Filter by Minimum Minutes Played")
    minute_cols = [col for col in combined_df.columns if 'min' in col.lower() or 'minutes' in col.lower()]
    if minute_cols:
        minute_col = st.selectbox("Select column for minutes filter:", minute_cols)
        min_minutes = st.slider(
            f"Minimum minutes played ({minute_col}):",
            min_value=0,
            max_value=int(combined_df[minute_col].max()),
            value=0,
            step=50
        )
        combined_df = combined_df[combined_df[minute_col] >= min_minutes]
        st.success(f"{len(combined_df)} players with at least {min_minutes} minutes.")

    # 4️⃣ Highlight Players
    st.header("4️⃣ Highlight Players")
    if "Player" in combined_df.columns:
        player_names = combined_df["Player"].dropna().unique().tolist()
        highlighted_players = st.multiselect(
            "Select up to 5 players to highlight:",
            options=player_names,
            max_selections=5
        )
    else:
        highlighted_players = []

    # 5️⃣ Select Metrics
    st.header("5️⃣ Select Metrics (Numeric Columns)")
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found in your data!")
        st.stop()

    selected_metrics = st.multiselect(
        "Select at least 2 numeric columns for PCA:",
        options=numeric_cols
    )

    if len(selected_metrics) < 2:
        st.warning("Please select at least 2 numeric columns.")
        st.stop()

    df_clean = combined_df.dropna(subset=selected_metrics + (["Player"] if "Player" in combined_df.columns else []))

    if df_clean.empty:
        st.warning("No valid data left after filters.")
        st.stop()

    # 6️⃣ Run PCA
    X = df_clean[selected_metrics].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df_clean["PCA1"] = coords[:, 0]
    df_clean["PCA2"] = coords[:, 1]

    # 7️⃣ Plot
    st.header("6️⃣ PCA Plot")

    fig = go.Figure()

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    leagues = df_clean["League"].unique()
    league_colors = {league: colors[i % len(colors)] for i, league in enumerate(leagues)}

    for league, color in league_colors.items():
        df_league = df_clean[df_clean["League"] == league]

        normal_players = df_league[
            ~df_league["Player"].isin(highlighted_players)
        ] if "Player" in df_league.columns else df_league

        highlighted = df_league[
            df_league["Player"].isin(highlighted_players)
        ] if "Player" in df_league.columns else pd.DataFrame()

        if not normal_players.empty:
            fig.add_trace(go.Scatter(
                x=normal_players["PCA1"],
                y=normal_players["PCA2"],
                mode="markers",
                marker=dict(size=8, color=color, opacity=0.7),
                name=league,
                text=normal_players["Player"] if "Player" in normal_players.columns else None,
                hovertemplate="<b>%{text}</b><br>PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<extra></extra>"
                if "Player" in normal_players.columns else "PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<extra></extra>"
            ))

        if not highlighted.empty:
            fig.add_trace(go.Scatter(
                x=highlighted["PCA1"],
                y=highlighted["PCA2"],
                mode="markers+text",
                marker=dict(size=12, color=color, symbol="diamond", line=dict(width=2, color="black")),
                text=highlighted["Player"],
                textposition="bottom center",
                name="Highlighted",
                hovertemplate="<b>%{text}</b><br>PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<extra></extra>"
            ))

    # Add loadings (vectors)
    for i, metric in enumerate(selected_metrics):
        fig.add_trace(go.Scatter(
            x=[0, pca.components_[0, i] * 3],
            y=[0, pca.components_[1, i] * 3],
            mode="lines+text",
            line=dict(color="blue", width=2),
            text=[None, metric],
            textposition="top center",
            showlegend=False
        ))

    fig.update_layout(
        title="PCA Analysis",
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        width=1200,
        height=800,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.success("✅ PCA plot generated successfully!")

if __name__ == "__main__":
    main()
