import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import base64
from PIL import Image
import io
import plotly.io as pio

st.set_page_config(page_title="PCA Analysis App", layout="wide")

def install_kaleido():
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaleido"])
        return True
    except:
        return False

def main():
    st.title("⚽ PCA Analysis - Physical/Technical Metrics")

    st.markdown("""
    Upload up to **10 XLS files**, filter by position, age range, minimum minutes played,
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
        df["League"] = file.name
        dfs.append(df)
        st.success(f"✅ File added: {file.name}")

    combined_df = pd.concat(dfs, ignore_index=True)

    # 2️⃣ Filter by Position
    st.header("2️⃣ Filter by Position")
    if "Position" in combined_df.columns:
        all_positions = combined_df["Position"].dropna().str.split(',').explode().str.strip().unique().tolist()
        positions = st.multiselect(
            "Select one or more positions to include:",
            options=sorted(all_positions)
        )
        if positions:
            mask = combined_df["Position"].str.split(',').apply(
                lambda x: any(pos.strip() in positions for pos in x) if isinstance(x, list) else False
            )
            combined_df = combined_df[mask]
            st.success(f"{len(combined_df)} players found for selected positions.")

    # 3️⃣ Filter by Age
    st.header("3️⃣ Filter by Age Range")
    if "Age" in combined_df.columns:
        min_age = int(combined_df["Age"].min())
        max_age = int(combined_df["Age"].max())
        
        age_range = st.slider(
            "Select age range:",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )
        combined_df = combined_df[
            (combined_df["Age"] >= age_range[0]) & 
            (combined_df["Age"] <= age_range[1])
        ]
        st.success(f"{len(combined_df)} players between ages {age_range[0]} and {age_range[1]}.")
    else:
        st.warning("No 'Age' column found in the data. Age filter will be skipped.")

    # 4️⃣ Filter by Minutes
    st.header("4️⃣ Filter by Minimum Minutes Played")
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

    # 5️⃣ Highlight Players
    st.header("5️⃣ Highlight Players")
    if "Player" in combined_df.columns:
        player_names = combined_df["Player"].dropna().unique().tolist()
        highlighted_players = st.multiselect(
            "Select up to 5 players to highlight:",
            options=player_names,
            max_selections=5
        )
    else:
        highlighted_players = []

    # 6️⃣ Select Metrics
    st.header("6️⃣ Select Metrics (Numeric Columns)")
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

    # Modified section to handle missing columns
    columns_to_keep = selected_metrics.copy()
    required_cols = ["Player", "Position", "League"]
    available_cols = [col for col in required_cols if col in combined_df.columns]
    columns_to_keep.extend(available_cols)

    if "Age" in combined_df.columns:
        columns_to_keep.append("Age")

    # Verify all columns exist before dropna
    existing_cols = [col for col in columns_to_keep if col in combined_df.columns]
    df_clean = combined_df.dropna(subset=existing_cols)

    if df_clean.empty:
        st.warning("No valid data left after filters.")
        st.stop()

    # 7️⃣ Run PCA
    X = df_clean[selected_metrics].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df_clean["PCA1"] = coords[:, 0]
    df_clean["PCA2"] = coords[:, 1]

    # 8️⃣ Plot
    st.header("7️⃣ PCA Plot")

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
            hover_text = normal_players.apply(lambda row: 
                f"<b>{row['Player']}</b><br>" +
                (f"Position: {row['Position']}<br>" if pd.notna(row.get('Position')) else "") +
                (f"Age: {int(row['Age'])}<br>" if pd.notna(row.get('Age')) else "") +
                f"PCA1: {row['PCA1']:.2f}<br>PCA2: {row['PCA2']:.2f}",
                axis=1
            )

            fig.add_trace(go.Scatter(
                x=normal_players["PCA1"],
                y=normal_players["PCA2"],
                mode="markers",
                marker=dict(size=8, color=color, opacity=0.7),
                name=league,
                text=hover_text,
                hoverinfo="text",
                hovertemplate="%{text}<extra></extra>"
            ))

        if not highlighted.empty:
            hover_text_highlighted = highlighted.apply(lambda row: 
                f"<b>{row['Player']}</b><br>" +
                (f"Position: {row['Position']}<br>" if pd.notna(row.get('Position')) else "") +
                (f"Age: {int(row['Age'])}<br>" if pd.notna(row.get('Age')) else "") +
                f"PCA1: {row['PCA1']:.2f}<br>PCA2: {row['PCA2']:.2f}",
                axis=1
            )

            for _, player_row in highlighted.iterrows():
                fig.add_trace(go.Scatter(
                    x=[player_row["PCA1"]],
                    y=[player_row["PCA2"]],
                    mode="markers+text",
                    marker=dict(size=12, color=color, symbol="diamond", line=dict(width=2, color="black")),
                    text=[player_row["Player"]],
                    textposition="bottom center",
                    name=player_row["Player"],
                    hovertext=hover_text_highlighted,
                    hoverinfo="text",
                    hovertemplate="%{hovertext}<extra></extra>",
                    legendgroup="highlighted",
                    showlegend=True
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

    # 9️⃣ Export Results
    st.header("8️⃣ Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export to HTML
        if st.button("Export to HTML"):
            html = fig.to_html(full_html=True, include_plotlyjs='cdn')
            st.download_button(
                label="Download HTML",
                data=html,
                file_name="pca_analysis.html",
                mime="text/html"
            )
    
    with col2:
        # Export to PNG with 300 DPI
        if st.button("Export to PNG (300 DPI)"):
            try:
                # Try to export with Kaleido
                img_bytes = fig.to_image(format="png", width=1200, height=800, scale=3)
                st.download_button(
                    label="Download PNG",
                    data=img_bytes,
                    file_name="pca_analysis.png",
                    mime="image/png"
                )
            except:
                st.warning("Kaleido package is required for PNG export. Installing now...")
                if install_kaleido():
                    st.success("Kaleido installed successfully! Please click the export button again.")
                else:
                    st.error("Failed to install Kaleido. Please install it manually with: pip install kaleido")

if __name__ == "__main__":
    main()
