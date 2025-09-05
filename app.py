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

# 2️⃣ Filter by Position (via Virtual Pitch - Opta)
st.header("2️⃣ Filter by Position (click on the pitch)")
# --- Tactical presets & mapping mode ---
with st.expander("⚙️ Tactical mapping", expanded=False):
    colt1, colt2 = st.columns(2)
    with colt1:
        mapping_mode = st.radio("Mapping mode", ["Roles (by formation)", "Zones (thirds + half-spaces)"], index=0)
    with colt2:
        phase = st.selectbox("Phase", ["Offensive", "Defensive"], index=0)

    formation = st.selectbox(
        "Formation (roles presets)",
        ["4-3-3", "4-2-3-1", "3-5-2", "4-4-2", "3-4-3"],
        index=0
    )

# Preset centroids per (formation, phase)
FORMATION_CENTROIDS = {
    ("4-3-3", "Offensive"): {
        "GK": (6,40),
        "RB": (90,14), "RCB": (86,28), "LCB": (86,52), "LB": (90,66),
        "DM": (74,40),
        "RCM": (68,30), "LCM": (68,50),
        "RW": (42,18), "CF": (30,40), "LW": (42,62)
    },
    ("4-3-3", "Defensive"): {
        "GK": (6,40),
        "RB": (96,16), "RCB": (92,30), "LCB": (92,50), "LB": (96,64),
        "DM": (80,40),
        "RCM": (76,32), "LCM": (76,48),
        "RW": (58,24), "CF": (54,40), "LW": (58,56)
    },
    ("4-2-3-1", "Offensive"): {
        "GK": (6,40),
        "RB": (90,16), "RCB": (86,30), "LCB": (86,50), "LB": (90,64),
        "RDM": (76,34), "LDM": (76,46),
        "RAM": (58,30), "CAM": (56,40), "LAM": (58,50),
        "ST": (30,40)
    },
    ("4-2-3-1", "Defensive"): {
        "GK": (6,40),
        "RB": (96,18), "RCB": (92,32), "LCB": (92,48), "LB": (96,62),
        "RDM": (82,36), "LDM": (82,44),
        "RAM": (66,34), "CAM": (64,40), "LAM": (66,46),
        "ST": (56,40)
    },
    ("3-5-2", "Offensive"): {
        "GK": (6,40),
        "RCB": (92,28), "CB": (90,40), "LCB": (92,52),
        "RWB": (82,18), "LWB": (82,62),
        "RDM": (74,34), "LDM": (74,46), "AM": (58,40),
        "RS": (30,36), "LS": (30,44)
    },
    ("3-5-2", "Defensive"): {
        "GK": (6,40),
        "RCB": (96,30), "CB": (94,40), "LCB": (96,50),
        "RWB": (90,24), "LWB": (90,56),
        "RDM": (82,36), "LDM": (82,44), "AM": (66,40),
        "RS": (58,38), "LS": (58,42)
    },
    ("4-4-2", "Offensive"): {
        "GK": (6,40),
        "RB": (90,16), "RCB": (86,30), "LCB": (86,50), "LB": (90,64),
        "RM": (60,24), "RCM": (68,34), "LCM": (68,46), "LM": (60,56),
        "RS": (32,36), "LS": (32,44)
    },
    ("4-4-2", "Defensive"): {
        "GK": (6,40),
        "RB": (96,18), "RCB": (92,32), "LCB": (92,48), "LB": (96,62),
        "RM": (72,28), "RCM": (78,36), "LCM": (78,44), "LM": (72,52),
        "RS": (60,38), "LS": (60,42)
    },
    ("3-4-3", "Offensive"): {
        "GK": (6,40),
        "RCB": (92,28), "CB": (90,40), "LCB": (92,52),
        "RWM": (78,22), "R6": (74,34), "L6": (74,46), "LWM": (78,58),
        "RW": (44,20), "CF": (30,40), "LW": (44,60)
    },
    ("3-4-3", "Defensive"): {
        "GK": (6,40),
        "RCB": (96,30), "CB": (94,40), "LCB": (96,50),
        "RWM": (86,26), "R6": (82,36), "L6": (82,44), "LWM": (86,54),
        "RW": (62,28), "CF": (58,40), "LW": (62,52)
    },
}

def get_role_centroids(formation, phase):
    return FORMATION_CENTROIDS.get((formation, phase), FORMATION_CENTROIDS[("4-3-3", "Offensive")])

# --- Zones definition (thirds + half-spaces) ---
# Coordinates are in Opta (x: 0->120 attack left->right, y: 0->80 bottom->top)
ZONES = [
    ("Defensive Left Wing",           (80,120), (0,20)),
    ("Defensive Left Half-space",     (80,120), (20,35)),
    ("Defensive Central",             (80,120), (35,45)),
    ("Defensive Right Half-space",    (80,120), (45,60)),
    ("Defensive Right Wing",          (80,120), (60,80)),

    ("Middle Left Wing",              (40,80), (0,20)),
    ("Middle Left Half-space",        (40,80), (20,35)),
    ("Middle Central",                (40,80), (35,45)),
    ("Middle Right Half-space",       (40,80), (45,60)),
    ("Middle Right Wing",             (40,80), (60,80)),

    ("Attacking Left Wing",           (0,40), (0,20)),
    ("Attacking Left Half-space",     (0,40), (20,35)),
    ("Attacking Central",             (0,40), (35,45)),
    ("Attacking Right Half-space",    (0,40), (45,60)),
    ("Attacking Right Wing",          (0,40), (60,80)),
]

def map_to_zone(x, y):
    for name, (x_min, x_max), (y_min, y_max) in ZONES:
        # x range is defined as [min,max), last band inclusive
        if (x_min <= x < x_max or (x == x_max and x_max == 120)) and (y_min <= y < y_max or (y == y_max and y_max == 80)):
            return name
    return "Unclassified"


from mplsoccer import Pitch
from PIL import Image
import numpy as np
from io import BytesIO

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    st.error("Missing dependency 'streamlit-image-coordinates'. Install with: pip install streamlit-image-coordinates")
    st.stop()

if "pitch_clicks" not in st.session_state:
    st.session_state.pitch_clicks = []
if "selected_roles" not in st.session_state:
    st.session_state.selected_roles = set()

def render_pitch_png(dpi=200):
    pitch = Pitch(pitch_type='opta', pitch_color='#f5f5f5', line_color='#222',
                  linewidth=1.5, line_zorder=2)
    fig, ax = pitch.draw(figsize=(8, 5.5), tight_layout=True)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    plt_img = Image.open(buf)
    return plt_img

pitch_img = render_pitch_png()

st.caption("Click positions on the Opta pitch to filter the table; use the buttons to undo or clear.")
coords = streamlit_image_coordinates(pitch_img, key="pitch_img", width=800)

c1, c2 = st.columns([1,1])
with c1:
    if st.button("↩️ Undo last click", use_container_width=True):
        if st.session_state.pitch_clicks:
            st.session_state.pitch_clicks.pop()
with c2:
    if st.button("🧹 Clear clicks", type="secondary", use_container_width=True):
        st.session_state.pitch_clicks.clear()
        st.session_state.selected_roles.clear()

if coords is not None and all(k in coords for k in ("x", "y", "width", "height")):
    x_px, y_px = coords["x"], coords["y"]
    w_px, h_px = coords["width"], coords["height"]
    x_opta = 120.0 * (x_px / w_px)
    y_opta = 80.0 * (1.0 - (y_px / h_px))
    st.session_state.pitch_clicks.append((x_opta, y_opta))


# Build role centroids from tactical selection (if in Roles mode)
if mapping_mode.startswith("Roles"):
    role_centroids = get_role_centroids(formation, phase)
else:
    role_centroids = {}  # not used in Zones mode

def nearest_role(xy):
    x, y = xy
    if not role_centroids:
        return None
    best_role, best_d = None, 1e9
    for r, (rx, ry) in role_centroids.items():
        d = (x - rx)**2 + (y - ry)**2
        if d < best_d:
            best_role, best_d = r, d
    return best_role
(xy):
    x, y = xy
    best_role, best_d = None, 1e9
    for r, (rx, ry) in role_centroids.items():
        d = (x - rx)**2 + (y - ry)**2
        if d < best_d:
            best_role, best_d = r, d
    return best_role

for xy in st.session_state.pitch_clicks:
    if mapping_mode.startswith("Roles"):
        label = nearest_role(xy)
    else:
        label = map_to_zone(*xy)
    if label:
        st.session_state.selected_roles.add(label)

st.write("**Selected labels (from clicks):**")
if st.session_state.selected_roles:
    st.write(", ".join(sorted(st.session_state.selected_roles)))
else:
    st.info("No positions selected yet. Click on the pitch above.")


if st.session_state.selected_roles:
    if mapping_mode.startswith("Roles") and "Position" in combined_df.columns:
        roles = sorted(st.session_state.selected_roles)
        mask = combined_df["Position"].fillna("").astype(str).str.upper().apply(
            lambda s: any(r in [p.strip().upper() for p in s.split(",")] for r in roles)
        )
        combined_df = combined_df[mask]
        st.success(f"{len(combined_df)} players in the selected roles.")
    elif not mapping_mode.startswith("Roles"):
        st.info("Zones mode ativo: rótulos servem para análise visual e export; não há filtro de DataFrame por zona a menos que seu dataset tenha coluna 'Zone'.")
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
    required_cols = ["Player", "Position", "League", "Team"]
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
                (f"Club: {row['Team']}<br>" if pd.notna(row.get('Team')) else "") +
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
                (f"Club: {row['Team']}<br>" if pd.notna(row.get('Team')) else "") +
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
