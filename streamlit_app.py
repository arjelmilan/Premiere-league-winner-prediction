import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# Set Page Config
st.set_page_config(
    page_title="Premier League Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load ML components and team stats database
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")
TEAM_STATS_PATH = os.path.join("artifacts", "team_rolling_stats.csv")

@st.cache_resource
def load_ml_components():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        st.error("Model or Preprocessor files not found in artifacts. Please run the training pipeline first.")
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

@st.cache_data
def load_team_stats():
    if not os.path.exists(TEAM_STATS_PATH):
        st.error(f"Team rolling stats file not found at {TEAM_STATS_PATH}. Please run the generator first.")
        return {}
    df = pd.read_csv(TEAM_STATS_PATH)
    return df.set_index("Team").to_dict(orient="index")

model, preprocessor = load_ml_components()
team_stats_dict = load_team_stats()
teams = sorted(list(team_stats_dict.keys())) if team_stats_dict else []

# Custom CSS for Glassmorphism Dashboard
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        /* Base page styling */
        .stApp {
            background: linear-gradient(135deg, #0d0f1e 0%, #151a3a 100%) !important;
            color: #f3f4f6 !important;
            font-family: 'Outfit', sans-serif !important;
        }

        h1, h2, h3, h4, h5, h6, p, span, div, label {
            font-family: 'Outfit', sans-serif !important;
        }

        /* Glassmorphic card styling */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            padding: 2rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }

        .home-card {
            border-top: 3px solid #6366f1;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), 0 0 20px rgba(99, 102, 241, 0.15);
        }

        .away-card {
            border-top: 3px solid #ec4899;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), 0 0 20px rgba(236, 72, 153, 0.15);
        }

        /* Custom Title & Headers */
        .app-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(to right, #818cf8, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
            letter-spacing: -0.5px;
        }

        .app-subtitle {
            text-align: center;
            color: #9ca3af;
            font-size: 1.15rem;
            font-weight: 300;
            margin-bottom: 2rem;
        }

        /* Section divider */
        .vs-divider {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
        }

        .vs-circle {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 2px solid rgba(255,255,255,0.08);
            width: 70px;
            height: 70px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.4rem;
            color: #9ca3af;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
        }

        /* Streamlit overrides for labels and input boxes */
        label[data-testid="stWidgetLabel"] {
            color: #9ca3af !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }

        div[data-baseweb="select"] > div {
            background-color: rgba(0, 0, 0, 0.3) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 12px !important;
            color: #f3f4f6 !important;
            padding: 4px !important;
        }

        input[type="number"] {
            background-color: rgba(0, 0, 0, 0.3) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 12px !important;
            color: #f3f4f6 !important;
        }

        /* Results & Winner Card */
        .winner-announcement {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.05);
            text-align: center;
            margin-bottom: 2rem;
        }

        .winner-title {
            font-size: 0.9rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 0.25rem;
        }

        .winner-name {
            font-size: 2.4rem;
            font-weight: 800;
            text-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
        }

        .home-win {
            color: #818cf8;
            text-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
        }

        .away-win {
            color: #f472b6;
            text-shadow: 0 0 15px rgba(236, 72, 153, 0.3);
        }

        .draw-win {
            color: #10b981;
            text-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
        }

        /* Probabilities Progress Chart */
        .prob-chart {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            margin-bottom: 1rem;
        }

        .prob-item {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .prob-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.95rem;
            font-weight: 600;
        }

        .prob-label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .prob-value {
            font-size: 1.15rem;
            font-weight: 800;
        }

        .prob-bar-bg {
            height: 14px;
            background: rgba(255,255,255,0.05);
            border-radius: 7px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.03);
        }

        .prob-bar-fill {
            height: 100%;
            border-radius: 7px;
        }

        .fill-home { background: linear-gradient(90deg, #4f46e5, #6366f1); }
        .fill-draw { background: linear-gradient(90deg, #059669, #10b981); }
        .fill-away { background: linear-gradient(90deg, #db2777, #ec4899); }

        /* Stats Comparison Panel */
        .stats-header {
            font-size: 1.5rem;
            font-weight: 700;
            text-align: center;
            color: #818cf8;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
        }

        .stat-row {
            display: grid;
            grid-template-columns: 2.5fr 1fr 2.5fr;
            align-items: center;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.03);
            margin-bottom: 0.8rem;
        }

        .stat-name {
            text-align: center;
            font-size: 0.85rem;
            font-weight: 600;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stat-val-home {
            text-align: left;
            font-weight: 700;
            font-size: 1.15rem;
            color: #c7d2fe;
        }

        .stat-val-away {
            text-align: right;
            font-weight: 700;
            font-size: 1.15rem;
            color: #fbcfe8;
        }

        .stat-bar-container {
            display: flex;
            height: 6px;
            background: rgba(255,255,255,0.05);
            border-radius: 3px;
            grid-column: 1 / span 3;
            margin-top: 0.5rem;
            overflow: hidden;
        }

        .stat-bar-home {
            background: #6366f1;
            height: 100%;
        }

        .stat-bar-away {
            background: #ec4899;
            height: 100%;
        }

        /* custom predict button alignment */
        div.stButton > button {
            background: linear-gradient(90deg, #6366f1 0%, #ec4899 100%) !important;
            border: none !important;
            border-radius: 30px !important;
            color: #f3f4f6 !important;
            font-size: 1.25rem !important;
            font-weight: 700 !important;
            padding: 0.6rem 3rem !important;
            cursor: pointer !important;
            box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3) !important;
            transition: all 0.3s ease !important;
            display: block !important;
            margin: 2rem auto 1rem auto !important;
            width: fit-content !important;
        }

        div.stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 15px 30px rgba(236, 72, 153, 0.4) !important;
            filter: brightness(1.1) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Main Title banner
st.markdown('<div class="app-title">Premier League Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Predict match outcomes using team ELO, rolling stats, and betting odds</div>', unsafe_allow_html=True)

# Input Forms Layout
col_home, col_vs, col_away = st.columns([5, 1, 5], gap="large")

with col_home:
    st.markdown('<div class="glass-card home-card">', unsafe_allow_html=True)
    st.markdown('<h3><i class="fa-solid fa-house-chimney" style="color: #818cf8;"></i> Home Team</h3>', unsafe_allow_html=True)
    
    # Filter away team if selected
    home_team = st.selectbox(
        "Select Home Team",
        options=teams,
        index=0 if teams else None,
        key="home_team_select"
    )
    
    home_odds = st.number_input(
        "Home Win Odds (B365H)",
        min_value=1.01,
        value=1.85,
        step=0.01,
        format="%.2f",
        key="home_odds_input"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_vs:
    st.markdown('<div class="vs-divider"><div class="vs-circle">VS</div></div>', unsafe_allow_html=True)

with col_away:
    st.markdown('<div class="glass-card away-card">', unsafe_allow_html=True)
    st.markdown('<h3><i class="fa-solid fa-plane-departure" style="color: #f472b6;"></i> Away Team</h3>', unsafe_allow_html=True)
    
    # Filter home team from choices to avoid same team selection
    away_teams_options = [t for t in teams if t != home_team] if teams else []
    
    away_team = st.selectbox(
        "Select Away Team",
        options=away_teams_options,
        index=0 if away_teams_options else None,
        key="away_team_select"
    )
    
    away_odds = st.number_input(
        "Away Win Odds (B365A)",
        min_value=1.01,
        value=4.20,
        step=0.01,
        format="%.2f",
        key="away_odds_input"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Draw Odds Card in the center
_, col_draw_center, _ = st.columns([3, 4, 3])
with col_draw_center:
    st.markdown('<div class="glass-card" style="padding: 1.5rem; text-align: center;">', unsafe_allow_html=True)
    draw_odds = st.number_input(
        "Draw Odds (B365D)",
        min_value=1.01,
        value=3.60,
        step=0.01,
        format="%.2f",
        key="draw_odds_input"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Action button
predict_btn = st.button("🔮 Predict Match Outcome", use_container_width=False)

if predict_btn:
    if not home_team or not away_team:
        st.error("Please select both Home and Away teams.")
    elif home_team == away_team:
        st.error("Home and Away teams cannot be the same.")
    elif model is None or preprocessor is None:
        st.error("Model or Preprocessor is not loaded. Ensure training pipeline was run.")
    else:
        # Proceed with prediction
        with st.spinner("Analyzing match dynamics..."):
            home_stats = team_stats_dict.get(home_team)
            away_stats = team_stats_dict.get(away_team)
            
            # Probability calculation from odds
            h_p = 1.0 / home_odds
            d_p = 1.0 / draw_odds
            a_p = 1.0 / away_odds
            
            # Construct features DataFrame in exact order
            features = {
                'Home_ELO': home_stats['ELO'],
                'Away_ELO': away_stats['ELO'],
                'ELO_Difference': home_stats['ELO'] - away_stats['ELO'],
                
                'HomeRollingGoalsScored': home_stats['RollingGoalsScored'],
                'HomeRollingGoalsConceeded': home_stats['RollingGoalsConceeded'],
                'HomeRollingShots': home_stats['RollingShots'],
                'HomeRollingPoints': home_stats['RollingPoints'],
                'HomeRollingSTarget': home_stats['RollingSTarget'],
                'HomeRollingFreekick': home_stats['RollingFreekick'],
                'HomeRollingCorner': home_stats['RollingCorner'],
                'HomeTotalPoints': home_stats['TotalPoints'],
                
                'AwayRollingGoalsScored': away_stats['RollingGoalsScored'],
                'AwayRollingGoalsConceeded': away_stats['RollingGoalsConceeded'],
                'AwayRollingShots': away_stats['RollingShots'],
                'AwayRollingPoints': away_stats['RollingPoints'],
                'AwayRollingSTarget': away_stats['RollingSTarget'],
                'AwayRollingFreekick': away_stats['RollingFreekick'],
                'AwayRollingCorner': away_stats['RollingCorner'],
                'AwayTotalPoints': away_stats['TotalPoints'],
                
                'H_P': h_p,
                'D_P': d_p,
                'A_P': a_p
            }
            
            features_df = pd.DataFrame([features])
            
            # Apply preprocessing scaler
            scaled_features = preprocessor.transform(features_df)
            
            # Predict probabilities and outcome
            prediction_idx = int(model.predict(scaled_features)[0])
            probabilities = model.predict_proba(scaled_features)[0] # [Home, Draw, Away]
            
            outcome_mapping = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            predicted_outcome = outcome_mapping.get(prediction_idx, "Unknown")
            
            h_prob = round(probabilities[0] * 100, 2)
            d_prob = round(probabilities[1] * 100, 2)
            a_prob = round(probabilities[2] * 100, 2)
            
            # Winner Class style
            winner_class = "home-win" if prediction_idx == 0 else ("draw-win" if prediction_idx == 1 else "away-win")
            
            # 1. Output prediction card
            st.markdown(f"""
                <div class="glass-card">
                    <div class="winner-announcement">
                        <div class="winner-title">Predicted Outcome</div>
                        <div class="winner-name {winner_class}">{predicted_outcome.upper()}</div>
                    </div>
                    
                    <div class="prob-chart">
                        <!-- Home Win -->
                        <div class="prob-item">
                            <div class="prob-meta">
                                <span class="prob-label"><i class="fa-solid fa-circle" style="color: #6366f1; font-size: 0.75rem;"></i> Home Win Probability</span>
                                <span class="prob-value">{h_prob}%</span>
                            </div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill fill-home" style="width: {h_prob}%;"></div>
                            </div>
                        </div>
                        
                        <!-- Draw -->
                        <div class="prob-item">
                            <div class="prob-meta">
                                <span class="prob-label"><i class="fa-solid fa-circle" style="color: #10b981; font-size: 0.75rem;"></i> Draw Probability</span>
                                <span class="prob-value">{d_prob}%</span>
                            </div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill fill-draw" style="width: {d_prob}%;"></div>
                            </div>
                        </div>
                        
                        <!-- Away Win -->
                        <div class="prob-item">
                            <div class="prob-meta">
                                <span class="prob-label"><i class="fa-solid fa-circle" style="color: #ec4899; font-size: 0.75rem;"></i> Away Win Probability</span>
                                <span class="prob-value">{a_prob}%</span>
                            </div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill fill-away" style="width: {a_prob}%;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # 2. Output team stats comparison
            # Calculate percentages for ratio bars
            def get_ratios(val_h, val_a):
                tot = val_h + val_a
                if tot == 0:
                    return 50.0, 50.0
                return (val_h / tot) * 100, (val_a / tot) * 100
                
            elo_h_pct, elo_a_pct = get_ratios(home_stats['ELO'], away_stats['ELO'])
            goals_h_pct, goals_a_pct = get_ratios(home_stats['RollingGoalsScored'], away_stats['RollingGoalsScored'])
            conc_h_pct, conc_a_pct = get_ratios(home_stats['RollingGoalsConceeded'], away_stats['RollingGoalsConceeded'])
            ppg_h_pct, ppg_a_pct = get_ratios(home_stats['RollingPoints'], away_stats['RollingPoints'])
            tot_h_pct, tot_a_pct = get_ratios(home_stats['TotalPoints'], away_stats['TotalPoints'])
            
            # helper for conceeded color inversion
            conc_home_color = "#10b981" if home_stats['RollingGoalsConceeded'] < away_stats['RollingGoalsConceeded'] else "#ec4899"
            conc_away_color = "#10b981" if away_stats['RollingGoalsConceeded'] < home_stats['RollingGoalsConceeded'] else "#ec4899"

            st.markdown(f"""
                <div class="glass-card">
                    <div class="stats-header">
                        <i class="fa-solid fa-chart-simple"></i>
                        <span>Team Rolling History & Stats Comparison</span>
                    </div>
                    
                    <!-- ELO Row -->
                    <div class="stat-row">
                        <div class="stat-val-home">{round(home_stats['ELO'], 1)}</div>
                        <div class="stat-name">ELO Rating</div>
                        <div class="stat-val-away">{round(away_stats['ELO'], 1)}</div>
                        <div class="stat-bar-container">
                            <div class="stat-bar-home" style="width: {elo_h_pct}%;"></div>
                            <div class="stat-bar-away" style="width: {elo_a_pct}%;"></div>
                        </div>
                    </div>
                    
                    <!-- Avg Goals Row -->
                    <div class="stat-row">
                        <div class="stat-val-home">{round(home_stats['RollingGoalsScored'], 2)}</div>
                        <div class="stat-name">Avg Goals Scored</div>
                        <div class="stat-val-away">{round(away_stats['RollingGoalsScored'], 2)}</div>
                        <div class="stat-bar-container">
                            <div class="stat-bar-home" style="width: {goals_h_pct}%;"></div>
                            <div class="stat-bar-away" style="width: {goals_a_pct}%;"></div>
                        </div>
                    </div>
                    
                    <!-- Avg Conceded Row -->
                    <div class="stat-row">
                        <div class="stat-val-home">{round(home_stats['RollingGoalsConceeded'], 2)}</div>
                        <div class="stat-name">Avg Goals Conceded</div>
                        <div class="stat-val-away">{round(away_stats['RollingGoalsConceeded'], 2)}</div>
                        <div class="stat-bar-container">
                            <div class="stat-bar-home" style="width: {conc_h_pct}%; background-color: {conc_home_color};"></div>
                            <div class="stat-bar-away" style="width: {conc_a_pct}%; background-color: {conc_away_color};"></div>
                        </div>
                    </div>
                    
                    <!-- PPG Row -->
                    <div class="stat-row">
                        <div class="stat-val-home">{round(home_stats['RollingPoints'], 2)}</div>
                        <div class="stat-name">PPG (Last 5)</div>
                        <div class="stat-val-away">{round(away_stats['RollingPoints'], 2)}</div>
                        <div class="stat-bar-container">
                            <div class="stat-bar-home" style="width: {ppg_h_pct}%;"></div>
                            <div class="stat-bar-away" style="width: {ppg_a_pct}%;"></div>
                        </div>
                    </div>
                    
                    <!-- Total Points Row -->
                    <div class="stat-row">
                        <div class="stat-val-home">{int(home_stats['TotalPoints'])}</div>
                        <div class="stat-name">Total Season Points</div>
                        <div class="stat-val-away">{int(away_stats['TotalPoints'])}</div>
                        <div class="stat-bar-container">
                            <div class="stat-bar-home" style="width: {tot_h_pct}%;"></div>
                            <div class="stat-bar-away" style="width: {tot_a_pct}%;"></div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
