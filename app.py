import os
import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils import load_object

# Initialize FastAPI App
app = FastAPI(
    title="Premier League Match Outcome Predictor",
    description="A FastAPI backend to predict Premier League match outcomes using historical stats, ELO ratings, and betting odds.",
    version="1.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File Paths
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")
TEAM_STATS_PATH = os.path.join("artifacts", "team_rolling_stats.csv")
TEMPLATES_DIR = "templates"

# Load ML components
try:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Model or Preprocessor files not found in artifacts. Please run the training pipeline first.")
    
    model = load_object(MODEL_PATH)
    preprocessor = load_object(PREPROCESSOR_PATH)
    print("Model and Preprocessor loaded successfully.")
except Exception as e:
    print(f"Error loading ML components: {e}")
    model = None
    preprocessor = None

# Load Team Rolling Stats Database
try:
    if not os.path.exists(TEAM_STATS_PATH):
        raise FileNotFoundError(f"Team rolling stats file not found at {TEAM_STATS_PATH}. Please run the generator first.")
    
    team_stats_df = pd.read_csv(TEAM_STATS_PATH)
    # Create lookup dict: team name -> stats dict
    team_stats_dict = team_stats_df.set_index("Team").to_dict(orient="index")
    print(f"Loaded statistics for {len(team_stats_dict)} Premier League teams.")
except Exception as e:
    print(f"Error loading team rolling stats: {e}")
    team_stats_dict = {}

# Pydantic schemas
class PredictRequest(BaseModel):
    home_team: str = Field(..., example="Arsenal")
    away_team: str = Field(..., example="Chelsea")
    b365h: float = Field(..., gt=1.0, description="Bet365 Home Win Odds", example=1.85)
    b365d: float = Field(..., gt=1.0, description="Bet365 Draw Odds", example=3.60)
    b365a: float = Field(..., gt=1.0, description="Bet365 Away Win Odds", example=4.20)

@app.get("/teams")
async def get_teams():
    """Return the list of all available teams in our statistics database."""
    if not team_stats_dict:
        return JSONResponse(status_code=500, content={"error": "Stats database is not loaded."})
    return sorted(list(team_stats_dict.keys()))

@app.post("/predict")
async def predict_match(request: PredictRequest):
    """Predict the match outcome and return winning probabilities."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Prediction models are not loaded on the server.")
        
    home = request.home_team
    away = request.away_team
    
    if home == away:
        raise HTTPException(status_code=400, detail="Home team and Away team cannot be the same.")
        
    # Check if teams exist in stats dictionary
    home_stats = team_stats_dict.get(home)
    away_stats = team_stats_dict.get(away)
    
    if not home_stats:
        raise HTTPException(status_code=404, detail=f"Statistics for home team '{home}' not found.")
    if not away_stats:
        raise HTTPException(status_code=404, detail=f"Statistics for away team '{away}' not found.")
        
    try:
        # Calculate winning probabilities from betting odds
        h_p = 1.0 / request.b365h
        d_p = 1.0 / request.b365d
        a_p = 1.0 / request.b365a
        
        # Build feature dictionary in the exact order required
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
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Preprocess features (scaling)
        scaled_features = preprocessor.transform(features_df)
        
        # Run classification model
        prediction_idx = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0].tolist()  # [Home, Draw, Away]
        
        outcome_mapping = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        predicted_outcome = outcome_mapping.get(prediction_idx, "Unknown")
        
        response_data = {
            "home_team": home,
            "away_team": away,
            "prediction": predicted_outcome,
            "probabilities": {
                "Home Win": round(probabilities[0] * 100, 2),
                "Draw": round(probabilities[1] * 100, 2),
                "Away Win": round(probabilities[2] * 100, 2)
            },
            "stats": {
                "home": {
                    "ELO": round(home_stats['ELO'], 1),
                    "RollingGoalsScored": round(home_stats['RollingGoalsScored'], 2),
                    "RollingGoalsConceeded": round(home_stats['RollingGoalsConceeded'], 2),
                    "RollingPoints": round(home_stats['RollingPoints'], 2),
                    "TotalPoints": int(home_stats['TotalPoints'])
                },
                "away": {
                    "ELO": round(away_stats['ELO'], 1),
                    "RollingGoalsScored": round(away_stats['RollingGoalsScored'], 2),
                    "RollingGoalsConceeded": round(away_stats['RollingGoalsConceeded'], 2),
                    "RollingPoints": round(away_stats['RollingPoints'], 2),
                    "TotalPoints": int(away_stats['TotalPoints'])
                }
            }
        }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serve the single-page HTML frontend."""
    html_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h2>Error: Frontend template index.html not found.</h2>", status_code=404)
    with open(html_path, "r") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
