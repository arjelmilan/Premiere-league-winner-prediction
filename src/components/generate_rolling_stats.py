import os
import pandas as pd
import numpy as np

def generate_team_rolling_stats(raw_data_path, output_path, window=5):
    print("Loading raw match data...")
    df = pd.read_csv(raw_data_path)
    
    # Standardize dates and sort by date
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # 1. ELO Rating Calculation
    print("Calculating ELO ratings...")
    all_teams = set(df['HomeTeam']).union(set(df['AwayTeam']))
    elo_ratings = {team: 1500.0 for team in all_teams}
    
    K = 100  # ELO volatility factor
    
    for idx, row in df.iterrows():
        home, away, r = row['HomeTeam'], row['AwayTeam'], row['FTR']
        h_elo = elo_ratings[home]
        a_elo = elo_ratings[away]
        
        # Expected outcomes
        E_H = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        E_A = 1 / (1 + 10 ** ((h_elo - a_elo) / 400))
        
        # Actual outcomes
        S_H = 1.0 if r == 'H' else (0.5 if r == 'D' else 0.0)
        S_A = 1.0 if r == 'A' else (0.5 if r == 'D' else 0.0)
        
        # Update ratings
        elo_ratings[home] = h_elo + K * (S_H - E_H)
        elo_ratings[away] = a_elo + K * (S_A - E_A)
        
    # 2. Melt and Calculate Rolling stats (without shift, to get post-match values)
    print("Calculating rolling stats and season points...")
    df['HomePoints'] = df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    df['AwayPoints'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 3})
    
    home_df = df[['Date', 'Season', 'HomeTeam', 'FTHG', 'FTAG', 'HS', 'HomePoints', 'HST', 'HF', 'HC']].copy()
    home_df.columns = ['Date', 'Season', 'Team', 'GoalsScored', 'GoalsConceeded', 'Shots', 'Points', 'STarget', 'Freekick', 'Corner']
    
    away_df = df[['Date', 'Season', 'AwayTeam', 'FTAG', 'FTHG', 'AS', 'AwayPoints', 'AST', 'AF', 'AC']].copy()
    away_df.columns = ['Date', 'Season', 'Team', 'GoalsScored', 'GoalsConceeded', 'Shots', 'Points', 'STarget', 'Freekick', 'Corner']
    
    melted = pd.concat([home_df, away_df], ignore_index=True)
    melted = melted.sort_values(by=['Date']).reset_index(drop=True)
    
    # Note: No shift(1) here because we want the state of the team AFTER their last match
    melted['RollingGoalsScored'] = melted.groupby('Team')['GoalsScored'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    melted['RollingGoalsConceeded'] = melted.groupby('Team')['GoalsConceeded'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    melted['RollingShots'] = melted.groupby('Team')['Shots'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    melted['RollingPoints'] = melted.groupby('Team')['Points'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    melted['RollingSTarget'] = melted.groupby('Team')['STarget'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    melted['RollingFreekick'] = melted.groupby('Team')['Freekick'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    melted['RollingCorner'] = melted.groupby('Team')['Corner'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    
    # Cumulative points within the current season (including the last match)
    melted['TotalPoints'] = melted.groupby(['Season', 'Team'])['Points'].transform(lambda x: x.cumsum())
    
    # 3. Extract the last match for each team to get the latest state
    print("Compiling latest team stats...")
    latest_rows = melted.sort_values('Date').groupby('Team').last().reset_index()
    
    # Select columns we want
    stats_cols = [
        'Team', 'RollingGoalsScored', 'RollingGoalsConceeded', 'RollingShots',
        'RollingPoints', 'RollingSTarget', 'RollingFreekick', 'RollingCorner', 'TotalPoints'
    ]
    team_stats = latest_rows[stats_cols].copy()
    
    # Add ELO rating
    team_stats['ELO'] = team_stats['Team'].map(elo_ratings)
    
    # Ensure correct columns and order
    team_stats = team_stats[[
        'Team', 'ELO', 'RollingGoalsScored', 'RollingGoalsConceeded', 'RollingShots',
        'RollingPoints', 'RollingSTarget', 'RollingFreekick', 'RollingCorner', 'TotalPoints'
    ]]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    team_stats.to_csv(output_path, index=False)
    print(f"Latest rolling stats successfully saved to {output_path}")
    print(team_stats.head())

if __name__ == "__main__":
    generate_team_rolling_stats(
        raw_data_path="artifacts/raw.csv",
        output_path="artifacts/team_rolling_stats.csv"
    )
