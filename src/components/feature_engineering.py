import pandas as pd
from dataclasses import dataclass
import os


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')


class FeatureEngineering():
    def __init__(self,path,window=5):
        self.df = pd.read_csv(path)
        self.window = window
    
    def get_elo(self):
        all_teams = set(self.df['HomeTeam']).union(set(self.df['AwayTeam']))
        elo_ratings = {team: 1500 for team in all_teams}

            # Arrays to store ELO before the match happens
        home_elo_before = []
        away_elo_before = []

        K = 32 # ELO volatility factor

        for idx, row in self.df.iterrows():
            home, away, r = row['HomeTeam'], row['AwayTeam'], row['FTR']
                
            # Store current ELOs before the match
            h_elo = elo_ratings[home]
            a_elo = elo_ratings[away]
            home_elo_before.append(h_elo)
            away_elo_before.append(a_elo)
                
            # Expected outcomes
            E_H = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
            E_A = 1 / (1 + 10 ** ((h_elo - a_elo) / 400))
                
            # Actual outcomes
            S_H = 1 if r == 'H' else (0.5 if r == 'D' else 0)
            S_A = 1 if r == 'A' else (0.5 if r == 'D' else 0)
                
            # Update ratings
            elo_ratings[home] = h_elo + K * (S_H - E_H)
            elo_ratings[away] = a_elo + K * (S_A - E_A)

            # Assign ELO columns
        self.df['Home_ELO'] = home_elo_before
        self.df['Away_ELO'] = away_elo_before

    def get_elo_difference(self):
        self.df['ELO_Difference'] = self.df['Home_ELO']-self.df['Away_ELO']

    def get_individual_teams(self):
        self.df['HomePoints'] = self.df['FTR'].map({'H':3,'D':1,'A':0})
        self.df['AwayPoints'] = self.df['FTR'].map({'H':0,'D':1,'A':3})


        home_df = self.df[['Date','Season','HomeTeam', 'FTHG','FTAG', 'HS', 'HomePoints','HST','HF','HC']].copy()
        home_df.columns = ['Date','Season', 'Team', 'GoalsScored','GoalsConceeded' ,'Shots', 'Points','STarget','Freekick','Corner'] 
        home_df['IsHome'] = True


        away_df = self.df[['Date','Season', 'AwayTeam', 'FTAG','FTHG','AS', 'AwayPoints','AST','AF','AC']].copy()
        away_df.columns = ['Date', 'Season','Team', 'GoalsScored','GoalsConceeded', 'Shots', 'Points','STarget','Freekick','Corner'] 
        away_df['IsHome'] = False

        melted = pd.concat([home_df, away_df], ignore_index=True)


        melted = melted.sort_values(by=['Date']).reset_index(drop=True)

        return melted
    
    def get_rolling_features(self,fill_values=None,window=5):
        melted = self.get_individual_teams()
        melted['RollingGoalsScored'] = melted.groupby('Team')['GoalsScored'].transform(lambda x: x.rolling(window , min_periods=1).mean().shift(1))
        melted['RollingGoalsConceeded'] = melted.groupby('Team')['GoalsConceeded'].transform(lambda x: x.rolling(window , min_periods=1).mean().shift(1))
        melted['RollingShots'] = melted.groupby('Team')['Shots'].transform(lambda x: x.rolling(window , min_periods=1).mean().shift(1))
        melted['RollingPoints'] = melted.groupby('Team')['Points'].transform(lambda x: x.rolling(window , min_periods=1).mean().shift(1))
        melted['RollingSTarget'] = melted.groupby('Team')['STarget'].transform(lambda x: x.rolling(window , min_periods=1).mean().shift(1))
        melted['RollingFreekick'] = melted.groupby('Team')['Freekick'].transform(lambda x: x.rolling(window , min_periods=1).mean().shift(1))
        melted['RollingCorner'] = melted.groupby('Team')['Corner'].transform(lambda x: x.rolling(window , min_periods=1).mean().shift(1))
        melted['TotalPoints'] = melted.groupby(['Season','Team'])['Points'].transform(lambda x:x.cumsum().shift(1))
        # melted['TotalGoalsScored'] = melted.groupby(['Season','Team'])['GoalsScored'].transform(lambda x:x.cumsum().shift(1))
        # melted['TotalGoalsConceeded'] = melted.groupby(['Season','Team'])['GoalsConceeded'].transform(lambda x:x.cumsum().shift(1))


        return melted

    def get_win_probabilities(self):
        self.df['H_P'] = 1/self.df['B365H']
        self.df['D_P'] = 1/self.df['B365D']
        self.df['A_P'] = 1/self.df['B365A']
    
    def drop_columns(self ,drop_cols):
        self.df.drop(columns = drop_cols , inplace = True)

    def get_final_df(self,fill_values=None):
        # Calculate ELO and ELO differences first
        self.get_elo()
        self.get_elo_difference()

        # Compute rolling features
        melted = self.get_rolling_features()
        home_stats = melted[melted['IsHome']].drop(columns=['GoalsScored','GoalsConceeded','Season', 'Shots', 'Points','IsHome','STarget','Freekick','Corner'])
        away_stats = melted[~melted['IsHome']].drop(columns=['GoalsScored','GoalsConceeded','Season', 'Shots', 'Points','IsHome','STarget','Freekick','Corner'])

        home_stats.columns = ['Date', 'HomeTeam','HomeRollingGoalsScored','HomeRollingGoalsConceeded','HomeRollingShots', 'HomeRollingPoints','HomeRollingSTarget','HomeRollingFreekick','HomeRollingCorner','HomeTotalPoints']
        away_stats.columns = ['Date', 'AwayTeam','AwayRollingGoalsScored','AwayRollingGoalsConceeded','AwayRollingShots', 'AwayRollingPoints','AwayRollingSTarget','AwayRollingFreekick','AwayRollingCorner','AwayTotalPoints']

        self.df = self.df.merge(home_stats,how = 'left' , on = ['Date','HomeTeam'])
        self.df = self.df.merge(away_stats,how = 'left' , on = ['Date','AwayTeam'])
        
        # Define default fill values for rolling stats
        if not fill_values:
            fill_values = {
                'HomeRollingPoints': 1.52, 'AwayRollingPoints': 1.24,
                'HomeRollingGoalsScored': 1.53, 'AwayRollingGoalsScored': 1.35,
                'HomeRollingGoalsConceeded': 1.35, 'AwayRollingGoalsConceeded': 1.53, 
                'HomeRollingShots': 12.59, 'AwayRollingShots': 12.70,
                'HomeRollingSTarget': 4.37, 'AwayRollingSTarget': 4.47,
                'HomeRollingFreekick': 11.08, 'AwayRollingFreekick': 11.0,
                'HomeRollingCorner': 5.08, 'AwayRollingCorner': 5.13,
                'HomeTotalPoints': 0, 'AwayTotalPoints': 0
            }
        
        self.df = self.df.fillna(value=fill_values)

        # Compute probabilities from betting odds
        self.get_win_probabilities()

        # Map Target variable
        label2idx = {'H': 0, 'D': 1, 'A': 2}
        self.df['Target'] = self.df['FTR'].map(label2idx)

        # Select only the features and target we need
        features_and_target = [
            'Home_ELO', 'Away_ELO', 'ELO_Difference',
            'HomeRollingGoalsScored', 'HomeRollingGoalsConceeded', 'HomeRollingShots',
            'HomeRollingPoints', 'HomeRollingSTarget', 'HomeRollingFreekick', 'HomeRollingCorner', 'HomeTotalPoints',
            'AwayRollingGoalsScored', 'AwayRollingGoalsConceeded', 'AwayRollingShots',
            'AwayRollingPoints', 'AwayRollingSTarget', 'AwayRollingFreekick', 'AwayRollingCorner', 'AwayTotalPoints',
            'H_P', 'D_P', 'A_P', 'Target'
        ]
        self.df = self.df[features_and_target]

        # Drop any row that still has NaNs in the selected columns (e.g. if betting odds were missing)
        self.df.dropna(inplace=True)

        return self.df


                
                    
