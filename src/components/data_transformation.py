import pandas as pd
from dataclasses import dataclass
import os
import pickle as pkl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import sys

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException   
from src.components.feature_engineering import FeatureEngineering


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')



class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        columns = [
            'Home_ELO', 'Away_ELO', 'ELO_Difference',
            'HomeRollingGoalsScored', 'HomeRollingGoalsConceeded', 'HomeRollingShots',
            'HomeRollingPoints', 'HomeRollingSTarget', 'HomeRollingFreekick', 'HomeRollingCorner', 'HomeTotalPoints',
            'AwayRollingGoalsScored', 'AwayRollingGoalsConceeded', 'AwayRollingShots',
            'AwayRollingPoints', 'AwayRollingSTarget', 'AwayRollingFreekick', 'AwayRollingCorner', 'AwayTotalPoints',
            'H_P', 'D_P', 'A_P'
        ]
        
        pipeline = Pipeline(
            steps=[('scaler', StandardScaler())]
        )

        preprocessor = ColumnTransformer([
            ('pipeline', pipeline, columns)
        ])

        return preprocessor

    def get_transformation(self,train_df , test_df):

        try:
            # Handle if FeatureEngineering objects are passed instead of DataFrames
            if hasattr(train_df, 'get_final_df'):
                train_df = train_df.get_final_df()
            if hasattr(test_df, 'get_final_df'):
                test_df = test_df.get_final_df()

            preprocessor_obj = self.get_transformer_object()

            target_column_name = 'Target'

            train_df_target = train_df[target_column_name]
            test_df_target = test_df[target_column_name]

            train_df_features = train_df.drop(columns=[target_column_name])
            test_df_features = test_df.drop(columns=[target_column_name])

            train_preprocessed = preprocessor_obj.fit_transform(train_df_features)
            test_preprocessed = preprocessor_obj.transform(test_df_features)

            train_arr = np.c_[train_preprocessed, np.array(train_df_target)]
            test_arr = np.c_[test_preprocessed, np.array(test_df_target)]

            save_object(
                self.data_transformation_config.preprocessor_obj_path,
                preprocessor_obj
            )

            return (train_arr ,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_path
                    )
        except Exception as e:
            raise CustomException(e,sys)







            




