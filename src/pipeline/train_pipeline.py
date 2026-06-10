import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.feature_engineering import FeatureEngineering
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

class TrainPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            logging.info("Starting training pipeline...")
            
            # 1. Ingest Data (Downloads and splits raw.csv into train.csv and test.csv)
            logging.info("Step 1: Data Ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            
            # 2. Feature Engineering & Transformation
            logging.info("Step 2 & 3: Feature Engineering and Transformation")
            train_fe = FeatureEngineering(train_data_path)
            test_fe = FeatureEngineering(test_data_path)
            
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.get_transformation(train_fe, test_fe)
            
            # 3. Model Training
            logging.info("Step 4: Model Training")
            model_trainer = ModelTraining()
            accuracy = model_trainer.model_trainer(train_arr, test_arr)
            
            print(f"Training pipeline completed successfully. Best Model Accuracy: {accuracy:.4f}")
            logging.info(f"Training pipeline completed successfully. Best Model Accuracy: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            logging.error(f"Error in TrainPipeline: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run()
