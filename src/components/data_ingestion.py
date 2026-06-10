import os
from src.logger import logging
from dataclasses import dataclass , field
import requests
from io import StringIO
import pandas as pd


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_link: list[str] = field(default_factory=lambda: [
        "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2021/E0.csv",
        "https://www.football-data.co.uk/mmz4281/1920/E0.csv"
    ])
    raw_data_path:str=os.path.join("artifacts","raw.csv")


class GetData:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def download_data(self):
        try:
            logging.info("Starting data download...")
            data_frames = []
            for url in self.ingestion_config.raw_data_link:
                response = requests.get(url)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text))
                df['Season'] = str(url.split('/')[-2])
                df['Season'] = df['Season'].astype(str)
                data_frames.append(df)

            raw_data = pd.concat(data_frames, ignore_index=True)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            raw_data.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info(f"Data downloaded successfully and saved to {self.ingestion_config.raw_data_path}")
        except Exception as e:
            logging.error(f"Error occurred while downloading data: {e}")
            raise e
        
class DataIngestion:
    """class for splitting the data into train and test where data from 2024-2025 season will be used for training and 
        data from 2025-2026 season will be used for testing"""
    
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.get_data = GetData()

    def initiate_data_ingestion(self):
        try:
            logging.info("Initiating data ingestion...")
            self.get_data.download_data()
            raw_data = pd.read_csv(self.ingestion_config.raw_data_path)
            raw_data['Date'] = pd.to_datetime(raw_data['Date'], dayfirst=True)
            raw_data = raw_data.sort_values(by='Date',ascending=False)

            train_data = raw_data[760:]
            test_data = raw_data[:760]

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info(f"Data ingestion completed successfully. Train data saved to {self.ingestion_config.train_data_path} and Test data saved to {self.ingestion_config.test_data_path}")


            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {e}")
            raise e

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    train_fe = FeatureEngineering(train_data_path)
    test_fe = FeatureEngineering(test_data_path)

    data_transformation = DataTransformation()
    train_arr , test_arr , preprocessor_path = data_transformation.get_transformation(train_fe,test_fe)

    model_trainer = ModelTraining()
    r2_score = model_trainer.model_trainer(train_arr , test_arr)
    print("R2 Square:", r2_score)
