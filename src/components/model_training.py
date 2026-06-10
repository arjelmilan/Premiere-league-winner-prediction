from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from src.utils import evaluate_model
from dataclasses import dataclass
import os
from src.exception import CustomException
import sys
from src.utils import save_object
from src.logger import logging
from sklearn.metrics import accuracy_score

class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')



class ModelTraining:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def model_trainer(self,train_arr , test_arr):
        try:
            X_train , y_train , X_test , y_test = (train_arr[:,:-1],
                                                train_arr[:,-1],
                                                test_arr[:,:-1],
                                                test_arr[:,-1])
            
            models  = {'Random Forest': RandomForestClassifier(),
            'SVC': SVC(probability=True),
            'XGBClassifier': XGBClassifier(),
            'LGBMClassifier': LGBMClassifier()}


            best_model_name , best_model ,best_model_score,report = evaluate_model(X_train , y_train , X_test , y_test , models)
            print(report)
            print("Best Model:", best_model_name)
            print("Best Score:", best_model_score)

            # if best_score<0.6:
            #     raise CustomException("No best model found")
            # logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Trained model saved at path: {self.model_trainer_config.trained_model_file_path}")
            
            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
            

        except Exception as e:
                raise CustomException(e,sys)

    