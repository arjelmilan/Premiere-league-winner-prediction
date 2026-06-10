import pickle
import os
from sklearn.metrics import accuracy_score

def save_object(file_path:str,obj):
    dir_path=os.path.dirname(file_path)
    os.makedirs(dir_path,exist_ok=True)

    with open(file_path,"wb") as file_obj:
        pickle.dump(obj,file_obj)

def load_object(file_path:str):
    if not os.path.exists(file_path):
        raise Exception(f"The file: {file_path} does not exist")

    with open(file_path,"rb") as file_obj:
        return pickle.load(file_obj)
    
def evaluate_model(X_train , y_train , X_test , y_test , models):
    report = {}
    for model_name , model in models.items():
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test , y_pred)
        report[model_name] = accuracy

    best_model_name = max(report , key = report.get)
    best_model = models[best_model_name]
    best_model_score = report[best_model_name]

    return best_model_name , best_model ,best_model_score,report