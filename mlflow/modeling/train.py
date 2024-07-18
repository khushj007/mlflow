import numpy as np
import pandas as pd
import pathlib
import yaml
import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.model_selection import GridSearchCV
import joblib


def get_data(path):
    return pd.read_csv(path / "train_processed.csv")

def best_model(x , y):
    hyperparameters = {
        "RandomForestClassifier": {
            "n_estimators": [10, 15, 20],
            "max_depth": [6, 8, 10],
            "max_features": ["sqrt", "log2", None],
        },
        "DecisionTreeClassifier": {
            "criterion": ["gini","entropy"],
            "max_depth":[6, 8, 10],
                
        },
    }
        
    models = {
        "RandomForestClassifier": RandomForestClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        }
    experiment_name = sys.argv[1]
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()
    best_model=None
    best_score=-1
    with mlflow.start_run() :
        for i in hyperparameters:
            model = GridSearchCV(estimator=models[i](),param_grid=hyperparameters[i],scoring={
                "accuracy":"accuracy",
                'f1_macro': 'f1_macro'},
                refit="f1_macro"
                ) 
            model.fit(x,y)
            if model.best_score_ > best_score :
                best_model = model.best_estimator_
                best_score = model.best_score_
    
    return best_model

def save_model(path,model):
    pathlib.Path(path).mkdir(parents=True,exist_ok=True)
    model_name=sys.argv[1]+"-model.joblib"
    joblib.dump(model,path / model_name)





def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_path = home_dir / "data" / "processed"
    params_path = home_dir / "params.yaml"
    models_path = home_dir / "models"


    params = yaml.safe_load(open(params_path))["model"]

    df = get_data(data_path)

    X = df.drop(columns="Species")
    y = df["Species"]

    model = best_model(X,y)
    
    save_model(models_path,model)

if __name__ == "__main__":
    main()