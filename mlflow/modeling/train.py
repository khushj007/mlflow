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
from hyperopt import  hp , fmin, tpe, space_eval ,Trials ,STATUS_OK
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_data(path):
    return pd.read_csv(path / "train_processed.csv")


def best_model(X_train, X_test, y_train, y_test):
    space = hp.choice('classifier_type', [
        {
            'type': 'RandomForestClassifier',
            'n_estimators': hp.quniform('rf_n_estimators', 10, 100, 10),
            'max_depth': hp.quniform('rf_max_depth', 10, 100, 10),
            'max_features': hp.choice('rf_max_features', ['sqrt', 'log2', None]),
            'estimator': RandomForestClassifier
        },
        {
            'type': 'DecisionTreeClassifier',
            'criterion': hp.choice('dt_criterion', ['gini', 'entropy']),
            'max_depth': hp.quniform('dt_max_depth', 10, 60, 10),
            'estimator': DecisionTreeClassifier
        }
    ])
    
    choice_maps = {
        'rf_max_features': ['sqrt', 'log2', None],
        'dt_criterion': ['gini', 'entropy']
    }
    
    experiment_name = sys.argv[1]
    mlflow.set_experiment(experiment_name)
    
    trials = Trials()
    with mlflow.start_run() as parent:


        def objective(params):
            type = params.pop("type")
            estimator = params.pop("estimator")

            if "max_depth" in params :
                params["max_depth"] = int(params["max_depth"])
            if "n_estimators" in params :
                params["n_estimators"] = int(params["n_estimators"])

            model = estimator(**params)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)

            return {"loss":-accuracy,"status":STATUS_OK,"estimator":estimator,"type":type}
        
        best = fmin(objective,space,tpe.suggest,5,trials=trials)



        best_params = {}

        if trials.best_trial["result"]["type"] == "RandomForestClassifier":
            for k , v in list(best.items())[1:]:
                if type(v) == int :
                    best_params[k[3:]] = choice_maps[k][v]
                else :
                    best_params[k[3:]] = int(v)

        if trials.best_trial["result"]["type"] == "DecisionTreeClassifier":
            for k , v in list(best.items())[1:]:
                if type(v) == int :
                    best_params[k[3:]] = choice_maps[k][v]
                else :
                    best_params[k[3:]] = int(v)

        estimator = trials.best_trial["result"]["estimator"](**best_params)
        estimator.fit(X_train,y_train)

        for i in trials.trials:
            with mlflow.start_run(nested=True) as child :
                mlflow.log_metric("accuracy" ,-i["result"]["loss"])
                keys_to_remove = [k for k, v in i["misc"]["vals"].items() if not v]
                for k in keys_to_remove:
                    del i["misc"]["vals"][k]
                mlflow.log_params(i["misc"]["vals"])
                mlflow.log_param("type",i["result"]["type"])


        mlflow.log_metric("accuracy" ,-trials.best_trial["result"]["loss"])
        mlflow.log_params(best_params)
        mlflow.log_param("type",trials.best_trial["result"]["type"])

    return estimator , trials.best_trial["result"]["type"] 
        

    
        

def save_model(path,model,estimator_name):
    pathlib.Path(path).mkdir(parents=True,exist_ok=True)
    model_name=estimator_name+"-model.joblib"
    joblib.dump(model,path / model_name)





def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    data_path = home_dir / "data" / "processed"
    params_path = home_dir / "params.yaml"
    models_path = home_dir / "models"


    params = yaml.safe_load(open(params_path))["model"]

    df = get_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=params["test_split"], random_state=42)

    model , estimator_name = best_model(X_train, X_test, y_train, y_test )
    
    save_model(models_path,model,estimator_name)

if __name__ == "__main__":
    main()