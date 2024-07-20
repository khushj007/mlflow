import numpy as np
import pandas as pd
import pathlib 
from features import feature_eng
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
import joblib
import sys
import mlflow
import matplotlib.pyplot as plt
import os


def get_data(path):
    return pd.read_csv(path / "test.csv")

def plot_cm(df,model,filename="default"):
    y_true = df.iloc[:,-1]
    y_pred = model.predict(df.iloc[:,:-1])
    
    fig, ax = plt.subplots()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    disp.plot(ax=ax)
    

    # with mlflow.start_run():
    #     mlflow.log_figure(fig, "confusionmatrix/fig.png")


    
    image_name = filename + 'confusion_matrix.png'
    save_directory = pathlib.Path(__file__).parent.parent / "reports/figures"
    save_directory.mkdir(parents=True, exist_ok=True)
    save_image_path = save_directory / image_name
    disp.figure_.savefig(save_image_path)
    plt.close(disp.figure_)




def main():
   

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent
    data_path = home_dir / "data" / "interim"
    params_path = home_dir / "params.yaml"

    model_directory = home_dir / "models"
    files = os.listdir(model_directory)
    model_files = [f for f in files if f.endswith('.joblib')]
    model_path = os.path.join(model_directory, model_files[0])

    model = joblib.load(model_path)

    df = get_data(data_path)
    feature_eng(df)

    plot_cm(df,model,model_files[0])


if __name__ == "__main__":
    main()