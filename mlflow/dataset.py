import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pathlib
import yaml

def get_data(path):
    df = pd.read_csv(path/"iris.csv")
    return df

def split_data(params,df):
    return train_test_split(df,test_size=params["split"])

def save_data(train,test,save_path):
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(save_path / 'train.csv', index=False)
    test.to_csv(save_path / 'test.csv', index=False)



def main():

    home_dir = pathlib.Path(__file__).parent.parent
    data_path = home_dir / "data"/ "raw"
    params_file = home_dir / "params.yaml"
    save_path = home_dir / "data" / "interim"

    params = yaml.safe_load(open(params_file))["dataset"]

    df = get_data(data_path)
    X,y = split_data(params,df)
    save_data(X,y,save_path)

if __name__ == "__main__":
    main()