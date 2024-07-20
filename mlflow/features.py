import pandas as pd 
import numpy as np
import pathlib 
from sklearn.preprocessing import LabelEncoder


def get_data(path):
    return pd.read_csv(path)

def feature_eng(df):
    le = LabelEncoder()
    df["Species"]=le.fit_transform(df["Species"])

def save_data(save_path,df):
    pathlib.Path(save_path).mkdir(parents = True, exist_ok = True)
    df.to_csv(save_path / "train_processed.csv" , index = False)

def main():
    curr_path = pathlib.Path(__file__)
    home_dir = curr_path.parent.parent
    data_path = home_dir / "data" / "interim" / "train.csv"
    save_path = home_dir / "data" / "processed"
    df = get_data(data_path)
    feature_eng(df)
    save_data(save_path,df)



if __name__ == "__main__":
    main()


