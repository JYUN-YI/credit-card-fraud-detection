import pandas as pd

def load_data(nrows=10000):
    DATA_PATH = "data/creditcard.csv"
    df = pd.read_csv(DATA_PATH, nrows=nrows)
    return df
