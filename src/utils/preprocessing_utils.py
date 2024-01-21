import pandas as pd

def data_split(dir):
    df = pd.read_csv(dir)
    X = df.drop(columns='phishing')
    y = df['phishing'].astype('int')
    return X,y