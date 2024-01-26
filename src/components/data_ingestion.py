from dataclasses import dataclass
from src.logger import logging
import pandas as pd
import os
from src.utils.ingestion_utils import DataCleaning,concat_x_y
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

@dataclass
class Ingestion_files_dir():
    os.makedirs('artifacts/cleaned_data/',exist_ok=True)
    cleaned_data = 'artifacts/cleaned_data/final_data.csv'
    train_data = 'artifacts/cleaned_data/train_data.csv'
    test_data = 'artifacts/cleaned_data/test_data.csv'

class Ingestion():
    def __init__(self, file='/content/Phishing/data/dataset_full.csv'):
        self.file = file
        self.dir = Ingestion_files_dir()

    def ingestion(self):
        df = pd.read_csv(self.file)
        logging.info(f'Data Read Succesfully from {self.file}')
        logging.info('Initiating Data Ingestion Method')
        clean = DataCleaning(df, 0.8, 0.8)
        df = clean.feature_scaling_df()

        X = df.drop(columns='phishing',axis=1)
        y = df['phishing']

        smote = SMOTE(sampling_strategy='all', k_neighbors=2)
        X, y = smote.fit_resample(X,y)

        df = concat_x_y(X,y)
        logging.info(f"Smoting Completed Succesfully The Value counts are {df['phishing'].value_counts()}")
        #print(f"Smoting Completed Succesfully The Vale counts are {df['phishing'].value_counts()}")
        df.to_csv(self.dir.cleaned_data,index=False)

        logging.info(f'Clean Data Saved to {self.dir.cleaned_data}')

        X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.25, random_state=42)
        train_df = concat_x_y(X_train,y_train)
        train_df.to_csv(self.dir.train_data,index=False)
        logging.info(f'Training Dataset Saved at {self.dir.train_data}')

        test_df = concat_x_y(X_test,y_test)
        test_df.to_csv(self.dir.test_data,index=False)
        logging.info(f'Testing Dataset Saved at {self.dir.test_data}')
        logging.info('Completed Data Ingestion Method')
        return {
            'cleaned_data' : self.dir.cleaned_data,
            'train_data' : self.dir.train_data,
            'test_data' : self.dir.test_data
        }
                
if __name__=='__main__':
    ingestion_process = Ingestion('/content/Phishing/data/dataset_full.csv')
    dir = ingestion_process.ingestion()
    print(dir['cleaned_data'])
    print(dir['train_data'])
    print(dir['test_data'])