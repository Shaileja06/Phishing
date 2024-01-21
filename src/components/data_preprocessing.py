from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import joblib
from dataclasses import dataclass
#from src.utils.ingestion_utils import concat_x_y
from src.utils.preprocessing_utils import data_split
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import Ingestion
import pandas as pd

@dataclass
class Preprocessing_files_dir():
    os.makedirs('artifacts/components',exist_ok=True)
    standar_scalar_dir = 'artifacts/components/standard.joblib'
    pca_dir = 'artifacts/components/pca.joblib'

class Preprocessing():
    def __init__(self,train_data,test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.dir = Preprocessing_files_dir()

    def preprocessing(self):
        logging.info('Initiating Data Preprocessing')
        #  Train data spliting to X,y
        X,y = data_split(self.train_data)

        # Standarization of the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        with open(self.dir.standar_scalar_dir,'wb') as f:
            joblib.dump(scaler,f)
        logging.info(f'Standardization Process Completed and Saved at {self.dir.standar_scalar_dir}')

        # PCA of the features
        pca = PCA(n_components=20)
        X = pca.fit_transform(X)
        with open(self.dir.pca_dir,'wb') as f:
            joblib.dump(pca,f)
        logging.info('PCA Process Completed')
        return X,y,self.dir.standar_scalar_dir,self.dir.pca_dir
        '''return {
            'X':X,
            'y':y,
            'standar_scalar_dir':self.dir.standar_scalar_dir,
            'pca_dir':self.dir.pca_dir
        }'''


if __name__=='__main__':
    ingestion_process = Ingestion('data\dataset_small.csv')
    dir = ingestion_process.ingestion()
    #print(dir['cleaned_data'])
    #print(dir['train_data'])
    #print(dir['test_data'])    

    preprocess_process = Preprocessing(dir['train_data'],dir['test_data']) 
    data = preprocess_process.preprocessing()
    ''' print(data['standar_scalar_dir'])
    print(data['pca_dir'])
    print(data['X'])
    print(data['y'])'''

    #print(data['X'].shape,data['y'].shape,len(data['y']))


