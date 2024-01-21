from dataclasses import dataclass
from tpot import TPOTClassifier
import os
import joblib
from src.utils.preprocessing_utils import data_split
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import Ingestion
from src.utils.model_training_utils import train_model,loadmodel
from src.components.data_preprocessing import Preprocessing

@dataclass
class Model_Training_files_dir():
    os.makedirs('artifacts/model',exist_ok=True)
    model_train_dir = 'artifacts/model/model.joblib'


class Model_Training():
    def __init__(self, X, y, standar_scalar_dir, pca_dir, test_data):
        logging.info(f'{X.shape} {y.shape}')
        logging.info(f'{X}')
        self.X = X
        self.y = y 
        self.standar_scalar_dir = standar_scalar_dir
        self.pca_dir = pca_dir
        self.test_data = test_data
        self.dir = Model_Training_files_dir()

    def start_training(self):
        logging.info('Model Training Initiated')
        dir2 = train_model(self.X,self.y,self.dir.model_train_dir)

        logging.info('Model Training Completed')

        # Test Data Load and Processing
        xt, yt = data_split(self.test_data)
        logging.info(f'Test Data Loaded Successfully')

        # Load scaler and pca models
        scaler = joblib.load(self.standar_scalar_dir)
        logging.info(f'Standardization Model loaded Successfully')
        xt = scaler.transform(xt)

        pca = joblib.load(self.pca_dir)
        logging.info(f'PCA Model loaded Successfully')
        xt = pca.transform(xt)

        # Load tpot model
        tpot = joblib.load(dir2)

        accuracy = tpot.score(xt, yt)
        logging.info(f"Test Accuracy: {accuracy}")

        return self.dir.model_train_dir 
    
if __name__=='__main__':
    ingestion_process = Ingestion('data\dataset_small.csv')
    dir = ingestion_process.ingestion()
    print(dir['cleaned_data'])
    print(dir['train_data'])
    print(dir['test_data'])    

    preprocess_process = Preprocessing(dir['train_data'],dir['test_data']) 
    data = preprocess_process.preprocessing()
    print(data['standar_scalar_dir'])
    print(data['pca_dir'])

    model_training_process = Model_Training(data['X'],data['y'],data['standar_scalar_dir'],data['pca_dir'],dir['test_data'])
    model_dir = model_training_process.start_training()
    print(model_dir)