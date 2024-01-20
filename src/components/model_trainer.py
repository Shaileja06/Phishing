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
    def __init__(self,X,y,standar_scalar_dir,pca_dir,test_data):
        self.X = X
        self.y = y
        self.standar_scalar_dir = standar_scalar_dir
        self.pca_dir = pca_dir
        self.test_data = test_data
        self.dir = Model_Training_files_dir()
        
    def start_training(self):
        logging.info('Model Training Initiated')
        
        try:
            #tpot = train_model(self.X, self.y)
            tpot_config = {
                'xgboost.XGBClassifier': {
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'n_estimators': range(50, 500),
                    'max_depth': range(3, 10),
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
                    'min_child_weight': range(1, 10)
                }
            }

            tpot = TPOTClassifier(generations=5, population_size=20, config_dict=tpot_config, verbosity=2, random_state=42, scoring='accuracy')
            tpot.fit(self.X, self.y)
            # Saving the trained TPOT classifier to a file in .joblib
            joblib.dump(tpot.fitted_pipeline_, self.dir.model_train_dir)
            logging.info('Model Training Completed')

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            return None  # Return None or handle the error as needed

        # Test Data Load and Processing
        xt, yt = data_split(self.test_data)
        logging.info(f'Test Data Loaded Succesfully')

        # Load scaler and pca models
        scaler = joblib.load(self.standar_scalar_dir)
        logging.info(f'Standardization Model loaded Successfully')
        xt = scaler.transform(xt)

        pca = joblib.load(self.pca_dir)
        logging.info(f'PCA Model loaded Successfully')
        xt = pca.transform(xt)

        # Load tpot model
        #tpot = joblib.load(self.dir.model_train_dir)

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