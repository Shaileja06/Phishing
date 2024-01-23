from dataclasses import dataclass
import os
import joblib
from src.utils.preprocessing_utils import data_split
from src.logger import logging
from src.components.data_ingestion import Ingestion
from src.components.data_preprocessing import Preprocessing
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials,space_eval
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score , precision_score, fbeta_score, confusion_matrix

@dataclass
class Model_Training_files_dir():
    os.makedirs('artifacts/model',exist_ok=True)
    model_train_dir = 'artifacts/model/model.joblib'


class Model_Training():
    def __init__(self, X, y, standar_scalar_dir, pca_dir, test_data,best_params=None):
        self.X = X
        self.y = y 
        self.standar_scalar_dir = standar_scalar_dir
        self.pca_dir = pca_dir
        self.test_data = test_data
        self.best_params = best_params
        self.dir = Model_Training_files_dir()

    def objective(self,space):
        model = XGBClassifier(
            learning_rate = space['learning_rate'],
            max_depth = space['max_depth'],
            min_child_weight = space['min_child_weight'],
            subsample = space['subsample'],
            gamma = space['gamma'],
            colsample_bytree = space['colsample_bytree'],
            n_estimators = space['n_estimators']
        )
        accuracy = cross_val_score(model, self.X, self.y, cv = 5).mean()

        # We aim to maximize accuracy, therefore we return it as a negative value
        return {'accuracy': accuracy, 'status': STATUS_OK }

    def start_hyperparameter_tunning(self):
        space = {
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'max_depth': hp.choice('max_depth', range(3, 15)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'gamma': hp.uniform('gamma', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'n_estimators': hp.choice('n_estimators', range(50, 500)),
        }
        trials = Trials()
        best = fmin(fn= self.objective,space= space,algo= tpe.suggest, max_evals = 80,trials= trials)
        # Retrieve the best parameters
        best_params = space_eval(space, best)
        logging.info(f"Best Hyperparameters:{best_params}")
        print("Best Hyperparameters:")
        print(best_params)
        return best_params

    def start_training(self,best_params):
        if best_params:
            self.best_params = best_params
        else:
            self.best_params = self.start_hyperparameter_tunning()
    
        xgb_hyp = XGBClassifier(
            learning_rate=self.best_params['learning_rate'],
            n_estimators =self.best_params['n_estimators'],
            max_depth=self.best_params['max_depth'],
            min_child_weight=self.best_params['min_child_weight'],
            subsample=self.best_params['subsample'],
            colsample_bytree=self.best_params['colsample_bytree']
        )

        xgb_hyp.fit(self.X, self.y)
        joblib.dump(xgb_hyp,self.dir.model_train_dir)

        # Evaluate the performance of the XGBoost classifier
        y_pred_xgb_hyp = xgb_hyp.predict(self.X)
        accuracy = accuracy_score(self.y,y_pred_xgb_hyp)
        precision = precision_score(self.y, y_pred_xgb_hyp)
        conf_matrix = confusion_matrix(self.y, y_pred_xgb_hyp)
        recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
        f2_score = fbeta_score(self.y, y_pred_xgb_hyp, beta=2)
        print('Trainind Data')
        print('Accuracy of XGBClassifier using HyperOPT:', accuracy)
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F2 Score: {f2_score}')

        #Test Data 
        xt, yt = data_split(self.test_data)
        print(f'Test Data Loaded Successfully')

        # Load scaler and pca models
        pca = joblib.load(self.pca_dir)
        print(f'PCA Model loaded Successfully')
        xt = pca.transform(xt)

        scaler = joblib.load(self.standar_scalar_dir)
        print(f'Standardization Model loaded Successfully')
        xt = scaler.transform(xt)

        y_pred_test = xgb_hyp.predict(xt)
        test_acc = accuracy_score(yt, y_pred_test)
        precision = precision_score(yt, y_pred_test)
        conf_matrix = confusion_matrix(yt, y_pred_test)
        recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
        f2_score = fbeta_score(yt, y_pred_test, beta=2)
        print('Testing Data')
        print('Accuracy of XGBClassifier using HyperOPT:', test_acc)
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F2 Score: {f2_score}')

        return self.dir.model_train_dir 
    
if __name__=='__main__':
    ingestion_process = Ingestion('/content/Phishing/data/dataset_full.csv')
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