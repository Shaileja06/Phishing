import pandas as pd
import numpy as np 
import joblib
from sklearn.metrics import accuracy_score,fbeta_score,confusion_matrix,precision_score
from src.logger import logging
from src.utils.preprocessing_utils import data_split

def evalulate_train_data(X,y,xgb_hyp):
    # Evaluate the performance of the XGBoost classifier
    y_pred_xgb_hyp = xgb_hyp.predict(X)
    accuracy = accuracy_score(y,y_pred_xgb_hyp)
    precision = precision_score(y, y_pred_xgb_hyp)
    conf_matrix = confusion_matrix(y, y_pred_xgb_hyp)
    recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    f2_score = fbeta_score(y, y_pred_xgb_hyp, beta=2)
    logging.info('Trainind Data')
    logging.info(f'Accuracy of XGBClassifier using HyperOPT: {accuracy}')
    logging.info(f'Precision: {precision}')
    logging.info(f'Recall: {recall}')
    logging.info(f'F2 Score: {f2_score}')

def evalulate_test_data(test_data,xgb_hyp):
    #Test Data 
    xt, yt = data_split(test_data)
    logging.info(f'Test Data Loaded Successfully')

    # Load scaler and pca models
    pca = joblib.load('artifacts\components\pca.joblib')
    logging.info(f'PCA Model loaded Successfully')
    xt = pca.transform(xt)

    scaler = joblib.load('artifacts\components\standard.joblib')
    logging.info(f'Standardization Model loaded Successfully')
    xt = scaler.transform(xt)

    y_pred_test = xgb_hyp.predict(xt)
    test_acc = accuracy_score(yt, y_pred_test)
    precision = precision_score(yt, y_pred_test)
    conf_matrix = confusion_matrix(yt, y_pred_test)
    recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    f2_score = fbeta_score(yt, y_pred_test, beta=2)
    logging.info('Testing Data')
    logging.info(f'Accuracy of XGBClassifier using HyperOPT: {test_acc}')
    logging.info(f'Precision: {precision}')
    logging.info(f'Recall: {recall}')
    logging.info(f'F2 Score: {f2_score}')

def start_validating_data():
    df3 = pd.read_csv('data\dataset_small.csv')
    df3 = df3[list(pd.read_csv('artifacts/cleaned_data/train_data.csv').columns)]
    X2 = df3.drop(columns='phishing',axis=1)
    Y2 = df3['phishing']
    pca = joblib.load('artifacts\components\pca.joblib')
    X2 = pca.fit_transform(X2)
    scaler = joblib.load('artifacts\components\standard.joblib')
    X2 = scaler.transform(X2)
    xgb_hyp = joblib.load('artifacts\model\model.joblib')
    ypred2 = xgb_hyp.predict(X2)
    accuracy = accuracy_score(Y2,ypred2)
    y_true = Y2
    y_pred = ypred2

    # Assuming y_true contains the true labels and y_pred contains the predicted labels
    precision = precision_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    f2_score = fbeta_score(y_true, y_pred, beta=2)
    logging.info('Validate Dataset')
    logging.info(f'Accuracy: {accuracy}')
    logging.info(f'Precision: {precision}')
    logging.info(f'Recall: {recall}')
    logging.info(f'F2 Score: {f2_score}')