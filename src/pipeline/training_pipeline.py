from sklearn.pipeline import Pipeline
from dataclasses import dataclass
import os
from src.components.data_ingestion import Ingestion
from src.components.data_preprocessing import Preprocessing
from src.components.model_trainer import Model_Training
import joblib
from src.logger import logging
from sklearn.metrics import accuracy_score,precision_score, fbeta_score, confusion_matrix
import pandas as pd

class Train_Pipeline():
  def start_training_pipeline(self):
    # Step 1: Ingestion
    ingestion_process = Ingestion('data\dataset_full.csv')
    dir = ingestion_process.ingestion()

    # Step 2: Preprocessing
    preprocess_process = Preprocessing(dir['train_data'], dir['test_data']) 
    data = preprocess_process.preprocessing()

    # Step 3: Model Training
    model_training_process = Model_Training(data['X'], data['y'], data['standar_scalar_dir'], data['pca_dir'], dir['test_data'])
    best_params = {'colsample_bytree': 0.6738406824277868, 'gamma': 0.7109523115417041, 'k_neighbour': 2, 'learning_rate': 0.060461943587600014, 'max_depth': 18, 'min_child_weight': 1.0, 'n_components': 5, 'n_estimators': 231, 'subsample': 0.915307830060938}
    model_dir = model_training_process.start_training(best_params)

    return model_dir

if __name__ == "__main__":
    # Create an instance of Train_Pipeline
    train_pipeline_instance = Train_Pipeline()

    # Start the training pipeline
    pipeline_dir = train_pipeline_instance.start_training_pipeline()
    print(f"Pipeline trained and saved to: {pipeline_dir}")

