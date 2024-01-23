from sklearn.pipeline import Pipeline
from dataclasses import dataclass
import os
from src.components.data_ingestion import Ingestion
from src.components.data_preprocessing import Preprocessing
from src.components.model_trainer import Model_Training
import joblib

@dataclass
class Pipeline_dir():
    os.makedirs('artifacts/pipe',exist_ok=True)
    pipeline_dir = 'artifacts/pipeline/model.joblib'


class Train_Pipeline():
  def __init__(self):
    self.dir = Pipeline_dir()

  def start_training_pipeline(self):
    # Step 1: Ingestion
    ingestion_process = Ingestion('/content/Phishing/data/dataset_full.csv')
    dir = ingestion_process.ingestion()

    # Step 2: Preprocessing
    preprocess_process = Preprocessing(dir['train_data'], dir['test_data']) 
    data = preprocess_process.preprocessing()

    # Step 3: Model Training
    model_training_process = Model_Training(data['X'], data['y'], data['standar_scalar_dir'], data['pca_dir'], dir['test_data'])
    model_dir = model_training_process.start_training()

    return model_dir

if __name__ == "__main__":
    # Create an instance of Train_Pipeline
    train_pipeline_instance = Train_Pipeline()

    # Start the training pipeline
    pipeline_dir = train_pipeline_instance.start_training_pipeline()

    print(f"Pipeline trained and saved to: {pipeline_dir}")

