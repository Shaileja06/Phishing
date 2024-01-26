from src.components.data_ingestion import Ingestion
from src.components.data_preprocessing import Preprocessing
from src.components.model_trainer import Model_Training

class Train_Pipeline():
  def start_training_pipeline(self):
    reply = input('Do your wish to Perform Hyperparamter Tunning Y/n')
    if reply == 'n' or reply=='N' :
      best_params = {'colsample_bytree': 0.6738406824277868, 'gamma': 0.7109523115417041, 'k_neighbour': 2, 'learning_rate': 0.060461943587600014, 'max_depth': 18, 'min_child_weight': 1.0, 'n_components': 5, 'n_estimators': 231, 'subsample': 0.915307830060938}
      # Step 1: Ingestion
      ingestion_process = Ingestion(file = 'data\dataset_full.csv',k_neighbors=int(best_params['k_neighbour']))
      dir = ingestion_process.start_ingestion()

      # Step 2: Preprocessing
      preprocess_process = Preprocessing(cleaned_data=dir['cleaned_data'],train_data=dir['train_data'], test_data=dir['test_data'],n_components=int(best_params['n_components'])) 
      data = preprocess_process.preprocessing()

      # Step 3: Model Training
      model_training_process = Model_Training(data['X'], data['y'],dir['test_data'],dir['cleaned_data'])
      model_dir = model_training_process.start_training(best_params)
      return model_dir

    else:
      model_hyperp_tunning = Model_Training()
      best_params = model_hyperp_tunning.start_hyperparameter_tunning()
      # Step 1: Ingestion
      ingestion_process = Ingestion('data\dataset_full.csv',int(best_params['k_neighbour']))
      dir = ingestion_process.start_ingestion()

      # Step 2: Preprocessing
      preprocess_process = Preprocessing(dir['cleaned_data'],dir['train_data'], dir['test_data'],int(best_params['n_components'])) 
      data = preprocess_process.preprocessing()

      # Step 3: Model Training
      model_training_process = Model_Training(data['X'], data['y'],dir['test_data'],dir['cleaned_data'])
      model_dir = model_training_process.start_training(best_params)
      return model_dir

       

if __name__ == "__main__":
    # Create an instance of Train_Pipeline
    train_pipeline_instance = Train_Pipeline()

    # Start the training pipeline
    pipeline_dir = train_pipeline_instance.start_training_pipeline()
    print(f"Pipeline trained and saved to: {pipeline_dir}")