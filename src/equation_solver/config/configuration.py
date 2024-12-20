
from equation_solver.constants import *
from equation_solver.utils.common import read_yaml, create_directories, save_json,get_unique_class_names
from equation_solver.entity.config_entity import (DataIngestionConfig,
                                                  PrepareBaseModelConfig,
                                                  TrainingConfig,
                                                  EvaluationConfig
                                                  )
import os

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            custom_model_URL=str(config.custom_model_URL),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_keep_dense=self.params.KEEP_DENSE,
            params_freeze_all=self.params.FREEZE_ALL,
            params_freeze_till=self.params.FREEZE_TILL,
        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "symbol_data")
        classes_ideal = get_unique_class_names(training_data)
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            classes_ideal=classes_ideal
        )

        return training_config
        
    # def get_evaluation_config(self) -> EvaluationConfig:
    #     eval_config = EvaluationConfig(
    #         path_of_model="artifacts/training/model.h5",
    #         training_data="artifacts/data_ingestion/symbol_data",
    #         mlflow_uri="https://dagshub.com/anandsr724/Handwritten_Equation_Solver.mlflow",
    #         all_params=self.params,
    #         params_image_size=self.params.IMAGE_SIZE,
    #         params_batch_size=self.params.BATCH_SIZE
    #     )
    #     return eval_config
    def get_evaluation_config(self) -> EvaluationConfig:
        training = self.config.training
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "symbol_data")
        classes_ideal = get_unique_class_names(training_data)

        eval_config = EvaluationConfig(
            path_of_model=Path(training.trained_model_path),
            training_data=Path(training_data),
            mlflow_uri="https://dagshub.com/anandsr724/Handwritten_Equation_Solver.mlflow",
            all_params=self.params,
            params_image_size=params.IMAGE_SIZE,
            params_batch_size=params.BATCH_SIZE,
            classes_ideal=classes_ideal
        )
        return eval_config