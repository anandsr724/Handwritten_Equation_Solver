import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from equation_solver.entity.config_entity import EvaluationConfig
from equation_solver.utils.common import save_json
import numpy as np
import os
from sklearn.model_selection import train_test_split

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.class_indices = {name: idx for idx, name in enumerate(self.config.classes_ideal)}
        self.model = self.load_model(self.config.path_of_model)

    def load_data(self):
        all_files = []
        for root, dirs, files in os.walk(self.config.training_data):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    class_name = os.path.basename(root)
                    if class_name in self.class_indices:
                        file_path = os.path.join(root, file)
                        all_files.append((file_path, class_name))

        # Shuffle the files
        np.random.shuffle(all_files)

        # Split file paths and labels
        file_paths, class_names = zip(*all_files)
        labels = [self.class_indices[class_name] for class_name in class_names]

        # Perform stratified split
        file_paths_train, file_paths_val, labels_train, labels_val = train_test_split(
            file_paths, labels, test_size=0.2, stratify=labels, random_state=34
        )

        # Create datasets
        val_ds = tf.data.Dataset.from_tensor_slices((file_paths_val, labels_val))

        # Apply preprocessing
        val_ds = val_ds.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch
        val_ds = val_ds.cache().batch(self.config.params_batch_size).prefetch(tf.data.AUTOTUNE)

        return val_ds

    def load_image(self, file_path, label):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.config.params_image_size[:-1])
        image = tf.image.rgb_to_grayscale(image)
        return image, label

    def load_model(self, path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.val_ds = self.load_data()
        self.score = self.model.evaluate(self.val_ds)
        self.save_score()


    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.tensorflow.log_model(self.model, "model", registered_model_name="Model768")
            else:
                mlflow.tensorflow.log_model(self.model, "model")