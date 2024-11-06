import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from equation_solver.entity.config_entity import TrainingConfig
from pathlib import Path 
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

class DataGenerator:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.class_indices = {name: idx for idx, name in enumerate(self.config.classes_ideal)}

    def load_data(self):
        all_files = []
        for root, dirs, files in os.walk(self.config.training_data):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    class_name = os.path.basename(root)
                    if class_name in self.class_indices:
                        file_path = os.path.join(root, file)
                        all_files.append((file_path, class_name))

        # print(f"Total files found: {len(all_files)}")
        # print(f"Number of classes: {len(self.class_indices)}")
        # print(f"Class indices: {self.class_indices}")

        # Shuffle the files
        np.random.shuffle(all_files)

        # Split file paths and labels
        file_paths, class_names = zip(*all_files)
        labels = [self.class_indices[class_name] for class_name in class_names]

        # Perform stratified split
        file_paths_train, file_paths_val, labels_train, labels_val = train_test_split(
            file_paths, labels, test_size=0.2, stratify=labels, random_state=34
        )

        # print(f"Training samples: {len(file_paths_train)}")
        # print(f"Validation samples: {len(file_paths_val)}")

        # Oversample the training data
        oversampler = RandomOverSampler(sampling_strategy='auto', random_state=54)
        file_paths_train_resampled, labels_train_resampled = oversampler.fit_resample(
            np.array(file_paths_train).reshape(-1, 1), labels_train
        )
        file_paths_train_resampled = file_paths_train_resampled.flatten()

        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((file_paths_train_resampled, labels_train_resampled))
        val_ds = tf.data.Dataset.from_tensor_slices((file_paths_val, labels_val))

        # Apply preprocessing
        train_ds = train_ds.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle and batch
        train_ds = train_ds.cache().shuffle(buffer_size=len(file_paths_train_resampled)).batch(self.config.params_batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().batch(self.config.params_batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds

    def load_image(self, file_path, label):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.config.params_image_size[:-1])
        image = tf.image.rgb_to_grayscale(image)
        return image, label
    
class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.batch_size = self.config.params_batch_size
        self.data_generator = DataGenerator(config)
        self.train_ds, self.val_ds = self.data_generator.load_data()
        # print(f"Using batch size: {self.batch_size}")

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        
        input_shape = self.model.input_shape
        # print(f"Model expects input shape: {input_shape}")
        # self.model.summary()

    def train_valid_generator(self):
        pass

    def train(self):
        # Calculate steps properly
        self.steps_per_epoch = len(self.train_ds)
        self.validation_steps = len(self.val_ds)

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train the model
        history = self.model.fit(
            self.train_ds,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.val_ds,
            verbose=1
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        
        return history

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)