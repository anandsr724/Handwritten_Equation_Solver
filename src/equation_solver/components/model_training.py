import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from equation_solver.entity.config_entity import TrainingConfig
from pathlib import Path 

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.batch_size = self.config.params_batch_size
        print(f"Using batch size: {self.batch_size}")

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        
        input_shape = self.model.input_shape
        print(f"Model expects input shape: {input_shape}")
        self.model.summary()

    def train_valid_generator(self):
        input_shape = self.model.input_shape
        color_mode = "grayscale" if input_shape[-1] == 1 else "rgb"
        # print(f"Using color mode: {color_mode}")

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.batch_size,
            interpolation="bilinear",
            color_mode=color_mode,
            class_mode='sparse'  # For sparse categorical crossentropy
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Validation generator - no shuffling
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        # Training generator - with shuffling
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        # # Print generator details for debugging
        # print(f"Training data shape: {self.train_generator.image_shape}")
        # print(f"Number of classes: {len(self.train_generator.class_indices)}")
        # print(f"Training batch size: {self.train_generator.batch_size}")
        # print(f"Steps per epoch: {len(self.train_generator)}")

    def train(self):
        # Calculate steps properly
        self.steps_per_epoch = len(self.train_generator)
        self.validation_steps = len(self.valid_generator)

        # Compile the model with appropriate loss and metrics
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train the model
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
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