import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import gdown
from equation_solver import logger
from pathlib import Path
from equation_solver.entity.config_entity import PrepareBaseModelConfig

# compenents -> prepare_base_model.py
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        '''
        Fetch data from the url
        '''
        try: 
            model_url = self.config.custom_model_URL
            model_download_dir = self.config.base_model_path
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading model from {model_url} into file {model_download_dir}")

            file_id = model_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,str(model_download_dir))

            logger.info(f"Downloaded model from {model_url} into file {model_download_dir}")

        except Exception as e:
            raise e
        
        # Load the downloaded model
        self.model = tf.keras.models.load_model(model_download_dir)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate, keep_dense):
        # Function to check if a layer is a convolutional layer
        def is_conv_layer(layer):
            return isinstance(layer, (
                tf.keras.layers.Conv2D,
                tf.keras.layers.Conv1D,
                tf.keras.layers.Conv3D,
                tf.keras.layers.DepthwiseConv2D,
                tf.keras.layers.SeparableConv2D
            ))
        
        # Function to check if a layer is a dense layer
        def is_dense_layer(layer):
            return isinstance(layer, tf.keras.layers.Dense)

        if freeze_all:
            # Freeze only convolutional layers
            for layer in model.layers:
                if is_conv_layer(layer):
                    layer.trainable = False
                elif is_dense_layer(layer):
                    layer.trainable = True
                # Other layers (like BatchNorm) associated with conv layers should also be frozen
                elif not is_dense_layer(layer) and not isinstance(layer, (tf.keras.layers.Dropout, tf.keras.layers.Flatten)):
                    layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            # Find the last convolutional layer
            last_conv_index = 0
            for i, layer in enumerate(model.layers):
                if is_conv_layer(layer):
                    last_conv_index = i
            
                    '''
            # Freeze layers up to the last convolutional layer
            for i, layer in enumerate(model.layers):
                if i <= last_conv_index:
                    layer.trainable = False
                else:
                    layer.trainable = True
                    '''
            # Freeze layers up to the last convolutional layer
            for i, layer in enumerate(model.layers):
                # print("enumerating: ", i , " layer: ", layer)
                if i <= freeze_till:
                    layer.trainable = False
                else:
                    layer.trainable = True

        if not keep_dense:
            # Get the output of the last convolutional layer
            last_conv_layer = None
            for layer in model.layers:
                if is_conv_layer(layer):
                    last_conv_layer = layer
            
            if last_conv_layer is None:
                raise ValueError("No convolutional layers found in the model")
            
            x = last_conv_layer.output

            # Flatten the output
            x = tf.keras.layers.Flatten()(x)

            # Add Dense layers similar to your original model
            x = tf.keras.layers.Dense(
                768, 
                activation='relu',
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                trainable=True  # Explicitly set dense layers as trainable
            )(x)
            x = tf.keras.layers.Dropout(0.25)(x)
            x = tf.keras.layers.Dense(
                128,
                activation='relu',
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                trainable=True
            )(x)
            x = tf.keras.layers.Dropout(0.25)(x)

            # Final output layer
            prediction = tf.keras.layers.Dense(
                classes,
                activation='softmax',
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                trainable=True
            )(x)

            full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)
        else:
            full_model = model

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        # Print trainable status of layers
        logger.info("Layer trainable status:")
        # print("\nLayer trainable status:")
        for layer in full_model.layers:
            # print(f"{layer.name}: {layer.trainable}")
            logger.info(f"{layer.name}: {layer.trainable}")


        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=self.config.params_freeze_all,
            freeze_till=self.config.params_freeze_till,
            learning_rate=self.config.params_learning_rate,
            keep_dense=self.config.params_keep_dense,
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)