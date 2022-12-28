import secrets
from pathlib import Path
from typing import Tuple

import mlflow
import tensorflow as tf
from anaconda.enterprise.server.common.sdk import demand_env_var
from anaconda.enterprise.server.contracts import BaseModel
from keras import Model
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_HEIGHT: int = 28
IMAGE_WIDTH: int = 28
BATCH_SIZE: int = 64

AUTOTUNE: int = tf.data.AUTOTUNE


class Trainer(BaseModel):
    def load_datasets(self, data_dir: Path) -> Tuple:
        # https://docs.python.org/3/library/secrets.html
        shuffle_seed: int = secrets.randbelow(1000)

        # Load our data sets
        # https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory

        train_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
            directory=data_dir,
            labels="inferred",
            label_mode="int",
            color_mode="rgb",
            batch_size=BATCH_SIZE,
            image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            shuffle=True,
            seed=shuffle_seed,
            # validation_split=0.2,
            validation_split=0.4,
            subset="training",
            interpolation="bilinear",
            crop_to_aspect_ratio=True,
        )

        val_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
            directory=data_dir,
            labels="inferred",
            label_mode="int",
            color_mode="rgb",
            batch_size=BATCH_SIZE,
            image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            shuffle=True,
            seed=shuffle_seed,
            # validation_split=0.2,
            validation_split=0.4,
            subset="validation",
            interpolation="bilinear",
            crop_to_aspect_ratio=True,
        )

        class_names = train_ds.class_names

        # Prepare data set for training / tf.python.data.ops.dataset_ops.DatasetV1.DatasetV1Adapter
        train_ds_prefetch = train_ds.shuffle(1000).cache(filename="/tmp/cache").prefetch(buffer_size=AUTOTUNE)
        val_ds_prefetch = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        data_split: dict = {"train": {}, "validation": {}}
        for class_name in class_names:
            data_split["train"][class_name] = []
            data_split["validation"][class_name] = []

        train_files = train_ds_prefetch.list_files(
            file_pattern=(data_dir / "**/*").resolve().as_posix(), seed=shuffle_seed
        )
        for file in train_files:
            file_path = Path(file.numpy().decode())
            item_name = file_path.stem
            item_label = file_path.parent.name
            data_split["train"][item_label].append(item_name)

        val_files = val_ds_prefetch.list_files(file_pattern=(data_dir / "**/*").resolve().as_posix(), seed=shuffle_seed)
        for file in val_files:
            file_path = Path(file.numpy().decode())
            item_name = file_path.stem
            item_label = file_path.parent.name
            data_split["validation"][item_label].append(item_name)

        return train_ds_prefetch, val_ds_prefetch, class_names, data_split, shuffle_seed
        # return train_ds, val_ds, class_names, data_split, shuffle_seed

    def build_model(self, num_classes: int) -> Model:
        # model input
        inputs = keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), name="inputs")

        # data augmentation layers
        data_augmentation_layer_1 = layers.RandomFlip(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))(inputs)
        data_augmentation_layer_2 = layers.RandomRotation(0.1)(data_augmentation_layer_1)
        data_augmentation_layer_3 = layers.RandomZoom(0.1)(data_augmentation_layer_2)

        # This layer acts as the new head during prediction
        data_augmentation_layer_4 = layers.Rescaling(1.0 / 255)(data_augmentation_layer_3)

        # convolutions
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
        conv_layer_1 = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(
            data_augmentation_layer_4
        )
        # conv_layer_2 = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(conv_layer_1)
        conv_layer_3 = layers.MaxPooling2D(padding="same")(conv_layer_1)

        conv_layer_4 = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(conv_layer_3)
        # conv_layer_5 = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(conv_layer_4)
        conv_layer_6 = layers.MaxPooling2D(padding="same")(conv_layer_4)

        conv_layer_7 = layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(conv_layer_6)
        # conv_layer_8 = layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(conv_layer_7)
        conv_layer_9 = layers.MaxPooling2D(padding="same")(conv_layer_7)
        conv_layer_10 = layers.Dropout(0.2)(conv_layer_9)

        # Dense Connections (disabled due to memory constraints)
        # fully_connected_layer_2 = layers.Dense(128, activation="relu")(conv_layer_10)
        # fully_connected_layer_3 = layers.Dense(128, activation="relu")(fully_connected_layer_2)
        # fully_connected_layer_4 = layers.Dense(128, activation="relu")(fully_connected_layer_3)

        # model end
        model_end_layer_1 = layers.Flatten()(conv_layer_10)
        model_end_layer_2 = layers.Dense(64, activation="relu")(model_end_layer_1)
        outputs = layers.Dense(num_classes)(model_end_layer_2)

        # build the model
        model: Model = keras.Model(inputs=inputs, outputs=outputs)

        # compile the model
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        return model

    def train(self, model, train_ds, val_ds):
        # train model
        epochs: int = 1
        # early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stop])

        return model, history

    def execute(self, data_dir: str):
        # Load Training Data
        train_ds, val_ds, class_names, data_split, shuffle_seed = self.load_datasets(data_dir=Path(data_dir))
        num_classes = len(class_names)

        print(f"Class Names: {class_names}, Count: {num_classes}")
        print(f"Shuffle Seed: {shuffle_seed}")

        # Build Model
        my_model: Model = self.build_model(num_classes)

        # Train Model
        mlflow.tensorflow.autolog(registered_model_name=demand_env_var(name="MLFLOW_REGISTERED_MODEL_NAME_TRAINING"), log_models=False)
        trained_model, history = self.train(my_model, train_ds, val_ds)

        # Generate inference time model
        prediction_model: Model = self.generate_inference_model(trained_model=trained_model)

        mlflow.tensorflow.log_model(
            model=prediction_model, registered_model_name=demand_env_var(name="MLFLOW_REGISTERED_MODEL_NAME_INFERENCE"), artifact_path="model"
        )
        mlflow.log_param(key="class_names", value=class_names)
        mlflow.log_param(key="shuffle_seed", value=shuffle_seed)

    def generate_inference_model(self, trained_model: Model) -> Model:
        new_inputs = tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), name="inputs")
        all_layers = [l for l in trained_model.layers[4:]]

        x = new_inputs
        for layer in all_layers:
            # Re-construct the model one layer at a time
            x = layer(x)

        return tf.keras.Model(inputs=new_inputs, outputs=x)
