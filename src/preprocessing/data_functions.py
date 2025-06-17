"""Module providing a dataframe for model task"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_image_label(image_path, label):
    """Function to process image and label"""

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    # Normalizar los píxeles a [0, 1]
    image = image / 255.0

    label = tf.cast(label, tf.float32)

    return image, label


def create_train_dataset(csv_path):
    """Function to read data and generate train and val datasets"""

    df_train_labels = pd.read_csv(csv_path)
    print(df_train_labels.head())

    image_paths = [
        os.path.join("data/images", f"{row['image_id']}.jpg")
        for index, row in df_train_labels.iterrows()
    ]
    labels = df_train_labels.iloc[:, 1:].values

    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(image_paths), tf.constant(labels)))

    # Mapear la función de carga a cada elemento del dataset
    dataset = dataset.map(load_image_label)

    return dataset


def split_train_test(dataset):
    """Function split dataset in train/test datasets"""

    images_list = []
    labels_list = []

    for image, label in dataset:
        images_list.append(image.numpy())  # Convertir la imagen a un array de NumPy
        labels_list.append(label.numpy())  # Convertir la etiqueta a un array de NumPy

    # Convertir listas a arrays de NumPy
    X = np.array(images_list)
    y = np.array(labels_list)

    # Dividir el DataFrame
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test