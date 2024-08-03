import tensorflow as tf
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from tensorflow.keras import layers, models # type: ignore
import matplotlib.pyplot as plt


class CnnModel:
    """CNN clasification model class"""

    def __init__(self, width, height, channel):
        """Generate a new CNN model from image width and height inputs"""

        self.image_width = width
        self.image_height = height
        self.image_channel = channel
        self.model = models.Sequential(
            [
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=(
                        self.image_width,
                        self.image_height,
                        self.image_channel,
                    ),
                ),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dense(
                    256,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                ),
                layers.Dropout(0.5),  # Añadir Dropout para evitar sobreajuste
                layers.Dense(128, activation="relu"),
                layers.Dense(7, activation="softmax"),
            ]
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

    def train(self, train_images, train_labels, epochs):
        """Train model"""

        history = self.model.fit(train_images, train_labels, epochs=epochs)

        acc = history.history['accuracy']
        loss = history.history['loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.show()

    def validation(self, test_images, test_labels):
        """Model validation with test examples"""

        y_true = []
        y_pred = []

        for images, labels in zip(test_images, test_labels):

            if len(images.shape) == 3:  # images tiene forma (224, 224, 3)
                images = np.expand_dims(images, axis=0)

            predictions = self.model.predict(images)

            y_true.extend([np.argmax(labels)])
            y_pred.append(np.argmax(predictions))

        return np.array(y_true), np.array(y_pred)

    def evaluate_model(self, y_true, y_pred, classes):
        """Calculate and display various metrics"""

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        print(f"Exactitud (Accuracy): {accuracy:.4f}")
        print(f"Precisión (Precision): {precision:.4f}")
        print(f"Recuerdo (Recall): {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        print("\nInforme de Clasificación:")
        print(classification_report(y_true, y_pred, target_names=classes))

    def plot_confusion_matrix(self, y_true, y_pred, clases):
        """Confusion matrix for validation analysis"""

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
        plt.xlabel('Predicciones')
        plt.ylabel('Valores Reales')
        plt.title('Matriz de Confusión')
        plt.show()

    def save_local(self):
        """Save model in a local path"""

        self.model.save('model/cnn_model.h5')

    def convert_to_tfl(self):
        """Convert tf model to tfl for mobile use"""

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        # Optimization techniques to reduce model size or improve performance
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open('cnn_model.tflite', 'wb') as f:
            f.write(tflite_model)
