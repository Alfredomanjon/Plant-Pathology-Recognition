import mlflow
import mlflow.keras
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

# from tensorflow.keras import layers, models  # type: ignore
import matplotlib.pyplot as plt
from keras import layers, models, regularizers, optimizers, losses


class CnnModel:
    """CNN clasification model class"""

    def __init__(self, width, height, channel):
        """Generate a new CNN model from image width and height inputs"""

        mlflow.set_tracking_uri("http://192.168.0.75:5000")
        mlflow.set_experiment("Planty")

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
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(256, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(
                    512,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(0.01),
                ),
                layers.Dropout(0.5),  # Añadir Dropout para evitar sobreajuste
                layers.Dense(7, activation="softmax"),
            ]
        )

        optimizer = optimizers.Adam(learning_rate=0.0001)

        self.model.compile(
            optimizer=optimizer,
            loss=losses.CategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

    def train(self, train_images, train_labels, epochs):
        """Train model"""

        history = self.model.fit(train_images, train_labels, epochs=epochs)

        acc = history.history["accuracy"]
        loss = history.history["loss"]

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="Training Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")

        # plt.show()

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

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print("\nInforme de Clasificación:")
        print(classification_report(y_true, y_pred, target_names=classes))

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        """Confusion matrix for validation analysis"""

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
        )
        plt.xlabel("Predicciones")
        plt.ylabel("Valores Reales")
        plt.title("Matriz de Confusión")
        plt.savefig("model/confusion_matrix.png")
        # plt.show()

        mlflow.log_artifact("model/confusion_matrix.png")

    def save_local(self):
        """Save model in a local path"""

        self.model.save("model/cnn_model.h5")
        # tfjs.converters.save_keras_model(self.model, "model/cnn_model.h5")

    def log_model_to_mlflow(self):
        """Log the model and metrics to MLflow"""

        # Log model summary
        # mlflow.tensorflow.log_model(self.model, artifact_path="models")
        # mlflow.log_artifacts("model/cnn_model.h5", artifact_path="model")
        with mlflow.start_run() as run:
            mlflow.keras.log_model(
                self.model, "cnn_model", keras_model_kwargs={"save_format": "h5"}
            )
        print("Model saved to MLflow.")
