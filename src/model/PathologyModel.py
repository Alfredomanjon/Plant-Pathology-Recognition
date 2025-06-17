"""Abstract class for define new implements models"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from src.model.InterfaceModelClass import ModelClass

class PathologyModel(ModelClass):
    """Pathology classification class implementation"""

    def __init__(self, data_dir, img_width, img_height, batch_size, validation_split, num_classes, mode="new"):
        """Generate a new CNN model from image width and height inputs"""

        self.data_dir = data_dir

        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.seed = 123

        self.num_classes = num_classes

        # Define model dataset
        self.train_ds, self.val_ds = self.load_dataset()

        if mode == "new":
            self.build_new_model()
        elif mode == "finetune":
            self.load_pretrained_model()
        elif mode == "load":
            print("[INFO] - Cargando modelo desde archivo")
            self.model = load_model("modelo_final_finetuned.keras")
        else:
            raise ValueError(f"[ERROR] Modo '{mode}' no reconocido. Usa 'new', 'finetune' o 'load'.")

        print("[INFO] - Modelo cargado correctamente")

        self.model.summary()

        
    def load_dataset(self):
        """Load dataset from directory"""
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_split,
            subset="training",
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.validation_split,
            subset="validation",
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)
        
        return train_ds, val_ds
    

    def _get_data_augmentation(self):
        """Devuelve el pipeline de aumento de datos."""
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ], name="data_augmentation")
    

    def build_new_model(self):
        """Construye un nuevo modelo de EfficientNetB3 con data augmentation y congelación de capas."""

        print("[INFO] - Entrenando modelo base")

        data_augmentation = self._get_data_augmentation()
            
        self.base_model = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_height, self.img_width, 3)
        )
        self.base_model.trainable = False

        inputs = tf.keras.Input(shape=(self.img_height, self.img_width, 3))
        x = data_augmentation(inputs)
        x = layers.Rescaling(1./255)(x)
        x = self.base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = models.Model(inputs, outputs)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    
    def load_pretrained_model(self):
        print("Entrenando modelo haciendo finetunning...")

        data_augmentation = self._get_data_augmentation()

        # Cargar el mejor modelo entrenado previamente (cabeza)
        model_pretrained = load_model("best_model_finetuned.keras")

        model_pretrained.summary()

        self.base_model = model_pretrained.get_layer('efficientnetb3')
        self.base_model.trainable = True

        # Descongelar solo las últimas capas del base_model (por ejemplo, las últimas 50)
        for layer in self.base_model.layers[:-50]:
            layer.trainable = False

        # Reconstruir el modelo completo con data augmentation
        inputs = tf.keras.Input(shape=(self.img_height, self.img_width, 3))
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = self.base_model(x, training=True)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = models.Model(inputs, outputs)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )


    def train(self, epochs):

        print("[INFO] - Entrenando modelo")

        finetune_checkpoint = ModelCheckpoint(
            'results/best_model_finetuned.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )

        finetune_early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )

        print("[INFO] - Checkpoint y Early Stopping activado")

        print(f"Capas entrenables: {sum([layer.trainable for layer in self.base_model.layers])}")

        # Entrenamiento de fine-tuning
        history_finetune = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,  
            #initial_epoch=3,
            callbacks=[finetune_checkpoint, finetune_early_stop]
        )

        # Guardar el modelo final
        self.save_local()

        self.validation(history_finetune)


    def validation(self, history):
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs_range = range(len(acc))

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Train Accuracy')
            plt.plot(epochs_range, val_acc, label='Val Accuracy')
            plt.legend(loc='lower right')
            plt.title('Accuracy over Epochs')

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Train Loss')
            plt.plot(epochs_range, val_loss, label='Val Loss')
            plt.legend(loc='upper right')
            plt.title('Loss over Epochs')

            plt.show()


    def evaluate_model(self, test_images, test_labels, class_names=None):
        """
        Evalúa el modelo sobre un conjunto de test y muestra métricas.
        """

        print("[INFO] - Evaluando modelo")

        # Obtener predicciones
        y_pred_probs = self.model.predict(test_images)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_labels

        # Accuracy
        accuracy = np.mean(y_pred == y_true)
        print(f"Accuracy: {accuracy:.4f}")

        # Classification report
        if class_names is not None:
            print(classification_report(y_true, y_pred, target_names=class_names))
        else:
            print(classification_report(y_true, y_pred))

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)


    def plot_confusion_matrix(self, y_true, y_pred, classes):
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()


    def save_local(self):

        self.model.save("results/modelo_final_finetuned.keras")
        