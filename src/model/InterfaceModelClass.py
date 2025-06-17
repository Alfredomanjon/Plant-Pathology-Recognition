"""Abstract class for define new implements models"""

from abc import ABC
from abc import abstractmethod

class ModelClass(ABC):
    """Abstract class implementation"""

    @abstractmethod
    def train(self, train_images, train_labels, epochs):
        pass

    @abstractmethod
    def validation(self, history):
        pass

    @abstractmethod
    def evaluate_model(self, y_true, y_pred, classes):
        pass

    @abstractmethod
    def plot_confusion_matrix(self, y_true, y_pred, classes):
        pass

    @abstractmethod
    def save_local(self):
        pass