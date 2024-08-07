"""Script for training flow"""

from src.data.data_functions import create_train_dataset, split_train_test
from src.model.model_train import CnnModel


dataset = create_train_dataset("data/train.csv")

X_train, X_test, y_train, y_test = split_train_test(dataset)

print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

model = CnnModel(224, 224, 3)

print("----- Starting training... -----")

model.train(X_train, y_train, 18)

model.save_local()

y_true, y_pred = model.validation(X_test, y_test)

clases = [
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
    "complex",
    "frog_eye_leaf_spot",
    "powdery_mildew",
]

model.evaluate_model(y_true, y_pred, clases)

model.plot_confusion_matrix(y_true, y_pred, clases)

model.log_model_to_mlflow()
