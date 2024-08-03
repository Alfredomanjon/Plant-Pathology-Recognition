"""Script for training flow"""

from src.data.data_functions import create_train_dataset, split_train_test
from src.model.model_train import CnnModel


dataset = create_train_dataset("data/train.csv")

X_train, X_test, y_train, y_test = split_train_test(dataset)

print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

print(X_train[0])
print(y_train[0])

model = CnnModel(224, 224, 3)

print("----- Starting training... -----")

model.train(X_train, y_train, 20)

y_true, y_pred = model.validation(X_test, y_test)

clases = ['Healthy', 'Multiple Diseases', 'Rust', 'Scab']
model.plot_confusion_matrix(y_true, y_pred, clases)

model.save_local()