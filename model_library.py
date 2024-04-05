import os
import warnings
import re
from joblib import load
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, space_eval
from joblib import dump
import numpy as np

warnings.filterwarnings("ignore")
directory = "Models/"


def loss_fn(cm):
    weights = np.array([
        [0, 1, 2, 4],
        [2, 0, 1, 2],
        [4, 2, 0, 1],
        [8, 4, 2, 0]
    ])

    cm = np.array(cm)
    return np.sum(cm * weights)


def find_params_grid(X_tr, y_tr, param_grid, path):
    kf = StratifiedKFold(n_splits=4)

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    def custom_loss_func(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return loss_fn(cm)

    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring={'custom_loss': custom_loss_func},
                               refit='custom_loss', verbose=3)

    grid_search.fit(X_tr, y_tr)

    print("Best parameters set found on development set:")
    print(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    dump(best_model, path + '_grid_best_model.joblib')

    return best_model


def save_model(model):
    next_index = _get_next_index()
    file_path = os.path.join(directory, f"best{next_index}.joblib")
    dump(model, file_path)
    print(f"Model saved to {file_path}")


def load_model(file_path):
    model = load(file_path)
    print(f"Model loaded from {file_path}")
    return model


def _get_next_index():
    max_index = 0
    pattern = re.compile(r"best(\d+)\.joblib")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index

    return max_index + 1


def print_model_metrics(path):
    model_data = load_model(path)

    if model_data:
        acc = model_data.get("acc", "N/A")
        lfn = model_data.get("lfn", "N/A")
        y_pred = model_data.get("y_pred", None)

        print("Model Metrics:")
        print(f"Accuracy: {acc}")
        print(f"Loss Function Value: {lfn}")
        print(y_pred)
