import data_managers
import yaml
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from data_managers import load_pipeline

with open("config.yaml", "r") as f:
    config_file = yaml.safe_load(f)

# loading the training data
X_test, y_test = data_managers.load_data(split=True, data_="test")

file_name = f"{config_file['files']['trained_model_name']}_{config_file['files']['version']}.pkl"

titanic_pipe = load_pipeline(file_name)

# make predictions for train set
class_ = titanic_pipe.predict(X_test)
pred = titanic_pipe.predict_proba(X_test)[:, 1]
# determine roc and accuracy
print("test roc-auc: {}".format(roc_auc_score(y_test, pred)))
print("test accuracy: {}".format(accuracy_score(y_test, class_)))
