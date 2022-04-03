import pipeline as pipe
import data_managers
import yaml

from urllib.parse import urlparse
from sklearn.metrics import roc_auc_score, accuracy_score

# loading the training data
X_train, y_train = data_managers.load_data(split=True, data_="train")

titanic_pipe = pipe.pipeline()

titanic_pipe.fit(X_train, y_train)

# make predictions for train set
class_ = titanic_pipe.predict(X_train)
pred = titanic_pipe.predict_proba(X_train)[:, 1]
# determine roc and accuracy
print("train roc-auc: {}".format(roc_auc_score(y_train, pred)))
print("train accuracy: {}".format(accuracy_score(y_train, class_)))


# Mlflow Configuration
with open("config.yaml", "r") as file:
    config_file = yaml.safe_load(file)


mlflow_config = config_file["mlflow_config"]
remote_server_uri = mlflow_config["remote_server_uri"]
mlflow.set_tracking_uri(remote_server_uri)

mlflow.set_experiment(mlflow_config["experiment_name"])

with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
    from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(
            l1_ratio=l1_ratio,
            alpha=alpha,
            random_state=random_state
        )
        model.fit(X_train_both, y_train_both)

        from sklearn.metrics import accuracy_score, f1_score,recall_score,precision_score

        predicted_qualities = model.predict(test_x)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("accuracy", float(round(accuracy_score(y_pred=predicted_qualities,y_true=test_y),3)))
        #mlflow.log_metric("accuracy", '99.8')
        mlflow.log_metric("f1", float(round(f1_score(y_pred=predicted_qualities,y_true=test_y),3)))
        mlflow.log_metric("recall", float(round(recall_score(y_pred=predicted_qualities,y_true=test_y),3)))
        mlflow.log_metric("precision", float(round(precision_score(y_pred=predicted_qualities,y_true=test_y),3)))
        
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(model, "model")
      




# saving the models
data_managers.save_model(titanic_pipe)
