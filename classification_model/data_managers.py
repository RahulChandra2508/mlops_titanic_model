import re
from typing import List
# to handle datasets
import pandas as pd
# to persist the model and the scaler
import joblib
from processing import features
import config
import yaml
from sklearn.pipeline import Pipeline
from pathlib import Path

from sklearn.model_selection import train_test_split

with open("config.yaml","r") as file:
    config_file = yaml.safe_load(file)

def load_data(split= False, data_: str= "train") -> pd.DataFrame:

    data = pd.read_csv(config_file["data_source"]["openml"])
    titanic_data = data.copy()
    
    # removing "?" and extracting lables from name column
    cleaning = features.Cleaning(variables= ["name"])
    cleaning.fit(titanic_data)
    titanic_data = cleaning.transform(titanic_data)

    # Correcting the data types for the variables with wrong data types
    titanic_data['fare'] = titanic_data['fare'].astype('float')
    titanic_data['age'] = titanic_data['age'].astype('float')

    # Droping unwanted columns
    titanic_data.drop(labels= config_file["attributes"]["drop_variables"], axis=1, inplace=True)

    if split:
        X_train, X_test, y_train, y_test = train_test_split(
                                                        titanic_data.drop(config_file["base"]["target_column"], axis=1),  # predictors
                                                        titanic_data[config_file["base"]["target_column"]],  # target, # target
                                                        test_size= config_file["base"]["test_size"],  # percentage of obs in test set
                                                        random_state= config_file["base"]["random_state"]  # seed to ensure reproducibility
                                                        )

        if data_== "train":
            return X_train, y_train

        elif data_== "test":
            return X_test, y_test

        elif data_== "none":
            return X_train, X_test, y_train, y_test
    else:
        return titanic_data

#Function to remove the old model
def remove_old_pipeline(files_to_keep: List[str] = ["__inti__.py"]):
    """
    Removes the old pipeline if it exits and replaces it with the latest model
    """
    for model_file in Path(config_file["path"]["TRAINED_DATASET_DIR"]).iterdir():
        if model_file.name in files_to_keep:
            model_file.unlink()

#Function to save the trained model
def save_model(pipeline_to_save: Pipeline):

    """ Saves the pipeline(pipeline.pkl) at trained _models directory """

    save_model_name = f"{config_file['files']['trained_model_name']}_{config_file['files']['version']}.pkl"
    save_path = Path(config_file["path"]["TRAINED_DATASET_DIR"])/save_model_name

    #config_file["files"]["version"] += 0.0.1

    remove_old_pipeline(files_to_keep= [save_model_name])

    joblib.dump(pipeline_to_save, save_path)

#Function to load the trained pipeline
def load_pipeline(file_name: str) -> Pipeline:
    
    """ Load the pretrained model """
    model_path = Path(config_file["path"]["TRAINED_DATASET_DIR"])/file_name
    trained_model = joblib.load(model_path)
    return trained_model

# Function to split the data into train, test data sets
def data_split(data: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
                                                        data.drop(config_file["base"]["target_column"], axis=1),  # predictors
                                                        data[config_file["base"]["target_column"]],  # target
                                                        test_size= config_file["base"]["test_size"],  # percentage of obs in test set
                                                        random_state= config_file["base"]["random_state"]  # seed to ensure reproducibility
                                                        )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    new_data = load_data()
    X_train, X_test, y_train, y_test = data_split(new_data)
    print(X_train)
    print(X_test)
    print(y_train)
    print(y_test)