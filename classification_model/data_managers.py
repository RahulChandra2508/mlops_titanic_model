import re

# to handle datasets
import pandas as pd
# to persist the model and the scaler
import joblib
from processing import features
import config
import yaml

from sklearn.model_selection import train_test_split

with open("config.yaml","r") as file:
    config_file = yaml.safe_load(file)

def load_data(split= False):

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
                                                        titanic_data[config_file["base"]["target_column"]],  # target
                                                        test_size= config_file["base"]["test_size"],  # percentage of obs in test set
                                                        random_state= config_file["base"]["random_state"]  # seed to ensure reproducibility
                                                        )
        return X_train, X_test, y_train, y_test
    else:
        return titanic_data

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