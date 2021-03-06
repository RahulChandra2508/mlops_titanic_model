import pandas as pd
import yaml

from classification_model.data_managers import load_data

from data_managers import *
from processing import features as fea
from pathlib import Path

with open("config.yaml", "r") as file:
    config_file = yaml.safe_load(file)


def test_ExtractLetterTransformer():
    # saving the data copy into data folder
    save_to_csv()
    new_data = pd.read_csv(Path(config_file["path"]["DATA"]) / "titanic_data.csv")

    assert new_data[config_file["attributes"]["cabin"][0]].iat[1] == "C22 C26"

    # ExtractLetterTransformer
    et_ins = fea.ExtractLetterTransformer(variables=config_file["attributes"]["cabin"])

    # when
    new_data_transformed = et_ins.transform(X=new_data)

    # then
    assert new_data_transformed[config_file["attributes"]["cabin"][0]].iat[1] == "C"
