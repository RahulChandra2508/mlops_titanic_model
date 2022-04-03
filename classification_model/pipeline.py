from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from processing import features as pp

# for imputation
from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer,
)

# for encoding categorical variables
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder

import yaml


with open("config.yaml", "r") as file:
    config_file = yaml.safe_load(file)


def pipeline():
    titanic_pipe = Pipeline(
        [
            (
                "categorical_imputation",
                CategoricalImputer(
                    imputation_method="missing",
                    variables=config_file["attributes"]["categorical_variables"],
                ),
            ),
            # add missing indicator to numerical variables
            (
                "missing_indicator",
                AddMissingIndicator(
                    variables=config_file["attributes"]["numeric_variables"]
                ),
            ),
            # impute numerical variables with the median
            (
                "median_imputation",
                MeanMedianImputer(
                    imputation_method="median",
                    variables=config_file["attributes"]["numeric_variables"],
                ),
            ),
            # Extract letter from cabin
            (
                "extract_letter",
                pp.ExtractLetterTransformer(
                    variables=config_file["attributes"]["cabin"]
                ),
            ),
            # == CATEGORICAL ENCODING ======
            # remove categories present in less than 5% of the observations (0.05)
            # group them in one category called 'Rare'
            (
                "rare_label_encoder",
                RareLabelEncoder(
                    tol=0.05,
                    n_categories=1,
                    variables=config_file["attributes"]["categorical_variables"],
                ),
            ),
            # encode categorical variables using one hot encoding into k-1 variables
            (
                "categorical_encoder",
                OneHotEncoder(
                    drop_last=True,
                    variables=config_file["attributes"]["categorical_variables"],
                ),
            ),
            # scale
            ("scaler", StandardScaler()),
            (
                "Logit",
                LogisticRegression(
                    C=config_file["model_params"]["C"],
                    random_state=config_file["model_params"]["random_state"],
                ),
            ),
        ]
    )

    return titanic_pipe
