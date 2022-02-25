# Creating a class to extract the 1st letter of the cabin column

import pandas as pd
import numpy as np
from typing import List
import re
from sklearn.base import BaseEstimator, TransformerMixin

class ExtractLetterTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,variables : List[str]):

        if not isinstance(variables, list):
            raise ValueError("Variable should be a list")

        self.variables = variables

    def fit(self, X : pd.DataFrame, y: pd.Series):
        return self
    
    def transform(self, X : pd.DataFrame) -> pd.DataFrame:
        
        X =  X.copy()
        
        for features in self.variables:
           X[features] = X[features].str[0]
        
        return X


class Cleaning(BaseEstimator, TransformerMixin):

    def __init__(self, variables : List[str]):
        
        if not isinstance(variables, list):
            raise ValueError("Expected a variables list")
        
        self.variables = variables

    def fit(self, X: pd.DataFrame):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        def get_title(passenger):
            line = passenger
            if re.search('Mrs', line):
                return 'Mrs'
            elif re.search('Mr', line):
                return 'Mr'
            elif re.search('Miss', line):
                return 'Miss'
            elif re.search('Master', line):
                return 'Master'
            else:
                return 'Other'

        X = X.replace("?", np.nan)

        for features in self.variables:
           X["title"] =  X[features].apply(get_title)
           
        return X