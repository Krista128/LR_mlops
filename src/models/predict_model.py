import pandas as pd
from src.features.build_features import (
    drop_features,
    add_new_features,
    make_dummies,
)


class predictorClass:
    def __init__(self, model: any):
        """
        model: target estimator
        """
        self.model = model

    def predict(self, input: dict) -> dict:
        """
        inputs: dict with raw features
        """
        df = pd.DataFrame(input, index=[0])
        df["Attrition"] = "Yes"

        df = add_new_features(df)
        df = make_dummies(df)
        df = drop_features(df)

        return {"Attrition": self.model.predict(df)[0]}
