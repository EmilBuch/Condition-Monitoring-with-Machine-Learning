import sys

sys.path.insert(0, "src")
from models.model_classes import (
    MLPRegressorClass,
    RFRegressorClass,
    XGBRegressorClass,
    SVRClass,
    KNNRegressorClass,
)

MODEL_DICTIONARY = {
    "MLPRegressor": [
        MLPRegressorClass,
        {
            "learning_rate": 0.03,
        },
    ],
    "RFRegressor": [
        RFRegressorClass,
        {
            "n_estimators": (10, 1000),
            "min_samples_leaf": (1, 100),
            "min_samples_split": (2, 100),
        },
    ],
    "XGBRegressor": [
        XGBRegressorClass,
        {
            "n_estimators": (10, 1000),
            "min_child_weight": (1, 100),
            "max_depth": (1, 10),
            "learning_rate": (0.001, 1),
            "min_split_loss": (0, 10),
            "colsample_bytree": (0.1, 1),
            "lambda": (0, 10),
            "alpha": (0, 10),
            "n_jobs": [1],
        },
    ],
    "KNNRegressor": [
        KNNRegressorClass,
        {
            "n_neighbors": (1, 1000),
            "p": [1, 2],
        },
    ],
    # "SVR": [
    #     SVRClass,
    #     {
    #         "C": (0.1, 100),
    #         "epsilon": (0.01, 1),
    #         "degree": (1, 5),
    #         "tol": (1e-4, 1e-1),
    #     },
    # ],
}
