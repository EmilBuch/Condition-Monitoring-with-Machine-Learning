import pandas as pd
import numpy as np
import optuna
from typing import Union, List, Dict, Literal, Optional, Any
import sys

sys.path.insert(0, "src/")
from utils.utils import compute_error_metrics
from tuning.hp_tuning import HyperParameterTuning
from models.model_classes import (
    MLPRegressorClass,
    BayesianRidgeClass,
    ElasticNetClass,
    LRClass,
    RFRegressorClass,
    XGBRegressorClass,
    SVRClass,
)


class ModelDirector:
    """
    A director class implementing a fluent interface for machine learning workflows.

    This class provides a streamlined pipeline for common machine learning tasks,
    allowing users to chain operations in a readable and maintainable way. It handles
    the entire ML workflow from data loading to model evaluation.

    Workflow steps:
    1. Data loading (load_dataframe)
    2. Train-test splitting (perform_train_test_split)
    3. Model selection (set_model)
    4. Hyperparameter tuning (tune_hyperparameters) - optional
    5. Model training (fit)
    6. Prediction (predict)
    7. Evaluation (evaluate)
    8. Results retrieval (get_results)

    Example:
        results = (
            ModelDirector()
            .load_dataframe(feature_cols, target_cols, df=data)
            .perform_train_test_split(test_size=0.2)
            .set_model(model)
            .fit()
            .predict()
            .evaluate()
            .get_results()
        )
    """

    def __init__(self):
        """
        Initialize the ModelDirector class.

        The ModelDirector class provides a fluent interface for machine learning workflows
        including data loading, train-test splitting, model selection, hyperparameter tuning,
        training, prediction, and evaluation.

        Attributes:
            df (pd.DataFrame): The input dataframe containing features and targets.
            X_train (pd.DataFrame): Features for the training set.
            X_test (pd.DataFrame): Features for the test set.
            y_train (pd.DataFrame): Target values for the training set.
            y_test (pd.DataFrame): Target values for the test set.
            model (object): The machine learning model to be used.
            predictions (np.ndarray): Predictions from the model on the test set.
            error_metrics (dict): Evaluation metrics for the model predictions.
            best_params (dict): Best hyperparameters found during tuning.
            feature_cols (list): Column names for the feature variables.
            target_cols (list): Column names for the target variables.
        """
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.predictions = None
        self.error_metrics = None
        self.best_params = None
        self.feature_cols = None
        self.target_cols = None

    @property
    def X_train(self) -> pd.DataFrame:
        """
        Get the training features.

        Returns:
            pd.DataFrame: The training features.
        """
        return self._X_train

    @X_train.setter
    def X_train(self, value: pd.DataFrame) -> None:
        """
        Set the training features.

        Args:
            value (pd.DataFrame): The training features.
        """
        self._X_train = value

    @property
    def y_train(self) -> pd.DataFrame:
        """
        Get the training target values.

        Returns:
            pd.DataFrame: The training target values.
        """
        return self._y_train

    @y_train.setter
    def y_train(self, value: pd.DataFrame) -> None:
        """
        Set the training target values.

        Args:
            value (pd.DataFrame): The training target values.
        """
        self._y_train = value

    @property
    def X_test(self) -> pd.DataFrame:
        """
        Get the test features.

        Returns:
            pd.DataFrame: The test features.
        """
        return self._X_test

    @X_test.setter
    def X_test(self, value: pd.DataFrame) -> None:
        """
        Set the test features.

        Args:
            value (pd.DataFrame): The test features.
        """
        self._X_test = value

    @property
    def y_test(self) -> pd.DataFrame:
        """
        Get the test target values.

        Returns:
            pd.DataFrame: The test target values.
        """
        return self._y_test

    @y_test.setter
    def y_test(self, value: pd.DataFrame) -> None:
        """
        Set the test target values.

        Args:
            value (pd.DataFrame): The test target values.
        """
        self._y_test = value

    def load_dataframe(
        self,
        feature_cols: Union[List[str], str],
        target_cols: Union[List[str], str],
        df: pd.DataFrame = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> "ModelDirector":
        """
        Load and prepare data for model training and evaluation.

        This method loads features and target variables from either a pandas DataFrame
        or from separate feature and target arrays. It stores the column names for features
        and targets, and ensures data is sorted by time for time series analysis.

        Args:
            feature_cols (Union[List[str], str]): Column name(s) for feature variables.
                Can be a single string or list of strings.
            target_cols (Union[List[str], str]): Column name(s) for target variables.
                Can be a single string or list of strings.
            df (pd.DataFrame, optional): DataFrame containing both features and target variables.
                Default is None.
            X (np.ndarray, optional): Array of feature values. Used only if df is None.
                Default is None.
            y (np.ndarray, optional): Array of target values. Used only if df is None.
                Default is None.

        Returns:
            ModelDirector: Self reference for method chaining.

        Raises:
            ValueError: If neither df nor both X and y are provided.

        Note:
            If a DataFrame is provided, it will be sorted by the 'Time' column and reindexed.
        """
        self.feature_cols = (
            feature_cols if isinstance(feature_cols, list) else [feature_cols]
        )
        self.target_cols = (
            target_cols if isinstance(target_cols, list) else [target_cols]
        )

        if df is not None:
            self.df = df
        elif X is not None and y is not None:
            data = np.column_stack((X, y))
            self.df = pd.DataFrame(data, columns=self.feature_cols + self.target_cols)
        else:
            raise ValueError("No data provided. Use either df or X and y.")
        return self

    def perform_train_test_split(
        self,
        test_size: float = 0.2,
        split_method: Literal["sequential", "yearly"] = "sequential",
        test_year: Optional[int] = None,
    ) -> "ModelDirector":
        """
        Perform timeseries train test split.

        Args:
            test_size: Proportion of data to use for testing (for sequential split)
            split_method: Method to use for splitting:
                - 'sequential': Use last test_size proportion of data for testing
                - 'yearly': Split by year (requires 'Year' column)
            test_year: Specific year to use for testing (for yearly split)

        Returns:
            self: For method chaining
        """
        if self.df is None:
            raise ValueError("Data not loaded. Use load_dataset first.")

        # Extract features and target
        X = self.df[self.feature_cols]
        y = self.df[self.target_cols]

        if split_method == "sequential":
            # Simple approach where last N% is test
            split_idx = int(len(self.df) * (1 - test_size))

            self.X_train = X.iloc[:split_idx]
            self.X_test = X.iloc[split_idx:]
            self.y_train = y.iloc[:split_idx]
            self.y_test = y.iloc[split_idx:]

        elif split_method == "yearly":
            if "Year" not in self.df.columns:
                raise ValueError("Year column not found in dataset")
            # Split by year
            years = sorted(self.df["Year"].unique())

            if test_year is None:
                test_year = years[-1]  # Use last year as test by default
            elif test_year not in years:
                raise ValueError(f"Test year {test_year} not found in dataset")

            train_mask = self.df["Year"] < test_year
            test_mask = self.df["Year"] >= test_year

            self.X_train = X[train_mask]
            self.X_test = X[test_mask]
            self.y_train = y[train_mask]
            self.y_test = y[test_mask]

        else:
            raise ValueError(f"Unsupported split_method: {split_method}")

        return self

    def set_model(self, model: Any) -> "ModelDirector":
        """
        Set the model to use.

        Args:
            model: Machine learning model with fit and predict methods

        Returns:
            self: For method chaining
        """
        self.model = model
        return self

    def tune_hyperparameters(
        self,
        turbine_id: int,
        param_search_space: Dict,
        n_trials: int = 100,
        cv_mode: Literal[
            "yearly", "monthly", "expanding", "rolling", "sliding"
        ] = "expanding",
        metric: Literal[
            "MAE",
            "MSE",
            "RMSE",
            "MAPE",
            "R2",
            "Bias",
            "Over_Predictions_Pct",
            "Under_Predictions_Pct",
            "Exact_Predictions_Pct",
            "Variance",
            "Standard_Deviation",
            "Max_Abs_Error",
            "Max_Error_Index",
            "Directional_Acc",
            "PPScore",
        ] = "MAE",
        optimization_direction: optuna.study.StudyDirection = optuna.study.StudyDirection.MINIMIZE,
        folds: Optional[int] = 5,
        savefile: Optional[bool] = True,
        filename: Optional[str] = "tuning_results.csv",
        experiment_name: str = None,
    ) -> "ModelDirector":
        """
        Tune hyperparameters of the model using Optuna framework.

        This method performs hyperparameter optimization using the Optuna framework
        with various cross-validation strategies appropriate for time series data.
        The best parameters found are automatically applied to the model.

        Args:
            param_search_space (Dict): Dictionary defining the hyperparameter search space.
                Format should be {param_name: (min_value, max_value)} for numeric parameters.
            n_trials (int, optional): Number of optimization trials to run. Default is 100.
            cv_mode (Literal["yearly", "monthly", "expanding", "rolling", "sliding"], optional):
                Cross-validation strategy for time series data. Default is "expanding".
                - "yearly": Split data by years
                - "monthly": Split data by months
                - "expanding": Use expanding window validation
                - "rolling": Use rolling window validation
                - "sliding": Use sliding window validation
            metric (Literal[...], optional): Evaluation metric to optimize. Default is "MAE".
                Available options include various error metrics like MAE, RMSE, R2, etc.
            optimization_direction (optuna.study.StudyDirection, optional):
                Direction of optimization (minimize or maximize). Default is to minimize.
            folds (Optional[int], optional): Number of cross-validation folds. Default is 5.
            savefile (Optional[bool], optional): Whether to save tuning results to file. Default is True.
            filename (Optional[str], optional): Name of the file to save results. Default is "tuning_results.csv".

        Returns:
            ModelDirector: Self reference for method chaining.

        Raises:
            ValueError: If model is not set or training data is not available.
            ValueError: If full dataset is not available for hyperparameter tuning.

        Note:
            The best parameters found will be automatically applied to the model.
            If the model has a `update_model_params` method, it will be used,
            otherwise parameters are set using attribute assignment.
        """
        if self.model is None:
            raise ValueError("Model not set. Use set_model first.")
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not available. Split the data first.")

        if self.df is not None:
            df = self.df.loc[self.X_train.index]
            # Extract features and targets for tuning
            X = self.X_train.values
            y = self.y_train.values

            if np.squeeze(y).ndim == 1 and y.ndim > 1:
                y = np.squeeze(y)

            # Create tuner object
            tuner = HyperParameterTuning(
                df=df,
                explainable_vars=X,
                feature_names=self.feature_cols,
                experiment_name=experiment_name,
                dependet_var=y,
                model=self.model,
                metric=metric,
                cv_mode=cv_mode,
                folds=folds,
            )

            # Run tuning
            self.best_params = tuner.run(
                turbine_id=turbine_id,
                param_search_space=param_search_space,
                n_trials=n_trials,
                optimization_direction=optimization_direction,
                savefile=savefile,
                filename=filename,
            )

            # Update model with best parameters
            if hasattr(self.model, "update_model_params"):
                self.model.update_model_params(self.best_params)
            else:
                # Fallback for models without update_model_params method
                for param, value in self.best_params.items():
                    if hasattr(self.model, param):
                        setattr(self.model, param, value)

        else:
            raise ValueError(
                "Full dataset required for hyperparameter tuning. Use load_dataset."
            )

        return self

    def fit(self) -> "ModelDirector":
        """
        Fit the model on training data.

        Returns:
            self: For method chaining
        """
        if self.model is None:
            raise ValueError("Model not set. Use set_model first.")
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not available. Split the data first.")

        if len(self.target_cols) == 1:
            self.model.fit(self.X_train, self.y_train.values.ravel())
        else:
            self.model.fit(self.X_train, self.y_train)
        return self

    def predict(self) -> "ModelDirector":
        """
        Make predictions on test data.

        Returns:
            self: For method chaining
        """
        if self.model is None:
            raise ValueError("Model not fitted. Use fit_model first.")
        if self.X_test is None:
            raise ValueError("Test data not available. Split the data first.")

        self.predictions = self.model.predict(self.X_test)
        return self

    def evaluate(
        self,
        include_mean_aggregate: Optional[bool] = False,
    ) -> "ModelDirector":
        """
        Evaluate predictions using error metrics from utils.py.

        Args:
            include_mean_aggregate: Whether to include aggregate metrics
            dependent_cols: List of column names for multi-dimensional outputs

        Returns:
            self: For method chaining
        """
        if self.predictions is None:
            raise ValueError("No predictions available. Use predict first.")

        self.error_metrics = compute_error_metrics(
            actual=self.y_test.values,
            forecast=self.predictions,
            dependent_cols=self.target_cols,
            include_mean_aggregate=include_mean_aggregate,
        )
        return self

    def get_results(self) -> Dict:
        """
        Get all results as a dictionary.

        Returns:
            Dictionary containing results
        """
        return {
            "model": self.model,
            "best_params": self.best_params,
            "y_pred": self.predictions,
            "y_true": self.y_test,
            "error_metrics": self.error_metrics,
        }


if __name__ == "__main__":
    import os

    # Example usage
    md = ModelDirector()

    df = pd.read_pickle(
        os.path.join(
            os.getcwd(), "data", "nbm_selector_data", "off_shore", "AAU_Park01.pkl"
        )
    )
    df = (
        df[(df["TurbineId"] == 4) & (df["is_stable"] == True)]
        .sort_values("Time")
        .reset_index()
    )

    explainable_vars = df.loc[
        :,
        (df.columns != "GridPower")
        & (df.columns != "Time")
        & (df.columns != "TurbineId")
        & (df.columns != "is_stable"),
    ]
    target_var = df["GridPower"]

    # Then chain the methods as needed
    md.load_dataframe(
        feature_cols=explainable_vars.columns.tolist(),
        target_cols=target_var.name,
        X=explainable_vars,
        y=target_var,
    ).perform_train_test_split(split_method="sequential")

    # linear_regression = md.set_model(LRClass()).fit().predict().evaluate().get_results()
    mlp = (
        md.set_model(MLPRegressorClass())
        # .tune_hyperparameters(
        #     param_search_space={
        #         "learning_rate": (0.00001, 0.01),
        #         "batch_size": (4, 128),
        #     },
        #     n_trials=10,
        #     cv_mode="rolling",
        # )
        .fit()
        .predict()
        .evaluate()
        .get_results()
    )
    # xgboost = md.set_model(XGBRegressorClass()).fit().predict().evaluate().get_results()
    # svr = (
    #     md.set_model(SVRClass({"kernel": "poly"}))
    #     .fit()
    #     .predict()
    #     .evaluate()
    #     .get_results()
    # )
    # random_forest = (
    #     md.set_model(RFRegressorClass()).fit().predict().evaluate().get_results()
    # )
    # elastic_net = (
    #     md.set_model(ElasticNetClass()).fit().predict().evaluate().get_results()
    # )
    # bayesian = (
    #     md.set_model(BayesianRidgeClass()).fit().predict().evaluate().get_results()
    # )

    # print(f"Linear Regression: {linear_regression['error_metrics']['GridPower']}")
    # print(f"Elastic Net: {elastic_net['error_metrics']['GridPower']}")
    # print(f"Bayesian Ridge: {bayesian['error_metrics']['GridPower']}")
    # print(f"SVR: {svr['error_metrics']['GridPower']}")
    # print(f"Random Forest: {random_forest['error_metrics']['GridPower']}")
    # print(f"XGBoost: {xgboost['error_metrics']['GridPower']}")
    print(f"MLP Params: {mlp['best_params']}")
    print(f"MLP Errors: {mlp['error_metrics']['GridPower']}")
