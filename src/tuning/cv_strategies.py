import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from typing import Union, Literal
from collections import defaultdict
from models.model_classes import MLPRegressorClass

sys.path.insert(0, "src/")
from utils.utils import (
    compute_error_metrics,
    multi_dimension_mean_formatter,
    reset_weights,
)


class TSCrossValidation:
    def __init__(self, df, explainable_vars: list, dependet_var: list, model: object):

        self.df = df
        self.explainable_vars = explainable_vars
        self.dependent_var = dependet_var
        self.result_df = pd.DataFrame()
        self.model = model

        self.df["Year"] = self.df["Time"].dt.year
        self.df["Month"] = self.df["Time"].dt.month

    def history_yearly_CV(self):
        """Rolling window cross validation. The folds are yearly based.
        The first year of the df is regarded as upstart year and removed from the CV.
        The model is trained on all previous years and tested on the following years.


        Returns:
            pandas.DataFrame: DataFrame with the MAE for each year.
        """
        error_list = []
        sorted_years = self.df["Year"].unique()
        sorted_years.sort()
        sorted_years = sorted_years[1:]

        for i in range(1, len(sorted_years)):
            training_years = self.df[self.df["Year"] < sorted_years[i]]
            self.model.train(training_years, self.explainable_vars, self.dependent_var)
            preds = self.model.pred(
                self.df[self.df["Year"] == sorted_years[i]], self.explainable_vars
            )
            mae = mean_absolute_error(
                self.df[self.df["Year"] == sorted_years[i]][self.dependent_var], preds
            )
            error_df = pd.DataFrame({"Year": sorted_years[i], "MAE": mae}, index=[0])
            error_list.append(error_df)

        self.result_df = pd.concat(error_list, ignore_index=True)
        print(self.result_df)
        return self.result_df

    def history_monthly_CV(self):
        """Rolling window cross validation. The folds are monthly based.
        The first year of the df is regarded as upstart year and removed from the CV.
        The model is trained on all previous years and tested on the following months.

        Returns:
            pandas.DataFrame: DataFrame with the MAE for each month.
        """
        error_list = []
        sorted_years = self.df["Year"].unique()
        sorted_years.sort()
        sorted_months = self.df["Month"].unique()
        sorted_months.sort()
        sorted_years = sorted_years[1:]

        for i in range(1, len(sorted_years)):
            for j in range(len(sorted_months)):
                if (
                    len(
                        self.df[
                            (self.df["Year"] == sorted_years[i])
                            & (self.df["Month"] == sorted_months[j])
                        ]
                    )
                    == 0
                ):
                    continue
                training_years = self.df[
                    (self.df["Year"] < sorted_years[i])
                    | (
                        (self.df["Year"] == sorted_years[i])
                        & (self.df["Month"] < sorted_months[j])
                    )
                ]
                self.model.train(
                    training_years, self.explainable_vars, self.dependent_var
                )
                preds = self.model.pred(
                    self.df[
                        (self.df["Year"] == sorted_years[i])
                        & (self.df["Month"] == sorted_months[j])
                    ],
                    self.explainable_vars,
                )
                mae = mean_absolute_error(
                    self.df[
                        (self.df["Year"] == sorted_years[i])
                        & (self.df["Month"] == sorted_months[j])
                    ][self.dependent_var],
                    preds,
                )
                error_df = pd.DataFrame(
                    {"Year": sorted_years[i], "Month": sorted_months[j], "MAE": mae},
                    index=[0],
                )
                error_list.append(error_df)
        self.result_df = pd.concat(error_list, ignore_index=True)
        print(self.result_df)
        return self.result_df

    def hp_monthly_CV(self):
        """Expanding window cross validation. The folds are monthly based.

        Returns:
            pandas.DataFrame: DataFrame with actual and predicted values.
        """
        predictions_list = []  # List to store predictions and actual values
        sorted_months = self.df["Month"].unique()
        sorted_months.sort()
        training_month = self.df[self.df["Month"] == sorted_months[0]]
        self.model.train(training_month, self.explainable_vars, self.dependent_var)

        for i in range(1, len(sorted_months)):
            # if there is no data for the month, skip
            if len(self.df[self.df["Month"] == sorted_months[i]]) == 0:
                continue
            preds = self.model.pred(
                self.df[self.df["Month"] == sorted_months[i]], self.explainable_vars
            )
            actuals = self.df[self.df["Month"] == sorted_months[i]][
                self.dependent_var
            ].values.reshape(-1)

            # Store predictions and actuals with the original index
            pred_df = pd.DataFrame(
                {"Predictions": preds, "Actuals": actuals},
                index=self.df[self.df["Month"] == sorted_months[i]].index,
            )
            predictions_list.append(pred_df)

            # concatenate the training data with the current month
            training_month = pd.concat(
                [training_month, self.df[self.df["Month"] == sorted_months[i]]]
            )
            self.model.train(training_month, self.explainable_vars, self.dependent_var)

        self.result_df = pd.concat(predictions_list)  # Combine all predictions
        return self.result_df  # Return the DataFrame with predictions and actuals

    def hp_yearly_CV(self):
        predictions_list = []  # List to store predictions and actual values
        sorted_years = self.df["Year"].unique()
        sorted_years.sort()
        training_year = self.df[self.df["Year"] == sorted_years[0]]
        self.model.train(training_year, self.explainable_vars, self.dependent_var)

        for i in range(1, len(sorted_years)):
            # if there is no data for the month, skip
            if len(self.df[self.df["Year"] == sorted_years[i]]) == 0:
                continue
            preds = self.model.pred(
                self.df[self.df["Year"] == sorted_years[i]], self.explainable_vars
            )
            actuals = self.df[self.df["Year"] == sorted_years[i]][
                self.dependent_var
            ].values.reshape(-1)

            # Store predictions and actuals with the original index
            pred_df = pd.DataFrame(
                {"Predictions": preds, "Actuals": actuals},
                index=self.df[self.df["Year"] == sorted_years[i]].index,
            )
            predictions_list.append(pred_df)

            # concatenate the training data with the current year
            training_year = pd.concat(
                [training_year, self.df[self.df["Year"] == sorted_years[i]]]
            )
            self.model.train(training_year, self.explainable_vars, self.dependent_var)

        self.result_df = pd.concat(predictions_list)  # Combine all predictions
        return self.result_df  # Return the DataFrame with predictions and actuals


class CrossValidationStrategizer:
    """
    A class that provides different cross-validation strategies for time series data.

    This class implements expanding window and rolling window cross-validation for time series data,
    with support for multi-dimensional outputs and custom metrics. It handles different input formats
    (arrays, lists, Series) and automatically formats the results.

    Attributes:
        df (pd.DataFrame): The dataset containing both features and target variables.
        explainable_var (np.ndarray | list | str | pd.Series): Features/independent variables used for prediction.
            Can be column name(s), Series, or array of values.
        dependent_var (np.ndarray | list | str | pd.Series): Target/dependent variables to predict.
            Can be column name(s), Series, or array of values.
        model (object): The machine learning model with fit() and predict() methods.
        metric (object | str, optional): Evaluation metric to calculate performance.
            Can be a string referencing predefined metrics or a callable custom metric function.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        explainable_var: np.ndarray | list[str] | str | pd.Series,
        dependent_var: np.ndarray | list[str] | str | pd.Series,
        model: object,
        metric: (
            object
            | Literal[
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
            ]
        ) = None,
    ):
        """
        Initialize the CrossValidationStrategizer with dataset and model parameters.

        Parameters:
            df (pd.DataFrame): DataFrame containing the dataset with both features and targets.
            explainable_var (np.ndarray | list[str] | str | pd.Series):
                The input features for the model. Can be:
                - Column name(s) in the DataFrame
                - NumPy array of values
                - Pandas Series
            dependent_var (np.ndarray | list[str] | str | pd.Series):
                The target variable(s) to predict. Can be:
                - Column name(s) in the DataFrame
                - NumPy array of values
                - Pandas Series
            model (object): Machine learning model with fit() and predict() methods.
            metric (object | str, optional): Metric to evaluate model performance.
                - If None: Calculates all available metrics
                - If string: Uses one of the predefined metrics
                - If callable: Uses custom metric function
                Defaults to None.
        """
        self.df = df
        self.explainable_var = explainable_var
        self.dependent_var = dependent_var
        self.model = model
        self.metric = metric

    def __find_matching_columns(self, array: np.ndarray) -> Union[str, list]:
        """
        Find column names in the DataFrame that match the values in the provided array.

        This private method is used to identify which columns in the DataFrame correspond
        to the values in the input array. It handles both single-dimensional and
        multi-dimensional arrays.

        Parameters:
            array (np.ndarray): The NumPy array to match against DataFrame columns.
                Can be single or multi-dimensional.

        Returns:
            list: A list of column names from the DataFrame whose values match the input array.

        Raises:
            ValueError: If no matching columns are found in the DataFrame.

        Notes:
            - For multi-dimensional arrays, it checks each dimension against all columns
            - The comparison is done using np.array_equal for exact matching
            - This is used primarily to identify target variable column names when the
              dependent_var parameter is passed as an array
        """
        matching_columns = []
        if np.squeeze(array).ndim > 1:
            for d in range(array.ndim):
                for col in self.df.columns:
                    if np.array_equal(self.df[col].values, array[d]):
                        matching_columns.append(col)
        else:
            for col in self.df.columns:
                if np.array_equal(self.df[col].values, array):
                    matching_columns.append(col)

        if not matching_columns:
            raise ValueError("No matching columns found in DataFrame")

        if len(matching_columns) == 1:
            return matching_columns[0]
        else:
            return matching_columns

    def __variable_formatter(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Format explainable and dependent variables into numpy arrays for model fitting.

        This private method handles the conversion of different input formats (column names,
        numpy arrays, pandas Series) into consistent numpy arrays that can be used for
        model training and evaluation.

        The method intelligently processes:
        - Column names (strings or lists of strings): Extracts values from DataFrame
        - Pandas Series: Extracts underlying numpy array
        - Numpy arrays: Uses as-is

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - X: Formatted explainable variables as numpy array
                - y: Formatted dependent variables as numpy array

        Notes:
            This method is called internally by cross-validation methods to ensure
            consistent data format regardless of how the user provided the input
            variables during initialization.
        """
        if (
            isinstance(self.explainable_var, list)
            and all(isinstance(x, str) for x in self.explainable_var)
            or isinstance(self.explainable_var, str)
        ):
            X = self.df[self.explainable_var].values
        elif isinstance(self.explainable_var, pd.Series):
            X = self.explainable_var.values
        else:
            X = self.explainable_var

        if (
            isinstance(self.dependent_var, list)
            and all(isinstance(x, str) for x in self.dependent_var)
            or isinstance(self.dependent_var, str)
        ):
            y = self.df[self.dependent_var].values
        elif isinstance(self.dependent_var, pd.Series):
            y = self.dependent_var.values
        else:
            y = self.dependent_var

        return X, y

    def expanding_window(self, folds: int = 5) -> Union[dict, np.ndarray]:
        """
        Perform expanding window cross-validation for time series data.

        In expanding window validation, the training set grows with each fold while the test set
        moves forward in time. This approach simulates making predictions with an increasing amount
        of historical data, which is realistic for time series forecasting applications.

        Parameters:
            folds (int, optional): Number of validation folds to perform. Defaults to 5.

        Returns:
            Union[dict, np.ndarray]:
                - For multiple outputs or when metric is None: A dictionary mapping dependent variable
                  names to their respective error metrics.
                - For single output with specified metric: Array of error values across folds.

        Notes:
            - Uses scikit-learn's TimeSeriesSplit for time-aware fold generation
            - Automatically handles reshaping of inputs based on dimensionality
            - Supports multi-dimensional outputs and various metric calculations
            - Results are organized by dependent variable for multi-output models

        Example:
            >>> cv = CrossValidationStrategizer(df, ["feature1", "feature2"], "target", model)
            >>> results = cv.expanding_window(folds=3)
            >>> print(results)  # Shows metrics for each fold
        """
        X, y = self.__variable_formatter()
        is_multidim_output = np.squeeze(y).ndim > 1
        dependent_cols = (
            self.__find_matching_columns(y)
            if isinstance(self.dependent_var, np.ndarray)
            else self.dependent_var
        )

        tscv = TimeSeriesSplit(n_splits=folds)
        if self.metric is None:
            scores_list = defaultdict(list)
        elif isinstance(self.metric, str) and is_multidim_output:
            scores_list = defaultdict(list)
        else:
            scores_list = np.array([])

        for i, (train_index, test_index) in enumerate(tscv.split(X=X, y=y)):
            if type(self.model).__name__ == "MLPRegressorClass":
                if self.model.model is not None:
                    params = self.model.params
                    path = self.model.path
                    # add path to params if it exists
                    model = MLPRegressorClass()
                    self.model = model
                    self.model.update_model_params(params)
                    self.model.path = path

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Reshape based on dimensionality
            if X.ndim == 1:
                X_train = X_train.reshape(-1, 1)
                X_test = X_test.reshape(-1, 1)

            if not is_multidim_output:
                y_train = y_train.reshape(-1, 1)

            if is_multidim_output and X.ndim == 1:
                y_test = y_test.flatten()

            # Fit and predict the model
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)

            if self.metric is None:
                score = compute_error_metrics(
                    y_test,
                    preds,
                    dependent_cols=dependent_cols,
                )
                for key, value in score.items():
                    scores_list[key].append(value)
            elif isinstance(self.metric, str):
                if not is_multidim_output:
                    score = compute_error_metrics(y_test, preds)[self.metric]
                    scores_list = np.append(scores_list, score)
                else:
                    score = compute_error_metrics(
                        y_test, preds, dependent_cols=dependent_cols
                    )
                    for key, value in score.items():
                        scores_list[key].append(value[self.metric])
            else:
                score = self.metric(y_test, preds, multioutput="raw_values")
                if scores_list.size == 0:
                    scores_list = score
                else:
                    if is_multidim_output:
                        scores_list = np.column_stack((scores_list, score))
                    else:
                        scores_list = np.append(scores_list, score)

        if isinstance(scores_list, defaultdict):
            scores_list = dict(scores_list)
        elif isinstance(scores_list, np.ndarray) and scores_list.ndim > 1:
            scores_list = dict(zip(dependent_cols, scores_list))
        else:
            scores_list = {dependent_cols[0]: scores_list}
        return scores_list

    def rolling_window(
        self, folds: int = 5, train_ratio: float = 0.3
    ) -> Union[dict, np.ndarray]:
        """
        Perform rolling window cross-validation for time series data.

        In rolling window validation, both training and test sets move forward in time, maintaining
        a consistent training set size. This approach simulates real-world forecasting scenarios
        where models are regularly retrained on recent data of fixed size to predict the future.

        Parameters:
            folds (int, optional): Number of validation folds to perform. Defaults to 5.
            train_ratio (float, optional): Proportion of the dataset to use for training in each fold.
                Must be between 0 and 1. Defaults to 0.3 (30% of data).

        Returns:
            Union[dict, np.ndarray]:
                - For multiple outputs or when metric is None: A dictionary mapping dependent variable
                  names to their respective error metrics.
                - For single output with specified metric: Array of error values across folds.

        Raises:
            ValueError: If the lengths of explainable and dependent variables don't match.

        Notes:
            - Uses scikit-learn's TimeSeriesSplit with fixed train size for time-aware fold generation
            - Controls the size of training data with train_ratio parameter
            - Automatically reshapes inputs based on dimensionality requirements
            - Supports multi-dimensional outputs and various metric calculations
            - Results are organized by dependent variable for multi-output models

        Example:
            >>> cv = CrossValidationStrategizer(df, ["feature1", "feature2"], "target", model)
            >>> results = cv.rolling_window(folds=3, train_ratio=0.4)
            >>> print(results)  # Shows metrics for each fold
        """
        X, y = self.__variable_formatter()
        is_multidim_output = np.squeeze(y).ndim > 1
        dependent_cols = (
            self.__find_matching_columns(y)
            if isinstance(self.dependent_var, np.ndarray)
            else self.dependent_var
        )

        data_length = len(y)
        if len(X) != data_length:
            raise ValueError(
                f"Explainable variable length {len(X)} does not match dependent variable length {data_length}"
            )
        num_train_samples = np.floor(data_length * train_ratio).astype(int)
        num_test_samples = np.floor((data_length - num_train_samples) // folds).astype(
            int
        )

        tscv = TimeSeriesSplit(
            n_splits=folds, max_train_size=num_train_samples, test_size=num_test_samples
        )
        if self.metric is None:
            scores_list = defaultdict(list)
        elif isinstance(self.metric, str) and is_multidim_output:
            scores_list = defaultdict(list)
        else:
            scores_list = np.array([])

        for i, (train_index, test_index) in enumerate(tscv.split(X=X, y=y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Reshape based on dimensionality
            if X.ndim == 1:
                X_train = X_train.reshape(-1, 1)
                X_test = X_test.reshape(-1, 1)

            if not is_multidim_output and len(y_train.shape) > 1:
                y_train = y_train.reshape(-1, 1)

            if is_multidim_output and X.ndim == 1:
                y_test = y_test.flatten()

            # Fit and predict the model
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)

            if self.metric is None:
                score = compute_error_metrics(
                    y_test,
                    preds,
                    dependent_cols=dependent_cols,
                )
                for key, value in score.items():
                    scores_list[key].append(value)
            elif isinstance(self.metric, str):
                if not is_multidim_output:
                    score = compute_error_metrics(y_test, preds)[self.metric]
                    scores_list = np.append(scores_list, score)
                else:
                    score = compute_error_metrics(
                        y_test, preds, dependent_cols=dependent_cols
                    )
                    for key, value in score.items():
                        scores_list[key].append(value[self.metric])
            else:
                score = self.metric(y_test, preds, multioutput="raw_values")
                if scores_list.size == 0:
                    scores_list = score
                else:
                    if is_multidim_output:
                        scores_list = np.column_stack((scores_list, score))
                    else:
                        scores_list = np.append(scores_list, score)

        if isinstance(scores_list, defaultdict):
            scores_list = dict(scores_list)
        elif isinstance(scores_list, np.ndarray) and scores_list.ndim > 1:
            scores_list = dict(zip(dependent_cols, scores_list))
        else:
            scores_list = {dependent_cols: scores_list}
        return scores_list


# Example usage function - not executed on import
def example():
    """
    Example demonstrating how to use CrossValidationStrategizer.

    This function shows a complete workflow:
    1. Loading data from a pickle file
    2. Filtering for a specific turbine
    3. Setting up explanatory and dependent variables
    4. Creating and running cross-validation
    5. Formatting and displaying results

    To run this example, call this function directly.
    """
    import sys

    sys.path.insert(0, "src/")
    from models.model_classes import XGBRegressorClass

    df = pd.read_pickle("data/processed_data/off_shore/AAU_Park01.pkl")
    df = df[df["TurbineId"] == 1]

    explainable_var = ["WindSpeed", "BladeLoadA", "AmbTemp"]
    dependent_var = ["GridPower", "PitchAngleA", "WdAbs"]

    model = XGBRegressorClass(df=df)

    tscv = CrossValidationStrategizer(
        df, explainable_var, dependent_var, model=model, metric=None
    )
    scores = tscv.expanding_window()

    # Check if any value is a list of dictionaries, and if so, format once
    has_dict_lists = any(
        isinstance(value, list)
        and value
        and all(isinstance(item, dict) for item in value)
        for value in scores.values()
    )
    if has_dict_lists:
        scores = multi_dimension_mean_formatter(scores)

    # Now print the results for all keys
    for key, value in scores.items():
        if isinstance(value, list) and all(isinstance(item, float) for item in value):
            print(f"{key} - MAE: {np.mean(value)}")
        elif isinstance(value, dict) and "MAE" in value:
            print(f"{key} - MAE: {value['MAE']}")
        else:
            # For numpy arrays or other numeric types
            print(f"{key} - MAE: {np.mean(value) if hasattr(value, 'mean') else value}")


# Uncomment to run the example:
# if __name__ == "__main__":
#     example()
