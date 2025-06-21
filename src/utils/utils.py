import re
import pandas as pd
import numpy as np
from typing import Literal
from scipy.stats import kstest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
import multiprocessing as mp
from typing import Union
import sys
import torch
from torch import nn

sys.path.insert(0, "src/")


def calculate_directional_accuracy(
    actual: Union[pd.Series, np.ndarray], forecast: Union[pd.Series, np.ndarray]
) -> dict:
    if not isinstance(actual, pd.Series):
        actual = pd.Series(actual)
    if not isinstance(forecast, pd.Series):
        forecast = pd.Series(forecast)

    # Calculate actual and forecasted changes
    actual_change = actual.diff()
    forecast_change = forecast.diff()

    # Determine directions (1 for increase, 0 for decrease or no change)
    actual_direction = (actual_change > 0).astype(int)
    forecast_direction = (forecast_change > 0).astype(int)

    # Calculate accuracy
    correct_predictions = (actual_direction == forecast_direction).sum()
    total_predictions = len(actual_direction.dropna())
    accuracy = (correct_predictions / total_predictions) * 100

    # Calculate confusion matrix elements
    true_pos = ((actual_direction == 1) & (forecast_direction == 1)).sum()
    true_neg = ((actual_direction == 0) & (forecast_direction == 0)).sum()
    false_pos = ((actual_direction == 0) & (forecast_direction == 1)).sum()
    false_neg = ((actual_direction == 1) & (forecast_direction == 0)).sum()

    return {
        "accuracy": accuracy,
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "confusion_matrix": {
            "true_positive": true_pos,
            "true_negative": true_neg,
            "false_positive": false_pos,
            "false_negative": false_neg,
        },
    }


def multi_dimension_mean_formatter(scores: dict, col_name: str = None) -> dict:
    """Helper function to format multi-dimensional metrics for display"""
    # Compute the mean for each metric within each variable
    results = {}

    for variable, metrics_list in scores.items():
        # Convert list of dicts to DataFrame
        if col_name:
            df = pd.DataFrame(metrics_list, columns=[col_name])
        else:
            df = pd.DataFrame(metrics_list)
        mean_values = df.mean().to_dict()
        results[variable] = mean_values

    return results


def _compute_single_dimension_metrics(actual: pd.Series, forecast: pd.Series) -> dict:
    """Helper function to compute metrics for a single dimension"""
    error = actual - forecast
    max_error_idx = np.abs(error).idxmax()

    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    r2 = r2_score(actual, forecast)
    bias = np.mean(error)
    mape = mean_absolute_percentage_error(actual, forecast)
    over_predictions = (error > 0).sum() / len(actual) * 100
    under_predictions = (error < 0).sum() / len(actual) * 100
    exact_predictions = (error == 0).sum() / len(actual) * 100
    variance = np.var(error)
    std = np.std(error)
    directional_accuracy = calculate_directional_accuracy(actual, forecast)

    # PPScore
    pps = 1 - (mae / mean_absolute_error(actual, np.full_like(actual, actual.median())))

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Bias": bias,
        "Over_Predictions_Pct": over_predictions,
        "Under_Predictions_Pct": under_predictions,
        "Exact_Predictions_Pct": exact_predictions,
        "Variance": variance,
        "Standard_Deviation": std,
        "Max_Abs_Error": np.abs(error).max(),
        "Max_Error_Index": max_error_idx,
        "Directional_Acc": directional_accuracy["accuracy"],
        "PPScore": pps,
    }


def compute_error_metrics(
    actual: Union[pd.Series, pd.DataFrame, np.ndarray],
    forecast: Union[pd.Series, pd.DataFrame, np.ndarray],
    dependent_cols: Union[str, list] = None,
    include_mean_aggregate: bool = False,
) -> dict:
    """
    Compute various error metrics between actual and forecasted values.

    This function handles both single-dimensional and multi-dimensional inputs:
    - For 1D inputs: returns a flat dictionary of metrics
    - For multidimensional inputs: returns a nested dictionary with metrics per dimension

    Parameters:
    -----------
    actual : Union[pd.Series, pd.DataFrame, np.ndarray]
        Actual observed values
    forecast : Union[pd.Series, pd.DataFrame, np.ndarray]
        Forecasted or predicted values
    dependent_cols : Union[str, list], optional
        List of column names for multi-dimensional inputs, by default None
    include_mean_aggregate : bool, optional
        Whether to include aggregate metrics across all dimensions, by default False

    Returns:
    --------
    dict
        Dictionary containing error metrics, either flat (for 1D) or nested by dimension
    """
    if isinstance(actual, np.ndarray) and dependent_cols:
        actual = pd.DataFrame(actual, columns=dependent_cols)
    elif isinstance(actual, np.ndarray):
        actual = pd.Series(actual.flatten())

    if isinstance(forecast, np.ndarray) and dependent_cols:
        forecast = pd.DataFrame(forecast, columns=dependent_cols)
    elif isinstance(forecast, np.ndarray):
        forecast = pd.Series(forecast.flatten())

    # Handle single dimension case (Series or 1D array)
    if isinstance(actual, pd.Series) and isinstance(forecast, pd.Series):
        return _compute_single_dimension_metrics(actual, forecast)

    # Handle multidimensional case (DataFrame)
    if isinstance(actual, pd.DataFrame) and isinstance(forecast, pd.DataFrame):
        # Ensure the DataFrames have the same columns
        if set(actual.columns) != set(forecast.columns):
            raise ValueError(
                "Actual and forecast DataFrames must have the same columns"
            )

        results = {}
        # Calculate metrics for each column/dimension
        for col in actual.columns:
            results[col] = _compute_single_dimension_metrics(actual[col], forecast[col])

        if include_mean_aggregate:
            # Add aggregate metrics across all dimensions
            flattened_actual = actual.values.flatten()
            flattened_forecast = forecast.values.flatten()
            results["aggregate"] = _compute_single_dimension_metrics(
                pd.Series(flattened_actual), pd.Series(flattened_forecast)
            )

        return results

    raise ValueError(
        "Actual and forecast must both be Series, both DataFrames, or both numpy arrays"
    )


def two_sample_KS_test(
    base_year_df: pd.DataFrame,
    test_year_df: pd.DataFrame,
    var_dist: str,
    test_mode="less",
):

    ks_stat, p_value = kstest(
        base_year_df[var_dist],
        test_year_df[var_dist],
        alternative=test_mode,
    )

    return ks_stat, p_value


def calculate_percentage_diff(
    df: pd.DataFrame,
    y_year: int,
    y_col: str,
    yhat_year: int,
    yhat_col: str,
) -> float:
    """
    Calculate the percentage difference between two columns in a DataFrame
    Args:
    df: DataFrame
    base_year: int
    base_col: str
    comparison_year: int
    comparison_col: str
    """
    y = df[df["Year"] == y_year][y_col].sum()
    yhat = df[df["Year"] == yhat_year][yhat_col].sum()

    return ((y - yhat) / y) * 100


class RobustScalerClass:
    """Robust Scaler Class for scaling data using RobustScaler
    The class is used to fit and transform, saves the fitted scalers in a dictionary
    for later inverse transformation.
    """

    def __init__(self, df):
        self.scaler_dict = {}
        self.cols = list(df.columns)
        self.col_cleaner(df)

    def col_cleaner(self, df):
        self.cols = [col for col in self.cols if df[col].dtype == float]
        if "WSE" in self.cols:
            self.cols.remove("WSE") or self.cols

    def fit_transform_scaler(self, df: pd.DataFrame):
        for col in self.cols:
            scaler = RobustScaler()
            scaler.fit(df[col].values.reshape(-1, 1))
            self.scaler_dict[col] = scaler
            df[col] = scaler.transform(df[col].values.reshape(-1, 1))
        return df

    def inverse_transform(self, df: pd.DataFrame):
        for col in self.cols:
            df[col] = self.scaler_dict[col].inverse_transform(
                df[col].values.reshape(-1, 1)
            )
        return df


def delta_IQR_computation(before_df: pd.DataFrame, after_df: pd.DataFrame) -> dict:
    """
    Computes the delta (change) in IQR for each column that is shared between two DataFrames.

    The delta IQR is calculated as:

        delta_iqr = (before_iqr - after_iqr) / before_iqr * 100

    where before_iqr and after_iqr are the interquartile ranges (75th percentile - 25th percentile)
    of the respective features.

    Args:
        before_df (pd.DataFrame): DataFrame containing the "before" values.
        after_df (pd.DataFrame): DataFrame containing the "after" values.

    Returns:
        dict: A dictionary with each common column name as key and its computed delta IQR as value.
    """
    # Find the columns shared between the two DataFrames
    common_columns = after_df.columns.intersection(before_df.columns)
    results = {}

    # Iterate through each shared column and compute the delta IQR
    for col in common_columns:
        before_feature = before_df[col]
        after_feature = after_df[col]

        before_iqr = np.percentile(before_feature, 75) - np.percentile(
            before_feature, 25
        )
        after_iqr = np.percentile(after_feature, 75) - np.percentile(after_feature, 25)

        # Avoid division by zero if before_iqr is 0
        if before_iqr == 0:
            delta_iqr = np.nan
        else:
            delta_iqr = (before_iqr - after_iqr) / before_iqr * 100

        results[f"delta_iqr_{col}"] = delta_iqr

    return results


def delta_data(before_df, after_df):
    len_b_df = len(before_df)
    d_data = (len_b_df - len(after_df)) / len_b_df * 100

    return d_data


def sort_sheets(input_path, output_path):
    excel_file = pd.ExcelFile(input_path)

    sorted_sheets = sorted(
        excel_file.sheet_names, key=lambda x: int(re.search(r"\d+", x).group())
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name in sorted_sheets:
            df = excel_file.parse(sheet_name)
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def reset_weights(m: torch.nn.Module):
    """Try resetting model weights to avoid weight leakage.

    Args:
        m (torch.nn.Module): The model to reset.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()
