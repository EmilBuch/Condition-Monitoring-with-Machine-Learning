import numpy as np
import pandas as pd
import sys

from sklearn import tree
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, "src/")
from tuning.cv_strategies import CrossValidationStrategizer


class PredictivePowerScore:
    """
    A class for calculating Predictive Power Scores (PPS) between variables in a dataset.

    The PPS measures the ability of one variable to predict another using a machine learning model.
    Unlike correlation, PPS can detect non-linear relationships and is asymmetric (the score of x predicting y
    may differ from y predicting x). The score is normalized between 0 (no predictive power) and 1 (perfect prediction).

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the data to analyze.
    df_index : str, default="Time"
        The column name to use as the dataframe index. The dataframe will be sorted by this index.
    model : sklearn estimator, default=DecisionTreeRegressor()
        The machine learning model used to calculate predictive power.
        Must implement fit() and predict() methods.
    eval_metric : function, default=mean_absolute_error
        The evaluation metric used to calculate the prediction score.
        Should take two arguments: y_true and y_pred.

    Methods
    -------
    score(x, y, folds=5, train_ratio=0.3, random_seed=123)
        Calculate the PPS of x predicting y.
    predictors(y, sorted=True, **kwargs)
        Find all predictors for a target variable y and their respective PPS.
    matrix(sorted=False, **kwargs)
        Calculate the PPS matrix for all pairs of variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from utils.pps import PredictivePowerScore
    >>>
    >>> # Create sample data
    >>> data = pd.DataFrame({
    ...     'Time': pd.date_range(start='2023-01-01', periods=100),
    ...     'A': np.random.rand(100),
    ...     'B': np.random.rand(100)
    ... })
    >>>
    >>> # Initialize PredictivePowerScore
    >>> pps = PredictivePowerScore(data)
    >>>
    >>> # Calculate PPS for one pair of variables
    >>> pps.score('A', 'B')
    >>>
    >>> # Get all predictors for variable 'B'
    >>> pps.predictors('B')
    >>>
    >>> # Get the entire PPS matrix
    >>> pps.matrix()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_index: str = "Time",
        model: object = tree.DecisionTreeRegressor(),
        eval_metric: object = mean_absolute_error,
    ):
        """
        Initialize the PredictivePowerScore calculator.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe containing the data for analysis.
            This dataframe should contain all features and target variables.

        df_index : str, default="Time"
            The column name to use as the dataframe index.
            The dataframe will be sorted by this index for temporal consistency.
            For time series data, this is typically a datetime column.

        model : sklearn estimator, default=DecisionTreeRegressor()
            The machine learning model used to calculate predictive power.
            Must implement fit() and predict() methods.
            Decision trees are used by default as they can capture non-linear relationships.

        eval_metric : function, default=mean_absolute_error
            The evaluation metric used to calculate prediction performance.
            Should take two arguments: y_true and y_pred.
            Mean absolute error is used by default for regression tasks.

        Notes
        -----
        The dataframe will be reindexed using the specified df_index column
        and sorted by this index during initialization.
        """
        self.df = df.set_index(df_index).sort_index()
        self.model = model
        self.eval_metric = eval_metric

    def _normalized_eval_score(self, model_mae, naive_mae):
        """
        Normalize the evaluation score to be between 0 and 1.

        This method calculates the normalized predictive power score by comparing
        the model's error to the baseline (naive) model's error. If the model performs
        worse than the baseline, the score is 0. Otherwise, the score is scaled
        to be between 0 and 1, where 1 indicates a perfect prediction.

        Parameters
        ----------
        model_mae : float
            The error metric value for the prediction model.
        naive_mae : float
            The error metric value for the naive baseline model.

        Returns
        -------
        float
            The normalized score between 0 and 1. A score of 0 means the model
            has no predictive power, while 1 means perfect prediction.
        """
        if model_mae > naive_mae:
            return 0
        else:
            return 1 - (model_mae / naive_mae)

    def _eval_score(self, y, model_score, **kwargs):
        """
        Calculate the PPS by comparing model performance to a naive baseline.

        This method creates a naive baseline prediction (using the median value of the target)
        and compares the model's performance against this baseline to calculate the PPS.

        Parameters
        ----------
        y : str
            The name of the target column.
        model_score : float
            The evaluation metric score of the predictive model.
        **kwargs : dict
            Additional parameters (not used in current implementation).

        Returns
        -------
        tuple
            A tuple containing (ppscore, baseline_score), where:
            - ppscore: float between 0 and 1, the normalized predictive power score
            - baseline_score: float, the evaluation metric score for the naive baseline model
        """
        score_df = self.df.copy()
        score_df["naive"] = score_df[y].median()
        baseline_score = self.eval_metric(
            score_df[y].to_numpy(), score_df["naive"].to_numpy()
        )  # true, pred

        ppscore = self._normalized_eval_score(model_score, baseline_score)
        return ppscore, baseline_score

    def _is_column_in_df(self, column):
        """
        Check if a column exists in the dataframe.

        This method safely checks if a column name exists in the dataframe's columns.

        Parameters
        ----------
        column : str
            The name of the column to check.

        Returns
        -------
        bool
            True if the column exists in the dataframe, False otherwise.
        """
        try:
            return column in self.df.columns
        except:
            return False

    def _calculate_model_cv_score_(self, target, feature, folds, train_ratio, **kwargs):
        """
        Calculate the cross-validated model score.

        This method uses the CrossValidationStrategizer to perform rolling window
        cross-validation and returns the mean error across all folds.

        Parameters
        ----------
        target : str
            The name of the target column to predict.
        feature : str
            The name of the feature column used for prediction.
        folds : int
            Number of cross-validation folds to use.
        train_ratio : float
            The ratio of data to use for training in each fold.
        **kwargs : dict
            Additional parameters to pass to the cross-validation strategy.

        Returns
        -------
        float
            The mean evaluation metric score across all cross-validation folds.
        """
        return np.mean(
            CrossValidationStrategizer(
                self.df, feature, target, self.model, self.eval_metric
            ).rolling_window(folds=folds, train_ratio=train_ratio)[target]
        )

    def _score(self, x, y, folds, train_ratio, random_seed):
        """
        Calculate the PPS and related metrics for a single feature-target pair.

        This is an internal method used by the public score() method after validation.
        It calculates the model score using cross-validation and compares it to
        the baseline to determine the predictive power score.

        Parameters
        ----------
        x : str
            The name of the feature column.
        y : str
            The name of the target column.
        folds : int
            Number of cross-validation folds to use.
        train_ratio : float
            The ratio of data to use for training in each fold.
        random_seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            A dictionary containing the following keys:
            - x: The feature column name
            - y: The target column name
            - ppscore: The calculated predictive power score
            - metric: The name of the evaluation metric used
            - baseline_score: The score of the naive baseline model
            - model_score: The score of the predictive model
            - model: The model object used for prediction
        """
        model_score = self._calculate_model_cv_score_(
            target=y,
            feature=x,
            folds=folds,
            train_ratio=train_ratio,
            random_seed=random_seed,
        )
        ppscore, baseline_score = self._eval_score(
            y, model_score, random_seed=random_seed
        )

        return {
            "x": x,
            "y": y,
            "ppscore": ppscore,
            "metric": self.eval_metric.__name__,
            "baseline_score": baseline_score,
            "model_score": model_score,
            "model": self.model,
        }

    def score(
        self,
        x: str,
        y: str,
        folds: int = 5,
        train_ratio: float = 0.3,
        random_seed: int = 123,
    ):
        """
        Calculate the Predictive Power Score between a feature and a target variable.

        This method computes how well one feature (x) can predict another feature (y)
        using cross-validated machine learning models. The score ranges from 0 (no predictive power)
        to 1 (perfect prediction).

        Parameters
        ----------
        x : str
            The name of the feature column (predictor).
        y : str
            The name of the target column to be predicted.
        folds : int, default=5
            Number of cross-validation folds to use.
        train_ratio : float, default=0.3
            The ratio of data to use for training in each fold.
            For example, 0.3 means 30% of the data is used for training.
        random_seed : int, default=123
            Random seed for reproducibility. If None, a random seed will be generated.

        Returns
        -------
        dict
            A dictionary containing the following keys:
            - x: The feature column name
            - y: The target column name
            - ppscore: The calculated predictive power score (0 to 1)
            - metric: The name of the evaluation metric used
            - baseline_score: The error score of the naive baseline model
            - model_score: The error score of the predictive model
            - model: The model object used for prediction

        Raises
        ------
        TypeError
            If the dataframe is not a pandas DataFrame.
        ValueError
            If the column names are not found in the dataframe.
        AssertionError
            If multiple columns with the same name are found.
        Exception
            Any other exceptions that occur during score calculation.

        Examples
        --------
        >>> pps = PredictivePowerScore(df)
        >>> pps.score('temperature', 'energy_consumption')
        {'x': 'temperature', 'y': 'energy_consumption', 'ppscore': 0.75, ...}
        """
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(
                f"The 'df' argument should be a pandas.DataFrame but you passed a {type(self.df)}\nPlease convert your input to a pandas.DataFrame"
            )
        if not self._is_column_in_df(x):
            raise ValueError(
                f"The 'x' argument should be the name of a dataframe column but the variable that you passed is not a column in the given dataframe.\nPlease review the column name or your dataframe"
            )
        if len(self.df[[x]].columns) >= 2:
            raise AssertionError(
                f"The dataframe has {len(self.df[[x]].columns)} columns with the same column name {x}\nPlease adjust the dataframe and make sure that only 1 column has the name {x}"
            )
        if not self._is_column_in_df(y):
            raise ValueError(
                f"The 'y' argument should be the name of a dataframe column but the variable that you passed is not a column in the given dataframe.\nPlease review the column name or your dataframe"
            )
        if len(self.df[[y]].columns) >= 2:
            raise AssertionError(
                f"The dataframe has {len(self.df[[y]].columns)} columns with the same column name {y}\nPlease adjust the dataframe and make sure that only 1 column has the name {y}"
            )

        if random_seed is None:
            from random import random

            random_seed = int(random() * 1000)

        try:
            return self._score(x, y, folds, train_ratio, random_seed)
        except Exception as exception:
            raise exception

    def _format_list_of_dicts(self, scores, sorted):
        """
        Format a list of score dictionaries into a pandas DataFrame.

        This internal method takes the raw score dictionaries from calculations
        and converts them into a structured DataFrame. If requested, the scores
        can be sorted by the predictive power score in descending order.

        Parameters
        ----------
        scores : list of dict
            A list of dictionaries, each containing the results of a PPS calculation.
        sorted : bool
            Whether to sort the scores by PPS value in descending order.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns for feature, target, PPS and other metrics.
        """
        if sorted:
            scores.sort(key=lambda item: item["ppscore"], reverse=True)

        df_columns = [
            "x",
            "y",
            "ppscore",
            "metric",
            "baseline_score",
            "model_score",
            "model",
        ]
        data = {column: [score[column] for score in scores] for column in df_columns}
        scores = pd.DataFrame.from_dict(data)

        return scores

    def predictors(self, y: str, sorted: bool = True, **kwargs):
        """
        Find all predictors for a target variable and their respective PPS.

        This method calculates the Predictive Power Score of every column in the dataframe
        (except the target column itself) predicting the specified target column.

        Parameters
        ----------
        y : str
            The name of the target column to be predicted.
        sorted : bool, default=True
            Whether to sort the results by PPS value in descending order.
        **kwargs : dict
            Additional arguments to pass to the score() method, such as:
            - folds: Number of cross-validation folds
            - train_ratio: Ratio of data to use for training
            - random_seed: Random seed for reproducibility

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing all features and their predictive power scores
            for the target variable, sorted by score if requested.

        Raises
        ------
        TypeError
            If the dataframe is not a pandas DataFrame.
        ValueError
            If the target column is not found in the dataframe or sorted is not a boolean.
        AssertionError
            If multiple columns with the same target name are found.

        Examples
        --------
        >>> pps = PredictivePowerScore(df)
        >>> pps.predictors('energy_consumption')
           x                 y  ppscore  ...
        0  temperature  energy_consumption  0.75  ...
        1  humidity     energy_consumption  0.43  ...
        """
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(
                f"The 'df' argument should be a pandas.DataFrame but you passed a {type(self.df)}\nPlease convert your input to a pandas.DataFrame"
            )
        if not self._is_column_in_df(y):
            raise ValueError(
                f"The 'y' argument should be the name of a dataframe column but the variable that you passed is not a column in the given dataframe.\nPlease review the column name or your dataframe"
            )
        if len(self.df[[y]].columns) >= 2:
            raise AssertionError(
                f"The dataframe has {len(self.df[[y]].columns)} columns with the same column name {y}\nPlease adjust the dataframe and make sure that only 1 column has the name {y}"
            )
        if not sorted in [True, False]:
            raise ValueError(
                f"""The 'sorted' argument should be one of [True, False] but you passed: {sorted}\nPlease adjust your input to one of the valid values"""
            )

        temp_scores = [
            self.score(column, y, **kwargs) for column in self.df if column != y
        ]

        return self._format_list_of_dicts(scores=temp_scores, sorted=sorted)

    def matrix(self, sorted: bool = False, **kwargs):
        """
        Calculate the complete Predictive Power Score matrix.

        This method computes the PPS for every possible pair of columns in the dataframe,
        generating a comprehensive matrix of predictive relationships between all variables.

        Parameters
        ----------
        sorted : bool, default=False
            Whether to sort the results by PPS value in descending order.
        **kwargs : dict
            Additional arguments to pass to the score() method, such as:
            - folds: Number of cross-validation folds
            - train_ratio: Ratio of data to use for training
            - random_seed: Random seed for reproducibility

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing all pairwise feature-target combinations and their
            respective predictive power scores.

        Raises
        ------
        TypeError
            If the dataframe is not a pandas DataFrame.
        ValueError
            If sorted is not a boolean.

        Examples
        --------
        >>> pps = PredictivePowerScore(df)
        >>> pps_matrix = pps.matrix()
        >>> # Convert to pivot table format if desired
        >>> pps_pivot = pps_matrix.pivot(index='x', columns='y', values='ppscore')
        """
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(
                f"The 'df' argument should be a pandas.DataFrame but you passed a {type(self.df)}\nPlease convert your input to a pandas.DataFrame"
            )
        if not sorted in [True, False]:
            raise ValueError(
                f"""The 'sorted' argument should be one of [True, False] but you passed: {sorted}\nPlease adjust your input to one of the valid values"""
            )

        scores = [self.score(x, y, **kwargs) for x in self.df for y in self.df]

        return self._format_list_of_dicts(scores=scores, sorted=sorted)
