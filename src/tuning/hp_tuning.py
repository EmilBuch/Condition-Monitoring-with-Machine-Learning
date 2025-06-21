import optuna
import os
import json
import warnings
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Literal, Union, Optional

import sys

sys.path.insert(0, "src/")
from utils.utils import compute_error_metrics, multi_dimension_mean_formatter
from tuning.cv_strategies import TSCrossValidation, CrossValidationStrategizer
from models.model_classes import XGBRegressorClass


MODELS_DIR = os.path.join(os.getcwd(), "models", date.today().strftime("%Y%m%d"))
os.makedirs(MODELS_DIR, exist_ok=True)


class HyperParameterTuning(CrossValidationStrategizer):
    """
    A class for automated hyperparameter optimization using cross-validation techniques.

    This class extends CrossValidationStrategizer to provide hyperparameter tuning capabilities
    through Optuna's optimization framework. It supports various cross-validation strategies
    such as expanding window, rolling window, yearly, monthly, and sliding window approaches.

    The class handles the full hyperparameter tuning workflow including:
    - Parameter space definition and sampling
    - Cross-validation execution
    - Result aggregation and evaluation
    - Persistence of tuning results

    It's designed to work with any model that implements an update_model_params method
    to incorporate new hyperparameter settings during optimization trials.

    Example
    -------
    ```python
    # Create model and define search space
    model = XGBRegressorClass()
    search_space = {
        "n_estimators": (1, 1000),
        "max_depth": (1, 10),
        "learning_rate": (0.005, 0.5)
    }

    # Initialize tuning with expanding window CV
    tuner = HyperParameterTuning(
        df, X_features, y_target, model, "MAE", "expanding", 5
    )

    # Run optimization
    best_params = tuner.run(
        param_search_space=search_space,
        n_trials=100,
        optimization_direction=optuna.study.StudyDirection.MINIMIZE
    )
    ```
    """

    def __init__(
        self,
        df: pd.DataFrame,
        explainable_vars: np.ndarray | list,
        dependet_var: np.ndarray | str,
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
        ),
        cv_mode: Literal[
            "yearly", "monthly", "expanding", "rolling", "sliding"
        ] = "expanding",
        folds: int = 5,
        train_ratio: float = 0.3,
        feature_names: Optional[list] = None,
        experiment_name: str = None,
    ):
        """
        Initialize a hyperparameter tuning instance for model optimization using cross-validation strategies.

        This class handles hyperparameter optimization through different cross-validation modes,
        including expanding window, rolling window, and others. It leverages Optuna for efficient
        parameter searching and evaluates models using specified metrics.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing both feature variables and target variables.
        explainable_vars : np.ndarray or list
            Feature variables used for prediction. Can be column names or pre-extracted values.
        dependet_var : np.ndarray or str
            Target variable to predict. Can be a column name or pre-extracted values.
        model : object
            Model object to be tuned. Must implement an update_model_params method.
        metric : str or callable
            Evaluation metric for model performance. If string, must be one of the predefined
            metrics (MAE, MSE, RMSE, etc.). If callable, should accept predictions and actual
            values and return a score.
        cv_mode : {"yearly", "monthly", "expanding", "rolling", "sliding"}, default="expanding"
            Cross-validation strategy to use for evaluating hyperparameters:
            - "yearly": Year-based splits
            - "monthly": Month-based splits
            - "expanding": Incrementally growing training window
            - "rolling": Fixed-size moving window
            - "sliding": Custom sliding window approach
        folds : int, default=5
            Number of cross-validation folds to use.
        train_ratio : float, default=0.3
            Ratio of data to use for training in rolling window validation.
            Only used when cv_mode="rolling".

        Attributes
        ----------
        cv_mode : str
            Selected cross-validation mode.
        folds : int
            Number of cross-validation folds.
        train_ratio : float
            Training data ratio for rolling window validation.

        Notes
        -----
        - The parent class CrossValidationStrategizer handles basic cross-validation functionality.
        - The class is designed to work with Optuna for hyperparameter optimization.
        - Results can be saved to CSV for later analysis and comparison.
        """
        super(HyperParameterTuning, self).__init__(
            df, explainable_vars, dependet_var, model, metric
        )
        self.cv_mode = cv_mode
        self.folds = folds
        self.train_ratio = train_ratio
        self.feature_names = feature_names
        self.experiment_name = experiment_name

    def __save_tuning_results(
        self,
        study: optuna.Study,
        turbine_id: int,
        param_search_space: dict,
        model_name: str,
        cv_mode: str,
        metric: str,
        filename: str = "tuning_results.csv",
        overwrite: bool = False,
    ):
        """
        Save hyperparameter tuning results to a CSV file.

        This method extracts key information from an Optuna study, including the best trial
        parameters, trial statistics, and optimization details, and saves them to a CSV file
        for later analysis and comparison.

        Parameters
        ----------
        study : optuna.Study
            The completed Optuna study containing trial information and results.
        param_search_space : dict
            Dictionary defining the hyperparameter search space used in the study.
        model_name : str
            Name of the model being tuned.
        cv_mode : str
            Cross-validation mode used for evaluation (e.g., "expanding", "rolling").
        metric : str
            Name of the evaluation metric used for optimization.
        filename : str, default="tuning_results.csv"
            Name of the CSV file to save results to. Will be saved in MODELS_DIR.
        overwrite : bool, default=False
            If True, overwrite existing file if it exists. If False, append to it.

        Returns
        -------
        None
            Results are saved to disk, nothing is returned.

        Notes
        -----
        - Saved information includes:
          - Timestamp of the tuning run
          - Best trial number and performance value
          - Best hyperparameters (JSON formatted)
          - Search space definition (JSON formatted)
          - Model name and cross-validation details
          - Optimization direction
          - Trial statistics (total, completed, pruned, failed)
        - The file is saved in the MODELS_DIR directory, which is defined at module level
        - If the file already exists, this method will either append to it or overwrite it
          depending on the overwrite parameter
        """
        # Count trial states
        pruned_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ]
        complete_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        failed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.FAIL
        ]

        # Get best trial
        best_trial = study.best_trial

        # Create results dictionary
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "turbine": turbine_id,
            "model_name": model_name,
            "best_trial_number": best_trial.number,
            "best_trial_value": best_trial.value,
            "explainable_vars": json.dumps(self.feature_names),
            "best_params": json.dumps(best_trial.params),
            "search_space": param_search_space,
            "cv_mode": cv_mode,
            "metric": metric,
            "optimization_direction": str(study.direction),
            "total_trials": len(study.trials),
            "completed_trials": len(complete_trials),
            "pruned_trials": len(pruned_trials),
            "failed_trials": len(failed_trials),
        }
        if self.feature_names != None:
            results["feature_cols"] = self.feature_names
        if self.experiment_name != None:
            results["experiment_name"] = self.experiment_name

        # Create DataFrame
        tuning_results_df = pd.DataFrame.from_dict(results, orient="index").T

        # Save to CSV
        filename = os.path.join(MODELS_DIR, filename)

        # Ensure the directory exists.
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Check if file exists.
        if os.path.exists(filename):
            if overwrite:
                warnings.warn(
                    f"File '{filename}' already exists and will be overwritten."
                )
                tuning_results_df.to_csv(filename, index=False)
            else:
                existing_df = pd.read_csv(filename)
                combined_df = pd.concat(
                    [existing_df, tuning_results_df], ignore_index=True
                )
                combined_df.to_csv(filename, index=False)
        else:
            tuning_results_df.to_csv(filename, index=False)

    def __prediction_formatter(
        self,
        preds: Union[dict, np.ndarray],
        score_type: Literal["mean", "sum", "median", "max", "min"] = "mean",
    ) -> float:
        """
        Format and aggregate prediction results from cross-validation.

        This method processes prediction results that may come in different formats
        (dictionary or array) and computes an aggregated score based on the specified
        metric and aggregation method.

        Parameters
        ----------
        preds : Union[dict, np.ndarray]
            Predictions to be evaluated. Can be:
            - Dictionary mapping fold IDs to prediction metrics
            - Dictionary mapping dimension IDs to metrics (for multi-dimensional predictions)
            - Direct numpy array of scores
        score_type : {"mean", "sum", "median", "max", "min"}, default="mean"
            Method to aggregate scores across folds or dimensions:
            - "mean": Average of all scores (default)
            - "sum": Sum of all scores
            - "median": Median of all scores
            - "max": Maximum score
            - "min": Minimum score

        Returns
        -------
        float
            Aggregated performance score based on the specified metric and aggregation method.

        Notes
        -----
        - For multi-dimensional predictions, the method first formats the results using
          the multi_dimension_mean_formatter utility
        - For single-dimension predictions in dictionary format, it extracts the values
        - The aggregation is performed based on the score_type parameter
        """

        def aggregate_scores(scores: Union[dict, np.ndarray], score_type: str) -> float:
            if isinstance(scores, dict):
                return {
                    "mean": np.mean([stats[self.metric] for stats in scores.values()]),
                    "sum": np.sum([stats[self.metric] for stats in scores.values()]),
                    "median": np.median(
                        [stats[self.metric] for stats in scores.values()]
                    ),
                    "max": np.max([stats[self.metric] for stats in scores.values()]),
                    "min": np.min([stats[self.metric] for stats in scores.values()]),
                }[score_type]
            else:
                return {
                    "mean": np.mean(scores),
                    "sum": np.sum(scores),
                    "median": np.median(scores),
                    "max": np.max(scores),
                    "min": np.min(scores),
                }[score_type]

        if isinstance(preds, dict):
            if len(preds) > 1:  # Multiple dimensions
                scores = multi_dimension_mean_formatter(preds, self.metric)
            else:  # Single dimension
                scores = next(iter(preds.values()))
        else:
            scores = preds
        return aggregate_scores(scores, score_type)

    def __objective_cv(self, trial: optuna.Trial, param_search_space: dict) -> float:
        """
        Objective function for Optuna hyperparameter optimization.

        This function is called for each trial in the hyperparameter optimization process.
        It interprets the parameter search space, suggests values for each parameter based
        on the search space definition, updates the model with these parameters, performs
        cross-validation, and returns the evaluation metric to be optimized.

        Parameters
        ----------
        trial : optuna.Trial
            Current optimization trial object used to suggest parameter values.
        param_search_space : dict
            Dictionary defining the hyperparameter search space. Supports multiple formats:
            - Tuple (min, max) for numeric parameters
            - Dictionary with "type", "min", "max", "step" for stepped parameters
            - List for categorical parameters
            - Single values for fixed parameters

        Returns
        -------
        float
            Evaluation score from cross-validation with the suggested parameters.
            This value will be minimized or maximized by Optuna depending on the
            direction specified when creating the study.

        Raises
        ------
        NotImplementedError
            If cv_mode is "yearly", "monthly", or "sliding", which are not yet implemented.
        ValueError
            If cv_mode is not one of the recognized modes.

        Notes
        -----
        - Parameter types are automatically inferred from the search space definition
        - For each parameter, appropriate Optuna suggest_* methods are called based on type
        - Cross-validation is performed using the specified mode (expanding, rolling, etc.)
        - The resulting predictions are formatted and aggregated to a single score
        """
        params = {}
        for name, space in param_search_space.items():
            if isinstance(space, tuple) and len(space) == 2:
                if all(isinstance(x, int) for x in space):
                    params[name] = trial.suggest_int(name, space[0], space[1])
                elif all(isinstance(x, (int, float)) for x in space):
                    params[name] = trial.suggest_float(name, space[0], space[1])
            elif (
                isinstance(space, dict)
                and space.get("type") == "float"
                and "step" in space
            ):
                params[name] = trial.suggest_float(
                    name, space["min"], space["max"], step=space["step"]
                )
            elif (
                isinstance(space, dict)
                and space.get("type") == "int"
                and "step" in space
            ):
                params[name] = trial.suggest_int(
                    name, space["min"], space["max"], step=space["step"]
                )
            elif isinstance(space, list):
                params[name] = trial.suggest_categorical(name, space)
            else:
                # Fixed value
                params[name] = space
        self.model.update_model_params(params)

        if self.cv_mode == "yearly":
            raise NotImplementedError("Yearly CV not implemented")
        elif self.cv_mode == "monthly":
            raise NotImplementedError("Monthly CV not implemented")
        elif self.cv_mode == "expanding":
            preds = self.expanding_window(folds=self.folds)
        elif self.cv_mode == "rolling":
            preds = self.rolling_window(folds=self.folds, train_ratio=self.train_ratio)
        elif self.cv_mode == "sliding":
            raise NotImplementedError("Sliding CV not implemented")
        else:
            raise ValueError(
                "Invalid cv_mode, must be 'yearly', 'monthly', 'expanding', 'rolling', or 'sliding'"
            )

        eval_score = self.__prediction_formatter(preds, "mean")
        return eval_score

    def run(
        self,
        param_search_space: dict,
        n_trials: int,
        turbine_id: int,
        optimization_direction: optuna.study.StudyDirection,
        sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(),
        pruner: optuna.pruners.BasePruner = optuna.pruners.SuccessiveHalvingPruner(),
        savefile: bool = True,
        filename: str = "tuning_results.csv",
        overwrite: bool = False,
    ) -> dict:
        """
        Execute the hyperparameter optimization process.

        This method runs the complete hyperparameter tuning workflow by creating an Optuna study,
        optimizing parameters according to the specified search space, and tracking the results.
        It handles the creation of trials, execution of the objective function via cross-validation,
        and optional persistence of results.

        Parameters
        ----------
        param_search_space : dict
            Dictionary defining the hyperparameter search space. The structure depends on parameter types:
            - For numeric ranges: (min, max) tuples
            - For stepped numeric values: {"type": "float"/"int", "min": val, "max": val, "step": val}
            - For categorical values: [val1, val2, ...]
            - For fixed values: direct value assignment
        n_trials : int
            Number of optimization trials to run. Higher values provide better results but
            take longer to complete.
        optimization_direction : optuna.study.StudyDirection
            Direction of optimization: MINIMIZE (lower is better) or MAXIMIZE (higher is better).
            Typically MINIMIZE for error metrics like MAE or MSE, MAXIMIZE for metrics like R².
        sampler : optuna.samplers.BaseSampler, default=optuna.samplers.TPESampler()
            Sampling algorithm for parameter suggestions. The default TPESampler uses
            Bayesian optimization to efficiently explore the search space.
        pruner : optuna.pruners.BasePruner, default=optuna.pruners.SuccessiveHalvingPruner()
            Strategy for early stopping unpromising trials to improve optimization efficiency.
        savefile : bool, default=True
            Whether to save tuning results to a CSV file.
        filename : str, default="tuning_results.csv"
            Name of the file to save results to (if savefile=True).
        overwrite : bool, default=False
            Whether to overwrite existing results file (if savefile=True and file exists).
            If False, results will be appended to the existing file.

        Returns
        -------
        dict
            Dictionary containing the best hyperparameters found during optimization.

        Notes
        -----
        - The method prints information about the best trial to stdout
        - When savefile=True, detailed results are saved using the __save_tuning_results method
        - The optimization uses the internal __objective_cv method which performs cross-validation
        - Early stopping can occur based on the pruner's strategy

        Example
        -------
        ```python
        tuner = HyperParameterTuning(df, X, y, model, "MAE")
        best_params = tuner.run(
            param_search_space={"n_estimators": (10, 100), "max_depth": (2, 8)},
            n_trials=50,
            optimization_direction=optuna.study.StudyDirection.MINIMIZE
        )
        ```
        """
        study = optuna.create_study(
            direction=optimization_direction,
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(
            lambda trial: self.__objective_cv(
                trial=trial, param_search_space=param_search_space
            ),
            n_trials=n_trials,
        )

        trial = study.best_trial
        print(f"\nBest trial: {trial.number}")
        print("\tValue: ", trial.value)
        print("\tParams: ")
        for key, value in trial.params.items():
            print("\t\t{}: {}".format(key, value))

        if savefile:
            self.__save_tuning_results(
                study=study,
                turbine_id=turbine_id,
                param_search_space=param_search_space,
                model_name=self.model.__class__.__name__,
                cv_mode=self.cv_mode,
                metric=(
                    self.metric
                    if isinstance(self.metric, str)
                    else self.metric.__name__
                ),
                filename=filename,
                overwrite=overwrite,
            )

        return trial.params


class XGBHyperParameterTuning(TSCrossValidation):
    def __init__(
        self,
        df: pd.DataFrame,
        explainable_vars: list,
        dependet_var: list,
        model: object,
        cv_mode: str,
    ):
        super(XGBHyperParameterTuning, self).__init__(
            df, explainable_vars, dependet_var, model
        )
        self.cv_mode = cv_mode

    def objective_cv(self, trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "reg_alpha": trial.suggest_int("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_int("reg_lambda", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 0, 10),
            "gamma": trial.suggest_int("gamma", 0, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.1, 1, step=0.01
            ),
            "enable_categorical": trial.suggest_categorical(
                "enable_categorical", [True]
            ),
            "n_jobs": trial.suggest_int("n_jobs", -1, -1, step=1),
        }
        self.model.update_model_params(params)

        if self.cv_mode == "yearly":
            preds = self.hp_yearly_CV()
        elif self.cv_mode == "monthly":
            preds = self.hp_monthly_CV()
        else:
            raise ValueError("Invalid cv_mode, must be 'yearly' or 'monthly'")

        # Compute error metrics
        error_metrics = compute_error_metrics(
            preds["Actuals"],
            preds["Predictions"],
        )

        # Set metric as the objective to minimize
        test_accuracy = error_metrics["MSE"]

        return test_accuracy

    def run_hp_tuning(self, n_trials: int) -> dict:
        study = optuna.create_study(
            direction=optuna.study.StudyDirection.MINIMIZE,
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.SuccessiveHalvingPruner(),
        )
        study.optimize(
            lambda trial: self.objective_cv(trial=trial),
            n_trials=n_trials,
        )

        pruned_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ]
        complete_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        print("\nStudy statistics: ")
        print("\tNumber of finished trials: ", len(study.trials))
        print("\tNumber of pruned trials: ", len(pruned_trials))
        print("\tNumber of complete trials: ", len(complete_trials))

        trial = study.best_trial
        print(f"\nBest trial: {trial.number}")
        print("\tValue: ", trial.value)
        print("\tParams: ")
        for key, value in trial.params.items():
            print("\t\t{}: {}".format(key, value))

        return trial.params


def example():
    df = pd.read_pickle(
        os.path.join(
            os.getcwd(), "data", "processed_data", "off_shore", "AAU_Park01.pkl"
        )
    )
    df = df[df["TurbineId"] == 1]

    explainable_var = df["WindSpeed"].values
    dependent_var = df["GridPower"].values
    # explainable_var = ["WindSpeed", "BladeLoadA"]
    # dependent_var = ["GridPower", "PitchAngleA"]

    model = XGBRegressorClass()
    search_space = {
        "n_estimators": (1, 10),
        "max_depth": (1, 5),
        "reg_alpha": (0, 10),
        "reg_lambda": (0, 5),
        "min_child_weight": (0, 10),
        "gamma": (0, 5),
        "learning_rate": (0.005, 0.5),
        "colsample_bytree": {"type": "float", "min": 0.1, "max": 1, "step": 0.01},
        "enable_categorical": [True],
        "n_jobs": -1,
    }

    tscv = HyperParameterTuning(
        df, explainable_var, dependent_var, model, "MAPE", "expanding", 5
    )
    best_params = tscv.run(
        param_search_space=search_space,
        n_trials=5,
        optimization_direction=optuna.study.StudyDirection.MINIMIZE,
    )


# Uncomment to run the example:
if __name__ == "__main__":
    example()
