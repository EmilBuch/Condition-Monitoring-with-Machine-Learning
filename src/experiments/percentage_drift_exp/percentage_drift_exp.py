import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import torch
import time

from typing import Dict, List, Optional
from parallelbar.wrappers import add_progress
from parallelbar import progress_starmap

sys.path.insert(0, "src")
from data_utility.data_utils import DataLoader, StabilityDataLoader
from utils.utils import sort_sheets, reset_weights
from models.model_director import ModelDirector
from models.model_classes import (
    XGBRegressorClass,
    KNNRegressorClass,
)
from models.model_params import MODEL_DICTIONARY
from sklearn.metrics import mean_absolute_percentage_error
from utils.decorators import mute_print
from visualization.visualize import qq_plot
import optuna
from utils.utils import RobustScalerClass

# Mute study prints
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(1)


class PercentageDriftExperiment:
    """
    Class to calculate the percentage drift between two dataframes.
    """

    def __init__(
        self,
        source_path: str,
        result_path: str,
        turbine_id: int,
        explainable_vars: List[str],
        target_var: str,
        models: Dict[str, list],
        stability_path: str = None,
        lock=None,
    ):
        self.source_path = source_path
        self.stability_path = stability_path
        self.result_path = result_path
        self.turbine_id = turbine_id
        self.explainable_vars = explainable_vars
        self.target_var = target_var
        self.models_dict = models
        if lock is not None:
            self.lock = lock

    def calculate_delta(
        self,
        predictions: List[float],
        data: pd.DataFrame,
    ) -> float:
        """
        Calculate the delta between the predicted and actual values.

        Args:
            model: The model to be used for prediction.
            data (pd.DataFrame): Dataframe containing the data.

        Returns:
            float: Delta value.
        """
        # Compute the prediction distribution
        # Get the actual values
        actual = data[self.target_var]

        # Calculate the delta
        delta = 100 * (sum(actual) - sum(predictions)) / sum(actual)
        return delta

    def drift_calculation(
        self,
        R_1_delta: float,
        R_2_delta: float,
    ) -> float:
        """
        Calculate the percentage drift between two dataframes.

        Args:
            R_1_delta (float): Delta value from the reference year.
            R_2_df (pd.DataFrame): Dataframe containing the target set of data.

        Returns:
            float: Percentage drift between the two years.
        """
        delta_drift = R_2_delta - R_1_delta
        return delta_drift

    @mute_print
    def run_experiment(self, n_trials: int = 100):

        # Load the turbine data
        data_loader = DataLoader(source_path=self.source_path)
        turbine_df = data_loader.load_turbine_data(turbine_id=self.turbine_id)
        turbine_df["year"] = pd.to_datetime(turbine_df["Time"]).dt.year
        turbine_df = turbine_df[turbine_df["is_stable"] == 1]
        turbine_df = turbine_df.sort_values(by=["Time"])
        turbine_df = turbine_df.set_index("Time")

        # Use the RobustScaler to scale the data

        scaler = RobustScalerClass(turbine_df[self.explainable_vars])
        turbine_df[self.explainable_vars] = scaler.fit_transform_scaler(
            turbine_df[self.explainable_vars]
        )
        stability_data_loader = StabilityDataLoader(stability_path=self.stability_path)
        # Load the data
        T_year, R_1, target_years = stability_data_loader.load_years(self.turbine_id)
        # Converting the years to ints
        target_years = eval(target_years)
        target_years = [int(i) for i in target_years]
        target_years.sort()
        results_df = pd.DataFrame(
            columns=[
                "Model",
                "Year",
                "R_1 delta",
                "R_2 delta",
                "Drift",
                "MAPE",
            ]
        )

        for model_name, model in self.models_dict.items():
            print(f"Turine id: {self.turbine_id}, model: {model_name}")

            model_instance = model[0]()
            # check if the class is an MLP from sklearn
            if type(model_instance).__name__ == "MLPRegressorClass":
                model[1]["turbine_id"] = self.turbine_id

            # Initiate the model director
            md = ModelDirector()
            md.set_model(model_instance)

            # Set training data for tuning and training
            train_df = turbine_df[turbine_df["year"] == T_year]

            # Set the test years as both R_1 and R_2
            test_df = turbine_df[turbine_df["year"] == R_1]
            for year in target_years:
                test_df = pd.concat([test_df, turbine_df[turbine_df["year"] == year]])
            test_df.sort_index()

            # Set the dataframes for the model director
            md.load_dataframe(
                feature_cols=self.explainable_vars,
                target_cols=self.target_var,
                df=turbine_df,
            )
            # Set X_train to be the training dataframe
            md.X_train = train_df[self.explainable_vars]
            md.y_train = train_df[self.target_var]

            # Set X_test to be the test dataframe
            md.X_test = test_df[self.explainable_vars]
            md.y_test = test_df[self.target_var]

            R_1_df = test_df[test_df["year"] == R_1]
            # Train the model
            md.tune_hyperparameters(
                turbine_id=self.turbine_id,
                param_search_space=model[1],
                n_trials=n_trials,
                metric="MAPE",
                experiment_name="perc_drift",
            )
            if type(md.model).__name__ == "MLPRegressorClass":
                md.model.model.apply(reset_weights)
                print("Reset model weights before final training")
            md.fit().predict()
            # Wrap predictions with index from X_test
            md.predictions = pd.Series(
                md.predictions, index=md.X_test.index, name="predictions"
            )
            assert md.X_test.index.equals(
                md.predictions.index
            ), "Indices mismatch between predictions and X_test"
            assert R_1_df.index.isin(
                md.predictions.index
            ).all(), "Not all R_1_df indices found in predictions"
            preds = md.predictions.loc[R_1_df.index]
            mape = mean_absolute_percentage_error(R_1_df[self.target_var], preds)
            print(f"Turbine ID: {self.turbine_id}")
            print(f"Train year: {T_year}")
            print(f"Reference year: {R_1}")
            print(f"MAPE for the reference year: {mape*100}")
            # Calculate the delta for R_1
            R_1_delta = self.calculate_delta(
                predictions=preds,
                data=R_1_df,
            )

            print(f"R_1 delta: {R_1_delta}")
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        {
                            "Model": [model_name],
                            "MAPE": [mape * 100],
                            "R_1 delta": [R_1_delta],
                            "Year": [R_1],
                            "T_year": [T_year],
                        }
                    ),
                ],
                ignore_index=True,
            )

            for year in target_years:
                # Get the test dataframe for the year
                R_2_df = test_df[test_df["year"] == year]

                # Get the predictions for R_2 by matching the index
                preds = md.predictions[R_2_df.index]

                # calculate the MAPE for preds vs R_2_df
                mape = mean_absolute_percentage_error(R_2_df[self.target_var], preds)

                # Calculate the delta for R_2
                R_2_delta = self.calculate_delta(
                    predictions=preds,
                    data=R_2_df,
                )

                # Calculate the drift
                drift = self.drift_calculation(R_1_delta, R_2_delta)

                print("----------------")
                print(f"Year: {year}")
                print(f"MAPE: {mape*100}")
                print(f"R_2 delta: {R_2_delta}")
                print(f"Drift: {drift}")
                # Append the results to the dataframe
                results_df = pd.concat(
                    [
                        results_df,
                        pd.DataFrame(
                            {
                                "Model": [model_name],
                                "MAPE": [mape * 100],
                                "R_1 delta": [R_1_delta],
                                "Year": [year],
                                "R_2 delta": [R_2_delta],
                                "Drift": [drift],
                                "T_year": [T_year],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
        return results_df


class MultiProcessDriftExperiment:
    def __init__(
        self,
        source_path: str,
        result_path: str,
        stability_path: str,
        xlsx_filename: str,
    ):
        self.source_path = source_path
        self.result_path = result_path
        self.stability_path = stability_path
        self.xlsx_filename = xlsx_filename

    def save_to_excel(
        self,
        result_df: pd.DataFrame,
        turbine_id: int,
        lock=None,
    ):
        """
        Save the results to an excel file.

        Args:
            result_df (pd.DataFrame): Dataframe containing the results.
            turbine_id (int): Turbine ID.
            lock: Lock object for multiprocessing.
            file_name (str): Name of the file to save the results to.
        """
        if not self.xlsx_filename.endswith(".xlsx"):
            raise ValueError("File name must end with .xlsx")
        filepath = f"{self.result_path}/{self.xlsx_filename}"

        with lock:
            if not os.path.exists(filepath):
                with pd.ExcelWriter(filepath, mode="w") as writer:
                    result_df.to_excel(
                        writer,
                        sheet_name=f"Turbine {turbine_id}",
                        index=True,
                    )
            else:
                with pd.ExcelWriter(
                    filepath, mode="a", if_sheet_exists="replace"
                ) as writer:
                    result_df.to_excel(
                        writer,
                        sheet_name=f"Turbine {turbine_id}",
                        index=True,
                    )

    @add_progress(error_handling="coerce")
    def run_experiment(
        self,
        turbine_id: int,
        explainable_vars: List[str],
        target_var: str,
        models: Dict[str, object],
        n_trials: int = 100,
        lock=None,
    ):

        # initialize the experiment class
        drift_exp = PercentageDriftExperiment(
            source_path=self.source_path,
            stability_path=self.stability_path,
            result_path=self.result_path,
            turbine_id=turbine_id,
            explainable_vars=explainable_vars,
            target_var=target_var,
            models=models,
            lock=lock,
        )
        # run the experiment
        results_df = drift_exp.run_experiment(n_trials=n_trials)
        # save the results to excel
        self.save_to_excel(
            result_df=results_df,
            turbine_id=turbine_id,
            lock=lock,
        )

    def run_parallel_experiment(
        self,
        turbines: List[int],
        explainable_vars: List[str],
        target_var: str,
        models: Dict[str, object],
        n_trials: int = 100,
        cpu_num: int = 4,
    ):
        manager = mp.Manager()
        lock = manager.Lock()
        res = progress_starmap(
            self.run_experiment,
            [
                (turbine_id, explainable_vars, target_var, models, n_trials, lock)
                for turbine_id in turbines
            ],
            n_cpu=cpu_num,
            total=len(turbines) * 2 - 1,
        )
        print(f"Errors during multiprocessing: {res}")


if __name__ == "__main__":
    dl = DataLoader(source_path="data/nbm_selector_data")
    turbine_df = dl.load_turbine_data(turbine_id=1)
    # remove the columns that are not needed
    # "GridPower", "Time", "TurbineId", "is_stable"
    explainable_vars = [
        i
        for i in turbine_df.columns
        if i
        not in [
            "GridPower",
            "Time",
            "TurbineId",
            "is_stable",
            "WSE",
        ]
    ]
    # explainable_vars = [
    #     "AmbTemp",
    #     "PitchAngleA",
    #     "PitchAngleB",
    #     "PitchAngleC",
    #     "WdAbs",
    #     "WindDirRel",
    #     "WindSpeed",
    # ]

    # explainable_vars = ["WindSpeed", "AmbTemp", ]
    print(f"Explainable variables used for experiment: {explainable_vars}")
    target_var = "GridPower"

    # initialize an XGBoost model
    xgb_params = {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "n_jobs": [1],
    }
    knn_params = {
        "n_neighbors": (1, 100),
    }

    mlp_params = {
        "epochs": [1000],
        "learning_rate": (0.1),
    }

    # create the dict to store the models and their hyperparameters
    model_dict = {
        "XGBoost": [XGBRegressorClass, xgb_params],
        "KNNReg": [KNNRegressorClass, knn_params],
        # "MLP": [MLPRegressorClass, mlp_params]
    }

    stability_data_loader = StabilityDataLoader(
        stability_path="data/nbm_selector_data/stability.xlsx"
    )

    experiment = PercentageDriftExperiment(
        source_path="data/nbm_selector_data",
        result_path="data/nbm_selector_data",
        turbine_id=4,
        explainable_vars=explainable_vars,
        target_var=target_var,
        models=MODEL_DICTIONARY,
        stability_path="data/nbm_selector_data/stability.xlsx",
    )
    # begin start time
    start_time = time.perf_counter()
    experiment.run_experiment(n_trials=2)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(elapsed_time / 60)
    # Load unique turbine ids
    # turbines = stability_data_loader.load_stable_turbines()
    # turbines = [1, 4]
    # print(f"Unique turbine ids: {turbines}")
    # # start the experiment in parallel
    # cpu_count = mp.cpu_count() - 1
    # experiment = MultiProcessDriftExperiment(
    #     source_path="data/nbm_selector_data",
    #     stability_path="data/nbm_selector_data/stability.xlsx",
    #     result_path="data/nbm_selector_data",
    #     xlsx_filename="drift_results_WS.xlsx",
    # )
    # experiment.run_parallel_experiment(
    #     turbines=turbines,
    #     explainable_vars=explainable_vars,
    #     target_var=target_var,
    #     models=MODEL_DICTIONARY,
    #     n_trials=5,
    #     cpu_num=cpu_count,
    # )

    # # sort the sheets in the excel file
    # sort_sheets(
    #     input_path="data/nbm_selector_data/drift_results_WS.xlsx",
    #     output_path="data/nbm_selector_data/drift_results_WS_sorted.xlsx",
    # )
