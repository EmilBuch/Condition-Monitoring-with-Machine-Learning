import os
import sys
import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import torch
import optuna

from typing import Dict, List, Optional
from parallelbar.wrappers import add_progress
from parallelbar import progress_starmap
from sklearn.metrics import mean_absolute_percentage_error

sys.path.insert(0, "src")
from data_utility.data_utils import DataLoader, StabilityDataLoader
from utils.utils import sort_sheets, reset_weights
from models.model_director import ModelDirector
from models.model_classes import XGBRegressorClass, MLPRegressorClass, KNNRegressorClass
from data_utility.data_utils import RobustScalerClass

from utils.decorators import mute_print

# Mute study prints
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(1)


class SensitivityAnalysis:
    def __init__(
        self,
        source_path: str,
        results_path: str,
        turbine_id: int,
        explainable_vars: List[str],
        target_var: str,
        models_dict: Dict[str, List],
        stability_path: str = "data/nbm_selector_data/stability.xlsx",
        lock=None,
    ):
        self.source_path = source_path
        self.results_path = results_path
        self.turbine_id = turbine_id
        self.explainable_vars = explainable_vars
        self.target_var = target_var
        self.models_dict = models_dict
        self.stability_path = stability_path
        if lock is not None:
            self.lock = lock

    def _percentage_drift(
        self,
        actual,
        predicted,
    ):
        """
        Calculate the percentage drift between two values.

        Args:

        """
        actual_sum = np.sum(actual)
        predicted_sum = np.sum(predicted)
        if actual_sum == 0:
            raise ValueError("Sum of r_1 is zero, division by zero!!")
        return (actual_sum - predicted_sum) / actual_sum * 100

    @mute_print
    def run_sensitivity_analysis(
        self,
        n_trials: int = 100,
    ):
        # Load the turbine data
        data_loader = DataLoader(source_path=self.source_path)
        turbine_df = data_loader.load_turbine_data(turbine_id=self.turbine_id)
        turbine_df["year"] = pd.to_datetime(turbine_df["Time"]).dt.year

        stability_data_loader = StabilityDataLoader(stability_path=self.stability_path)
        # Load the data
        T_year, R_1, target_years = stability_data_loader.load_years(self.turbine_id)
        # Converting the years to ints
        target_years = eval(target_years)
        target_years = [int(i) for i in target_years]
        # append R_1 to the list target_years
        target_years = [int(R_1)] + target_years
        all_years = [int(T_year)] + target_years

        print(f"Target years: {target_years}")
        print(f"Years: {all_years}")

        # Slice the dataframe to get the years
        turbine_df = turbine_df[turbine_df["year"].isin(all_years)]
        turbine_df = turbine_df[turbine_df["is_stable"] == 1]
        turbine_df = turbine_df.sort_values(by=["Time"])
        turbine_df = turbine_df.set_index("Time")

        scaler = RobustScalerClass(turbine_df[self.explainable_vars])
        turbine_df[self.explainable_vars] = scaler.fit_transform_scaler(
            turbine_df[self.explainable_vars]
        )

        if "year" not in self.explainable_vars:
            self.explainable_vars.append("year")
            print("Year added to explainable variables")

        results_df = pd.DataFrame(columns=["Model", "Year", "Drift", "MAPE", "T_year"])
        for model_name, model in self.models_dict.items():
            print(f"Turine id: {self.turbine_id}, model: {model_name}")
            model_instance = model[0]()
            # check if the class is an MLP from sklearn
            if type(model_instance).__name__ == "MLPRegressorClass":
                model[1]["turbine_id"] = self.turbine_id

            md = ModelDirector()
            md.set_model(model_instance).load_dataframe(
                feature_cols=self.explainable_vars,
                target_cols=self.target_var,
                df=turbine_df,
            )
            md.X_train = turbine_df[self.explainable_vars]
            md.y_train = turbine_df[self.target_var]

            # if type(md.model).__name__ == "MLPRegressorClass":
            #     print("Skipping hyperparameter tuning for MLP model")
            #     # add the turbine_id to the model parameters
            #     model[1]["turbine_id"] = self.turbine_id
            #     md.model.update_model_params(model[1])
            # else:
            md.tune_hyperparameters(
                self.turbine_id,
                param_search_space=model[1],
                n_trials=n_trials,
                metric="MAPE",
                experiment_name="Sensitivity",
            )
            if type(md.model).__name__ == "MLPRegressorClass":
                md.model.model.apply(reset_weights)
                print("Reset model weights before final training")

            # We create the test df for sensitivity analysis
            base_df = turbine_df[turbine_df["year"] == T_year]
            md.X_test = base_df[self.explainable_vars]
            md.y_test = base_df[self.target_var]

            md.fit().predict()
            drift = self._percentage_drift(
                actual=md.y_test,
                predicted=md.predictions,
            )
            mape = mean_absolute_percentage_error(
                y_true=md.y_test,
                y_pred=md.predictions,
            )
            print(f"Drift for base year {T_year}: {drift}")
            print(f"MAPE for base year {T_year}: {mape}")
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        {
                            "Model": [model_name],
                            "Year": None,
                            "Drift": [drift],
                            "MAPE": [mape * 100],
                            "T_year": [T_year],
                        }
                    ),
                ],
                ignore_index=True,
            )

            # Create synthetic data and predict on yearly basis
            for year in target_years:
                year_df = turbine_df[turbine_df["year"] == year]
                year_df["year"] = T_year

                md.X_test = year_df[self.explainable_vars]
                md.y_test = year_df[self.target_var]
                md.predict()
                drift = self._percentage_drift(
                    actual=md.y_test,
                    predicted=md.predictions,
                )
                mape = mean_absolute_percentage_error(
                    y_true=md.y_test,
                    y_pred=md.predictions,
                )
                print("-----------------")
                print(f"Drift for year {year}: {drift}")
                print(f"MAPE for year {year}: {mape}")
                results_df = pd.concat(
                    [
                        results_df,
                        pd.DataFrame(
                            {
                                "Model": [model_name],
                                "Year": [year],
                                "Drift": [drift],
                                "MAPE": [mape * 100],
                                "T_year": [T_year],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
        print(results_df)
        return results_df


class MultiProcessSensitivityAnalysis:
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
        sens_exp = SensitivityAnalysis(
            source_path=self.source_path,
            stability_path=self.stability_path,
            results_path=self.result_path,
            turbine_id=turbine_id,
            explainable_vars=explainable_vars,
            target_var=target_var,
            models_dict=models,
            lock=lock,
        )
        # run the experiment
        results_df = sens_exp.run_sensitivity_analysis(
            n_trials=n_trials,
        )
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
        # create a lock for multiprocessing
        manager = mp.Manager()
        lock = manager.Lock()
        # create a pool of workers
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
    # Example usage
    source_path = "data/nbm_selector_data"
    results_path = "data/nbm_selector_data"
    stability_path = "data/nbm_selector_data/stability.xlsx"
    turbine_id = 6
    dl = DataLoader(source_path="data/nbm_selector_data")
    turbine_df = dl.load_turbine_data(turbine_id=turbine_id)
    explainable_vars = [
        i
        for i in turbine_df.columns
        if i not in ["GridPower", "Time", "TurbineId", "is_stable", "WSE"]
    ]
    explainable_vars = [
        "WindSpeed",
    ]
    print(f"Explainable variables used for experiment: {explainable_vars}")
    print("Adding 'year' as an explanatory var inside the experiment")
    target_var = "GridPower"

    xgb_params = {
        "n_estimators": (50, 500),
        "max_depth": (3, 7),
        "learning_rate": (0.01, 0.1),
        "n_jobs": [1],
    }
    knn_params = {
        "n_neighbors": (1, 100),
    }

    mlp_params = {
        "epochs": 1000,
        "learning_rate": 0.03,
    }

    # create the dict to store the models and their hyperparameters
    model_dict = {
        "XGBoost": [XGBRegressorClass, xgb_params],
        # "KNNReg": [KNNRegressorClass, knn_params],
        "MLP": [MLPRegressorClass, mlp_params],
    }
    sa = SensitivityAnalysis(
        source_path,
        results_path,
        turbine_id,
        explainable_vars,
        target_var,
        model_dict,
        stability_path,
    )

    # begin start time
    start_time = time.perf_counter()
    sa.run_sensitivity_analysis(n_trials=5)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(elapsed_time / 60)

    stability_data_loader = StabilityDataLoader(
        stability_path="data/nbm_selector_data/stability.xlsx"
    )
    # Load unique turbine ids
    turbines = stability_data_loader.load_stable_turbines()
    turbines = [1, 4]
    print(f"Unique turbine ids: {turbines}")
    # start the experiment in parallel
    cpu_count = mp.cpu_count() - 2

    # experiment = MultiProcessSensitivityAnalysis(
    #     source_path=source_path,
    #     result_path=results_path,
    #     stability_path=stability_path,
    #     xlsx_filename="sensitivity_results_WS.xlsx",
    # )
    # experiment.run_parallel_experiment(
    #     turbines=turbines,
    #     explainable_vars=explainable_vars,
    #     target_var=target_var,
    #     models=model_dict,
    #     n_trials=5,
    #     cpu_num=cpu_count,
    # )
    # sort_sheets(
    #     input_path=f"{results_path}/sensitivity_results_WS.xlsx",
    #     output_path=f"{results_path}/sensitivity_results_WS_sorted.xlsx",
    # )
