##################################
### Attempt at builder pattern ###
##################################

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Literal
from parallelbar.wrappers import add_progress
from parallelbar import progress_starmap
import sys
import os
import time
import warnings
import multiprocessing as mp
import pandas as pd

sys.path.insert(0, "src/")

from utils.pps import PredictivePowerScore
from data_utility.data_utils import DataLoader, GMMFilterClass
from utils.plotting_utils import iterative_plot_feeder
from utils.utils import delta_IQR_computation, delta_data
from visualization.visualize import (
    plot_aggregated_ts_frequency_percentage,
    plot_K_SCORE_comparison_TS,
)

# turn off pandas future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


class ExpBuilder(ABC):
    """Abstract class for the Experiment Builder, defining the methods that need to be implemented by concrete builders.

    Args:
        ABC (_type_): Abstract Base Class, used to define abstract methods.
    """

    @property
    @abstractmethod
    def experiment(self) -> Any:
        pass

    @abstractmethod
    def data_loader(self) -> Any:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class ConcreteGMMExpBuilder(ExpBuilder):
    """Concrete builder class for the GMM Experiment.
    -----------------------------------------------
    This class implements the methods defined in the abstract builder class.
    The purpose of the concrete builder is to implement a specific experiment builder.

    I.e. Concrete Builders are supposed to provide their own methods for
    retrieving results. That's because various types of builders may create
    entirely different products that don't follow the same interface.
    Therefore, such methods cannot be declared in the base Builder interface
    (at least in a statically typed programming language).

    Args:
        ExpBuilder (): Abstract Builder class
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the experiment instance to a new instance of the GMMExperiment class."""
        self._experiment = GMMExperiment()

    @property
    def experiment(self) -> GMMExperiment:
        """Returns the experiment instance.

        Returns:
            GMMExperiment: Instance of the GMMExperiment class
        """
        experiment = self._experiment
        self.reset()
        return experiment

    def col_cleaner(self, cols_to_exclude: list = None):
        """
        Function to clean the column names.

        Adds both the columns used for the experiments and the result dicts for the product.
        """
        cols_to_exclude = list(set(cols_to_exclude))
        cols = [
            col
            for col in self._experiment.turbine_df.columns
            if (self._experiment.turbine_df.dtypes[col] == float or col == "Time")
            and col not in cols_to_exclude
        ]

        self._experiment.add_cols(cols)
        dict_cols = [col for col in cols if col != "Time" and col != "GridPower"]
        self._experiment.add_dict_cols(dict_cols)

    def _translate_col_names(self, columns: list, append_str: str) -> list:
        """
        Function to append a string to the column name

        """
        return [f"{append_str}_{col}" for col in columns]

    def create_result_df(self):
        """
        Function to create a result dataframe. The result dataframe should be held by the
        product before experiment runs
        """
        # raise error if self_experiment.turbine_df and self_experiment.granularity does not exists

        if not hasattr(self._experiment, "turbine_df"):
            raise AttributeError("Turbine dataframe not found")

        if not hasattr(self._experiment, "granularity"):
            raise AttributeError("Granularity is not set")

        cols = self._experiment.dict_cols.copy()
        pps_cols = self._translate_col_names(cols, "pps")
        delta_iqr_cols = self._translate_col_names(cols, "delta_iqr")
        result_cols = ["Turbine", "Year"]

        if self._experiment.granularity == "quarter":
            result_cols.append("Quarter")

        elif self._experiment.granularity == "month":
            result_cols.append("Month")

        result_cols += ["K", "sumpps", "delta_data"]
        result_df = pd.DataFrame(columns=result_cols + pps_cols + delta_iqr_cols)

        self._experiment.add_result_df(result_df)
        self._experiment.add_iqr_cols(delta_iqr_cols)

    def _pivot_to_dict(self, pps_df: pd.DataFrame) -> dict:
        """
        Function to convert the pps_df to a dictionary
        """
        pps_pivot_df = pps_df.pivot_table(
            index=None, columns="x", values="ppscore", aggfunc="first"
        )
        sorted_columns = pps_pivot_df.iloc[0].sort_values(ascending=False).index
        pps_pivot_df = pps_pivot_df[sorted_columns]
        pps_pivot_df = pps_pivot_df.reset_index(drop=True)
        pps_pivot_df.columns = self._translate_col_names(pps_pivot_df.columns, "pps")
        pps_pivot_dict = pps_pivot_df.to_dict(orient="records")[0]

        return pps_pivot_dict

    def data_loader(self, turbine_id: int):
        """fetches turbine data by using the mapping dataframe.
        The data_loader adds the turbine dataframe to the product.

        Args:
            turbine_id (int): turbine id
            map_df (pd.DataFrame): mapping dataframe, sits on the product.
        """

        if os.path.exists(os.path.join(os.getcwd(), "data", "voting_system_data")):
            dl = DataLoader(source_path="data/voting_system_data")
        else:
            dl = DataLoader(source_path="data/processed_data")
        df = dl.load_turbine_data(turbine_id)
        park_num = dl.fetch_park_number(turbine_id)

        self._experiment.add_turbine_df(df), self._experiment.add_turbine_id(
            turbine_id
        ), self._experiment.add_park_num(park_num)

    def _apply_K_mixture_model(self, k: int, seed: int = 42) -> pd.DataFrame:
        """Apply the K-GMM to the dataframe sitting on the product.

        The Dataframe on the product is not permutated.

        Args:
            k (int): k GMM

        Returns:
            pd.DataFrame: Returns two dataframes: filtered_df and outlier_df.
        """
        copy_df = self._experiment.turbine_df.copy()

        gmm_filter = GMMFilterClass()
        params = {"n_components": k, "random_state": seed}
        filtered_df, outlier_df = gmm_filter.GMM_filtering(copy_df, params=params)

        return filtered_df, outlier_df

    def _run_experiment_fold(
        self,
        fold_df: pd.DataFrame,
        sliced_df: pd.DataFrame,
        k: int,
        year: int = None,
        granularity_dict: dict = None,
    ):
        """Function to run the experiment on a provided fold

        This function expects a sliced (according to granularity) dataframe, calculates
        the pps and deltas, and appends the results to the product result dataframe.

        Args:
            fold_df (pd.DataFrame): The sliced filtered df belonging to this fold
            sliced_df (pd.DataFrame): The sliced turbine (not GMM filtered) belonging to this fold
            k (int): k used for the filtering
            year (int, optional): year is the highest granularity, and thus, always used.
        """
        # if the fold_df is empty, add a NaN result
        if fold_df.empty:
            self._add_NaN_result(year, k, **granularity_dict)
            return

        if fold_df.shape[0] < 10:
            self._add_NaN_result(year, k, **granularity_dict)
            return

        # fold_df = fold_df.drop(["AmbTemp", "WdAbs", "WindDirRel"], axis=1)
        pps = PredictivePowerScore(fold_df[self._experiment.cols])
        fold_pps_df = pps.predictors("GridPower", sorted=True)
        fold_sumpps = sum(fold_pps_df.ppscore)
        fold_avgpps = fold_pps_df.ppscore.mean()
        pps_dict = self._pivot_to_dict(fold_pps_df)

        if k == 0:
            delta_iqr_dict = dict.fromkeys(self._experiment.iqr_cols, 0)
            self._add_result(
                year=year,
                k=k,
                sumpps=fold_sumpps,
                avgpps=fold_avgpps,
                delta_data=0,
                pps_dict=pps_dict,
                delta_iqr_dict=delta_iqr_dict,
                **granularity_dict,
            )

        else:
            delta_iqr_dict = delta_IQR_computation(
                sliced_df[self._experiment.dict_cols],
                fold_df[self._experiment.dict_cols],
            )

            delta_data_val = delta_data(
                sliced_df[self._experiment.cols], fold_df[self._experiment.cols]
            )
            self._add_result(
                year=year,
                k=k,
                sumpps=fold_sumpps,
                avgpps=fold_avgpps,
                delta_data=delta_data_val,
                pps_dict=pps_dict,
                delta_iqr_dict=delta_iqr_dict,
                **granularity_dict,
            )

    def _get_granularity_dict(self, granularity_val) -> dict:
        """Extracts the granularity dictionary based on the experiment's granularity setting.

        Args:
            fold_df (pd.DataFrame): The df sliced for the fold

        Returns:
            dict: dict with the granularity value
        """
        granularity_dict = {}
        if self._experiment.granularity == "quarter":
            granularity_dict["Quarter"] = granularity_val

        elif self._experiment.granularity == "month":
            granularity_dict["Month"] = granularity_val

        return granularity_dict

    def _add_result(
        self,
        year: int,
        k: int,
        sumpps: float,
        avgpps: float,
        delta_data: float,
        pps_dict: dict,
        delta_iqr_dict: dict,
        **kwargs,
    ):
        """
        Function to add the results to the _experiment result dataframe

        This function utilizes a concat that requires the result dataframe to already exists on the product
        """
        result_dict = {
            "Turbine": self._experiment.turbine_id,
            "Year": year,
            "K": k,
            "sumpps": sumpps,
            "avgpps": avgpps,
            "delta_data": delta_data,
            **pps_dict,
            **delta_iqr_dict,
        }

        if kwargs:
            result_dict.update(kwargs)
        self._experiment.result_df = pd.concat(
            [self._experiment.result_df, pd.DataFrame(result_dict, index=[0])]
        )

    def _add_NaN_result(self, year: int, k: int, **kwargs):
        """
        Function to add a NaN result to the result dataframe
        """
        delta_iqr_dict = dict.fromkeys(self._experiment.iqr_cols, None)
        result_dict = {
            "Turbine": self._experiment.turbine_id,
            "Year": year,
            "K": k,
            "sumpps": None,
            "delta_data": None,
            **delta_iqr_dict,
        }
        if kwargs:
            result_dict.update(kwargs)
        self._experiment.result_df = pd.concat(
            [self._experiment.result_df, pd.DataFrame(result_dict, index=[0])]
        )

    def set_granularity(self, granularity: Literal["year", "quarter", "month"]):
        """
        Function to set the product granularity
        """
        self._experiment.add_granularity(granularity)

    def _granularity_fold(
        self,
        k_filtered_df: pd.DataFrame,
        k: int,
    ):
        """Dataframe slicer for granularities

        Args:
            granularity (Literal[&quot;year&quot;, &quot;quarter&quot;, &quot;month&quot;]): Granularity set on the product
            k_filtered_df (pd.DataFrame): Dataframe with applied (or not) K GMMs
            k (int): The set K
        """
        years = k_filtered_df["Time"].dt.year.unique()
        if self._experiment.granularity == "quarter":
            quarters = self._experiment.turbine_df["Time"].dt.quarter.unique()
        elif self._experiment.granularity == "month":
            months = self._experiment.turbine_df["Time"].dt.month.unique()

        for year in years:
            if self._experiment.granularity == "year":
                year_df = k_filtered_df[k_filtered_df["Time"].dt.year == year]
                sliced_df = self._experiment.turbine_df[
                    self._experiment.turbine_df["Time"].dt.year == year
                ]
                self._run_experiment_fold(
                    fold_df=year_df, sliced_df=sliced_df, k=k, year=year
                )

            elif self._experiment.granularity == "quarter":
                for quarter in quarters:
                    granularity_dict = self._get_granularity_dict(quarter)

                    sliced_df = self._experiment.turbine_df[
                        (self._experiment.turbine_df["Time"].dt.year == year)
                        & (self._experiment.turbine_df["Time"].dt.quarter == quarter)
                    ]
                    quarter_df = k_filtered_df[
                        (k_filtered_df["Time"].dt.year == year)
                        & (k_filtered_df["Time"].dt.quarter == quarter)
                    ]
                    self._run_experiment_fold(
                        fold_df=quarter_df,
                        sliced_df=sliced_df,
                        k=k,
                        year=year,
                        granularity_dict=granularity_dict,
                    )

            elif self._experiment.granularity == "month":
                for month in months:
                    granularity_dict = self._get_granularity_dict(month)
                    sliced_df = self._experiment.turbine_df[
                        (self._experiment.turbine_df["Time"].dt.year == year)
                        & (self._experiment.turbine_df["Time"].dt.month == month)
                    ]
                    month_df = k_filtered_df[
                        (k_filtered_df["Time"].dt.year == year)
                        & (k_filtered_df["Time"].dt.month == month)
                    ]
                    self._run_experiment_fold(
                        fold_df=month_df,
                        sliced_df=sliced_df,
                        k=k,
                        year=year,
                        granularity_dict=granularity_dict,
                    )

    def _save_outlier_df(self, outlier_df: pd.DataFrame, k: int):
        """Saves the outlier "Time" column in a pickle format for later filtering
        Args:
            outlier_df (pd.DataFrame): The outlier dataframe generated by the DataWrangler.GMM_filtering
            k (int): The K-amount of mixtures used
        """
        outlier_df = outlier_df[["Time"]]
        base_path = f"data/outlier_dfs"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        turbine_path = f"{base_path}/Turbine{self._experiment.turbine_id}"
        if not os.path.exists(turbine_path):
            os.makedirs(turbine_path)
        outlier_df.to_pickle(f"{turbine_path}/outlier_df_{k}.pkl")

    def run_experiment(self, max_k: int, outlier_plots: bool = True):
        """
        Runs the experiment for each K.
        """

        for k in range(0, max_k + 1):
            if k == 0:
                self._granularity_fold(self._experiment.turbine_df, k)
            else:
                filtered_df, outlier_df = self._apply_K_mixture_model(k)
                self._save_outlier_df(outlier_df, k)
                self._granularity_fold(filtered_df, k)

                if outlier_plots:
                    self._plot_aggregated_outliers(
                        outlier_df,
                        k,
                    )

    def save_as_excel(self, lock=None, file_name=None):
        """
        Function to save the result dataframe as an Excel file safely.
        """
        if not file_name:
            raise ValueError("Please provide a file name.")

        file_path = f"{self._experiment.result_path}/{file_name}"

        with lock:
            if not os.path.exists(f"{file_path}"):
                with pd.ExcelWriter(f"{file_path}", mode="w") as writer:
                    self._experiment.result_df.T.to_excel(
                        writer,
                        sheet_name=f"Turbine {self._experiment.turbine_id}",
                        index=True,
                    )
            else:
                with pd.ExcelWriter(
                    f"{file_path}",
                    mode="a",
                    if_sheet_exists="replace",
                ) as writer:
                    self._experiment.result_df.T.to_excel(
                        writer,
                        sheet_name=f"Turbine {self._experiment.turbine_id}",
                        index=True,
                    )
            # lock.release()
        # print(f"Saved results for Turbine {self._experiment.turbine_id}")

    def add_result_path(self, result_path: str):
        """
        Function to add the result path to the product

        Args:
            result_path (str): path

        """
        self._experiment.add_result_path(result_path)

    def _plot_aggregated_outliers(self, outlier_df: pd.DataFrame, k: int):
        """Plotting outliers removed by GMM filtering.
        The plots are aggregated monthly, and the Y-axis is calculated as percentage of total data removed of that month

        Args:
            outlier_df (pd.DataFrame): The outlier dataframe generated by the DataWrangler.GMM_filtering
            k (int): The K-amount of mixtures used
        """
        outlier_graph_path = f"Outlier_graphs/Park{self._experiment.park_num}/Turbine{self._experiment.turbine_id}"
        plot_aggregated_ts_frequency_percentage(
            df=outlier_df,
            scale_df=self._experiment.turbine_df,
            savefig=True,
            overwrite=True,
            filename=f"{outlier_graph_path}/Outlier_TS_K{k}.png",
        )


class GMMExperiment:

    def __init__(self) -> None:
        pass

    def add_result_path(self, result_path: str) -> None:
        self.result_path = result_path

    def add_cols(self, cols: list) -> None:
        self.cols = cols

    def add_turbine_map(self, turbine_map: pd.DataFrame) -> None:
        self.turbine_map = turbine_map

    def add_turbine_df(self, turbine_df: pd.DataFrame) -> None:
        self.turbine_df = turbine_df

    def add_turbine_id(self, turbine_id: int) -> None:
        self.turbine_id = turbine_id

    def add_result_df(self, result_df: pd.DataFrame) -> None:
        self.result_df = result_df

    def add_park_num(self, park_num: int) -> None:
        self.park_num = park_num

    def add_granularity(self, granularity: Literal["year", "quarter", "month"]) -> None:
        self.granularity = granularity

    def add_dict_cols(self, dict_cols: list) -> None:
        self.dict_cols = dict_cols

    def add_iqr_cols(self, iqr_cols: list) -> None:
        self.iqr_cols = iqr_cols


class Director:
    """
    The Director orchestrates the experiment execution using multiple builders
    in parallel via multiprocessing.
    """

    def __init__(self) -> None:
        self._builder = None

    @property
    def builder(self) -> ExpBuilder:
        return self._builder

    @builder.setter
    def builder(self, builder: ExpBuilder) -> None:
        self._builder = builder

    @add_progress(error_handling="coerce")
    def single_turbine_experiment(
        self,
        turbine_id: int,
        granularity: str,
        max_k: int,
        lock=None,
        file_name=None,
        cols_to_exclude: list = None,
    ):
        """
        Runs an experiment for a single turbine, with an individual progress bar tracking K iterations.
        This function creates a new builder instance to avoid multiprocessing issues.
        """
        builder = ConcreteGMMExpBuilder()

        builder.data_loader(turbine_id)
        builder.col_cleaner(cols_to_exclude)
        builder.set_granularity(granularity)
        builder.create_result_df()
        builder.run_experiment(max_k, outlier_plots=False)
        builder.add_result_path("src/experiments")
        builder.save_as_excel(lock, file_name=file_name)

        return

    def execute_experiments(
        self,
        turbine_ids: list,
        granularity: str,
        max_k: int,
        file_name: str = None,
        cols_to_exclude: list = None,
    ) -> None:
        """
        Runs turbine experiments in parallel, ensuring each process gets its own independent builder instance.
        """

        if not file_name:
            raise ValueError("Please provide a file name.")
        manager = mp.Manager()
        lock = manager.Lock()

        res = progress_starmap(
            self.single_turbine_experiment,
            [
                (
                    turbine_id,
                    granularity,
                    max_k,
                    lock,
                    file_name,
                    cols_to_exclude,
                )
                for turbine_id in turbine_ids
            ],
            n_cpu=mp.cpu_count(),
            total=2 * len(turbine_ids) - 1,
        )
        if res:
            print("-------------------------------------------")
            print(f"Errors during multiprocessing: {res}")
            print("-------------------------------------------")


if __name__ == "__main__":

    start_time = time.time()
    turbine_ids = [i for i in range(1, 118, 1)]
    director = Director()
    builder = ConcreteGMMExpBuilder()
    director.builder = builder
    director.single_turbine_experiment(
        turbine_id=44,
        granularity="quarter",
        max_k=5,
        file_name="GMM_results_test_run.xlsx",
        cols_to_exclude=[
            "WSE",
            "AmbTemp",
            "WdAbs",
            "WindDirRel",
            "PitchAngleA",
            "PitchAngleB",
            "PitchAngleC",
        ],
    )
    # director.execute_experiments(
    #     turbine_ids, "quarter", 5, file_name="GMM_results.xlsx"
    # )
    end_time = time.time()
    print(f"Execution time: {end_time - start_time}")

    # kwargs = {
    #     "savefig": True,
    #     "overwrite": True,
    # }

    # iterative_plot_feeder(
    #     result_path="src/experiments/GMM_results_test_run.xlsx",
    #     save_path="K_SCORE_Comparison",
    #     granularity="quarter",
    #     plotter_function=plot_K_SCORE_comparison_TS,
    #     **kwargs,
    # )
