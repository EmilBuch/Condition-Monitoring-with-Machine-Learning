from typing import List, Literal
import pandas as pd
from shutil import copy2
import sys
import os
import numpy as np
import multiprocessing as mp
from typing import Literal, List
from shutil import copy2
from collections import defaultdict
from pymoo.indicators.hv import Hypervolume
from abc import ABC, abstractmethod
from parallelbar.wrappers import add_progress
from parallelbar import progress_starmap

sys.path.insert(0, "src/")
from models.model_classes import GMMClass
from utils.utils import RobustScalerClass

from data_utility.make_dataset import DataWrangler
from data_utility.voting_system import VotingSystem
import warnings

# turn off pandas future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
# Turn of chains warnings from pandas
pd.options.mode.chained_assignment = None  # default='warn'


class DataLoader:
    """
    Provides utilities for loading data on turbine and park levels.

    REQUIRES:
    ______________________
    turbine_mapping.xlsx to exists in the data folder.

    ...

    Attributes
    ----------
    source_path (str):
        The path to the processed data.
    mapping_path (str):
        The path to the turbine mapping.
    destination_path (str):
        The path to the destination data (if needed).
    map_df (pd.DataFrame):
        The turbine mapping dataframe.

    Methods
    -------
    load_turbine_data(turbine_id: int) -> pd.DataFrame:
        Loads in the turbine data based on the turbine_id by using the mapping dataframe.
    create_file_name(park: int)-> str:
        Creates a file name for the park, given the park in number.
    save_park_data(park: int, df: pd.DataFrame):
        Saves the park data as a pickle file.
    load_park_data(park: int) -> pd.DataFrame:
        Loads in the park data as a Dataframe
    fetch_park_number(self, turbine_id: int) -> int:
        Fetches the park number based on the turbine_id.
    fetch_park_list() -> list:
        Fetches the park list from the turbine mapping.
    fetch_turbine_list() -> list:
        Fetches the turbine list from the turbine mapping.
    """

    def __init__(
        self,
        source_path: str = "data/processed_data",
        mapping_path: str = "data/turbine_mapping.xlsx",
        destination_path: str = None,
    ):

        self.source_path = source_path
        self.mapping_path = mapping_path
        self.destination_path = destination_path
        self.map_df = pd.read_excel(mapping_path)
        self._convert_turbine_map()

    def _convert_turbine_map(self):
        """
        Converts the turbine map xlsx to a useable dataframe
        """

        def _convert_to_range(range_str: str) -> range:
            start, end = map(int, range_str.split("-"))
            return range(start, end + 1)

        self.map_df["Range"] = self.map_df["Range"].apply(_convert_to_range)

    def load_turbine_data(self, turbine_id: int) -> pd.DataFrame:
        """Loads in the turbine data based on the turbine_id by using the mapping dataframe.

        Args:
            turbine_id (int): turbine id

        Returns:
            pd.DataFrame: turbine data
        """

        match = self.map_df[self.map_df["Range"].apply(lambda r: turbine_id in r)]
        park = match["Park"].iloc[0]
        park_num = int(str.split(park, "Park")[1])
        if park_num <= 5:
            type_path = "off_shore/"
        else:
            type_path = "on_shore/"

        path = f"{self.source_path}/{type_path}AAU_{park}.pkl"
        df = pd.read_pickle(path)
        df = df[df["TurbineId"] == turbine_id]

        return df

    def create_file_name(self, park: int) -> str:
        """
        Creates a file name for the park, given the park in number.

        Args:
            park (int): park number

        Returns:
            str: The file name.
        """
        return f"AAU_Park{str(park).zfill(2)}"

    def save_park_data(self, park: int, df: pd.DataFrame):
        """
        Saves the park data as a pickle file.

        Args:
            park (int): The park number.
            df (pd.DataFrame): The data to be saved.
        """
        park_name = self.create_file_name(park)

        if park <= 5:
            type_path = "off_shore/"
            if not os.path.exists(f"{self.destination_path}/{type_path}"):
                os.makedirs(f"{self.destination_path}/{type_path}")
        else:
            type_path = "on_shore/"
            if not os.path.exists(f"{self.destination_path}/{type_path}"):
                os.makedirs(f"{self.destination_path}/{type_path}")

        path = f"{self.destination_path}/{type_path}{park_name}.pkl"

        df.to_pickle(path)

    def load_park_data(self, park: int) -> pd.DataFrame:
        """Loads in the park data based on the park number.

        Args:
            park (int): park number

        Returns:
            pd.DataFrame: park data
        """

        if park <= 5:
            type_path = "off_shore/"
        else:
            type_path = "on_shore/"

        path = f"{self.source_path}/{type_path}AAU_Park{str(park).zfill(2)}.pkl"
        df = pd.read_pickle(path)

        return df

    def fetch_park_number(self, turbine_id: int) -> int:
        """
        Fetches the park number based on the turbine_id.

        Args:
            turbine_id (int): The turbine id.

        Returns:
            int: The park number.
        """
        match = self.map_df[self.map_df["Range"].apply(lambda r: turbine_id in r)]
        park = match["Park"].iloc[0]
        park_num = int(str.split(park, "Park")[1])

        return park_num

    def fetch_park_list(self) -> list:
        """
        Fetches the park list from the turbine mapping.

        Returns:
            list: List of park numbers.
        """
        park_list = self.map_df["Park"].unique()
        park_list = [int(str.split(park, "Park")[1]) for park in park_list]
        return park_list

    def fetch_turbine_list(self) -> list:
        """
        Fetches the turbine list from the turbine mapping.

        Returns:
            list: List of turbine ids.
        """
        turbine_list = self.map_df["Range"].apply(lambda r: list(r)).sum()
        return turbine_list


class StabilityDataLoader:

    def __init__(self, stability_path: str = "data/nbm_selector_data/stability.xlsx"):
        self.stability_df = pd.read_excel(stability_path)

    def load_stable_turbines(self):
        """
        Loads the turbine ids that appears in the stability.xlsx file
        """
        stable_turbines = self.stability_df["TurbineId"].unique()
        return stable_turbines

    def load_years(self, turbine_id: int):
        """
        For a given turbine, fetches:
            -the training year
            -the reference year
            -the list of target years
        Args:
            turbine_id (int): The turbine id.
        Returns:
            tuple: The training year, reference year, and target years.
        """
        turbine = self.stability_df[self.stability_df["TurbineId"] == turbine_id]

        T_year = turbine["T_year"].values[0]
        R_year = turbine["R_1"].values[0]
        target_years = turbine["R_2"].values[0]

        return T_year, R_year, target_years


class KFinder:
    """
    Provides utilities for finding the best K for the Gaussian Mixture Model.
    This class utilized the excel file generated from the GMM experiment.

    ...

    Attributes
    ----------
    result_path (str):
        The path to the GMM results.
    sheets (list):
        List of sheet names. Created on initialization
    turbine_id (int):
        The turbine Id
    sheet_name (str):
        The sheet name in the GMM excel file. Created on initialization

    Methods
    -------
    _sheet_name_fetcher:
        Fetches the sheet names from the excel file.
    _match_sheet_turbine(sheets, turbine_id):
        Matches the turbine id with the sheet names.
    load_sheet_df:
        Loads the sheet dataframe, using the turbine_id attribute.
    find_best_PPS(sheet_name, score_name):
        Finds the best K and PPS score for the given sheet and score name.
    find_dynamic_PPS:
        Finds K dynamically over time. A K is selected at granularity level.
    find_best_gran_k:
        Finds the optimal K for a given time slice.
    granularity_outlier_filter:
        Utilize the outlier dataframes to filter out K-selected outliers.
    """

    def __init__(
        self,
        result_path: str,
        turbine_id: int,
    ):
        self.result_path = result_path
        self.turbine_id = turbine_id
        self.sheets = self._sheet_name_fetcher()
        self.sheet_name = self._match_sheet_turbine(self.sheets, self.turbine_id)

    def _sheet_name_fetcher(self) -> list:
        """
        Fetches the sheet names from the excel file.

        Returns:
            list: List of sheet names.
        """

        xls = pd.ExcelFile(self.result_path)
        return xls.sheet_names

    def _match_sheet_turbine(self, sheets: list, turbine_id: int) -> str:
        """Matches the turbine id with the sheet names.

        Args:
            sheets (list): List of sheet names.
            turbine_id (int): The turbine id.

        Raises:
            ValueError: If the turbine name is not found in the sheets.

        Returns:
            str: The turbine name.
        """

        turbine_name = f"Turbine {turbine_id}"
        if turbine_name in sheets:
            return turbine_name
        else:
            raise ValueError(f"{turbine_name} not found in the sheets.")

    def load_sheet_df(self) -> pd.DataFrame:
        """
        Loads the sheet dataframe.

        Returns:
            pd.DataFrame: The loaded dataframe.
        """

        df = pd.read_excel(self.result_path, sheet_name=self.sheet_name)
        df = df.T
        df.columns = df.iloc[0]
        df = df.drop("Unnamed: 0")
        df = df.reset_index(drop=True)

        return df

    def find_best_PPS(
        self,
        score_name: str,
    ) -> tuple:
        """
        THIS BEHAVIOUR IS DEPRECATED

        Finds the best K and PPS score for the given sheet and score name.

        Args:
            sheet_name (str): The sheet name.
            score_name (str): The score name to be used for filtering.

        Returns:
            tuple: The best K and PPS score.
        """

        df = self.load_sheet_df()

        # CURRENTLY USING SUM TO FIND THE BEST K
        grouped_df = df.groupby("K")[score_name].sum()

        best_k = grouped_df.idxmax()
        best_pps = grouped_df.max()

        return int(best_k), float(best_pps)

    @abstractmethod
    def find_dynamic_PPS(
        self,
        objectives: list,
        weights: list,
        ref_point: list,
        granularity: Literal["Year", "Quarter", "Month"],
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """This method is used to filter the data at a given granularity, by dynamically selecting
        the K for a period that minimizes a given objective function (the objectives provided).
        It will filter the dataframe based on the given objectives and weights, using the pre-determined K GMM filters.

        Args:
            objectives (list): The objectives to minimize by the objective function.
                these objectives has to exists in the excel file.
            weights (list): The weights for the objectives.
            ref_point (list): The referential point used to calculate the hyperplane by the objective function.
            granularity (Literal[&quot;Year&quot;, &quot;Quarter&quot;, &quot;Month&quot;]):
                The chosen granularity. It can never be smaller or larger than what exists in the GMM excel file.
            df (pd.DataFrame): The df that needs filtering

        Returns:
            tuple(pd.DataFrame, pd.DataFrame):
            The filtered df and a dataframe with selected K's for filtering period.
        """
        # create an empty dataframe to store the selected K's for a period
        k_df = pd.DataFrame(columns=["Year", granularity, "K"])
        sheet_df = self.load_sheet_df()
        years = sheet_df["Year"].unique()
        gran_vals = sheet_df[granularity].unique()
        years.sort()
        gran_vals.sort()
        for year in years:
            for gran_val in gran_vals:
                gran_sheet_df = sheet_df[
                    (sheet_df["Year"] == year) & (sheet_df[granularity] == gran_val)
                ]

                if gran_sheet_df.empty:
                    continue

                # find the best k for this period
                k = self.find_best_gran_k(
                    gran_sheet_df,
                    objectives=objectives,
                    weights=weights,
                    ref_point=ref_point,
                )
                # save the k selection history
                k_df = pd.concat(
                    [
                        k_df,
                        pd.DataFrame(
                            {
                                "Year": [year],
                                granularity: [gran_val],
                                "K": [k],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
                if k == 0:
                    continue

                # If a K is selected, we filter the data by using outlier dataframes,
                # that was calculated when running the K experiment
                else:
                    outlier_path = (
                        f"data/outlier_dfs/Turbine{self.turbine_id}/outlier_df_{k}.pkl"
                    )
                    outlier_df = pd.read_pickle(outlier_path)
                    df, num_filtered = self.granularity_outlier_filter(
                        df=df,
                        outlier_df=outlier_df,
                        year=year,
                        gran_slice=gran_val,
                        granularity=granularity,
                    )
        return df, k_df

    def find_best_gran_k(
        self,
        gran_sheet_df: pd.DataFrame,
        objectives: list,
        weights: np.ndarray,
        ref_point: np.ndarray,
    ) -> int:
        """
        Finds the best K for the given granularity.

        If the objective "delta_data" is given, the column is normalized to a 0-1 range

        If the objective "avgpps" is given, the column is multiplied by -1, to enforce
        a minimization objective.

        Args:
            gran_sheet_df (pd.DataFrame): The dataframe to be filtered.
            objectives (list): The objectives to minimize by the objective function.
            weights (list): The weights associated with the objectives.
            ref_point (list): The reference point to calculate the hypervolume.

        Returns:
            int: The best K.
        """
        # in case we want to use the same dataframe for multiple iterations
        copy_df = gran_sheet_df.copy()

        def _normalise_delta(df, column):
            """
            Normalise delta_data by dividing it by 100. We subtract delta_data from 1, to
            normalise it to a range between 0 and 1. This is done to make the data easier to
            interpret and compare. The function takes a dataframe and a column name as input
            and returns the normalised data.
            """
            normalised_data = df[column] / 100

            return normalised_data

        if "delta_data" in copy_df.columns and "delta_data" in objectives:
            copy_df["delta_data"] = _normalise_delta(copy_df, "delta_data")
        if "avgpps" in copy_df.columns and "avgpps" in objectives:
            copy_df["avgpps"] = copy_df["avgpps"] * -1

        k_dict = defaultdict(int)
        for k in copy_df["K"].unique():
            if k == 0:
                continue
            # we apply the weights for scaling before calculating the hypervolume
            F = copy_df[copy_df["K"] == k][objectives].to_numpy(dtype=float) * weights
            hv = Hypervolume(ref_point=ref_point)
            k_dict[int(k)] = hv(F)

        # Check if all values in k_dict are NaN or if the dictionary is empty
        if all(np.isnan(value) for value in k_dict.values()):  # or len(k_dict) == 0:
            return 0
        return max(k_dict, key=k_dict.get)

    def granularity_outlier_filter(
        self,
        df: pd.DataFrame,
        outlier_df: pd.DataFrame,
        year: int,
        granularity: Literal["Year", "Quarter", "Month"],
        gran_slice: int = None,
    ) -> pd.DataFrame:
        """
        Applies the specific GMM filter with the provided model,
        only on the provided granularity slice of the dataframe.

        Args:
            df (pd.DataFrame): The dataframe to be filtered.
            outlier_df (pd.DataFrame): The dataframe containing the outliers.
            year (int): The year to be used for filtering.
            granularity (Literal): The granularity slice to be used for filtering.
                Can be "Year", "Quarter", or "Month".
            gran_slice (int): The granularity slice to be used for filtering.

        Returns:
            pd.DataFrame: The updated dataframe.
        """
        copy_outlier_df = outlier_df.copy()

        copy_outlier_df["Year"] = copy_outlier_df["Time"].dt.year
        if granularity == "Year":
            sliced_df = copy_outlier_df[copy_outlier_df["Year"] == year].copy()
        if granularity == "Quarter":
            copy_outlier_df["Quarter"] = copy_outlier_df["Time"].dt.quarter
            sliced_df = copy_outlier_df[
                (copy_outlier_df["Quarter"] == int(gran_slice))
                & (copy_outlier_df["Year"] == int(year))
            ]
        if granularity == "Month":
            copy_outlier_df["Month"] = copy_outlier_df["Time"].dt.month
            sliced_df = copy_outlier_df[
                copy_outlier_df["Month"] == gran_slice & copy_outlier_df["Year"] == year
            ].copy()
        elif granularity not in ["Year", "Quarter", "Month"]:
            raise ValueError("Granularity must be 'Year', 'Quarter', or 'Month'.")

        df = df[~df.index.isin(sliced_df.index)]
        return df, len(sliced_df)


class GMMFilterClass:
    """
    Provides utilities for filtering data with Gaussian Mixture Model.

    ...

    Attributes
    ---------
        None

    Methods
    --------
        GMM_filtering(df: pd.DataFrame, params: dict) -> pd.DataFrame:
            Applies Gaussian Mixture Model to the data.
        GMM_best_K_filter(turbine_id: int, df: pd.DataFrame, result_path: str, score_name: str) -> pd.DataFrame:
            Applies the best K filter to the data.
        save_as_excel(save_path: str, lock: mp.Lock, file_name: str, result_df: pd.DataFrame):
            Function to save the result dataframe as an Excel file safely.
        dynamic_GMM_filtering():
            Filters the data dynamically, by considering periods of granularity, and selecting optimal K's
    """

    def __init__(
        self,
    ):
        pass

    def GMM_filtering(self, df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
        """
        Applies Gaussian Mixture Model to the data.

        Args:
            df (pd.DataFrame): The dataframe to be filtered.
            params (dict): The parameters for the GMM model (if needed).

        Returns:
            pd.DataFrame: The filtered dataframe & outlier df
        """
        model = GMMClass(params=params)

        scaler = RobustScalerClass(df)
        scaled_df = scaler.fit_transform_scaler(df)

        model.fit(scaled_df, scaler.cols)

        # Score samples
        scores = model.score_samples(scaled_df, scaler.cols)

        # Calculate box-plot filters
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter out the rows that are outside the bounds
        outliers = df[(scores < lower_bound) | (scores > upper_bound)]
        df = df[(scores > lower_bound) & (scores < upper_bound)]

        # Inverse transform the scaled data
        df = scaler.inverse_transform(df)
        outliers = scaler.inverse_transform(outliers)

        return df, outliers

    def dynamic_GMM_filtering(
        self,
        turbine_id: int,
        df: pd.DataFrame,
        result_path: str,
        objectives: list,
        weights: list,
        ref_point: list,
        granularity: Literal["Year", "Quarter", "Month"],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies the GMM filtering to the data.

        Args:
            turbine_id (int): The turbine id.
            df (pd.DataFrame): The dataframe to be filtered.
            result_path (str): The path to the GMM results.
            weights (list): The weights given to the objective funtion.
            ref_point (list): The ref point to calculate the hyper plane from.
            objectives (list): The list of objectives to be used for filtering.
            granularity (Literal): The granularity slice to be used for filtering.
                Can be "Year", "Quarter", or "Month".

        Returns:
            pd.DataFrame: The filtered dataframe.
            pd.DataFrame: The dataframe with the selected K's.
        """
        kf = KFinder(
            result_path=result_path,
            turbine_id=turbine_id,
        )

        filtered_df, k_df = kf.find_dynamic_PPS(
            objectives=objectives,
            weights=weights,
            ref_point=ref_point,
            granularity=granularity,
            df=df,
        )

        return filtered_df, k_df

    def GMM_best_K_filter(
        self,
        turbine_id: int,
        df: pd.DataFrame,
        result_path: str,
        score_name: str,
    ) -> pd.DataFrame:
        """
        DEPRECATED

        Applies the best K filter to the data.

        Args:
            turbine_id (int): The turbine id.
            df (pd.DataFrame): The dataframe to be filtered.
            result_path (str): The path to the GMM results.
            score_name (str): The score name to be used for filtering.

        Returns:
            pd.DataFrame: The filtered dataframe.
        """
        kf = KFinder(result_path)

        sheet_names = kf._sheet_name_fetcher()
        sheet = kf._match_sheet_turbine(sheet_names, turbine_id)
        best_k, _ = kf.find_best_PPS(sheet, score_name)
        if best_k == 0:
            return df, best_k
        else:
            filtered_df, _ = self.GMM_filtering(df, {"n_components": best_k})
            return filtered_df, best_k

    def save_as_excel(
        self,
        save_path: str,
        lock=None,
        file_name=None,
        result_df: pd.DataFrame = None,
    ):
        """
        Function to save the result dataframe as an Excel file safely.

        Args:
            save_path (str): The save path for the excel file.
            lock (mp.Lock): The lock for multiprocessing.
            file_name (str): The filename for the excel file.
            result_df (pd.DataFrame): The result dataframe to be saved.
        """

        file_path = f"{save_path}/{file_name}"

        with lock:
            if not os.path.exists(file_path):
                with pd.ExcelWriter(file_path, mode="w") as writer:
                    result_df.to_excel(writer, sheet_name="Results", index=False)

            else:
                existing_df = pd.read_excel(file_path, sheet_name="Results")
                existing_df = pd.concat([existing_df, result_df], ignore_index=True)
                with pd.ExcelWriter(file_path, mode="w") as writer:
                    existing_df.to_excel(writer, sheet_name="Results", index=False)


class MultiProcessGMMDirector:
    """
    Pipeline for applying GMM filtering to multiple Park dataframes.

    ...

    Attributes
    ----------
    turbine_ids (list):
        List of turbine ids to be filtered.
    score_name (str):
        The score name to be used for filtering.
    save_path (str):
        The save path for the filtered data and the excel file.
    xlsx_filename (str):
        The filename for the excel file.
    result_path (str):
        The path to the GMM results (from the experiment).

    Methods
    -------
    _apply_Park_GMM_filter(park: int, lock=None):
        Applies the GMM filter to the park data.
    apply_GMM_filter():
        Applies the GMM filter to the park dataframes under multiprocessing.
    """

    def __init__(
        self,
        park_list: list,
        turbine_ids: list,
        save_path: str = "data/k_filtered_data",
        source_path: str = "data/voting_system_data",
        destination_path: str = "data/k_filtered_data",
        result_path: str = "src/experiments/GMM_results.xlsx",
        k_excel_filename: str = "Selected_Ks.xlsx",
    ):
        self.park_list = park_list
        self.turbine_ids = turbine_ids
        self.result_path = result_path
        self.save_path = save_path
        self.xlsx_filename = k_excel_filename
        self.dl = DataLoader(source_path=source_path, destination_path=destination_path)

        if not os.path.exists(os.path.join(os.getcwd(), self.save_path)):
            os.makedirs(os.path.join(os.getcwd(), self.save_path), exist_ok=True)

    @add_progress(error_handling="coerce")
    def _apply_dynamic_gmm_filter(
        self,
        park: int,
        objectives: list,
        granularity: Literal["Year", "Quarter", "Month"],
        weights: list,
        ref_point: list,
        lock,
    ):

        turbine_df_list = []
        df = self.dl.load_park_data(park)

        for turbine_id in self.turbine_ids:
            if park != self.dl.fetch_park_number(turbine_id):
                continue
            turbine_df = df[df["TurbineId"] == turbine_id]
            turbine_df_copy = turbine_df.copy()
            filter_class = GMMFilterClass()
            filtered_turbine_df, k_df = filter_class.dynamic_GMM_filtering(
                turbine_id,
                turbine_df_copy,
                self.result_path,
                objectives=objectives,
                weights=weights,
                ref_point=ref_point,
                granularity=granularity,
            )
            turbine_df_list.append(filtered_turbine_df)
            if lock is not None:
                self.save_as_excel(
                    k_df,
                    turbine_id,
                    lock=lock,
                    file_name=self.xlsx_filename,
                )
        park_df = pd.concat(turbine_df_list, ignore_index=True)
        self.dl.save_park_data(
            park,
            park_df,
        )

    def apply_MP_dynamic_gmm_filter(
        self,
        objectives: list,
        granularity: Literal["Year", "Quarter", "Month"],
        weights: list,
        ref_point: list,
    ):
        manager = mp.Manager()
        lock = manager.Lock()
        res = progress_starmap(
            self._apply_dynamic_gmm_filter,
            [
                (
                    park,
                    objectives,
                    granularity,
                    weights,
                    ref_point,
                    lock,
                )
                for park in self.park_list
            ],
            n_cpu=mp.cpu_count(),
            total=len(self.park_list) * 2 - 1,
        )
        print(f"Errors during multiprocessing: {res}")

    def save_as_excel(
        self,
        k_df,
        turbine,
        lock=None,
        file_name=None,
    ):
        """
        Function to save the result dataframe as an Excel file safely.
        """
        if not file_name:
            raise ValueError("Please provide a file name.")

        file_path = f"{self.save_path}/{self.xlsx_filename}"

        with lock:
            if not os.path.exists(f"{file_path}"):
                with pd.ExcelWriter(f"{file_path}", mode="w") as writer:
                    k_df.to_excel(
                        writer,
                        sheet_name=f"Turbine {turbine}",
                        index=True,
                    )
            else:
                with pd.ExcelWriter(
                    f"{file_path}",
                    mode="a",
                    if_sheet_exists="replace",
                ) as writer:
                    k_df.to_excel(
                        writer,
                        sheet_name=f"Turbine {turbine}",
                        index=True,
                    )


class IQRPitchVsWindSpeedFiltering:
    """
    Applies IQR filtering to the pitch angle data, where each pitch angle value is considered a group.
    Intented for Park level operations.

    Attributes
    ----------
    pitch_list (list):
        List of pitch angles to be filtered.
    source_path (str):
        The source path for the data.
    destination_path (str):
        The destination path for the data.

    Methods
    -------
    _filter_iqr_grps(group, pitch_list: list) -> pd.DataFrame:
        The outer function creates the heirarchical groupby object and utilize the inner IQR filtering on the "WindSpeed" column.
    apply_iqr_filtering():
        Applies IQR filtering to Windspeed for every value of pitch angle.

    """

    def __init__(
        self,
        pitch_list: list = None,
        source_path: str = "data/k_filtered_data",
        destination_path: str = "data/iqr_filtered_data",
    ):
        self.pitch_list = pitch_list
        self.source_path = source_path
        self.destination_path = destination_path
        if pitch_list is None:
            self.pitch_list = ["PitchAngleA", "PitchAngleB", "PitchAngleC"]
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

    def _filter_iqr_grps(self, group, pitch_list: list) -> pd.DataFrame:
        """
        The outer function creates the heirarchical groupby object and utilize the inner IQR filtering on the "WindSpeed" column.

        Args:
            group: A groupby object, grouped on TurbineId's
            pitch_list: A list of pitch angles for filtering

        Returns:
            pd.DataFrame: The filtered dataframe
        """

        # We add the ugly group list to dodge pandas warnings
        group_list = []
        for pitch in pitch_list:
            group[pitch + "_group"] = group[pitch].apply(str)
            group_list.append(pitch + "_group")

        pitch_groups = group.groupby(group_list)

        def _inner_iqr_filter(group: pd.DataFrame) -> pd.DataFrame:
            """
            Applies IQR filtering on "WindSpeed" for each unique value in the pitch columns.

            Args:
                group: A dataframe grouped on the pitch angles.

            Returns:
                pd.DataFrame: The filtered dataframe.
            """

            Q1 = group["WindSpeed"].quantile(0.25)
            Q3 = group["WindSpeed"].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            return group[
                (lower_bound <= group["WindSpeed"])
                & (group["WindSpeed"] <= upper_bound)
            ]

        filtered_groups = pitch_groups.apply(_inner_iqr_filter, include_groups=False)
        return filtered_groups

    def apply_iqr_filtering(
        self,
    ):
        """
        Applies IQR filtering to the pitch angle data.
        """

        dl = DataLoader(
            source_path=self.source_path,
            destination_path=self.destination_path,
        )

        dw = DataWrangler(destination_path=self.destination_path)

        for park in dl.map_df["Park"].unique():
            print("-------------------------")
            print(f"Applying filter to park {park}")
            park_num = int(str.split(park, "Park")[1])
            df = dl.load_park_data(park_num)
            park_name = dl.create_file_name(park_num)

            df["TurbineGrp"] = df["TurbineId"]

            filtered_df = (
                df.groupby("TurbineGrp")
                .apply(
                    self._filter_iqr_grps,
                    self.pitch_list,
                    include_groups=False,
                )
                .reset_index(drop=True)
            )

            dw.save_dataframe(filtered_df, file_name=park_name, print_states=False)


class VotingFilterPipeline:
    """
    DEPRECATED: This class is no longer used and will be removed in future versions.
    Pipeline for applying voting system filtering to processed data.

    This class uses the VotingSystem to identify anomalous data entries in turbine data
    and creates new filtered pickle files with these anomalies removed.

    Attributes
    ----------
    source_path : str
        The source path for the processed data
    destination_path : str
        The destination path for the filtered data
    voting_threshold : float
        The threshold for the voting system to identify anomalies
    elected_handler : str
        The method to handle anomalies ('average' or 'remove')
    modify_by_period : bool
        Whether to modify by period or across all periods
    strict_mode : bool
        Whether to use strict mode for anomaly detection
    strict_mode_threshold : float
        The threshold ratio for strict mode
    dl : DataLoader
        DataLoader instance for loading turbine/park data

    Methods
    -------
    _apply_voting_filter(turbine_id: int, lock=None):
        Applies voting filter to data for a specific turbine
    apply_voting_filter(turbine_ids: list):
        Applies voting filter to multiple turbines with parallel processing
    """

    def __init__(
        self,
        granularity: Literal["year", "quarter", "month"] = "quarter",
        store_backup: bool = True,
    ):
        """
        Initialize the voting filter pipeline.

        Parameters
        ----------
        granularity : str, optional
            Granularity of dataframe, default is "quarter"
        """
        self.granularity = granularity
        self.store_backup = store_backup

    def format_gmm_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format the GMM dataframe to match the voting system requirements.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to format

        Returns
        -------
        pd.DataFrame
            The formatted dataframe
        """
        # Add your formatting logic here if needed
        gmm_results = df.copy()
        gmm_results = gmm_results.T
        gmm_results.columns = gmm_results.iloc[0]
        gmm_results = gmm_results.drop("Unnamed: 0")
        gmm_results = gmm_results.reset_index(drop=True)
        gmm_results["Year"] = gmm_results["Year"].astype(int)
        return gmm_results

    def reverse_format_gmm_df(self, formatted_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse the formatting applied by _format_gmm_df to return the dataframe to its original shape.

        Parameters
        ----------
        formatted_df : pd.DataFrame
            The formatted dataframe to be reversed

        Returns
        -------
        pd.DataFrame
            The dataframe in its original format
        """
        # Create a copy to avoid modifying the input
        df = formatted_df.copy()

        # Add back the index as a column called 'index'
        df = df.reset_index()

        # Transpose the dataframe
        df = df.T

        # Set the first row as the column names
        df.columns = df.iloc[0]

        # Drop the first row which now contains column names
        df = df.iloc[1:]

        # Insert the 'Unnamed: 0' row with the original column names
        df = df.reset_index()

        return df

    def overwrite_sheet(self, writer, df: pd.DataFrame, sheet_name: str):
        """
        Overwrite the existing sheet in the Excel file with the new dataframe.

        Parameters
        ----------
        writer : pd.ExcelWriter
            The Excel writer object
        df : pd.DataFrame
            The dataframe to write to the sheet
        sheet_name : str
            The name of the sheet to overwrite
        """
        if sheet_name in writer.book.sheetnames:
            idx = writer.book.sheetnames.index(sheet_name)
            writer.book.remove(writer.book.worksheets[idx])
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    def recompute_pps(self, df: pd.DataFrame, pps_columns: List[str]) -> pd.DataFrame:
        """
        Recompute the PPS values in the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to recompute PPS values

        Returns
        -------
        pd.DataFrame
            The dataframe with recomputed PPS values
        """
        df_copy = df.copy()
        # Recalculate sumpps (sum of all valid PPS values)
        if "sumpps" in df_copy.columns:
            df_copy["sumpps"] = df_copy[pps_columns].sum(axis=1, skipna=True)

        # Recalculate avgpps (average of all valid PPS values)
        if "avgpps" in df_copy.columns:
            # Count non-NaN values for each row
            valid_counts = df_copy[pps_columns].count(axis=1)
            # Sum of valid PPS values
            pps_sums = df_copy[pps_columns].sum(axis=1, skipna=True)
            # Calculate average (avoiding division by zero)
            df_copy["avgpps"] = pps_sums / valid_counts.replace(0, np.nan)

        return df_copy

    @add_progress(error_handling="coerce")
    def _apply_voting_filter(self, turbine_id: int, lock=None):
        """
        Parameters
        ----------
        turbine_id : int
            The turbine ID to process
        lock : multiprocessing.Lock, optional
            Lock for multiprocessing
        """
        try:
            # Read the Excel file for this turbine's sheet
            modified_gmm_path = os.path.join(
                os.getcwd(), "src", "experiments", "modified_GMM_results.xlsx"
            )
            gmm_path = os.path.join(
                os.getcwd(), "src", "experiments", "GMM_results.xlsx"
            )
            try:
                turbine_sheet = f"Turbine {turbine_id}"
                modified_gmm_df = pd.read_excel(
                    modified_gmm_path, sheet_name=turbine_sheet
                )
                gmm_df = pd.read_excel(gmm_path, sheet_name=turbine_sheet)
            except Exception as e:
                print(
                    f"Warning: Could not read Excel sheet for Turbine {turbine_id}: {e}"
                )
                return

            # Make a copy of gmm_df to update
            updated_gmm_df = self.format_gmm_df(gmm_df)
            pps_columns = [
                col for col in updated_gmm_df.columns if col.startswith("pps_")
            ]
            if not pps_columns:
                print(
                    f"Warning: No PPS columns found in Excel sheet for Turbine {turbine_id}"
                )
                return

            # Process based on granularity
            if self.granularity == "year":
                # Group by year
                year_groups = modified_gmm_df.groupby("Year")
                for year, year_data in year_groups:
                    # Check NaN values in PPS columns for this year
                    for pps_col in pps_columns:
                        if year_data[pps_col].isna().any():
                            # Update the corresponding data in gmm_df
                            year_mask = updated_gmm_df["Year"] == year
                            updated_gmm_df.loc[year_mask, pps_col] = np.nan

            elif self.granularity == "quarter":
                # Group by year and quarter
                for _, row in modified_gmm_df.iterrows():
                    year = row.get("Year")
                    quarter = row.get("Quarter")

                    if pd.isna(year) or pd.isna(quarter):
                        continue

                    # Check if any PPS column has NaN in this row
                    has_nan = any(pd.isna(row[pps_col]) for pps_col in pps_columns)

                    if has_nan:
                        # Find corresponding rows in gmm_df (by year and quarter)
                        mask = (updated_gmm_df["Year"] == year) & (
                            updated_gmm_df["Quarter"] == quarter
                        )

                        # For each PPS column with NaN, update the gmm_df
                        for pps_col in pps_columns:
                            if pd.isna(row[pps_col]):
                                updated_gmm_df.loc[mask, pps_col] = np.nan

            elif self.granularity == "month":
                # Group by year and month
                for _, row in modified_gmm_df.iterrows():
                    year = row.get("Year")
                    month = row.get("Month")

                    if pd.isna(year) or pd.isna(month):
                        continue

                    # Check if any PPS column has NaN in this row
                    has_nan = any(pd.isna(row[pps_col]) for pps_col in pps_columns)

                    if has_nan:
                        # Find corresponding rows in gmm_df (by year and month)
                        mask = (updated_gmm_df["Year"] == year) & (
                            updated_gmm_df["Month"] == month
                        )

                        # For each PPS column with NaN, update the gmm_df
                        for pps_col in pps_columns:
                            if pd.isna(row[pps_col]):
                                updated_gmm_df.loc[mask, pps_col] = np.nan

            updated_gmm_df = self.reverse_format_gmm_df(
                self.recompute_pps(updated_gmm_df, pps_columns)
            )

            if self.store_backup:
                backup_path = gmm_path.replace(".xlsx", "_backup.xlsx")
                if not os.path.exists(backup_path) and os.path.exists(gmm_path):
                    copy2(gmm_path, backup_path)
            # Save the updated DataFrame back to the Excel file
            if lock:
                with lock:
                    with pd.ExcelWriter(gmm_path, mode="a") as writer:
                        self.overwrite_sheet(writer, updated_gmm_df, turbine_sheet)
            else:
                with pd.ExcelWriter(gmm_path, mode="a") as writer:
                    self.overwrite_sheet(writer, updated_gmm_df, turbine_sheet)

        except Exception as e:
            print(f"Error processing Turbine {turbine_id}: {e}")
            raise

    def apply_voting_filter(self, turbine_ids=None):
        """
        Apply voting filter to multiple turbines with parallel processing.

        Parameters
        ----------
        turbine_ids : list, optional
            List of turbine IDs to process. If None, all turbines are processed.
        """
        if turbine_ids is None:
            dl = DataLoader()
            turbine_ids = dl.fetch_turbine_list()

        manager = mp.Manager()
        lock = manager.Lock()

        res = progress_starmap(
            self._apply_voting_filter,
            [(turbine_id, lock) for turbine_id in turbine_ids],
            n_cpu=mp.cpu_count(),
            total=len(turbine_ids) * 2 - 1,
        )


if __name__ == "__main__":
    # Example usage
    path = os.path.abspath(
        os.path.join(os.getcwd(), "src", "experiments", "GMM_results.xlsx")
    )
    k_finder = KFinder(
        result_path=path,
        turbine_id=11,
    )

    dl = DataLoader(source_path="data/voting_system_data")
    df = dl.load_park_data(2)
    df = df[df["TurbineId"] == 11]

    k_finder.find_dynamic_PPS(
        objectives=["avgpps", "delta_data"],
        weights=[0.6, 0.4],
        ref_point=[0, 1],
        granularity="Quarter",
        df=df,
    )
