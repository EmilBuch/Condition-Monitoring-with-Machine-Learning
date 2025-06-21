import sys
import pandas as pd
from typing import Any, Literal
from abc import abstractmethod, ABC
import multiprocessing as mp
import numpy as np


sys.path.insert(0, "src/")
from data_utility.data_utils import DataLoader, StabilityDataLoader
from data_utility.voting_system import VotingSystem
from visualization.visualize import (
    plot_K_SCORE_comparison_TS,
    plot_2d_scatter,
    plot_scatter_with_density,
    plot_joint_distribution,
    plot_fastkde_scatter,
    plot_2d_scatter_with_marginals,
    plot_heatmap,
    box_plots,
    stable_period_plots,
)
from parallelbar.wrappers import add_progress
from parallelbar import progress_starmap


class MultiProcessingPlotterParent:
    """Base class for plotters utilizing multiprocessing"""

    def __init__(
        self, turbine_list: list[int], source_path: str = None, save_path: str = None
    ):
        self.dl = DataLoader(source_path=source_path)
        self.turbine_list = turbine_list
        self.source_path = source_path
        self.save_path = save_path

    @abstractmethod
    def plotter_method(
        self,
        function: callable,
        turbine_id: int,
    ):
        pass

    @abstractmethod
    def iterative_plotter(self, function: callable, save_path: str, **kwargs):
        """Iteratively plots the function for all turbines in the turbine_list in parallel"""
        pass

    def _homogen_replacement(df: pd.DataFrame, variable: str) -> str:
        """Replaces the homogenized values in the column with the original values

        This function checks if the variable name is in the groups of the VotingSystem class.
        The variable names are expected to have an ending format like 'A' or 'B', to be consistent
        with the groups defined in the VotingSystem class. For example: 'BladeLoadA' belongs to 'BladeLoad'

        Args:
            df (pd.DataFrame): The dataframe
            variable (str): The column to be replaced

        Returns:
            str: The new column name
        """
        input_variable = variable
        voting_class = VotingSystem()
        groups = voting_class.groups
        dim_check = False
        if variable[:-1] in groups:
            for col in groups[variable[:-1]]:
                if col == variable:
                    continue
                if not dim_check and df[col].isna().all():
                    variable = col
                    dim_check = True
                    print(
                        f"Input {input_variable} was changed to {variable} due to NaNs"
                    )
        if not dim_check:
            raise ValueError("No dimension found without NaNs")
        return variable

    def two_dim_plot_prepper(
        self, turbine_id: int, X_col: str, Y_col: str, save_path: str
    ) -> tuple[pd.DataFrame, str, str, str]:
        """Prepares variables for two-dimensional plotting
        Args:
            turbine_id (int): Turbine ID
            X_col (str): x column
            Y_col (str): y column
            save_path (str): Where to save plots in subfolder

        Raises:
            ValueError: X_col is not in the dataframe
            ValueError: Y_col is not in the dataframe

        Returns:
            _type_: tuple[pd.DataFrame, str, str, str]
        """
        df = self.dl.load_turbine_data(turbine_id)
        park = self.dl.fetch_park_number(turbine_id)

        if df[X_col].isna().all():
            X_col = self._homogen_replacement(df, X_col)
        if df[Y_col].isna().all():
            Y_col = self._homogen_replacement(df, Y_col)
        if X_col not in df.columns:
            raise ValueError(f"{X_col} is not in the dataframe")
        if Y_col not in df.columns:
            raise ValueError(f"{Y_col} is not in the dataframe")

        new_save_path = f"{save_path}/Park{str(park).zfill(2)}/Turbine{turbine_id}"

        return df, X_col, Y_col, new_save_path


class TwoDimensionalMultiProcessPlotterPLT(MultiProcessingPlotterParent):
    """
    Class for plotters utilizing multiprocessing and two-dimensional plotting with plt

    This class works with the following plotting functions:
    - plot_scatter_with_density
    - plot_3d_density
    - plot_2d_scatter
    - plot_fastkde_scatter

    ...

    Arguments:
        turbine_list (list[int]):
            List of turbine IDs
        source_path (str):
            Path to the source data
        save_path (str):
            The directory where the plots will be saved

    Attr:
    -----------
        dl (DataLoader):
            DataLoader instance for loading turbine data
        turbine_list (list[int]):
            List of turbine IDs
        source_path (str):
            Path to the source data
        save_path (str):
            The directory where the plots will be saved

    Methods:
    ----------
        plotter_method(function, turbine_id, X_col, Y_col, kwargs):
            Plots the function for the turbine_id

    Parent Methods:
    ----------
        _homogen_replacement(df, variable):
            Replaces the homogenized values in the column with the original values
        two_dim_plot_prepper(turbine_id, X_col, Y_col):
            Prepares variables for two-dimensional plotting
        iterative_plotter(function, X_col, Y_col, save_path, **kwargs):
            Iteratively plots the function for all turbines in the turbine_list in parallel
    """

    def __init__(
        self,
        turbine_list: list[int],
        source_path: str = None,
        save_path: str = None,
    ):
        super().__init__(turbine_list, source_path, save_path)

    def plotter_method(
        self,
        function: callable,
        turbine_id: int,
        X_col: str,
        Y_col: str,
        save_path,
        kwargs: dict,
    ):
        """Populates the function for the turbine_id and saves the plot

        Args:
            function (callable): The plotting function
            turbine_id (int): Turbine ID
            X_col (str): X column
            Y_col (str): Y column
            save_path (str): Where to save plots in subfolder
            **kwargs: Additional arguments to be passed to the plotting function of choice
        """

        df, X_col, Y_col, turbine_save_path = self.two_dim_plot_prepper(
            turbine_id, X_col, Y_col, save_path
        )

        function(
            df[X_col],
            df[Y_col],
            filename=turbine_save_path,
            **kwargs,
        )

    def iterative_plotter(
        self,
        function: callable,
        X_col: str,
        Y_col: str,
        save_path: str,
        **kwargs,
    ):
        """Iteratively plots the function for all turbines in the turbine_list

        Args:
            function (callable): The plotting function
            X_col (str): X column
            Y_col (str): Y column
            save_path (str): Where to save plots in subfolder
            **kwargs: Additional arguments to be passed to the plotting function
        """

        progress_starmap(
            self.plotter_method,
            [
                (function, turbine_id, X_col, Y_col, save_path, kwargs)
                for turbine_id in self.turbine_list
            ],
            n_cpu=mp.cpu_count(),
            total=len(self.turbine_list),
        )


class TwoDimensionalPlotterSNS(TwoDimensionalMultiProcessPlotterPLT):
    """
    Class for plotters utilizing multiprocessing and two-dimensional plotting with sns
    This Class inherits functionality from the MultiProcessingPlotterParent class and
    the TwoDimMultiProcessPlotterPLT class, but implements its own plotter_method for sns functionality.

    This class works with the following plotting functions:
    - plot_joint_distribution
    - plot_2d_scatter_with_marginals
    - plot_heatmap
    - box_plots

    ...

    Arguments:
        turbine_list (list[int]):
            List of turbine IDs
        source_path (str):
            Path to the source data
        save_path (str):
            The directory where the plots will be saved

    Attr:
    -----------
        dl (DataLoader):
            DataLoader instance for loading turbine data
        turbine_list (list[int]):
            List of turbine IDs
        source_path (str):
            Path to the source data
        save_path (str):
            The directory where the plots will be saved

    Methods:
    ----------
        plotter_method(function, turbine_id, X_col, Y_col, kwargs):
            Plots the function for the turbine_id

    Parent Methods:
    ----------
        _homogen_replacement(df, variable):
            Replaces the homogenized values in the column with the original values
        two_dim_plot_prepper(turbine_id, X_col, Y_col):
            Prepares variables for two-dimensional plotting
        iterative_plotter(function, X_col, Y_col, save_path, **kwargs):
            Iteratively plots the function for all turbines in the turbine_list in parallel
    """

    def __init__(
        self,
        turbine_list: list[int],
        source_path: str = None,
        save_path: str = None,
    ):
        super().__init__(turbine_list, source_path, save_path)

    def plotter_method(
        self,
        function: callable,
        turbine_id: int,
        X_col: str,
        Y_col: str,
        save_path: str,
        kwargs: dict,
    ):
        """Plots the function for the turbine_id

        Args:
            function (callable): The plotting function
            turbine_id (int): Turbine ID
            df (pd.DataFrame): Dataframe
            **kwargs: Additional arguments to be passed to the plotting function
        """

        df, X_col, Y_col, turbine_save_path = self.two_dim_plot_prepper(
            turbine_id, X_col, Y_col, save_path
        )

        function(
            df,
            X_col,
            Y_col,
            filename=turbine_save_path,
            **kwargs,
        )


def iterative_plot_feeder(
    result_path: str = "src/experiments/GMM_experiments/gmm_results_general.xlsx",
    save_path: str = "K_SCORE_Comparison",
    granularity: Literal["year", "quarter", "month"] = "year",
    plotter_function: callable = None,
    **kwargs,
):
    """Iteratively plots the K_SCORE comparison for each turbine in the result_path(excel file)

    IMPORTANT:
    ----------
        The excel files granularity should always match the chosen granularity

    Args:
        result_path (str, optional): Where the excel file exists. Defaults to "src/experiments/GMM_experiments/gmm_results_general.xlsx".
        save_path (str, optional): Where to save plots in subfolder. Defaults to "K_SCORE_Comparison".
        map_path (str, optional): turbine mapping. Defaults to "data/turbine_mapping.xlsx".
        **kwargs: Additional arguments to be passed to the plot function
    """

    def _sheet_fetcher(path: str):
        """Returns the names of the sheets in an Excel file as a list

        Args:
            path (str): Path to the Excel file
        """
        xls = pd.ExcelFile(path)
        return xls.sheet_names

    def _fetch_park(turbine_id: int) -> int:
        """fetches park number by using the turbine_mapping.xlsx file

        Args:
            turbine_id (int): turbine id
            path (str): path to the turbine_mapping.xlsx file
        """

        dl = DataLoader()
        park_num = dl.fetch_park_number(turbine_id)
        return park_num

    # Load Excel file
    sheet_names = _sheet_fetcher(result_path)

    for sheet in sheet_names:
        turbine_id = int(sheet.split(" ")[1])
        df = pd.read_excel(result_path, sheet_name=sheet)
        df = df.T
        df.columns = df.iloc[0]
        df = df.drop("Unnamed: 0")
        df = df.reset_index(drop=True)
        df["Year"] = df["Year"].astype(int)

        # create cols for plotting
        if granularity == "year":
            time_col = "Year"
            df = df.sort_values(by=["Year"])
            cols = [col for col in df.columns if col not in ["Year", "K", "Turbine"]]
        elif granularity == "quarter":
            df["Quarter"] = df["Quarter"].astype(int)
            time_col = "Year_Quarter"
            df[time_col] = df["Year"].astype(str) + " : Q" + df["Quarter"].astype(str)
            df = df.sort_values(by=["Year", "Quarter"])
            # drop month and year
            cols = [
                col
                for col in df.columns
                if col not in ["Year", "Quarter", "K", "Turbine", "Year_Quarter"]
            ]

        elif granularity == "month":
            df["Month"] = df["Month"].astype(int)
            time_col = "Year_Month"
            df[time_col] = df["Year"].astype(str) + " : M" + df["Month"].astype(str)
            df = df.sort_values(by=["Year", "Month"])
            cols = [
                col
                for col in df.columns
                if col not in ["Year", "Month", "K", "Turbine", "Year_Month"]
            ]
        else:
            raise ValueError(
                "Invalid granularity. Choose from 'year', 'quarter', 'month'"
            )

        park = _fetch_park(turbine_id)

        for col in cols:
            # Update save path
            new_save_path = f"{save_path}/Park{str(park).zfill(2)}/{sheet}/{col}"

            # Update kwargs
            kwargs["score_col"] = col
            kwargs["filename"] = f"{new_save_path}"
            kwargs["title"] = f"K SCORE Comparison for {col}, Turbine {turbine_id}"

            # if "pps_" in col or "avgpps" in col:
            #     kwargs["limit_yaxis"] = True
            # else:
            #     kwargs["limit_yaxis"] = False
            # Plot
            plotter_function(df, time_col, **kwargs)


def plot_yearly_comparison_stable(
    turbine_id: int, stable_loader: object, data_loader: object
):
    T_year, R_1, target_years = stable_loader.load_years(turbine_id)
    target_years = eval(target_years)
    target_years = [int(i) for i in target_years]
    turbine_df = data_loader.load_turbine_data(turbine_id)
    years = [T_year, R_1] + target_years

    turbine_df = turbine_df[turbine_df["is_stable"] == 1]
    turbine_df["year"] = pd.to_datetime(turbine_df["Time"]).dt.year
    turbine_df = turbine_df[turbine_df["year"].isin(years)]

    stable_period_plots(df=turbine_df, turbine_id=turbine_id, file_name=None)


if __name__ == "__main__":
    # kwargs = {
    #     "savefig": True,
    #     "overwrite": True,
    #     "xlabel": "PitchAngle",
    #     "ylabel": "WindSpeed",
    #     "title": "Scatter Plot",
    # }

    # turbines = [i for i in range(1, 118, 1)]
    # # Create an instance of the TwoDimensionalPlotter
    # plotter = TwoDimensionalMultiProcessPlotterPLT(
    #     turbines, source_path="data/k_filtered_data"
    # )

    # # Iteratively plot the function for all turbines
    # plotter.iterative_plotter(
    #     plot_fastkde_scatter,
    #     X_col="PitchAngleA",
    #     Y_col="WindSpeed",
    #     save_path="plot_fastkde_scatter",
    #     **kwargs,
    # )
    dataloader = DataLoader(source_path="data/nbm_selector_data")
    stable_loader = StabilityDataLoader()
    turbines = stable_loader.load_stable_turbines()

    # testing the plot function
    for turbine in turbines:
        plot_yearly_comparison_stable(turbine, stable_loader, dataloader)
