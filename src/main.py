import sys
import os
from typing import List
import pandas as pd
import numpy as np
import warnings
import multiprocessing as mp
import torch

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)

sys.path.insert(0, "src/")

from data_utility.make_dataset import MatLabConverter, DataWrangler
from experiments.GMM_experiments.exp_builder import Director, ConcreteGMMExpBuilder
from experiments.percentage_drift_exp.percentage_drift_exp import (
    MultiProcessDriftExperiment,
)
from experiments.sensitivity_analysis.sensitivity_exp import (
    MultiProcessSensitivityAnalysis,
)
from data_utility.voting_system import VotingSystem
from data_utility.data_utils import (
    MultiProcessGMMDirector,
    IQRPitchVsWindSpeedFiltering,
    DataLoader,
    StabilityDataLoader,
)
from data_utility.nbm_selector import StabilitySelector
from utils.plotting_utils import (
    iterative_plot_feeder,
    plot_K_SCORE_comparison_TS,
)
from utils.utils import sort_sheets
from models.model_params import MODEL_DICTIONARY

torch.set_num_threads(1)


def get_removed_turbines() -> List[int]:
    """Get list of turbines that should be excluded from experiments"""
    removed_turbines_path = os.path.join(os.getcwd(), "data", "removed_turbines.xlsx")
    if os.path.exists(removed_turbines_path):
        removed_df = pd.read_excel(removed_turbines_path)
        return removed_df["turbine_id"].unique().tolist()
    return []


def data_preprocessing(mlc: MatLabConverter, dw: DataWrangler) -> None:
    """Convert and wrangle data"""
    mlc.recursive_data_converter()
    dw.recursive_wrangle()


def iqr_filtering() -> None:
    """Apply IQR filtering on the pitch angle"""
    print("----------------------------------")
    print("Applying IQR filtering on the pitch angle")
    iqr_filter = IQRPitchVsWindSpeedFiltering(
        pitch_list=["PitchAngleA", "PitchAngleB", "PitchAngleC"],
        source_path="data/processed_data",
    )
    iqr_filter.apply_iqr_filtering()


def pps(director: Director, turbine_ids: List[int]) -> None:
    """Execute the basic PPS experiment"""
    print("----------------------------------")
    print("Executing the basic PPS experiment for Voting System")
    columns_to_remove = ["WSE", "AmbTemp", "WdAbs", "WindDirRel"]
    director.execute_experiments(
        turbine_ids,
        "quarter",
        0,
        file_name="pps.xlsx",
        cols_to_exclude=columns_to_remove,
    )


def voting_system(turbine_ids: List[int]) -> List[int]:
    """Execute the voting system experiment"""
    print("----------------------------------")
    print("Executing the Voting System experiment")
    dl = DataLoader(
        source_path="data/iqr_filtered_data",
        destination_path="data/voting_system_data",
    )  # Initialize DataLoader for data handling

    vs = VotingSystem(
        data_loader=dl, threshold=0.09, elected_handler="remove", modify_by_period=True
    )
    vs.vote(turbine_ids, strict_mode=True)

    # Filter out removed turbines
    removed_turbines = get_removed_turbines()
    filtered_turbine_ids = [t for t in turbine_ids if t not in removed_turbines]
    print(f"Excluding {len(removed_turbines)} turbines marked for removal")

    return filtered_turbine_ids


def gmm_experiment(director: Director, turbine_ids: List[int]) -> None:
    """Execute the GMM experiment"""
    print("----------------------------------")
    print("Executing the K-GMM experiments")
    columns_to_remove = [
        "WSE",
        "AmbTemp",
        "WdAbs",
        "WindDirRel",
        "PitchAngleA",
        "PitchAngleB",
        "PitchAngleC",
    ]
    director.execute_experiments(
        turbine_ids,
        "quarter",
        5,
        file_name="GMM_results.xlsx",
        cols_to_exclude=columns_to_remove,
    )


def plot_gmm_results() -> None:
    """Plot the GMM experiment results"""
    print("----------------------------------")
    print("Plotting the GMM experiment results")
    kwargs = {"savefig": True, "overwrite": True}
    iterative_plot_feeder(
        result_path="src/experiments/GMM_results.xlsx",
        granularity="quarter",
        plotter_function=plot_K_SCORE_comparison_TS,
        **kwargs,
    )


def dynamic_k_selection(
    turbine_ids: list, weights: list = [0.6, 0.4], ref_point: list = [0, 1]
) -> None:
    # Get all unqiue park ids
    dl = DataLoader(source_path="data/voting_system_data")
    park_ids = []
    for turbine_id in turbine_ids:
        park = dl.fetch_park_number(turbine_id)
        if park is None:
            print(f"Park number not found for turbine {turbine_id}.")
            continue
        if park not in park_ids:
            park_ids.append(park)

    gmm_director = MultiProcessGMMDirector(
        park_list=park_ids,
        turbine_ids=turbine_ids,
        source_path="data/voting_system_data",
        destination_path="data/k_filtered_data",
    )
    # Filter the data
    print("Finding the best K by looking in result file")
    gmm_director.apply_MP_dynamic_gmm_filter(
        granularity="Quarter",
        weights=np.array(weights),
        objectives=["avgpps", "delta_data"],
        ref_point=np.array(ref_point),
    )


def select_stable_turbines(
    threshold: float = 0.8,
    std_threshold: float = 0.03,
    window_size: int = 3,
) -> None:
    """Select stable turbines based on GMM results"""
    print("----------------------------------")
    print("Selecting stable turbines and periods based on GMM results")
    # Run the stability selector for all parks
    StabilitySelector().run_stability_selector(
        stability_trheshold=threshold,
        std_cutoff=std_threshold,
        window_size=window_size,
    )


def run_experiment_1(
    explainable_vars: list = None,
    target_var: str = "GridPower",
    n_trials: int = 50,
    xlsx_name: str = "perc_drift_results.xlsx",
):
    print("----------------------------------")
    print("Running the drift experiment")
    # Class for fetching stability results
    stability_data_loader = StabilityDataLoader()
    turbines = stability_data_loader.load_stable_turbines()
    turbines = [4, 6, 7, 17, 26, 80, 81, 88, 96, 105, 108, 110]
    print(f"Running drift experiment for {len(turbines)} turbines")
    if explainable_vars is None:
        dl = DataLoader(source_path="data/nbm_selector_data")
        turbine_df = dl.load_turbine_data(turbine_id=1)
        # remove the columns that are not needed
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

    experiment = MultiProcessDriftExperiment(
        source_path="data/nbm_selector_data",
        stability_path="data/nbm_selector_data/stability.xlsx",
        result_path="data/nbm_selector_data",
        xlsx_filename=xlsx_name,
    )
    cpu_count = mp.cpu_count()
    experiment.run_parallel_experiment(
        turbines=turbines,
        explainable_vars=explainable_vars,
        target_var=target_var,
        models=MODEL_DICTIONARY,
        n_trials=n_trials,
        cpu_num=cpu_count,
    )


def run_experiment_2(
    explainable_vars: list = None,
    target_var: str = "GridPower",
    n_trials: int = 50,
    xlsx_name: str = "sensitivity_results.xlsx",
):
    print("----------------------------------")
    print("Running the sensitivity experiment")
    # Class for fetching stability results
    stability_data_loader = StabilityDataLoader()
    turbines = stability_data_loader.load_stable_turbines()
    turbines = [4, 80]
    print(f"Running sensitivity experiment for {len(turbines)} turbines")

    if explainable_vars is None:
        dl = DataLoader(source_path="data/nbm_selector_data")
        turbine_df = dl.load_turbine_data(turbine_id=1)
        # remove the columns that are not needed
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
    # Run the sensitivity analysis
    experiment = MultiProcessSensitivityAnalysis(
        source_path="data/nbm_selector_data",
        stability_path="data/nbm_selector_data/stability.xlsx",
        result_path="data/nbm_selector_data",
        xlsx_filename=xlsx_name,
    )
    cpu_count = mp.cpu_count()
    experiment.run_parallel_experiment(
        turbines=turbines,
        explainable_vars=explainable_vars,
        target_var=target_var,
        models=MODEL_DICTIONARY,
        n_trials=n_trials,
        cpu_num=cpu_count,
    )


def run_full_pipeline():
    # Init and run data preprocessing classes and functions
    mlc = MatLabConverter(
        source_path="data/raw_data", intermediate_path="data/intermediate_data"
    )
    dw = DataWrangler(
        source_path="data/intermediate_data",
        destination_path="data/processed_data",
    )
    data_preprocessing(mlc, dw)
    iqr_filtering()

    # Init dataloader for GMM exp and voting
    dl = DataLoader()
    turbine_ids = dl.fetch_turbine_list()

    # Clasess for running k-experiment
    director = Director()
    builder = ConcreteGMMExpBuilder()
    director.builder = builder

    # Run voting, and re-evaluate id's
    pps(director, turbine_ids)
    turbine_ids = voting_system(turbine_ids)

    # Run GMM experiment, update results
    gmm_experiment(director, turbine_ids)
    plot_gmm_results()

    # Final filter with dynamic K filering
    dynamic_k_selection(turbine_ids)
    select_stable_turbines(threshold=0.8, std_threshold=0.03, window_size=3)

    # # Run the drift experiment
    run_experiment_1()
    # Run the sensitivity experiment
    run_experiment_2()
    # sort the sheets in the excel file
    sort_sheets(
        input_path="data/nbm_selector_data/perc_drift_results.xlsx",
        output_path="data/nbm_selector_data/drift_results_sorted.xlsx",
    )
    sort_sheets(
        input_path="data/nbm_selector_data/sensitivity_results.xlsx",
        output_path="data/nbm_selector_data/sensitivity_results_sorted.xlsx",
    )


if __name__ == "__main__":
    #########################################
    ####### RUN INDIVIDUAL FUNCTIONS ########
    #########################################
    # # Classes for loading converting and wrangling the source data
    # mlc = MatLabConverter(
    #     source_path="data/raw_data", intermediate_path="data/intermediate_data"
    # )
    # dw = DataWrangler(
    #     source_path="data/intermediate_data",
    #     destination_path="data/processed_data",
    # )

    # data_preprocessing(mlc, dw)
    # iqr_filtering()

    # # Init dataloader for GMM exp and voting
    # dl = DataLoader()
    # turbine_ids = dl.fetch_turbine_list()
    # park_ids = dl.fetch_park_list()

    # # Clasess for running k-experiment
    # director = Director()
    # builder = ConcreteGMMExpBuilder()
    # director.builder = builder

    # # Run voting, and re-evaluate id's
    # # pps(director, turbine_ids)
    # turbine_ids = voting_system(turbine_ids)

    # # Run GMM experiment, update results
    # gmm_experiment(director, turbine_ids)
    # plot_gmm_results()

    # # Final filter with dynamic K filering
    # dynamic_k_selection(turbine_ids)

    # select_stable_turbines(
    #     turbine_ids=turbine_ids, threshold=0.8, std_threshold=0.03, window_size=3
    # )  # Select stable turbines based on GMM results

    #########################################
    ########### RUN FULL PIPELINE ###########
    #########################################
    run_full_pipeline()

    run_experiment_1(
        explainable_vars=["WindSpeed", "AmbTemp"],
        target_var="GridPower",
        n_trials=50,
        xlsx_name="perc_drift_results_WS_AT.xlsx",
    )
    run_experiment_2(
        explainable_vars=["WindSpeed", "AmbTemp"],
        target_var="GridPower",
        n_trials=50,
        xlsx_name="sensitivity_results_WS_AT.xlsx",
    )
    # sort the sheets in the excel file
    sort_sheets(
        input_path="data/nbm_selector_data/perc_drift_results_WS_AT.xlsx",
        output_path="data/nbm_selector_data/perc_drift_results_WS_AT_sorted.xlsx",
    )
    sort_sheets(
        input_path="data/nbm_selector_data/sensitivity_results_WS_AT.xlsx",
        output_path="data/nbm_selector_data/sensitivity_results_WS_AT_sorted.xlsx",
    )
