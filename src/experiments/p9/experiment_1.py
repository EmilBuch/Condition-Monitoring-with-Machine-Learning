import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "src/")

from tuning.hp_tuning import XGBHyperParameterTuning
from models.model_classes import XGBRegressorClass
from visualization.visualize import ks_test_plots
from utils.utils import two_sample_KS_test
from utils.utils import compute_error_metrics, calculate_percentage_diff


def experiment_intro_year_vs_test_year(
    df: pd.DataFrame,
    model: object,
    explainable: list,
    dependent: list,
    train_year: list,
    test_year: list,
):
    train_df = df[df["Time"].dt.year.isin(train_year)]
    test_df = df[df["Time"].dt.year.isin(test_year)].copy()

    model.train(train_df, explainable, dependent)
    test_df["Predictions"] = model.pred(test_df, explainable)
    test_df["Residuals"] = test_df[dependent[0]] - test_df["Predictions"]

    return test_df


if __name__ == "__main__":
    # df = pd.read_pickle("data/processed_data/vestas_data_final.pkl")
    # df["Year"] = df["Time"].dt.year
    # # df["Month"] = df["Time"].dt.month

    # # POTENTIAL LIST OF TURBINE CANDIDATES #
    # candidate_list = [86, 87, 89, 90, 91, 92]
    # # candidate_list = [91]

    df = pd.read_pickle("data/0.09_nbm_selector_data/off_shore/AAU_Park01.pkl")
    df = df[df["is_stable"] == True]
    # candidate_list = [1, 4, 6, 7, 9]
    candidate_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df["Year"] = df["Time"].dt.year

    training_year = 2019
    trials = 50
    fig_save_path = f"experiment_1/{training_year}/"
    results_save_path = (
        f"src/experiments/p9/experiment_1_results_{training_year}_percent.xlsx"
    )
    param_save_path = (
        f"src/experiments/p9/experiment_1_best_params_{training_year}.xlsx"
    )

    # Test years
    test_years = list(range(training_year + 1, 2024))
    # Sort the test years in ascending order
    test_years.sort()

    # Data variables
    dependent = ["GridPower"]
    explainable = list(df.columns)
    utility_cols = [
        "GridPower",
        "TurbineId",
        "WSE",
        "Time",
        "Park",
        "TurbineType",
        "WindSpeedBin",
        "Year",
    ]
    for col in utility_cols:
        if col in explainable:
            explainable.remove(col)

    # create empty DataFrame to store the KS test results
    ks_test_results_df = pd.DataFrame(
        columns=[
            "TurbineId",
            "Base Year",
            "MAPE",
            "Year",
            "KS Statistic",
            "P-Value",
            "Percent Diff",
        ]
    )

    experiment_1_best_params = pd.DataFrame()

    for turbine in candidate_list:
        xgb_model = XGBRegressorClass()
        turbine_df = df[df["TurbineId"] == turbine]
        hyper_param_df = turbine_df[turbine_df["Year"] == training_year]
        if hyper_param_df.empty:
            print(f"No data for turbine {turbine} in year {training_year}.")
            continue

        ts_cv = XGBHyperParameterTuning(
            hyper_param_df, explainable, dependent, xgb_model, cv_mode="monthly"
        )

        # find the best set of hyperparameters
        best_model_params = ts_cv.run_hp_tuning(n_trials=trials)
        tuned_model = XGBRegressorClass(params=best_model_params)

        # Keys to save as cols:
        col_keys = list(best_model_params.keys())
        col_keys.insert(0, "TurbineId")
        print(col_keys)

        # col_keys to save the best model parameters
        best_model_params["TurbineId"] = turbine
        experiment_1_best_params = pd.concat(
            [experiment_1_best_params, pd.DataFrame(best_model_params, index=[0])]
        )

        experiment_df = experiment_intro_year_vs_test_year(
            turbine_df,
            tuned_model,
            explainable,
            dependent,
            train_year=[training_year],
            test_year=test_years,
        )

        print(f"Running KS Test for Turbine {turbine}")
        for year in test_years:
            if year == test_years[0] and year in experiment_df["Year"].unique():
                # Calculate the MAPE for the base year
                mape = compute_error_metrics(
                    experiment_df[experiment_df["Year"] == year][dependent[0]],
                    experiment_df[experiment_df["Year"] == year]["Predictions"],
                )["MAPE"]
                ks_test_results_df = pd.concat(
                    [
                        ks_test_results_df,
                        pd.DataFrame(
                            {
                                "TurbineId": [experiment_df["TurbineId"].unique()[0]],
                                "Base Year": [test_years[0]],
                                "MAPE": [mape],
                            }
                        ),
                    ]
                )
                continue

            ks_stat, p_value = two_sample_KS_test(
                experiment_df[experiment_df["Year"] == test_years[0]],
                experiment_df[experiment_df["Year"] == year],
                var_dist="Residuals",
            )

            # Calculate the percentage difference between the base year and the test year
            percent_diff = calculate_percentage_diff(
                experiment_df,
                test_years[0],
                "GridPower",
                year,
                "Predictions",
            )
            ks_test_results_df = pd.concat(
                [
                    ks_test_results_df,
                    pd.DataFrame(
                        {
                            "TurbineId": [experiment_df["TurbineId"].unique()[0]],
                            "Base Year": [test_years[0]],
                            "Year": [year],
                            "KS Statistic": [ks_stat],
                            "P-Value": [p_value],
                            "Percent Diff": [percent_diff],
                        }
                    ),
                ]
            )
            print(f"KS Test for {test_years[0]} vs {year}")
            print(f"KS Statistic: {ks_stat}")
            print(f"P-Value: {p_value}")
            print("\n")

            save_path = (
                fig_save_path + f"{turbine}_{test_years[0]}_vs_{year}_residuals.png"
            )

            ks_test_plots(
                experiment_df,
                test_years[0],
                year,
                ks_stat,
                p_value,
                save_path,
            )

    # saving results in an excel file
    ks_test_results_df.to_excel(
        results_save_path,
        index=False,
    )
    experiment_1_best_params.to_excel(
        param_save_path,
        index=False,
    )
