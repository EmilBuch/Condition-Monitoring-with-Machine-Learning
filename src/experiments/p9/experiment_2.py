import sys
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

sys.path.insert(0, "src/")

from tuning.hp_tuning import XGBHyperParameterTuning
from models.model_classes import XGBRegressorClass
from visualization.visualize import ks_test_plots
from scipy.stats import wasserstein_distance
from utils.utils import (
    compute_error_metrics,
    calculate_percentage_diff,
    two_sample_KS_test,
)

pd.options.mode.chained_assignment = None


if __name__ == "__main__":

    # Candidate list of turbines
    # Fjernet: 86, 87
    # candidate_list = [89, 90, 91, 92]
    # candidate_list = [91]
    # df = pd.read_pickle("data/processed_data/vestas_data_final.pkl")

    df = pd.read_pickle("data/nbm_selector_data/off_shore/AAU_Park01.pkl")
    df = df[df["is_stable"] == True]
    candidate_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df["Year"] = df["Time"].dt.year

    trials = 50
    start_year = 2019
    save_path = f"experiment_2/{start_year}/"
    results_save_path = f"src/experiments/p9/experiment_2_results_{start_year}.xlsx"
    param_save_path = f"src/experiments/p9/experiment_2_best_params_{start_year}.xlsx"
    energy_loss_path = f"src/experiments/p9/experiment_2_energy_loss_{start_year}.xlsx"

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
    ]
    for col in utility_cols:
        if col in explainable:
            explainable.remove(col)

    # create empty DataFrame to store the test results
    experiment_2_df = pd.DataFrame(
        columns=[
            "TurbineId",
            "MAPE",
            "base_year",
            "Year",
            "KS Statistic",
            "P-Value",
            "Mean",
            "Perc Diff",
            "A_Perc Diff",
            "EMD",
            "sd",
        ]
    )

    experiment_2_best_params = pd.DataFrame()
    energy_loss_df = pd.DataFrame()

    for turbine in candidate_list:
        xgb_model = XGBRegressorClass()
        turbine_df = df[df["TurbineId"] == turbine]
        turbine_df = turbine_df[turbine_df["Year"] < 2024]
        turbine_df = turbine_df[turbine_df["Year"] >= start_year]

        # make "Year" catagorical
        turbine_df["Year"] = turbine_df["Year"].astype("category")

        ts_cv = XGBHyperParameterTuning(
            turbine_df, explainable, dependent, xgb_model, cv_mode="yearly"
        )
        best_model_params = ts_cv.run_hp_tuning(n_trials=trials)

        # train the model on the entire dataset
        xgb_model.update_model_params(best_model_params)
        xgb_model.train(turbine_df, explainable, dependent)

        # save the best model parameters to a DataFrame
        col_keys = list(best_model_params.keys())
        print(col_keys)

        # col_keys to save the best model parameters
        best_model_params["TurbineId"] = turbine
        experiment_2_best_params = pd.concat(
            [experiment_2_best_params, pd.DataFrame(best_model_params, index=[0])]
        )

        # synthetic dataset. Consists of starting year, and the next years as synthetic data
        list_of_years = turbine_df["Year"].unique()
        list_of_years = list_of_years[list_of_years != start_year]
        # sort the years in ascending order
        list_of_years = sorted(list_of_years)

        # synthetic dataset for the base year
        synthetic_df = turbine_df[turbine_df["Year"] == start_year]
        for year in list_of_years:
            year_df = turbine_df[turbine_df["Year"] == start_year].copy()
            year_df["Year"] = year
            synthetic_df = pd.concat([synthetic_df, year_df])
        # residuals for the synthetic dataset
        synthetic_df["Predictions"] = xgb_model.pred(synthetic_df, explainable)
        synthetic_df["Residuals"] = (
            synthetic_df["GridPower"] - synthetic_df["Predictions"]
        )

        mean_start_year = synthetic_df[synthetic_df["Year"] == start_year][
            "Residuals"
        ].mean()
        sd_start_year = synthetic_df[synthetic_df["Year"] == start_year][
            "Residuals"
        ].std()
        print(f"Mean Residual for {start_year}: {mean_start_year:.2f}")

        # Calculate the MAPE for the entire dataset
        turbine_df["Predictions"] = xgb_model.pred(turbine_df, explainable)
        mape = compute_error_metrics(
            turbine_df["GridPower"],
            turbine_df["Predictions"],
        )["MAPE"]

        print(f"MAPE for {turbine}: {mape:.2f}")

        NBM_perc_dif = calculate_percentage_diff(
            synthetic_df, start_year, "GridPower", start_year, "Predictions"
        )

        # concat the mean and sd to the experiment_2_df
        experiment_2_df = pd.concat(
            [
                experiment_2_df,
                pd.DataFrame(
                    {
                        "TurbineId": [turbine],
                        "MAPE": [mape],
                        "Mean": [mean_start_year],
                        "A_Perc Diff": [NBM_perc_dif],
                        "sd": [sd_start_year],
                    }
                ),
            ]
        )

        # run the tests and quantify the results
        for year in list_of_years:

            # synthetic dataset for loss estimation
            synthetic_df_loss_estimation = turbine_df[turbine_df["Year"] == year]
            nbm_year_df = turbine_df[turbine_df["Year"] == year].copy()
            nbm_year_df["Year"] = start_year
            synthetic_df_loss_estimation = pd.concat(
                [synthetic_df_loss_estimation, nbm_year_df]
            )
            # residuals for the synthetic dataset
            synthetic_df_loss_estimation["Predictions"] = xgb_model.pred(
                synthetic_df_loss_estimation, explainable
            )
            synthetic_df_loss_estimation["Residuals"] = (
                synthetic_df_loss_estimation["GridPower"]
                - synthetic_df_loss_estimation["Predictions"]
            )
            # calculate the energy loss and concatenate to the energy_loss_df
            energy_loss_perc = calculate_percentage_diff(
                synthetic_df_loss_estimation,
                year,
                "GridPower",
                start_year,
                "Predictions",
            )
            energy_loss_df = pd.concat(
                [
                    energy_loss_df,
                    pd.DataFrame(
                        {
                            "TurbineId": [turbine],
                            "Energy_Loss": [energy_loss_perc],
                            "Year": year,
                            "NBM_Year": [start_year],
                            "GridPower_Year": [
                                synthetic_df_loss_estimation[
                                    synthetic_df_loss_estimation["Year"] == year
                                ]["GridPower"].sum()
                            ],
                            "Predictions_Year": [
                                synthetic_df_loss_estimation[
                                    synthetic_df_loss_estimation["Year"] == start_year
                                ]["Predictions"].sum()
                            ],
                            "Energy_Loss_Year": [
                                synthetic_df_loss_estimation[
                                    synthetic_df_loss_estimation["Year"] == year
                                ]["GridPower"].sum()
                                - synthetic_df_loss_estimation[
                                    synthetic_df_loss_estimation["Year"] == start_year
                                ]["Predictions"].sum()
                            ],
                        }
                    ),
                ]
            )
            min_value = min(
                synthetic_df[synthetic_df["Year"] == start_year]["Residuals"].min(),
                synthetic_df[synthetic_df["Year"] == year]["Residuals"].min(),
            )
            max_value = max(
                synthetic_df[synthetic_df["Year"] == start_year]["Residuals"].max(),
                synthetic_df[synthetic_df["Year"] == year]["Residuals"].max(),
            )
            # domain range is the maximum range between min and max
            domain_range = max_value - min_value

            # Test drift hypothesis
            ks_stat, p_value = two_sample_KS_test(
                synthetic_df[synthetic_df["Year"] == start_year],
                synthetic_df[synthetic_df["Year"] == year],
                var_dist="Residuals",
                test_mode="less",
            )
            comparison_year = synthetic_df[synthetic_df["Year"] == year][
                "Residuals"
            ].mean()

            mean_diff = comparison_year - mean_start_year
            mean_diff_percent = (mean_diff / domain_range) * 100
            comparison_sd = synthetic_df[synthetic_df["Year"] == year][
                "Residuals"
            ].std()

            # Compute the Earth Mover's Distance
            emd = wasserstein_distance(
                synthetic_df[synthetic_df["Year"] == start_year]["Residuals"],
                synthetic_df[synthetic_df["Year"] == year]["Residuals"],
            )

            # Compute the percentage effect difference between the two years
            year_perc_diff = calculate_percentage_diff(
                synthetic_df, start_year, "GridPower", year, "Predictions"
            )

            # concat the results to the experiment_2_df
            experiment_2_df = pd.concat(
                [
                    experiment_2_df,
                    pd.DataFrame(
                        {
                            "TurbineId": [turbine],
                            "base_year": [start_year],
                            "Year": [year],
                            "KS Statistic": [ks_stat],
                            "P-Value": [p_value],
                            "Mean": [comparison_year],
                            "Perc Diff": [mean_diff_percent],
                            "A_Perc Diff": [year_perc_diff],
                            "EMD": [emd],
                            "sd": [comparison_sd],
                        }
                    ),
                ]
            )
            png_save = save_path + f"{turbine}_{start_year}_vs_{year}_residuals.png"
            print(png_save)
            ks_test_plots(
                synthetic_df,
                start_year,
                year,
                ks_stat,
                p_value,
                png_save,
            )

            print(f"Mean Residual Difference for {year}: {mean_diff:.2f}")
            print(
                f"Mean Residual percentage Difference for {year}: {mean_diff_percent:.2f}%"
            )
            print(f"EMD for {year}: {emd:.2f}")
            print(f"A_Perc Diff for {year}: {year_perc_diff:.2f}")
            print("\n")

    # save the results DataFrame to xlsx
    experiment_2_df.to_excel(results_save_path)
    experiment_2_best_params.to_excel(param_save_path)
    energy_loss_df.to_excel(energy_loss_path)
