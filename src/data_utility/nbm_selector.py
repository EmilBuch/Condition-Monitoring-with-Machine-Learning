import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from typing import Dict, List, Optional

sys.path.insert(0, "src")
from data_utility.data_utils import DataLoader


# Set paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER = os.path.join(PROJECT_ROOT, os.pardir, "data", "k_filtered_data")
EXPERIMENT_FOLDER = os.path.join(PROJECT_ROOT, "experiments")


def _load_gmm_results(turbine_id: int) -> pd.DataFrame:
    """
    Load and preprocess the GMM results for a given turbine from an Excel file.

    This function reads GMM results from an Excel workbook where each turbine's data is stored
    in a separate sheet named "Turbine {turbine_id}". It transposes the data, sets the first row
    as the header, and resets the index. The "Year" column is converted to an integer type.
    Additionally, the function loads an auxiliary Excel file ("GMM_results_filtered.xlsx") to determine
    the optimal number of clusters (Best K) for the specified turbine and filters the GMM results accordingly.
    The results are then sorted by "Year" and either "Quarter" or "Month", based on the available time data,
    and will raise a ValueError if neither column is found.

    Args:
        turbine_id (int): The unique identifier of the turbine for which the GMM results should be loaded.

    Returns:
        pd.DataFrame: A processed DataFrame containing the filtered GMM results for the turbine.

    Raises:
        ValueError: If neither "Quarter" nor "Month" columns are present in the GMM results.
    """
    gmm_path = os.path.join(EXPERIMENT_FOLDER, "GMM_results.xlsx")
    gmm_results = pd.read_excel(gmm_path, sheet_name=f"Turbine {turbine_id}")

    # Transform the data
    gmm_results = gmm_results.T
    gmm_results.columns = gmm_results.iloc[0]
    gmm_results = gmm_results.drop("Unnamed: 0")
    gmm_results = gmm_results.reset_index(drop=True)
    gmm_results["Year"] = gmm_results["Year"].astype(int)

    # Load best K values from filtered results
    filtered_path = os.path.join(DATA_FOLDER, "Selected_Ks.xlsx")
    best_k_df = pd.read_excel(filtered_path, sheet_name=f"Turbine {turbine_id}")

    # Create a list to store rows that match the best K values
    filtered_rows = []

    # Loop through each row in the best_k_df
    for _, row in best_k_df.iterrows():
        year = row["Year"]
        k = row["K"]

        # Check if we're dealing with quarterly or monthly data
        if "Quarter" in gmm_results.columns and "Quarter" in best_k_df.columns:
            q = row["Quarter"]
            mask = (
                (gmm_results["Year"] == year)
                & (gmm_results["Quarter"] == q)
                & (gmm_results["K"] == k)
            )
        elif "Month" in gmm_results.columns and "Month" in best_k_df.columns:
            m = row["Month"]
            mask = (
                (gmm_results["Year"] == year)
                & (gmm_results["Month"] == m)
                & (gmm_results["K"] == k)
            )
        else:
            raise ValueError(
                "Cannot match time periods between best_k_df and gmm_results"
            )

        matching_rows = gmm_results[mask]
        if not matching_rows.empty:
            filtered_rows.append(matching_rows)

    # Combine all filtered rows or return empty DataFrame if none found
    gmm_results = pd.concat(filtered_rows, ignore_index=True)

    # Check if "Quarter" exists in columns, otherwise use "Month"
    if "Quarter" in gmm_results.columns:
        gmm_results = gmm_results.sort_values(by=["Year", "Quarter"])
    elif "Month" in gmm_results.columns:
        gmm_results = gmm_results.sort_values(by=["Year", "Month"])
    else:
        raise ValueError("Neither 'Quarter' nor 'Month' columns found in gmm_results")

    return gmm_results


def _identify_stable_periods(
    gmm_results: pd.DataFrame,
    threshold: float = 0.78,
    std_cutoff: float = 0.06,
    window_size: int = 2,
) -> pd.DataFrame:
    """
    Identify stable periods in GMM results based on average predictive power and volatility.

    This function determines the stability of each period by checking if the
    'avgpps' value exceeds a given threshold and if the rolling standard deviation
    (calculated over a defined window) is below the specified cutoff. Periods meeting
    both conditions are flagged as stable. Additionally, it assigns a unique identifier
    to consecutive stable segments.

    Args:
        gmm_results (pd.DataFrame): DataFrame containing GMM results with an 'avgpps' column.
        threshold (float, optional): The cutoff for 'avgpps' above which a period is considered stable. Default is 0.78.
        std_cutoff (float, optional): The maximum allowable rolling standard deviation for a period to be deemed stable. Default is 0.06.
        window_size (int, optional): The number of periods included in the rolling calculation for standard deviation. Default is 2.

    Returns:
        pd.DataFrame: The updated DataFrame with the following new columns:
            - 'above_threshold': Boolean flag indicating if 'avgpps' is above the threshold.
            - 'rolling_std': The rolling standard deviation of 'avgpps'.
            - 'low_volatility': Boolean flag indicating if 'rolling_std' is below the cutoff (or NaN).
            - 'stable': Boolean flag set when both above_threshold and low_volatility conditions are true.
            - 'stable_run_id': An integer identifier for each consecutive group of stable periods.

    Notes:
        - Periods with insufficient data for the rolling calculation will have NaN for 'rolling_std' and are treated as low volatility.
    """
    gmm_results["above_threshold"] = gmm_results["avgpps"] > threshold

    gmm_results["rolling_std"] = (
        gmm_results["avgpps"].rolling(window=window_size, center=True).std()
    )

    gmm_results["low_volatility"] = (
        gmm_results["rolling_std"] < std_cutoff
    ) | gmm_results["rolling_std"].isna()

    gmm_results["stable"] = (
        gmm_results["above_threshold"] & gmm_results["low_volatility"]
    )

    gmm_results["stable_run_id"] = (
        gmm_results["stable"] != gmm_results["stable"].shift()
    ).cumsum()

    return gmm_results


def _filter_turbine_by_stability(
    data_loader: DataLoader, turbine_id: int, gmm_results: pd.DataFrame
):
    """
    Load and enrich turbine data with stability information derived from GMM results.

    This function uses the provided DataLoader instance to load the time series data for
    a given turbine. It then extracts the year from the turbine's timestamp and, based on
    the presence of either a 'Quarter' or 'Month' column in the GMM results, appends the
    corresponding time grouping to the turbine data. The stability flag from the GMM results
    is merged into the turbine data, after which the time grouping columns are removed and
    any missing stability values are filled with False (denoting instability). Finally, the
    stability column is renamed to 'is_stable' for clarity.

    Args:
        data_loader (DataLoader): An instance of DataLoader for accessing turbine data.
        turbine_id (int): The unique identifier of the turbine for which data is to be loaded.
        gmm_results (pd.DataFrame): DataFrame containing GMM results, including time grouping
            (either 'Quarter' or 'Month') and a 'stable' flag indicating stability.

    Returns:
        pd.DataFrame: A DataFrame with the original turbine data enriched with a binary
        'is_stable' column, which flags periods of stability based on the merged GMM results.

    Raises:
        ValueError: If neither 'Quarter' nor 'Month' columns exist in the provided gmm_results.
        Exception: Propagates exceptions encountered during turbine data loading, adding context
                   about the turbine_id.
    """
    try:
        # Load turbine data using the data loader
        turbine_data = data_loader.load_turbine_data(turbine_id)

        if turbine_data.empty:
            print(f"Warning: No data found for turbine {turbine_id}")
    except Exception as e:
        raise Exception(f"Error loading data for turbine {turbine_id}: {e}")
    turbine_data["Year"] = turbine_data["Time"].dt.year

    if "Quarter" in gmm_results.columns:
        # Extract the year, quarter, and stability information from gmm_results
        stability_info = gmm_results[["Year", "Quarter", "stable"]].copy()
        turbine_data["Quarter"] = turbine_data["Time"].dt.quarter
    elif "Month" in gmm_results.columns:
        # Extract the year, month, and stability information from gmm_results
        stability_info = gmm_results[["Year", "Month", "stable"]].copy()
        turbine_data["Month"] = turbine_data["Time"].dt.month
    else:
        raise ValueError("Neither 'Quarter' nor 'Month' columns found in gmm_results")

    if "Quarter" in turbine_data.columns:
        # Merge the stability information into the turbine data
        turbine_data = pd.merge(
            turbine_data, stability_info, on=["Year", "Quarter"], how="left"
        )
        # Drop the Year and Quarter columns since they are no longer needed
        turbine_data = turbine_data.drop(columns=["Year", "Quarter"])
    elif "Month" in turbine_data.columns:
        # Merge the stability information into the turbine data
        turbine_data = pd.merge(
            turbine_data, stability_info, on=["Year", "Month"], how="left"
        )
        # Drop the Year and Month columns since they are no longer needed
        turbine_data = turbine_data.drop(columns=["Year", "Month"])

    # Fill any NaN values in the 'stable' column with False (consider them unstable by default)
    turbine_data["stable"] = turbine_data["stable"].fillna(False)

    # Rename the column to make it more clear
    turbine_data = turbine_data.rename(columns={"stable": "is_stable"})

    return turbine_data


def _visualize_stability(
    gmm_results: pd.DataFrame,
    threshold: float,
    turbine_id: int,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize the stability of turbine performance over time using a time-indexed plot.

    This function builds a visualization where the x-axis represents time derived from the
    'Year' and 'Quarter' columns of the provided GMM results, and the y-axis represents the
    average predictive power score ('avgpps'). Each data point is color-coded to indicate
    its stability status (blue for stable, red for unstable), and consecutive points are connected
    with line segments colored based on the continuity of the stability status
    (blue when both points are stable, red when both are unstable, gray for mixed transitions).
    A horizontal dashed line is drawn at the specified threshold value to help contextualize
    the stability decision boundary. The plot is either displayed interactively or saved to a file
    if a save_path is provided.

    Args:
        gmm_results (pd.DataFrame): DataFrame containing GMM results, including the calculated
            stability flags and time components ('Year' and 'Quarter').
        threshold (float): The threshold value used for determining stability and for drawing
            the horizontal reference line.
        turbine_id (int): Identifier for the turbine, used in the plot title.
        save_path (Optional[str], optional): File path to save the visualization. If None,
            the plot is displayed interactively.

    Returns:
        None

    Raises:
        None
    """
    # Create a time index for better x-axis representation
    gmm_results["time_index"] = (
        gmm_results["Year"] + (gmm_results["Quarter"].astype(float) - 1) / 4
    )

    # Create figure
    plt.figure(figsize=(14, 7))

    # Add horizontal line for threshold
    plt.axhline(
        y=threshold,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Threshold ({threshold})",
    )

    # Plot the points
    plt.scatter(
        gmm_results["time_index"],
        gmm_results["avgpps"],
        c=gmm_results["stable"].map({True: "blue", False: "red"}),
        s=50,
        zorder=5,
    )

    # Plot the line segments with appropriate colors
    for i in range(len(gmm_results) - 1):
        current_point = gmm_results.iloc[i]
        next_point = gmm_results.iloc[i + 1]

        # X and Y coordinates for the current segment
        x_values = [current_point["time_index"], next_point["time_index"]]
        y_values = [current_point["avgpps"], next_point["avgpps"]]

        # Determine color based on stability
        if current_point["stable"] and next_point["stable"]:
            color = "blue"
        elif not current_point["stable"] and not next_point["stable"]:
            color = "red"
        else:
            color = "gray"  # Mixed segment

        plt.plot(x_values, y_values, color=color, linewidth=2, zorder=3)

    # Customize the plot
    plt.grid(True, linestyle="--", alpha=0.7)
    title = f"Turbine {turbine_id} - Predictive Power Score over Time with Stability Highlight"
    plt.title(title, fontsize=16)
    plt.ylabel("Average PPS", fontsize=14)
    plt.xlabel("Time (Year : Quarter)", fontsize=14)

    # Add custom legend
    legend_elements = [
        plt.Line2D(
            [0], [0], color="blue", marker="o", linestyle="-", label="Stable Periods"
        ),
        plt.Line2D(
            [0], [0], color="red", marker="o", linestyle="-", label="Unstable Periods"
        ),
        plt.Line2D(
            [0], [0], color="green", linestyle="--", label=f"Threshold ({threshold})"
        ),
        plt.Line2D([0], [0], color="gray", linestyle="-", label="Transition"),
    ]
    plt.legend(handles=legend_elements)

    # Customize x-axis ticks
    x_ticks = gmm_results["time_index"].values
    x_tick_labels = [
        f"{int(year)} : Q{int(float(quarter))}"
        for year, quarter in zip(gmm_results["Year"], gmm_results["Quarter"])
    ]
    plt.xticks(x_ticks, x_tick_labels, rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def analyze_turbine_stability(
    data_loader: DataLoader,
    park_id: int,
    turbine_ids: Optional[List[int]] = None,
    threshold: float = 0.78,
    std_cutoff: float = 0.06,
    window_size: int = 3,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Analyze and update the stability status of turbines within a wind park.

    This function performs stability analysis across all turbines in a given wind park.
    It retrieves the park data and extracts the turbine IDs, then iterates over each turbine:
      - Loads the corresponding GMM results from an Excel file.
      - Identifies stable periods by comparing the 'avgpps' values against a threshold and
        evaluating rolling standard deviation against a cutoff.
      - Merges the derived stability information into the raw turbine data.
      - Optionally visualizes the stability analysis by generating and saving figures.
    Finally, the updated park data is saved using the provided DataLoader.

    Args:
        data_loader (DataLoader): Instance used to load park and turbine data.
        park_id (int): Identifier of the wind park whose turbines are analyzed.
        threshold (float, optional): Cutoff for 'avgpps' above which periods are considered stable. Default is 0.78.
        std_cutoff (float, optional): Maximum allowable rolling standard deviation for declaring stability. Default is 0.06.
        window_size (int, optional): Number of periods to include in the rolling standard deviation calculation. Default is 3.
        save_path (Optional[str], optional): Directory where stability visualization figures are saved.
            If None, visualizations are not generated.

    Returns:
        Dict: Dictionary containing the outcome of the stability analysis, including the processed
        park data or error details if the analysis fails for any turbine.

    Raises:
        Exception: Propagates exceptions related to loading or processing turbine data.
    """
    try:
        park_df = data_loader.load_park_data(park_id)
    except FileNotFoundError:
        return

    if turbine_ids is None:
        turbine_ids = park_df["TurbineId"].unique()

    updated_park_df = None

    for turbine_id in turbine_ids:
        if park_id != data_loader.fetch_park_number(turbine_id):
            continue
        try:
            gmm_results = _load_gmm_results(turbine_id)

            # Identify stable periods
            gmm_results = _identify_stable_periods(
                gmm_results,
                threshold=threshold,
                std_cutoff=std_cutoff,
                window_size=window_size,
            )
        except Exception as e:
            print(f"Error analyzing GMM results for turbine {turbine_id}: {e}")
            return {"turbine_id": turbine_id, "error": str(e)}

        updated_turbine_df = _filter_turbine_by_stability(
            data_loader, turbine_id, gmm_results
        )

        if updated_park_df is None:
            updated_park_df = updated_turbine_df
        else:
            updated_park_df = pd.concat(
                [updated_park_df, updated_turbine_df], ignore_index=True
            )

        # Visualize stability
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig_path = os.path.join(save_path, f"turbine_{turbine_id}.png")
            _visualize_stability(gmm_results, threshold, turbine_id, fig_path)

    data_loader.save_park_data(park_id, updated_park_df)


def _example_run():
    dest_path = os.path.abspath(
        os.path.join(PROJECT_ROOT, os.pardir, "data", "nbm_selector_data")
    )

    dl = DataLoader(DATA_FOLDER, destination_path=dest_path)

    save_path = os.path.join(
        PROJECT_ROOT,
        os.pardir,
        "figures",
        datetime.now().strftime("%Y%m%d"),
        "nbm_year",
    )

    for park_id in dl.fetch_park_list():
        analyze_turbine_stability(
            data_loader=dl,
            park_id=park_id,
            threshold=0.78,
            std_cutoff=0.08,
            window_size=3,
            # save_path=save_path,
        )


class StabilitySelector:
    def __init__(
        self,
        source_path: str = "data/k_filtered_data",
        destination_path: str = "data/nbm_selector_data",
        excel_destination_path: str = "data/nbm_selector_data/stability.xlsx",
    ):

        self.source_path = source_path
        self.desination_path = destination_path
        self.excel_destination_path = excel_destination_path
        self.dl = DataLoader(
            source_path,
            destination_path=destination_path,
        )

    def run_stability_selector(
        self,
        stability_trheshold: float = 0.8,
        std_cutoff: float = 0.08,
        window_size: int = 3,
    ):
        """
        Run the stability selector for all parks in the source path.

        This method iterates through all parks in the source path and identifies stable periods
        for each turbine. The results are saved in the specified Excel destination path.

        The stable period is defined as 2 years in a row after 2018, with no flags in the
        'is_stable' column. The period cannot start later than 2020 and must be at least
        2 years long.
        The results are saved in an Excel file with the name 'stability_results.xlsx'.

        The results in the excel file include the following columns:
        - 'TurbineId': The ID of the turbine.
        - 'T_year': The year specified for training.
        - 'R_1': The year for the following the training period.
        - 'R_2': A list of following years after R_1.
        """
        # Create the destination directory if it doesn't exist
        if not os.path.exists(self.desination_path):
            print(f"Creating directory: {self.desination_path}")
            os.makedirs(self.desination_path, exist_ok=True)

        # If the excel file already exists, remove it
        if os.path.exists(self.excel_destination_path):
            os.remove(self.excel_destination_path)

        # Get the list of parks
        park_list = self.dl.fetch_park_list()

        # Iterate through each park and analyze stability
        for park_id in park_list:
            analyze_turbine_stability(
                data_loader=self.dl,
                park_id=park_id,
                threshold=stability_trheshold,
                std_cutoff=std_cutoff,
                window_size=window_size,
                # save_path=destination_path,
            )

        # update the source path for the dl
        if self.dl.source_path != self.desination_path:
            self.dl.source_path = self.desination_path

        # Select the stable periods
        stable_df = pd.DataFrame()
        for park_id in park_list:
            try:
                park_df = self.dl.load_park_data(park_id)
                turbine_list = park_df["TurbineId"].unique()
                turbine_list.sort()
                for turbine_id in turbine_list:
                    turbine_df = park_df[park_df["TurbineId"] == turbine_id]
                    result_df = self._select_stable_periods(turbine_id, turbine_df)
                    if result_df is not None:
                        stable_df = pd.concat([stable_df, result_df], ignore_index=True)
            except Exception as e:
                print(f"Error loading data for park {park_id}: {e}")
                continue
        # Save the results to an Excel file
        stable_df.to_excel(
            os.path.join(self.excel_destination_path),
            index=False,
            sheet_name="Stable Periods",
        )
        print(f"Stable periods saved to {os.path.join(self.excel_destination_path)}")

    def _select_stable_periods(
        self, turbine_id: int, turbine_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Select stable periods from the park DataFrame.

        This method filters the park DataFrame to identify stable periods based on the
        'is_stable' column. It returns a DataFrame with the following columns:
        - 'TurbineId': The ID of the turbine.
        - 'T_year': The year specified for training.
        - 'R_1': The year for the following the training period.
        - 'R_2': A list of following years after R_1.

        The T_year is the year of the first stable period, and R_1 is the year
        immediately following T_year, that is also stable. R_2 is a list of years
        that are following R_1 and are also stable. A stable period is defined as
        a year with no flags in the 'is_stable' column (I.e. no quarter is missing).
        The period cannot start later than 2020 and must be at least 2 years long.
        """

        # Filter the DataFrame to include only stable periods
        stable_df = turbine_df[turbine_df["is_stable"] == True]

        # Create a new DataFrame to store the results
        results = []

        # Iterate through each turbine in the stable DataFrame
        stable_df["Year"] = stable_df["Time"].dt.year
        stable_df["Quarter"] = stable_df["Time"].dt.quarter
        years = stable_df["Year"].unique()

        # Check if there are at least 3 years of data
        if len(years) < 3:
            return None

        # Iterate through the years and find stable periods
        # The T_year is the earliest possible year for any stable period
        stable_years = []
        for year in years:
            if year >= 2018:
                # Check if all quarters are present for the year
                quarters = stable_df[stable_df["Year"] == year]["Quarter"].unique()
                if len(quarters) == 4:
                    stable_years.append(year)

        # Check if there are at least 3 stable years
        if len(stable_years) < 3:
            return None

        stable_years.sort()

        # Check if the first stable year is before 2020
        if stable_years[0] > 2020:
            return None
        # Check if the first stable year is before 2018
        if stable_years[0] < 2018:
            return None

        # Check if R_1 is the year immediately following T_year
        if stable_years[1] != stable_years[0] + 1:
            return None

        # Check if R_2 is a list of years that are following R_1 and are also stable
        for i in range(2, len(stable_years)):
            if stable_years[i] != stable_years[i - 1] + 1:
                return None

        # T_year is the first element in the stable_years list
        T_year = stable_years[0]
        # R_1 is the year immediately following T_year
        R_1 = stable_years[1]
        # R_2 is a list of years that are following R_1 and are also stable
        R_2 = stable_years[2:]
        # Append the results to the list
        results.append(
            {
                "TurbineId": turbine_id,
                "T_year": T_year,
                "R_1": R_1,
                "R_2": R_2,
            }
        )
        # Create a DataFrame from the results list
        results_df = pd.DataFrame(results)
        return results_df


if __name__ == "__main__":
    dest_path = os.path.abspath(
        os.path.join(PROJECT_ROOT, os.pardir, "data", "nbm_selector_data")
    )
    excel_dest_path = os.path.abspath(
        os.path.join(
            PROJECT_ROOT, os.pardir, "data", "nbm_selector_data", "stability.xlsx"
        )
    )
    dl = DataLoader()
    park_list = dl.fetch_park_list()

    # Run the stability selector for all parks
    StabilitySelector(
        source_path=DATA_FOLDER,
        destination_path=dest_path,
        excel_destination_path=excel_dest_path,
    ).run_stability_selector(
        stability_trheshold=0.8,
        std_cutoff=0.03,
        window_size=3,
    )
