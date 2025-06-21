import numpy as np
import pandas as pd
import os
from typing import Literal, Dict, List, Set, Tuple, Optional, Union

import sys

sys.path.insert(0, "src/")
from data_utility.make_dataset import DataWrangler


class VotingSystem:
    def __init__(
        self,
        data_loader,
        threshold: float = 0.05,
        elected_handler: Literal["average", "remove"] = "remove",
        modify_by_period: bool = False,
        granularity: Literal["Quarter", "Month"] = "Quarter",
    ):
        """
        Initialize the voting system for anomaly detection in turbine data.

        The voting system uses distance-based strategies to identify anomalous dimensions
        in groups of related features (e.g., BladeLoad or PitchAngle measurements from different sensors).

        Parameters
        ----------
        threshold : float, optional
            The threshold for the distance-based strategy to identify anomalies.
            Higher values are more tolerant of differences between sensors, default is 0.05.
        elected_handler : Literal["average", "remove"], optional
            How to handle detected anomalous dimensions:
            - "average": Replace anomalous values with average of good values
            - "remove": Set anomalous values to NaN, default is "remove"
        modify_by_period : bool, optional
            If True, modifications are applied only to specific time periods.
            If False, modifications are applied across all time periods for affected dimensions.
            Default is False.
        """
        self.dl = data_loader  # Initialize the DataLoader instance
        self.threshold = threshold  # Set the threshold for the anomaly strategies
        self.elected_handler = elected_handler  # Set the elected handler method
        self.modify_by_period = (
            modify_by_period  # Flag to modify by period else dimnesions
        )
        self.granularity = granularity  # Set the granularity (Quarter or Month)

        # Homogeneous groups of dimensions to compare
        self.groups = {
            "BladeLoad": ["BladeLoadA", "BladeLoadB", "BladeLoadC"],
            "PitchAngle": ["PitchAngleA", "PitchAngleB", "PitchAngleC"],
        }
        # Create mapping for corresponding dimensions across groups
        self.dimension_correspondence = {
            "BladeLoadA": "PitchAngleA",
            "BladeLoadB": "PitchAngleB",
            "BladeLoadC": "PitchAngleC",
            "PitchAngleA": "BladeLoadA",
            "PitchAngleB": "BladeLoadB",
            "PitchAngleC": "BladeLoadC",
        }
        # Create DataFrame to track removed turbines
        self.removed_turbines = pd.DataFrame(
            columns=["turbine_id", "period", "dimension_group", "reason"]
        )
        self.removed_dimensions = (
            {}
        )  # Track removed dimensions for each turbine and period
        self.df = None  # Store the original dataframe

    def distance_based_strategy(self) -> Dict[str, Dict[str, int]]:
        """
        Implements the distance-based voting strategy using PPS (Performance Persistence Score) values.

        This method compares the PPS scores between pairs of features within the same group.
        If the distance between scores exceeds the threshold, both features vote for each other
        as potential anomalies.

        The voting is performed for each time period in the data.

        Returns
        -------
        Dict[str, Dict[str, int]]
            A nested dictionary where:
            - Outer keys are time periods in the format "YYYY-Q" (Year-Quarter) or "YYYY-M" (Year-Month)
            - Inner keys are feature names
            - Values are the number of votes each feature received in that period
        """
        votes_by_period = {}

        # Process each time period (row) in the dataframe
        for time_idx in range(len(self.df)):
            if "Quarter" in self.df.columns:
                period = f"{self.df['Year'].iloc[time_idx]}-Q{self.df['Quarter'].iloc[time_idx]}"
            elif "Month" in self.df.columns:
                period = f"{self.df['Year'].iloc[time_idx]}-M{self.df['Month'].iloc[time_idx]}"
            votes_by_period[period] = {}

            # Initialize votes for each feature in this time period
            features = [feature for group in self.groups.values() for feature in group]
            votes = {feature: 0 for feature in features}

            for group_name, group_features in self.groups.items():
                for i, feature1 in enumerate(group_features):
                    for j, feature2 in enumerate(group_features):
                        if i < j:  # Only compare each pair once
                            # Get PPS scores for this time period
                            pps_col1 = f"pps_{feature1}"
                            pps_col2 = f"pps_{feature2}"

                            # Skip if either column doesn't exist or has NaN value
                            if (
                                pps_col1 not in self.df.columns
                                or pps_col2 not in self.df.columns
                                or pd.isna(self.df[pps_col1].iloc[time_idx])
                                or pd.isna(self.df[pps_col2].iloc[time_idx])
                            ):
                                continue

                            # Compute the distance between PPS scores
                            distance = abs(
                                self.df[pps_col1].iloc[time_idx]
                                - self.df[pps_col2].iloc[time_idx]
                            )

                            # If the distance exceeds the threshold, both features vote for each other
                            if distance > self.threshold:
                                votes[feature1] += 1
                                votes[feature2] += 1

            votes_by_period[period] = votes

        return votes_by_period

    def _has_complete_disagreement(
        self, problematic_feature: str, turbine_id: int, period: str
    ) -> bool:
        feature_group = None
        for group_name, group_features in self.groups.items():
            if problematic_feature in group_features:
                feature_group = group_name
                break

        if feature_group:
            # Add this dimension to the removed list
            if (
                problematic_feature
                not in self.removed_dimensions[turbine_id][feature_group]
            ):
                self.removed_dimensions[turbine_id][feature_group].append(
                    problematic_feature
                )

            # Check if all dimensions in this group would be removed
            group_dimensions = self.groups[feature_group]
            removed_in_group = self.removed_dimensions[turbine_id][feature_group]

            if len(removed_in_group) == len(group_dimensions):
                # All dimensions in the group would be removed, so remove the entire turbine
                self.removed_turbines = pd.concat(
                    [
                        self.removed_turbines,
                        pd.DataFrame(
                            [
                                {
                                    "turbine_id": turbine_id,
                                    "period": period,
                                    "dimension_group": feature_group,
                                    "threshold": self.threshold,
                                    "reason": "Partial disagreement",
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
                if self.modify_by_period:
                    return False
                else:
                    return True
            else:
                return False

    def determine_elected(
        self,
        votes_by_period: Dict[str, Dict[str, int]],
        turbine_id: int,
        strict_mode: bool = False,
        strict_mode_threshold: float = 2.0,
    ) -> Optional[Dict[str, List[str]]]:
        """
        Determine which features are identified as anomalous based on voting results.

        This method applies various rules to interpret the voting results and identify
        which features (dimensions) are likely anomalous. It tracks removed dimensions
        and can identify entire turbines for removal in certain scenarios.

        Parameters
        ----------
        votes_by_period : Dict[str, Dict[str, int]]
            A nested dictionary with time periods as keys and vote counts as values,
            as returned by distance_based_strategy()
        turbine_id : int
            Identifier of the turbine being processed
        strict_mode : bool, optional
            If True, enables additional detection rules for features with fewer votes,
            default is False
        strict_mode_threshold : float, optional
            Ratio threshold used in strict mode to identify significant differences,
            default is 2.0

        Returns
        -------
        Optional[Dict[str, List[str]]]
            A dictionary mapping time periods to lists of elected (anomalous) features.
            Returns None if the entire turbine should be removed.
        """
        elected_by_period = {}
        # Initialize tracking for this turbine and period if not exists
        if turbine_id not in self.removed_dimensions:
            self.removed_dimensions[turbine_id] = {
                group: [] for group in self.groups.keys()
            }

        for period, votes in votes_by_period.items():
            # Count how many features have each number of votes
            vote_counts = {}
            for feature, vote_count in votes.items():
                if vote_count not in vote_counts:
                    vote_counts[vote_count] = []
                vote_counts[vote_count].append(feature)

            # Apply the rules
            elected = []
            if 2 in vote_counts and len(vote_counts[2]) == 3:
                # Determine which dimension group has all three features with 2 votes
                for group_name, group_features in self.groups.items():
                    if all(feature in vote_counts[2] for feature in group_features):
                        self.removed_turbines = pd.concat(
                            [
                                self.removed_turbines,
                                pd.DataFrame(
                                    [
                                        {
                                            "turbine_id": turbine_id,
                                            "period": period,
                                            "dimension_group": group_name,
                                            "threshold": self.threshold,
                                            "reason": "Total disagreement",
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )
                        if self.modify_by_period:
                            # elected.extend(vote_counts[2])
                            elected.extend(self.groups["BladeLoad"])
                        else:
                            return None
            elif 2 in vote_counts and len(vote_counts[2]) == 1:
                if self.elected_handler == "average":
                    # If only one feature received 2 votes, elect it and replace the other two with the average
                    # elected.extend(vote_counts[2])
                    elected.extend(self.groups["BladeLoad"])
                elif self.elected_handler == "remove":
                    problematic_feature = vote_counts[2][0]
                    if not self._has_complete_disagreement(
                        problematic_feature, turbine_id, period
                    ):
                        # elected.append(problematic_feature)
                        elected.extend(self.groups["BladeLoad"])
                    elif self.modify_by_period:
                        elected.extend(self.groups["BladeLoad"])
                    else:
                        return None

                else:
                    raise ValueError(
                        "Invalid elected_handler value; should be 'average' or 'remove'"
                    )
            elif strict_mode and 1 in vote_counts and len(vote_counts[1]) == 2:
                # Check if both dimensions with 1 vote are from the same group
                same_group = False
                group_name = None
                for grp_name, grp_features in self.groups.items():
                    if all(feature in grp_features for feature in vote_counts[1]):
                        same_group = True
                        group_name = grp_name
                        break

                if same_group:
                    # Find the third dimension in the group
                    third_dimension = [
                        f for f in self.groups[group_name] if f not in vote_counts[1]
                    ][0]

                    # Calculate distances between the dimensions with 1 vote and the third dimension
                    distances = (
                        {}
                    )  # Changed to dictionary to track which feature has which distance
                    for feature in vote_counts[1]:
                        pps_col1 = f"pps_{feature}"
                        pps_col2 = f"pps_{third_dimension}"

                        # Skip if columns don't exist or have NaN values for this period
                        if "Quarter" in self.df.columns:
                            period_parts = period.split("-Q")
                            year = int(period_parts[0])
                            quarter = int(float(period_parts[1]))
                            row_mask = (self.df["Year"] == year) & (
                                self.df["Quarter"] == quarter
                            )
                        elif "Month" in self.df.columns:
                            period_parts = period.split("-M")
                            year = int(period_parts[0])
                            month = int(float(period_parts[1]))
                            row_mask = (self.df["Year"] == year) & (
                                self.df["Month"] == month
                            )

                        if (
                            pps_col1 in self.df.columns
                            and pps_col2 in self.df.columns
                            and not pd.isna(self.df.loc[row_mask, pps_col1].values[0])
                            and not pd.isna(self.df.loc[row_mask, pps_col2].values[0])
                        ):

                            distance = abs(
                                self.df.loc[row_mask, pps_col1].values[0]
                                - self.df.loc[row_mask, pps_col2].values[0]
                            )
                            distances[feature] = distance

                    if len(distances) == 2:  # Only proceed if we have both distances
                        # Calculate distance between the two dimensions with 1 vote
                        feature1, feature2 = vote_counts[1]

                        if (
                            distances[feature1] > 0 and distances[feature2] > 0
                        ):  # Avoid division by zero
                            ratio = distances[feature1] / distances[feature2]

                            if ratio >= strict_mode_threshold:
                                # feature1 is significantly more distant from third_dimension
                                if not self._has_complete_disagreement(
                                    feature1, turbine_id, period
                                ):
                                    # elected.append(feature1)
                                    elected.extend(self.groups["BladeLoad"])
                                elif self.modify_by_period:
                                    elected.extend(self.groups["BladeLoad"])
                                else:
                                    return None
                            elif 1 / ratio >= strict_mode_threshold:
                                # feature2 is significantly more distant from third_dimension
                                if not self._has_complete_disagreement(
                                    feature2, turbine_id, period
                                ):
                                    # elected.append(feature2)
                                    elected.extend(self.groups["BladeLoad"])
                                elif self.modify_by_period:
                                    elected.extend(self.groups["BladeLoad"])
                                else:
                                    return None
                            else:
                                # No significant difference
                                continue

            elected_by_period[period] = elected

        return elected_by_period

    def handle_elected(self, elected_by_period: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Handle the elected (anomalous) features by modifying the dataframe.

        This method implements the strategy specified by elected_handler:
        - For "average": Replaces anomalous values with the average of non-anomalous values
        - For "remove": Sets anomalous values to NaN

        The modifications are applied either by period or across all periods based on
        the modify_by_period setting.

        Parameters
        ----------
        elected_by_period : Dict[str, List[str]]
            A dictionary mapping time periods to lists of elected (anomalous) features

        Returns
        -------
        pd.DataFrame
            A modified copy of the original dataframe with anomalous features handled
            according to the specified strategy
        """
        # Make a copy of the dataframe to avoid modifying the original
        modified_df = self.df.copy()

        # If there are no elected features across any periods, return the unmodified dataframe
        if not any(elected_by_period.values()):
            return modified_df

        if self.modify_by_period:
            # Handle each period independently
            for period, elected in elected_by_period.items():
                if not elected:
                    continue

                if "Quarter" in modified_df.columns:
                    period_parts = period.split("-Q")
                    year = int(period_parts[0])
                    quarter = int(float(period_parts[1]))

                    # Find the row for this period
                    period_mask = (modified_df["Year"] == year) & (
                        modified_df["Quarter"] == quarter
                    )
                elif "Month" in modified_df.columns:
                    period_parts = period.split("-M")
                    year = int(period_parts[0])
                    month = int(float(period_parts[1]))

                    # Find the row for this period
                    period_mask = (modified_df["Year"] == year) & (
                        modified_df["Month"] == month
                    )

                if self.elected_handler == "average":
                    # Process each group separately for averaging
                    for group_name, features in self.groups.items():
                        elected_in_group = [
                            feature for feature in elected if feature in features
                        ]
                        if elected_in_group:
                            non_elected = [
                                feature
                                for feature in features
                                if feature not in elected_in_group
                            ]

                            # Skip if there are no non-elected features or their columns don't exist
                            non_elected_cols = [
                                f"pps_{feature}" for feature in non_elected
                            ]
                            if not non_elected or not all(
                                col in modified_df.columns for col in non_elected_cols
                            ):
                                continue

                            # Calculate average of non-elected features for this period
                            avg_value = modified_df.loc[
                                period_mask, non_elected_cols
                            ].mean(axis=1)

                            # Replace values for elected features with the average
                            for feature in elected_in_group:
                                col = f"pps_{feature}"
                                if col in modified_df.columns:
                                    modified_df.loc[period_mask, col] = avg_value

                elif self.elected_handler == "remove":
                    # Set values to NaN for elected features only in this period
                    for feature in elected:
                        col = f"pps_{feature}"
                        if col in modified_df.columns:
                            modified_df.loc[period_mask, col] = float("nan")

        else:
            # Original implementation: Handle elected dimensions across all periods
            # Collect all unique elected features across all periods
            all_elected_features = set()
            for elected in elected_by_period.values():
                all_elected_features.update(elected)

            # Identify all affected columns based on the elected features
            affected_columns = []
            for feature in all_elected_features:
                affected_columns.append(f"pps_{feature}")

                # Add corresponding feature column
                corresponding_feature = self.dimension_correspondence.get(feature)
                if corresponding_feature:
                    affected_columns.append(f"pps_{corresponding_feature}")

            if self.elected_handler == "average":
                # Handle each period separately for averaging
                for period, elected in elected_by_period.items():
                    if not elected:
                        continue

                    if "Quarter" in modified_df.columns:
                        period_parts = period.split("-Q")
                        year = int(period_parts[0])
                        quarter = int(float(period_parts[1]))

                        # Find the row for this period
                        period_mask = (modified_df["Year"] == year) & (
                            modified_df["Quarter"] == quarter
                        )
                    elif "Month" in modified_df.columns:
                        period_parts = period.split("-M")
                        year = int(period_parts[0])
                        month = int(float(period_parts[1]))

                        # Find the row for this period
                        period_mask = (modified_df["Year"] == year) & (
                            modified_df["Month"] == month
                        )

                    # Process each group separately for averaging
                    for group_name, features in self.groups.items():
                        elected_in_group = [
                            feature
                            for feature in all_elected_features
                            if feature in features
                        ]
                        if elected_in_group:
                            non_elected = [
                                feature
                                for feature in features
                                if feature not in elected_in_group
                            ]

                            # Skip if there are no non-elected features or their columns don't exist
                            non_elected_cols = [
                                f"pps_{feature}" for feature in non_elected
                            ]
                            if not non_elected or not all(
                                col in modified_df.columns for col in non_elected_cols
                            ):
                                continue

                            # Calculate average of non-elected features for this period
                            avg_value = modified_df.loc[
                                period_mask, non_elected_cols
                            ].mean(axis=1)

                            # Replace values for elected features with the average
                            for feature in elected_in_group:
                                col = f"pps_{feature}"
                                if col in modified_df.columns:
                                    modified_df.loc[period_mask, col] = avg_value

            elif self.elected_handler == "remove":
                # For removal, we set the values to NaN across all periods
                for col in affected_columns:
                    if col in modified_df.columns:
                        modified_df[col] = float("nan")

        # Recalculate sumpps and avgpps columns
        # Get all pps_ columns
        pps_columns = [col for col in modified_df.columns if col.startswith("pps_")]

        if pps_columns:
            # Recalculate sumpps (sum of all valid PPS values)
            if "sumpps" in modified_df.columns:
                modified_df["sumpps"] = modified_df[pps_columns].sum(
                    axis=1, skipna=True
                )

            # Recalculate avgpps (average of all valid PPS values)
            if "avgpps" in modified_df.columns:
                # Count non-NaN values for each row
                valid_counts = modified_df[pps_columns].count(axis=1)
                # Sum of valid PPS values
                pps_sums = modified_df[pps_columns].sum(axis=1, skipna=True)
                # Calculate average (avoiding division by zero)
                modified_df["avgpps"] = pps_sums / valid_counts.replace(0, np.nan)

        return modified_df

    def remove_turbine_data(
        self, park_id: int, modified_pps_dfs: pd.DataFrame, drop_periods: bool = True
    ) -> None:
        turbine_df_list = []

        df = self.dl.load_park_data(park_id)

        for turbine_id in df["TurbineId"].unique():
            if park_id != self.dl.fetch_park_number(turbine_id):
                continue
            turbine_df = df[df["TurbineId"] == turbine_id].copy()

            # Load the GMM results for the turbine
            modified_pps_df = modified_pps_dfs[turbine_id]
            turbine_data = turbine_df.copy()
            turbine_data["Year"] = turbine_data["Time"].dt.year
            if self.granularity == "Quarter":
                turbine_data["Quarter"] = turbine_data["Time"].dt.quarter
            elif self.granularity == "Month":
                turbine_data["Month"] = turbine_data["Time"].dt.month
            else:
                raise ValueError(
                    "Neither 'Quarter' nor 'Month' columns found in gmm_results"
                )

            pps_cols = [
                col for col in modified_pps_df.columns if col.startswith("pps_")
            ]
            if self.granularity in modified_pps_df.columns:
                pps_info = modified_pps_df[["Year", self.granularity] + pps_cols].copy()

            if self.granularity in turbine_data.columns:
                # Merge the stability information into the turbine data
                turbine_data = pd.merge(
                    turbine_data, pps_info, on=["Year", self.granularity], how="left"
                )

                # Handle NaN values in pps_ columns
                for pps_col in pps_cols:
                    # Extract the corresponding column name without 'pps_' prefix
                    base_col = pps_col[4:]  # Remove 'pps_' prefix
                    if base_col in turbine_data.columns:
                        if drop_periods:
                            turbine_data = turbine_data.drop(
                                turbine_data[turbine_data[pps_col].isna()].index
                            )
                        else:
                            turbine_data.loc[turbine_data[pps_col].isna(), base_col] = (
                                np.nan
                            )

            # Drop temporary columns
            turbine_data = turbine_data.drop(
                columns=["Year", self.granularity] + pps_cols
            )
            turbine_df_list.append(turbine_data)

        # Concatenate the filtered dataframes
        filtered_df = pd.concat(turbine_df_list, ignore_index=True)

        dw = DataWrangler(destination_path="data/voting_system_data")
        park_name = self.dl.create_file_name(park_id)
        dw.save_dataframe(filtered_df, file_name=park_name, print_states=False)

    def vote(
        self,
        turbine_ids: List[int],
        strict_mode: bool = False,
        strict_mode_threshold: float = 2.0,
    ):
        """
        Run the complete voting process for identifying and handling anomalies in turbine data.

        This method:
        1. Loads data for each turbine ID
        2. Runs the distance-based voting strategy
        3. Determines which features are anomalous
        4. Applies the specified strategy to handle anomalies
        5. Saves the modified data and removed turbine information to files

        Parameters
        ----------
        turbine_ids : List[int]
            List of turbine IDs to process
        strict_mode : bool, optional
            Whether to use strict mode for anomaly detection, default is False
        strict_mode_threshold : float, optional
            The threshold ratio for strict mode, default is 2.0

        Returns
        -------
        None
            Results are saved to Excel files in the specified directories
        """
        modified_dfs = {}

        for turbine_id in turbine_ids:
            try:
                gmm_results = pd.read_excel(
                    os.path.join(os.getcwd(), "src", "experiments", "pps.xlsx"),
                    sheet_name=f"Turbine {turbine_id}",
                )
            except:
                print(f"WARNING: Failed to load data for turbine {turbine_id}")
                continue
            # Reshaping the dataframe
            gmm_results = gmm_results.T
            gmm_results.columns = gmm_results.iloc[0]
            gmm_results = gmm_results.drop("Unnamed: 0")
            gmm_results = gmm_results.reset_index(drop=True)
            gmm_results["Year"] = gmm_results["Year"].astype(int)
            gmm_results = gmm_results[gmm_results["K"] == 0]

            self.df = gmm_results.copy()

            votes_by_period = self.distance_based_strategy()
            elected_by_period = self.determine_elected(
                votes_by_period,
                turbine_id,
                strict_mode,
                strict_mode_threshold,
            )
            if elected_by_period:
                modified_df = self.handle_elected(elected_by_period)
                if modified_df is not None:
                    modified_dfs[turbine_id] = modified_df

        # Save the removed turbines dataframe to an Excel file
        if not self.modify_by_period:
            output_path = os.path.join(os.getcwd(), "data", "removed_turbines.xlsx")
            self.removed_turbines.to_excel(output_path, index=False)

        for park_id in self.dl.fetch_park_list():
            self.remove_turbine_data(park_id=park_id, modified_pps_dfs=modified_dfs)


if __name__ == "__main__":
    vs = VotingSystem(threshold=0.03, elected_handler="remove", modify_by_period=True)
    vs.vote(list(range(1, 118)), strict_mode=True, strict_mode_threshold=2.0)
