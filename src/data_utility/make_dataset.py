import os
import pandas as pd
import numpy as np
import sys
from typing import Literal
from datetime import datetime
from scipy.io import loadmat
from sklearn.preprocessing import minmax_scale
from scipy.stats import chi2

sys.path.insert(0, "src/")

# Turn of chains warnings from pandas
pd.options.mode.chained_assignment = None  # default='warn'
DTYPE_DICT = {
    "TurbineId": "int",
    "AmbTemp": "float64",
    "BladeLoadA": "float64",
    "BladeLoadB": "float64",
    "BladeLoadC": "float64",
    "GridPower": "float64",
    "PitchAngleA": "float64",
    "PitchAngleB": "float64",
    "PitchAngleC": "float64",
    "Time": "datetime64[ns]",
    "WSE": "float64",
    "WdAbs": "float64",
    "WindDirRel": "float64",
    "WindSpeed": "float64",
}


class MatLabConverter:
    """
    Recursively takes Matlab files, which should have the format:
        AAU_park#_year,
    extends the files to a park level dataframe and saves it as csv or pickle format.

    Attributes
    ----------
        source_path (os.path): path to source data, matlab files
        intermediate_path (str): path to intermediate_data folder
        file_format (str): save as either "csv" or "pickle"
        turbine_mapping_dict (dict): maps the turbines to parks
        file_df (pd.Dataframe): file indexer
        current_turbine_index_addition (int): adds id offset to ensure unique id's across parks
        id_mapping (dict): maps the id's across parks

    """

    def __init__(
        self, source_path: os.path, intermediate_path: str, file_format: str = "pickle"
    ):
        """
        Args:
            source_path (os.path): path to source data, matlab files
            intermediate_path (str): path to intermediate_data folder
            file_format (str): save as either "csv" or "pickle"
        """
        self.source_path = source_path
        self.intermediate_path = intermediate_path
        self.file_format = file_format
        self.turbine_mapping_dict = {}
        self.file_df = None
        self.current_turbine_index_addition = 1
        self.id_mapping = {}

    def file_indexer(self):
        """
        Indexes the files in the source path. Columns are:
            - Park
            - File_name
        Returns: pd.DataFrame
        """
        print("Indexing files.....")
        park_list = []
        file_list = []
        for root, dirs, files in os.walk(self.source_path):
            for file in files:
                if file.endswith(".mat"):
                    park_number = int(file.split("_Park")[1].split("_")[0])
                    park_list.append(park_number)
                    file_list.append(file)

        df_index = pd.DataFrame({"Park": park_list, "File_name": file_list})
        df_index = df_index.sort_values(by="Park")
        df_index.reset_index(drop=True, inplace=True)
        print("Files indexed")
        self.file_df = df_index

    def mat_file_converter(self, file: str) -> pd.DataFrame:
        """Converts matlab files to dataframes

        Args:
            file (str): the input (matlab)file

        Returns:
            pd.DataFrame: converted file
        """
        keys_to_drop = ["__header__", "__version__", "__globals__"]
        offset = datetime(1970, 1, 1).toordinal() + 366  # Time offset

        if file.endswith(".mat"):
            matlab_data = loadmat(os.path.join(self.source_path, file))

            # dropping metadata
            for key in keys_to_drop:
                del matlab_data[key]

            # unpacking keys
            for key in matlab_data.keys():
                matlab_data[key] = matlab_data[key].flatten()

            # set current time
            df_park = pd.DataFrame.from_dict(matlab_data)
            df_park["Time"] = pd.to_datetime(
                df_park["Time"].to_numpy() - offset, unit="D"
            ).round("s")

            # Setting the data types
            df_park = df_park.astype(DTYPE_DICT)

            return df_park

    def recursive_data_converter(self):
        """
        Recursively caller for converting the matlab files.

        Returns:
            nothing. Saves the dataframes at provided location (self.intermediate_path)
        """

        # start time
        start_time = datetime.now()

        # Indexing files
        self.file_indexer()

        # Converting data
        print("Converting data.....")
        unique_parks = sorted(self.file_df["Park"].unique())
        for park in unique_parks:
            print("------------------------")
            print(f"Loading files for park: {park}")
            park_df = pd.DataFrame()
            park_df_list = []
            park_files = self.file_df[self.file_df["Park"] == park]["File_name"]
            for file in park_files:
                df_park_yearly = self.mat_file_converter(file)
                park_df = pd.concat([park_df, df_park_yearly])
                park_df_list.append(df_park_yearly)

            park_df = pd.concat(park_df_list)

            # TurbineId offset based on park number
            if park not in self.turbine_mapping_dict.keys():
                print(f"Adding park {park} to the mapping....")
                self.turbine_mapping_dict[park] = {self.current_turbine_index_addition}

                self.current_turbine_index_addition += len(
                    park_df["TurbineId"].unique()
                )

            # Adding TurbineId offset
            park_df["TurbineId"] = (
                park_df["TurbineId"] + list(self.turbine_mapping_dict[park])[0] - 1
            )

            # Creating a mapping for reporting
            if park not in self.id_mapping.keys():
                self.id_mapping[f"Park{str(park)}"] = list(
                    range(
                        self.current_turbine_index_addition
                        - len(park_df["TurbineId"].unique()),
                        self.current_turbine_index_addition,
                    )
                )

            print(f"Dataframe shape: {park_df.shape}")
            print(f"Unique TurbineIds: {park_df['TurbineId'].nunique()}")
            print(
                f"Min ID: {park_df['TurbineId'].min()}, Max ID: {park_df['TurbineId'].max()}"
            )

            # Saving the dataframe to the intermediate path
            intermediate_file_name = file.split("_")[0] + "_" + file.split("_")[1]

            self.save_dataframe(file_name=intermediate_file_name, park_df=park_df)

        # convert id_mapping to dataframe
        turbine_mapping_df = pd.DataFrame(
            {
                "Park": [
                    f"Park{str(park[4:]).zfill(2)}" for park in self.id_mapping.keys()
                ],
                "Range": [
                    f"{min(vals)}-{max(vals)}" for vals in self.id_mapping.values()
                ],
            }
        )

        # Use OS parent directory to save the turbine mapping
        turbine_mapping_df.to_excel(
            os.path.join(self.intermediate_path, os.pardir, "turbine_mapping.xlsx"),
            index=False,
            header=True,
            columns=["Park", "Range"],
        )

        # end time
        end_time = datetime.now()
        print(f"Time spend converting data: {end_time - start_time}")

    def save_dataframe(self, file_name: str, park_df: pd.DataFrame):
        """Saves the dataframe to the intermediate path.
        Supports csv and pickle formats.

        Args:
            file_name (str): Name of the file
            park_df (pd.DataFrame): dataframe at park level

        Raises:
            ValueError: Raise if format not supported
        """
        if self.file_format == "pickle":
            park_df.to_pickle(
                os.path.join(self.intermediate_path, f"{file_name}.pkl"), protocol=4
            )
            print(
                "Dataframe saved as pickle at: ",
                os.path.join(self.intermediate_path, f"{file_name}.pkl"),
            )
        elif self.file_format == "csv":
            park_df.to_csv(os.path.join(self.intermediate_path, f"{file_name}.csv"))
            print(
                "Dataframe saved as csv at: ",
                os.path.join(self.intermediate_path, f"{file_name}.csv"),
            )
        else:
            raise ValueError("File format not supported")


class DataWrangler:
    """
    Wrangles the data to a format that is suitable for modelling.

    ...

    Attributes
    ----------
    source_parth (str):
        path to source data, either csv or pickle
    destination_path (str):
        path to save the processed data
    output_format (str):
        save as either "csv" or "pkl"

    Methods
    -------
    format_loading:
        Loads the data from the source path. Supports csv and pickle formats.
    recursive_wrangle:
        Recursive caller for the wrangler
    data_filtering:
        The method to filter the data with preset rules
    save_dataframe:
        Saves the dataframe to the destination path
    minmax_normalizer:
        Applies a minmax normalization to the data.
    z_normalizer:
        Applies z normalisation to the data
    filter_iqr_outliers:
        Fundemental iqr filter
    bivariate_norm_filter:
        Filters the data based on a bivariate fit
    _mahalanobis:
        Filter the data based on the mahalanobis distance



    """

    def __init__(
        self,
        source_path: str = "data/intermediate_data",
        destination_path: str = "data/processed_data",
        output_format: str = "pkl",
    ):
        """
        Args:
            source_path (str): path to source data, either csv or pickle
            destination_path (str): path to save the processed data
            output_format (str): save as either "csv" or "pkl"
        """
        self.source_path = source_path
        self.destination_path = destination_path
        self.output_format = output_format
        self.removed_rows = None

    def format_loading(self, file: str):
        """
        Loads the data from the source path.
        Supports csv and pickle formats.
        """
        print("Loading data.....")
        input_format = file.split(".")[-1]

        if input_format == "csv":
            df = pd.read_csv(self.source_path + "/" + file)
        elif input_format == "pkl":
            df = pd.read_pickle(self.source_path + "/" + file)
        else:
            raise ValueError("File format not supported")

        print(f"File loaded: {file}, shape: {df.shape}")
        return df

    def recursive_wrangle(
        self,
    ):
        """
        Recursively wrangle the data to a format that is suitable for modelling.
        """

        print("Wrangling data.....")

        for root, dirs, files in os.walk(self.source_path):
            for file in files:
                print("------------------------")
                print(f"Wrangling file: {file}")
                df = self.format_loading(file)
                # apply the method to the data
                df = self.data_filtering(df)
                self.save_dataframe(df, file.split(".")[0])

    def data_filtering(
        self,
        df: pd.DataFrame,
        pitch_list: list = ["PitchAngleA", "PitchAngleB", "PitchAngleC"],
    ) -> pd.DataFrame:
        """data_filtering is a collection of functions that filters the data based on filtering rules.

        Args:
            df (pd.DataFrame): Dataframe for filtering
            pitch_list (list): Collection of variables for iqr filtering


        Applies:
            - Binning of wind speed (Removes wind speeds below 5 and above 20)
            - Dropping rows with NaN values
            - Filter out rows where the GridPower is below 5% of the maximum power
            - Drop curtailment unless the turbine is at rated power (max power, 95% quantile)

        Returns:
            pd.DataFrame: wrangled dataframe
        """
        print("Wrangling data.....")

        original_shape = df.shape[0]

        # Drop rows with NaN values
        df = df.dropna()

        # Drop "AmbTemp" values below -10
        df = df[df["AmbTemp"] >= -10]

        # create a new column for the 5% power and rated power
        for turbine in df["TurbineId"].unique():
            # define rated power as the 0.95 quantile of the GridPower
            five_percent_power = (
                df[df["TurbineId"] == turbine]["GridPower"].max() * 0.05
            )
            # create a new column with the 5% power for later filtering
            df.loc[df["TurbineId"] == turbine, "FivePercentPower"] = five_percent_power
            rated_power = df[df["TurbineId"] == turbine]["GridPower"].quantile(0.95)
            df.loc[df["TurbineId"] == turbine, "RatedPower"] = rated_power

        # filter out rows where the GridPower is below 5% of the maximum power
        df = df[df["GridPower"] >= df["FivePercentPower"]]

        # filter out positive values for PitchAngles's
        df = df[
            (df["PitchAngleA"] <= 0)
            & (df["PitchAngleB"] <= 0)
            & (df["PitchAngleC"] <= 0)
        ]

        # filter positive values for BladeLoadA, B and C
        df = df[
            (df["BladeLoadA"] < 0) & (df["BladeLoadB"] < 0) & (df["BladeLoadC"] < 0)
        ]

        ### FILTER FOR NOMINAL POWER EXCLUSION ###
        df = df[
            (df["PitchAngleA"] <= 0)
            & (df["PitchAngleB"] <= 0)
            & (df["PitchAngleC"] <= 0)
            | (
                (df["GridPower"] >= df["RatedPower"])
                & (df["PitchAngleA"] > 0)
                & (df["PitchAngleB"] > 0)
                & (df["PitchAngleC"] > 0)
            )
        ]

        ### FILTER OUT 2024 ###
        df = df[df["Time"].dt.year < 2024]

        # We generally create this column for pandas behavior on grouping
        df["TurbinePitchGrp"] = df["TurbineId"]
        df["TurbineBivariateGrp"] = df["TurbineId"]

        # Apply the IQR filtering
        print(f"Shape before IQR filtering: {df.shape}")
        df = (
            df.groupby("TurbinePitchGrp")
            .apply(filter_iqr_outliers, pitch_list=pitch_list, include_groups=False)
            .reset_index(drop=True)
        )
        print(f"Shape after IQR filtering: {df.shape}")

        # Apply the bivariate filtering
        print(f"Shape before bivariate filtering: {df.shape}")
        df = (
            df.groupby("TurbineBivariateGrp")
            .apply(bivariate_norm_filter, pitch_list=pitch_list, include_groups=False)
            .reset_index(drop=True)
        )
        print(f"Shape after bivariate filtering: {df.shape}")

        # remove the columns only used for wrangling
        df = df.drop(labels=["FivePercentPower", "RatedPower"], axis="columns")

        # Prints:
        self.removed_rows = original_shape - df.shape[0]
        print(f"Removed total rows: {self.removed_rows}")

        print(f"Removed rows %: {(self.removed_rows / original_shape)*100}")

        return df

    def minmax_normalizer(self, df: pd.DataFrame, granularity: str = "TurbineId"):
        """
        Applies a min-max normalization to columns with float typing.
        Normalization is done per turbine per deffault.

        !!Note!!:
        Self.df is overwritten with the normalized data.

        Args:
            granularity (str): "TurbineId" or "Park", directly used as column name
        """
        print("Normalizing data.....")

        if granularity not in df.columns:
            raise ValueError("Granularity not found in dataframe")

        minmax_list = []

        for instance in df[granularity].unique():
            instance_df = df[df[granularity] == instance]
            for col in instance_df.columns:
                if instance_df[col].dtype == np.float64:
                    instance_df[col] = minmax_scale(instance_df[col])
            minmax_list.append(instance_df)

        df = None
        df = pd.concat(minmax_list)
        print("Data normalized")

    def z_normalizer(self, granularity: str = "TurbineId"):
        """
        !!WARNING!!
        Z-score normalization assumes a normal distribution of the data.

        Applies a z-score normalization to columns with float typing.
        Normalization is done per turbine per deffault.

        Args:
            granularity (str): "TurbineId" or "Park", directly used as column name
        """
        print("Normalizing data.....")

        if granularity not in self.df.columns:
            raise ValueError("Granularity not found in dataframe")

        z_list = []
        for instance in self.df[granularity].unique():
            instance_df = self.df[self.df[granularity] == instance]
            for col in instance_df.columns:
                if instance_df[col].dtype == np.float64:
                    instance_df[col] = (instance_df[col] - instance_df[col].mean()) / (
                        instance_df[col].std()
                    )
            z_list.append(instance_df)

        self.df = None
        self.df = pd.concat(z_list)
        print("Data normalized")

    def save_dataframe(
        self, df: pd.DataFrame, file_name: str, print_states: bool = True
    ):
        """
        Saves the dataframe to the destination path.
        Supports csv and pickle formats.

        Args:
            file_name (str): name of the file to save
            print_states (bool): print statements
        """
        if not os.path.exists(os.path.join(self.destination_path, "on_shore")):
            if print_states:
                print("Creating on_shore folder")
            os.makedirs(os.path.join(self.destination_path, "on_shore"))
        if not os.path.exists(os.path.join(self.destination_path, "off_shore")):
            if print_states:
                print("Creating off_shore folder")
            os.makedirs(os.path.join(self.destination_path, "off_shore"))

        if print_states:
            print("Saving dataframe, shape: ", df.shape)

        # sort based on turbine type
        turbine_type = None
        if int(file_name.split("_Park")[1].split(".")[0]) >= 6:
            turbine_type = "on_shore"
        elif int(file_name.split("_Park")[1].split(".")[0]) < 6:
            turbine_type = "off_shore"
        else:
            raise ValueError("Turbine type not found")

        if self.output_format == "pkl":
            df.to_pickle(
                os.path.join(
                    self.destination_path,
                    turbine_type,
                    file_name + "." + self.output_format,
                ),
                protocol=4,
            )
            if print_states:
                print(
                    "Dataframe saved as pickle at: ",
                    os.path.join(
                        self.destination_path,
                        turbine_type,
                        file_name + "." + self.output_format,
                    ),
                )
        elif self.output_format == "csv":
            df.to_csv(
                os.path.join(
                    self.destination_path, file_name + "." + self.output_format
                )
            )
            if print_states:
                print(
                    "Dataframe saved as csv at: ",
                    os.path.join(
                        self.destination_path, file_name + "." + self.output_format
                    ),
                )
        else:
            raise ValueError("File format not supported")


# Turbine based IQR filtering for PitchAngles
def _lower_upper_pitch_filter(group: pd.DataFrame, pitch_list: list) -> pd.DataFrame:
    """Filter out rows where the PitchAngles are outside the bounds.
    This function is meant for hard filters close to the median value for pitch angles.

    Args:
        group (pd.DataFrame): Turbines
        pitch_list (list): List of columns to filter

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    lower_bound = group[pitch_list].median() - 0.1
    upper_bound = group[pitch_list].median() + 0.1
    return group[
        ((group[pitch_list] > lower_bound) & (group[pitch_list] < upper_bound)).any(
            axis=1
        )
    ]


def filter_iqr_outliers(group: pd.DataFrame, pitch_list: list) -> pd.DataFrame:
    """Filter out rows where the PitchAngles are outside the bounds, based on the IQR.

    Args:
        group (pd.DataFrame): Turbine grouping
        pitch_list (list): List of columns to filter

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    Q1 = group[pitch_list].quantile(0.25)
    Q3 = group[pitch_list].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return group[
        ((group[pitch_list] >= lower_bound) & (group[pitch_list] <= upper_bound)).any(
            axis=1
        )
    ]


def bivariate_norm_filter(
    group: pd.DataFrame, pitch_list: list, alpha: float = 0.90
) -> pd.DataFrame:
    """
    Filter out rows where the PitchAngles Vs WindSpeed are outside the bounds,
    based on the parameters of the bivariate normal distribution.

    The filter is applied by the Mahalanobis distance, with each point being filtered
    by a hypothesis thesis under the chi-squared dist with significance level alpha.

    Args:
        group (pd.DataFrame): Turbine grouping
        pitch_list (list): List of columns to filter
        alpha (float): Significance level. Lower alpha applies stricter filter

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    def _is_pos_def(A) -> bool:
        """
        Checks if a given matrix is positive semi-definite (PSD).

        A matrix is considered positive semi-definite if:
        - It is symmetric.
        - All its eigenvalues are non-negative (or equivalently, it has a valid Cholesky decomposition).

        Args:
            A (np.ndarray): The matrix to check. Should be a square 2D NumPy array.

        Returns:
            bool: True if the matrix is symmetric and positive semi-definite, False otherwise.
        """
        if np.allclose(A, A.T):
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False

    def _mahalanobis(x, mean, inv_cov) -> float:
        """
        Computes the Mahalanobis distance between a point and a multivariate distribution.

        The Mahalanobis distance accounts for the covariance between variables and is
        useful for identifying multivariate outliers.

        Args:
            x (np.ndarray): A 1D array representing the data point.
            mean (np.ndarray): A 1D array representing the mean of the distribution.
            inv_cov (np.ndarray): The inverse of the covariance matrix of the distribution.

        Returns:
            float: The Mahalanobis distance between the point `x` and the distribution defined by `mean` and `inv_cov`.
        """
        diff = x - mean
        return np.dot(np.dot(diff, inv_cov), diff.T)

    rv_dict = {}
    inv_cov_dict = {}
    mean_dict = {}

    # Find the multivariate normal per pitch
    for pitch in pitch_list:
        subset = group[[pitch, "WindSpeed"]]
        mean = subset.mean()
        cov = subset.cov().values
        inv_cov = np.linalg.inv(cov)

        if not _is_pos_def(inv_cov):
            raise ValueError(
                "inv_cov is not PSD. Make sure the inverse covariance matrix are PSD for the Mahalanobis distance"
            )

        rv_dict[pitch] = (mean, inv_cov)
        mean_dict[pitch] = mean
        inv_cov_dict[pitch] = inv_cov

    # Compute Mahalanobis distances and filter
    mask = np.ones(len(group), dtype=bool)
    for pitch in pitch_list:
        subgroup = group[[pitch, "WindSpeed"]].to_numpy()
        distances = np.array(
            [
                _mahalanobis(x, mean_dict[pitch].values, inv_cov_dict[pitch])
                for x in subgroup
            ]
        )
        # alpha cutoff for 2D chi-squared
        threshold = chi2.ppf(alpha, df=2)
        pitch_mask = distances < threshold
        mask &= pitch_mask  # Keep only if it's not an outlier in ALL pitch dimensions

    return group[mask]


if __name__ == "__main__":
    mlc = MatLabConverter(
        source_path="data/raw_data", intermediate_path="data/intermediate_data"
    )

    dw = DataWrangler(
        source_path="data/intermediate_data",
        destination_path="data/processed_data",
    )

    mlc.recursive_data_converter()
    dw.recursive_wrangle(method="data_filtering")
