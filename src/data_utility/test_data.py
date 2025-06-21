import pandas as pd
import sys
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from sklearn.preprocessing import minmax_scale

sys.path.insert(0, "src/")

# Turn of chains warnings from pandas
pd.options.mode.chained_assignment = None  # default='warn'
from visualization.visualize import plot_2d_scatter
from data_utility.data_utils import IQRPitchVsWindSpeedFiltering, DataLoader
from data_utility.make_dataset import DataWrangler


if __name__ == "__main__":

    dw = DataWrangler(
        source_path="data/intermediate_data",
        destination_path="data/processed_data",
    )

    iqr_filter = IQRPitchVsWindSpeedFiltering(
        source_path="data/processed_data", destination_path="data/iqr_filtered_data"
    )

    dl_pre = DataLoader()

    # mlc.recursive_data_converter()
    # dw.recursive_wrangle(method="data_filtering")
    iqr_filter.apply_iqr_filtering()
