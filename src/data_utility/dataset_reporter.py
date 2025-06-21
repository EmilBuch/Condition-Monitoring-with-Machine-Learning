import os
import pandas as pd
from make_dataset import DataWrangler


if __name__ == "__main__":

    PATH = "data/intermediate_data"

    print("Creating NaN report for all parks")
    print("------------------------------------------")
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".pkl"):
                df_park_x = pd.read_pickle(os.path.join(root, file))

                print(f"Processing {file}")
                park = file.split("Park")[1].split(".")[0]
                print(f"Park: {park}")
                print(f"Shape: {df_park_x.shape}")

                df_park_x["Year"] = df_park_x["Time"].dt.year
                df_park_x = df_park_x[
                    df_park_x.columns.difference(["Time", "TurbineId"])
                ]

                df_park_x = (
                    df_park_x.isnull()
                    .groupby([df_park_x["Year"]])
                    .sum()
                    .astype(int)
                    .div(df_park_x.groupby([df_park_x["Year"]]).size(), axis=0)
                    .round(3)
                    * 100
                )
                df_park_x = df_park_x[df_park_x.columns.difference(["Year", "Park"])]
                if not os.path.exists("data/processed_data/data_NaN_report.xlsx"):
                    with pd.ExcelWriter(
                        "data/processed_data/data_NaN_report.xlsx", mode="w"
                    ) as writer:
                        df_park_x.T.to_excel(
                            writer, sheet_name=f"Park {park}", index=True
                        )
                else:
                    with pd.ExcelWriter(
                        "data/processed_data/data_NaN_report.xlsx",
                        mode="a",
                        if_sheet_exists="replace",
                    ) as writer:
                        df_park_x.T.to_excel(
                            writer, sheet_name=f"Park {park}", index=True
                        )

    print("Creating Zero report for all parks")
    print("------------------------------------------")

    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".pkl"):
                df_park_x = pd.read_pickle(os.path.join(root, file))
                park = file.split("Park")[1].split(".")[0]

                df_park_x["Year"] = df_park_x["Time"].dt.year
                df_park_x = df_park_x[
                    df_park_x.columns.difference(["Time", "TurbineId"])
                ]

                df_park_x = (df_park_x == 0).groupby([df_park_x["Year"]]).sum().astype(
                    int
                ).div(df_park_x.groupby([df_park_x["Year"]]).size(), axis=0).round(
                    3
                ) * 100
                df_park_x = df_park_x[df_park_x.columns.difference(["Year", "Park"])]

                if not os.path.exists("data/processed_data/data_zero_report.xlsx"):
                    with pd.ExcelWriter(
                        "data/processed_data/data_zero_report.xlsx", mode="w"
                    ) as writer:
                        df_park_x.T.to_excel(
                            writer, sheet_name=f"Park {park}", index=True
                        )
                else:
                    with pd.ExcelWriter(
                        "data/processed_data/data_zero_report.xlsx",
                        mode="a",
                        if_sheet_exists="replace",
                    ) as writer:
                        df_park_x.T.to_excel(
                            writer, sheet_name=f"Park {park}", index=True
                        )
