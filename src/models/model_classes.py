import xgboost as xgb
import numpy as np
import pandas as pd
import os
from datetime import date
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from early_stopping_pytorch import EarlyStopping
from utils.utils import reset_weights

np.random.seed(8)


class BaseModel(ABC):
    @abstractmethod
    def fit(
        self,
        explainable_var: np.ndarray | list | str,
        dependent_var: np.ndarray | list | str,
        df: pd.DataFrame = None,
    ):
        pass

    @abstractmethod
    def predict(
        self, explainable_var: np.ndarray | list | str, df: pd.DataFrame = None
    ):
        pass


class XGBRegressorClass(BaseModel):
    def __init__(self, df: pd.DataFrame = None, params: dict = None):
        self.df = df
        self.params = params

        self.model = None
        if self.params:
            self.update_model_params(self.params)
        else:
            self.model = xgb.XGBRegressor(seed=1234)

    def update_model_params(self, params):
        self.params = params
        self.model = xgb.XGBRegressor(**self.params)

    def train(self, df, explainable_vars: list, dependent_var: list):
        self.model.fit(df[explainable_vars], df[dependent_var])

    def fit(
        self,
        explainable_var: np.ndarray | list | str,
        dependent_var: np.ndarray | list | str,
        df: pd.DataFrame = None,
    ):
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]
        if isinstance(dependent_var, str):
            dependent_var = [dependent_var]

        if df is not None:
            self.model.fit(df[explainable_var], df[dependent_var])
        elif self.df is not None and isinstance(explainable_var, list):
            self.model.fit(self.df[explainable_var], self.df[dependent_var])
        else:
            self.model.fit(explainable_var, dependent_var)

    def predict(
        self, explainable_var: np.ndarray | list | str, df: pd.DataFrame = None
    ):
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]

        if df is not None:
            return self.model.predict(df[explainable_var])
        elif self.df is not None and isinstance(explainable_var, list):
            return self.model.predict(self.df[explainable_var])
        else:
            return self.model.predict(explainable_var)

    def pred(self, df, explainable_vars: list):
        predictions = self.model.predict(df[explainable_vars])
        return predictions


class KNNRegressorClass(BaseModel):
    def __init__(self, df: pd.DataFrame = None, params: dict = None):
        self.df = df
        self.params = params

        self.model = None
        if self.params:
            self.update_model_params(self.params)
        else:
            self.model = KNeighborsRegressor()

    def update_model_params(self, params):
        self.params = params
        self.model = KNeighborsRegressor(**self.params)

    def train(self, df, explainable_vars: list, dependent_var: list):
        self.model.fit(df[explainable_vars], df[dependent_var])

    def fit(
        self,
        explainable_var: np.ndarray | list | str,
        dependent_var: np.ndarray | list | str,
        df: pd.DataFrame = None,
    ):
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]
        if isinstance(dependent_var, str):
            dependent_var = [dependent_var]

        if df is not None:
            self.model.fit(df[explainable_var], df[dependent_var])
        elif self.df is not None and isinstance(explainable_var, list):
            self.model.fit(self.df[explainable_var], self.df[dependent_var])
        else:
            self.model.fit(explainable_var, dependent_var)

    def predict(
        self, explainable_var: np.ndarray | list | str, df: pd.DataFrame = None
    ):
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]

        if df is not None:
            return self.model.predict(df[explainable_var])
        elif self.df is not None and isinstance(explainable_var, list):
            return self.model.predict(self.df[explainable_var])
        else:
            return self.model.predict(explainable_var)


class GMMClass(BaseModel):
    def __init__(self, params: dict = None):
        self.params = params
        self.model = None
        if self.params:
            self.model = GaussianMixture(**self.params)
        else:
            self.model = GaussianMixture(random_state=1234)

    def fit(
        self,
        df: pd.DataFrame,
        explainable_var: np.ndarray | list | str,
        # dependent_var: np.ndarray | list | str = None,
    ):
        self.model.fit(df[explainable_var])

    def predict(self, df: pd.DataFrame, explainable_var: np.ndarray | list | str):
        predictions = self.model.predict(df[explainable_var])
        return predictions

    def score_samples(self, df: pd.DataFrame, vars: list):
        return self.model.score_samples(df[vars])

    def goodness_fit(self, df: pd.DataFrame, vars: list):
        print(f"BIC: {self.model.bic(df[vars])}")
        print(f"Params: {self.model.get_params()}")
        print(f"Converged: {self.model.converged_}")


class LRClass(BaseModel):
    def __init__(self, df: pd.DataFrame = None, params: dict = None):
        self.df = df
        self.params = params

        self.model = None
        if self.params:
            self.update_model_params(self.params)
        else:
            self.model = LinearRegression()

    def update_model_params(self, params):
        self.params = params
        self.model = LinearRegression(**self.params)

    def fit(
        self,
        explainable_var: np.ndarray | list | str,
        dependent_var: np.ndarray | list | str,
        df: pd.DataFrame = None,
    ):
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]
        if isinstance(dependent_var, str):
            dependent_var = [dependent_var]

        if df is not None:
            self.model.fit(df[explainable_var], df[dependent_var])
        elif self.df is not None and isinstance(explainable_var, list):
            self.model.fit(self.df[explainable_var], self.df[dependent_var])
        else:
            self.model.fit(explainable_var, dependent_var)

    def predict(
        self, explainable_var: np.ndarray | list | str, df: pd.DataFrame = None
    ):
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]

        if df is not None:
            return self.model.predict(df[explainable_var])
        elif self.df is not None and isinstance(explainable_var, list):
            return self.model.predict(self.df[explainable_var])
        else:
            return self.model.predict(explainable_var)


class SVRClass(BaseModel):
    def __init__(self, params: dict = None):
        self.df = None
        self.params = params

        self.model = None
        if self.params:
            self.update_model_params(self.params)
        else:
            self.model = SVR()

    def update_model_params(self, params):
        self.params = params
        self.model = SVR(**self.params)

    def fit(
        self,
        explainable_var: np.ndarray | list | str,
        dependent_var: np.ndarray | list | str,
        df: pd.DataFrame = None,
    ):
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]
        if isinstance(dependent_var, str):
            dependent_var = [dependent_var]

        if df is not None:
            self.model.fit(df[explainable_var], df[dependent_var])
        elif self.df is not None and isinstance(explainable_var, list):
            self.model.fit(self.df[explainable_var], self.df[dependent_var])
        else:
            self.model.fit(explainable_var, dependent_var)

    def predict(
        self, explainable_var: np.ndarray | list | str, df: pd.DataFrame = None
    ):
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]

        if df is not None:
            return self.model.predict(df[explainable_var])
        elif self.df is not None and isinstance(explainable_var, list):
            return self.model.predict(self.df[explainable_var])
        else:
            return self.model.predict(explainable_var)


class RFRegressorClass(BaseModel):
    def __init__(self, params: dict = None):
        self.df = None
        self.params = params

        self.model = None
        if self.params:
            self.update_model_params(self.params)
        else:
            self.model = RandomForestRegressor(random_state=1234)

    def update_model_params(self, params):
        self.params = params
        self.model = RandomForestRegressor(**self.params)

    def fit(
        self,
        explainable_var: np.ndarray | list | str,
        dependent_var: np.ndarray | list | str,
        df: pd.DataFrame = None,
    ):
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]
        if isinstance(dependent_var, str):
            dependent_var = [dependent_var]

        if df is not None:
            self.model.fit(df[explainable_var], df[dependent_var])
        elif self.df is not None and isinstance(explainable_var, list):
            self.model.fit(self.df[explainable_var], self.df[dependent_var])
        else:
            self.model.fit(explainable_var, dependent_var.ravel())

    def predict(
        self, explainable_var: np.ndarray | list | str, df: pd.DataFrame = None
    ):
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]

        if df is not None:
            return self.model.predict(df[explainable_var])
        elif self.df is not None and isinstance(explainable_var, list):
            return self.model.predict(self.df[explainable_var])
        else:
            return self.model.predict(explainable_var)


class MLPRegressorClass(BaseModel):
    # Default parameters
    default_params = {
        "hidden_sizes": [16, 32],
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 1000,
    }

    def __init__(self, params: dict = None):
        self.df = None
        self.model = None
        self.params = params
        self.path = None
        if self.params:
            self.update_model_params({self.default_params, self.params})
        else:
            self.params = self.default_params

    def update_model_params(self, params):
        self.params = {**self.params, **params}
        if "turbine_id" in self.params.keys():
            self.path = os.path.join(
                os.getcwd(),
                "models",
                date.today().strftime("%Y%m%d"),
                "checkpoints",
            )
            if not os.path.exists(self.path):
                os.makedirs(self.path)

            self.path = os.path.join(
                self.path, "turbine_" + str(self.params["turbine_id"]) + ".pt"
            )
            del self.params["turbine_id"]
        if self.df is not None:
            self.model = self._build_model(self.df.shape[1])
        if self.model is not None:
            self.model.apply(reset_weights)
            print("Reset weights with update_model_params")

    def _build_model(self, input_size, output_size=1):
        layers = []
        prev_size = input_size

        for hidden_size in self.params["hidden_sizes"]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        return nn.Sequential(*layers)

    def fit(
        self,
        explainable_var: np.ndarray | list | str,
        dependent_var: np.ndarray | list | str,
        df: pd.DataFrame = None,
    ):
        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        # Process input parameters
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]
        if isinstance(dependent_var, str):
            dependent_var = [dependent_var]
        # print the max and min value of "BladeLoadA" from the df
        if df is not None:
            print(
                f"Max BladeLoadA: {df['BladeLoadA'].max()}, Min BladeLoadA: {df['BladeLoadA'].min()}"
            )
        # Get data
        if df is not None:
            self.df = df
            X = df[explainable_var].values
            y = df[dependent_var].values
        elif self.df is not None and isinstance(explainable_var, list):
            X = self.df[explainable_var].values
            y = self.df[dependent_var].values
        else:
            X = np.array(explainable_var)
            y = np.array(dependent_var)

        # Reshape y if needed
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Build model
        input_size = X.shape[1]
        self.model = self._build_model(input_size).to(device)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)

        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)

        # Calculate sizes
        train_size = int(0.8 * len(dataset))

        # Create indices for the split (no shuffling)
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, len(dataset)))

        # Create subsets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_loader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.params["batch_size"], shuffle=False
        )
        dataloader = DataLoader(
            dataset, batch_size=self.params["batch_size"], shuffle=True
        )

        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])
        criterion = nn.L1Loss()
        max_target_val = torch.max(y_tensor).item()
        delta_percent = 0.01
        # Calculate delta based on max target value
        delta = max_target_val * delta_percent
        print(f"Delta: {delta}, Max target value: {max_target_val}")
        # Train the model
        self.model.train()
        early_stopping = EarlyStopping(
            patience=15, verbose=False, delta=delta, path=self.path
        )
        val_plot_list = []
        for epoch in range(self.params["epochs"]):
            self.model.train()
            val_list = []
            loss_list = []
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            train_loss = np.average(loss_list)
            loss_list = []
            # Validation step
            self.model.eval()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_list.append(loss.item())
            val_loss = np.average(val_list)
            val_plot_list.append(val_loss)

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch-15}")
                break

        self.model.load_state_dict(torch.load(self.path, weights_only=True))

    def predict(
        self, explainable_var: np.ndarray | list | str, df: pd.DataFrame = None
    ):

        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        # Process input parameters
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]

        # Get data
        if df is not None:
            X = df[explainable_var].values
        elif self.df is not None and isinstance(explainable_var, list):
            X = self.df[explainable_var].values
        else:
            X = np.array(explainable_var)

        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()

        # Reshape to 1D array
        predictions = predictions[:, 0]

        return predictions


class ElasticNetClass(BaseModel):
    def __init__(self, params: dict = None):
        self.df = None
        self.params = params

        self.model = None
        if self.params:
            self.update_model_params(self.params)
        else:
            self.model = ElasticNet(random_state=1234)

    def update_model_params(self, params):
        self.params = params
        self.model = ElasticNet(**self.params)

    def fit(
        self,
        explainable_var: np.ndarray | list | str,
        dependent_var: np.ndarray | list | str,
        df: pd.DataFrame = None,
    ):
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]
        if isinstance(dependent_var, str):
            dependent_var = [dependent_var]

        if df is not None:
            self.model.fit(df[explainable_var], df[dependent_var])
        elif self.df is not None and isinstance(explainable_var, list):
            self.model.fit(self.df[explainable_var], self.df[dependent_var])
        else:
            self.model.fit(explainable_var, dependent_var)

    def predict(
        self,
        explainable_var: np.ndarray | list | str,
        df: pd.DataFrame = None,
    ):
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]

        if df is not None:
            return self.model.predict(df[explainable_var])
        elif self.df is not None and isinstance(explainable_var, list):
            return self.model.predict(self.df[explainable_var])
        else:
            return self.model.predict(explainable_var)


class BayesianRidgeClass(BaseModel):
    def __init__(self, params: dict = None):
        self.df = None
        self.params = params

        self.model = None
        if self.params:
            self.update_model_params(self.params)
        else:
            self.model = BayesianRidge()

    def update_model_params(self, params):
        self.params = params
        self.model = BayesianRidge(**self.params)

    def fit(
        self,
        explainable_var: np.ndarray | list | str,
        dependent_var: np.ndarray | list | str,
        df: pd.DataFrame = None,
    ):
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]
        if isinstance(dependent_var, str):
            dependent_var = [dependent_var]

        if df is not None:
            self.model.fit(df[explainable_var], df[dependent_var])
        elif self.df is not None and isinstance(explainable_var, list):
            self.model.fit(self.df[explainable_var], self.df[dependent_var])
        else:
            self.model.fit(explainable_var, dependent_var)

    def predict(
        self,
        explainable_var: np.ndarray | list | str,
        df: pd.DataFrame = None,
    ):
        if isinstance(explainable_var, str):
            explainable_var = [explainable_var]

        if df is not None:
            return self.model.predict(df[explainable_var])
        elif self.df is not None and isinstance(explainable_var, list):
            return self.model.predict(self.df[explainable_var])
        else:
            return self.model.predict(explainable_var)
