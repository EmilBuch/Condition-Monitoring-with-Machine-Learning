from datetime import date
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import pandas as pd
import numpy as np
import os
import warnings
import sys
import plotly.express as px
import plotly
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from typing import Union, Optional, List, Tuple, Callable, Any, Literal
import fastkde
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, "src/")
from data_utility.data_utils import KFinder, DataLoader
from scipy.stats import norm
from scipy.stats import expon
from scipy.interpolate import interp1d
from scipy.stats import linregress
from utils.decorators import mute_print


VISUALIZATION_DIR = os.path.join(
    os.getcwd(), "figures", date.today().strftime("%Y%m%d")
)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


def _savefig_helper(
    fig: Union[
        matplotlib.figure.Figure,
        plotly.graph_objs._figure.Figure,
        sns.axisgrid._BaseGrid,
    ],
    filename: str,
    overwrite: bool = False,
) -> None:
    """
    Helper function to save figures with proper error handling and path management.

    Parameters
    ----------
    fig : Union[matplotlib.figure.Figure, plotly.graph_objs._figure.Figure, sns.axisgrid._BaseGrid]
        The figure object to save. Can be a matplotlib Figure, Plotly Figure, or Seaborn grid.
    filename : str
        The name of the file to save. Will be placed in the VISUALIZATION_DIR directory.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
    """

    def _savefig(fig, filename):
        # Explicit type checking for robustness.
        if isinstance(fig, matplotlib.figure.Figure):
            if hasattr(fig, "savefig"):
                fig.savefig(filename)
            elif hasattr(fig, "write_html"):
                fig.write_html(filename)
        elif isinstance(fig, plotly.graph_objs._figure.Figure):
            if hasattr(fig, "write_html"):
                fig.write_html(filename)
            elif hasattr(fig, "savefig"):
                fig.savefig(filename)
        elif isinstance(fig, sns.axisgrid._BaseGrid):
            if hasattr(fig, "savefig"):
                fig.savefig(filename)
        else:
            raise ValueError(
                "Unsupported figure type. The figure must be either a matplotlib, seaborn or a Plotly figure."
            )

    filename = os.path.join(VISUALIZATION_DIR, filename)

    # Ensure the directory exists.
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Check if file exists.
    if os.path.exists(filename):
        if overwrite:
            warnings.warn(f"File '{filename}' already exists and will be overwritten.")
            _savefig(fig, filename)
        else:
            warnings.warn(
                f"File '{filename}' already exists and will not be overwritten."
            )
    else:
        _savefig(fig, filename)


def ks_test_plots(
    residual_df: pd.DataFrame,
    year_A: int,
    year_B: int,
    ks_stat: float,
    p_value: float,
    filename: str,
    overwrite: bool = False,
):
    """Plot the residuals distribution for two years and include the KS test results.

    Parameters
    ----------
    residual_df : pd.DataFrame
        Residuals dataframe
    year_A : int
        Base year
    year_B : int
        Comparison year
    ks_stat : float
        KS statistic
    p_value : float
        P-value
    filename : str
        Name of the file to save the figure to
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists
    """
    fig, ax = plt.subplots()

    ax.hist(
        residual_df[residual_df["Year"] == year_A]["Residuals"],
        bins=50,
        alpha=0.5,
    )

    ax.hist(
        residual_df[residual_df["Year"] == year_B]["Residuals"],
        bins=50,
        alpha=0.5,
    )

    # label the distributions with their colors
    ax.legend([f"{year_A}", f"{year_B}"])
    ax.set_title(f"Residuals Distribution for {year_A} vs {year_B}")
    # include the KS test results and the p-value, at max 5 decimal places, left aligned
    ax.text(
        0.05,
        0.95,
        f"KS Statistic: {ks_stat:.5f}\nP-Value: {p_value:.5f}",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )

    _savefig_helper(fig, filename, overwrite)
    plt.close(fig)


def residual_comparison_plot(
    residual_df: pd.DataFrame,
    year_A: int,
    year_B: int,
    save_path: str,
):
    """Plot the residuals distribution for two years.

    Args:
        residual_df (pd.DataFrame): Residuals dataframe
        year_A (int): Base year
        year_B (int): Comparison year
        save_path (str): Path to save the plot
    """
    plt.hist(
        residual_df[residual_df["Year"] == year_A]["Residuals"],
        bins=50,
        alpha=0.5,
    )

    plt.hist(
        residual_df[residual_df["Year"] == year_B]["Residuals"],
        bins=50,
        alpha=0.5,
    )

    # label the distributions with their colors
    plt.legend([f"{year_A}", f"{year_B}"])
    plt.title(f"Residuals Distribution for {year_A} vs replacement year {year_B}")
    # include the KS test results and the p-value, at max 5 decimal places, left aligned
    plt.savefig(save_path)
    plt.clf()


def plot_3d_scatter(
    x: Union[pd.Series, List[float]],
    y: Union[pd.Series, List[float]],
    z: Union[pd.Series, List[float]],
    xlabel: str = "X",
    ylabel: str = "Y",
    zlabel: str = "Z",
    marker: str = "o",
    cmap: str = "viridis",
    savefile: bool = False,
    filename: str = "3d_scatter.png",
    overwrite: bool = False,
) -> None:
    """
    Create a 3D scatter plot with matplotlib.

    Parameters
    ----------
    x : Union[pd.Series, List[float]]
        Data for x-axis.
    y : Union[pd.Series, List[float]]
        Data for y-axis.
    z : Union[pd.Series, List[float]]
        Data for z-axis and color mapping.
    xlabel : str, default="X"
        Label for x-axis.
    ylabel : str, default="Y"
        Label for y-axis.
    zlabel : str, default="Z"
        Label for z-axis.
    marker : str, default="o"
        Marker style for scatter points.
    cmap : str, default="viridis"
        Colormap used for coloring points.
    savefile : bool, default=False
        Whether to save the figure to a file.
    filename : str, default="3d_scatter.png"
        Name of the file to save the figure to.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(x, y, z, marker=marker, c=z, cmap=cmap)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # Optionally add a colorbar to show the mapping of colors to z values.
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5, pad=0.1)

    if savefile:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


def plot_3d_scatter_interactive(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    color_variable: Optional[str] = None,
    title: Optional[str] = None,
    savefile: bool = False,
    filename: str = "3d_interactive_scatter_plot.html",
    overwrite: bool = False,
) -> None:
    """
    Create an interactive 3D scatter plot using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    x : str
        Column name for x-axis data.
    y : str
        Column name for y-axis data.
    z : str
        Column name for z-axis data.
    color_variable : Optional[str], default=None
        Column name to use for coloring points. If None, all points have the same color.
    title : Optional[str], default=None
        Title for the plot.
    savefile : bool, default=False
        Whether to save the figure to a file.
    filename : str, default="3d_interactive_scatter_plot.html"
        Name of the file to save the figure to (HTML format).
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
    """
    # Create an interactive 3D scatter plot
    fig = px.scatter_3d(df, x=x, y=y, z=z, color=color_variable, title=title)

    if savefile:
        _savefig_helper(fig, filename, overwrite)
    else:
        fig.show()


def plot_scatter_with_density(
    x: Union[pd.Series, List[float]],
    y: Union[pd.Series, List[float]],
    xlabel: str = "X",
    ylabel: str = "Y",
    title: Optional[str] = None,
    cmap: str = "viridis",
    savefig: bool = False,
    filename: str = "scatter-density_plot.png",
    overwrite: bool = False,
) -> None:
    """
    Create a scatter plot where points are colored by their density.

    This function calculates the density of points using a Gaussian KDE (Kernel Density Estimation)
    and colors the scatter points based on this density. Denser regions appear with a different
    color according to the selected colormap. Points are sorted by density so that the densest
    points appear on top.

    Parameters
    ----------
    x : Union[pd.Series, List[float], np.ndarray]
        Data for x-axis. Will be converted to numpy array internally.
    y : Union[pd.Series, List[float], np.ndarray]
        Data for y-axis. Will be converted to numpy array internally.
    xlabel : str, default="X"
        Label for the x-axis.
    ylabel : str, default="Y"
        Label for the y-axis.
    title : Optional[str], default=None
        Title for the plot. If None, no title is displayed.
    cmap : str, default="viridis"
        Colormap used for density visualization. Any matplotlib colormap name.
    savefig : bool, default=False
        Whether to save the figure to a file.
    filename : str, default="scatter-density_plot.png"
        Name of the file to save the figure to if savefig is True.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
        This function doesn't return any value but shows or saves the plot.
    """
    if len(x.unique()) < 2:
        y = np.asarray(y)
        density = gaussian_kde(y)(y)
        idx = density.argsort()
    elif len(y.unique()) < 2:
        x = np.asarray(x)
        density = gaussian_kde(x)(x)
        idx = density.argsort()
    else:
        xy = np.vstack([x, y])
        density = gaussian_kde(xy)(xy)
        idx = density.argsort()

    x_sorted, y_sorted, density_sorted = (
        np.array(x)[idx],
        np.array(y)[idx],
        density[idx],
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Create a scatter plot where color indicates density.
    sc = ax.scatter(x_sorted, y_sorted, c=density_sorted, s=50, cmap=cmap)
    fig.colorbar(sc, ax=ax, label="Density")

    # Set labels and title.
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if ax.get_legend_handles_labels()[1]:
        ax.legend()

    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


def plot_fastkde_scatter(
    x: Union[pd.Series, List[float]],
    y: Union[pd.Series, List[float]],
    xlabel: str = "X",
    ylabel: str = "Y",
    title: Optional[str] = None,
    fill_value=0.0,
    cmap_scatter: str = "viridis",
    cmap_contour: str = "Blues",
    contour: bool = True,
    s: int = 10,
    padding: bool = False,
    savefig: bool = False,
    filename: str = "scatter-density_plot.png",
    overwrite: bool = False,
):
    """Create a scatter plot with density estimation using fastkde.
    This function uses fastkde to compute the density of points in a 2D space and
    visualize it using a scatter plot. The points are colored based on their estimated
    density, and an optional contour plot can be added to show the density levels.

    Furthermore, if either x or y is constant, a 1D KDE is computed and plotted with no contour
    (contours are not meaningful in 1D).

    Args:
        x (Union[pd.Series, List[float]]): x-axis data.
        y (Union[pd.Series, List[float]]): y-axis data.
        xlabel (str, optional): Label the x-axis.
        ylabel (str, optional): Label the y-axis.
        title (Optional[str], optional):  Set the title of the plot
        fill_value (float, optional): Fillers for interpolation. (should not be touched)
        cmap_scatter (str, optional): Choose the colormap for scatter points.
        cmap_contour (str, optional): Choose the colormap for contour lines.
        contour (bool, optional): Set contour lines for the plot (2D only, highlights density).
        s (int, optional): Size of scatter points.
        padding (bool, optional): Add padding to the axes limits. This does not expand the kde!
        savefig (bool, optional): Whether to save the figure to a file.
        filename (str, optional): Name of the file to save the figure to.
        overwrite (bool, optional): Whether to overwrite the file if it already exists.
    """

    # fastkde requires numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Check if x and y are constant
    x_is_const = len(np.unique(x)) == 1
    y_is_const = len(np.unique(y)) == 1

    # Handle 1D case
    if x_is_const or y_is_const:
        varying = x if not x_is_const else y
        constant_val = x[0] if x_is_const else y[0]
        label = xlabel if not x_is_const else ylabel
        const_axis = ylabel if not x_is_const else xlabel

        # Run 1D KDE
        pdf_1d = fastkde.pdf(varying, var_names=[label])
        kde_x = pdf_1d.coords[label].values
        density = pdf_1d.values

        interp = interp1d(kde_x, density, bounds_error=False, fill_value=fill_value)
        point_densities = interp(varying)

        # Plot as scatter with KDE coloring
        fig, ax = plt.subplots()

        if x_is_const:
            sc = ax.scatter(
                [constant_val] * len(varying),
                varying,
                c=point_densities,
                cmap=cmap_scatter,
                s=s,
                edgecolor="none",
            )
            ax.plot([constant_val] * len(kde_x), kde_x, color="k", lw=1, alpha=0.5)
            ax.set_xlabel(const_axis)
            ax.set_ylabel(label)
        else:
            sc = ax.scatter(
                varying,
                [constant_val] * len(varying),
                c=point_densities,
                cmap=cmap_scatter,
                s=s,
                edgecolor="none",
            )
            ax.plot(kde_x, [constant_val] * len(kde_x), color="k", lw=1, alpha=0.5)
            ax.set_xlabel(label)
            ax.set_ylabel(const_axis)

        fig.colorbar(sc, ax=ax, label="Estimated Density")
        ax.set_title(title or f"1D KDE of {label}")

        if savefig:
            _savefig_helper(fig, filename, overwrite)
        else:
            plt.show()
        plt.close(fig)
        return

    # 2D KDE case
    pdf = fastkde.pdf(x, y, var_names=[xlabel, ylabel])
    pdf_values = pdf.values

    # Extract axis names and grid centers
    dim_y, dim_x = pdf.dims
    xcenters = pdf.coords[dim_x].values
    ycenters = pdf.coords[dim_y].values

    # Interpolate pointwise density from the KDE grid
    interp = RegularGridInterpolator(
        (ycenters, xcenters),
        pdf_values,
        bounds_error=False,
        fill_value=fill_value,
    )
    points = np.vstack([y, x]).T
    point_densities = interp(points)

    Xgrid, Ygrid = np.meshgrid(xcenters, ycenters)

    fig, ax = plt.subplots()

    if padding:
        x_pad = 0.1 * (x.max() - x.min())
        y_pad = 0.1 * (y.max() - y.min())
        ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
        ax.set_ylim(y.min() - y_pad, y.max() + y_pad)

    if contour:
        ax.contourf(Xgrid, Ygrid, pdf_values, levels=100, cmap=cmap_contour, alpha=1)
    sc = ax.scatter(x, y, c=point_densities, cmap=cmap_scatter, s=s, edgecolor="none")
    fig.colorbar(sc, ax=ax, label="Estimated Density")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if ax.get_legend_handles_labels()[1]:
        ax.legend()

    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


def plot_joint_distribution(
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: Optional[str] = None,
    savefig: bool = False,
    filename: str = "3d_joint-dist_plot.png",
    kind: str = "kde",
    hue: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Create a joint distribution plot showing both scatter and marginal distributions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    x : str
        Column name for x-axis data.
    y : str
        Column name for y-axis data.
    xlabel : str, default="X"
        Label for the x-axis.
    ylabel : str, default="Y"
        Label for the y-axis.
    title : Optional[str], default=None
        Title for the plot. If None, no title is displayed.
    savefig : bool, default=False
        Whether to save the figure to a file.
    filename : str, default="3d_joint-dist_plot.png"
        Name of the file to save the figure to.
    kind : str, default="kde"
        Kind of plot to draw. Options include 'scatter', 'kde', 'hist', 'hex', 'reg', 'resid'.
    hue : Optional[str], default=None
        Column name to use for grouping/coloring data points. If None, all points have the same color.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
        This function doesn't return any value but shows or saves the plot.
    """
    # Create a joint plot using the specified kind.
    fig = sns.jointplot(data=df, x=x, y=y, kind=kind, hue=hue)
    fig.set_axis_labels(xlabel, ylabel, fontsize=12)
    if title:
        fig.figure.suptitle(title, y=1.02, fontsize=14)

    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()


def plot_3d_density(
    x: Union[pd.Series, List[float], np.ndarray],
    y: Union[pd.Series, List[float], np.ndarray],
    xlabel: str = "X",
    ylabel: str = "Y",
    title: Optional[str] = None,
    grid_size: int = 100,
    cmap: str = "viridis",
    savefig: bool = False,
    filename: str = "3d_density_plot.png",
    overwrite: bool = False,
) -> None:
    """
    Create a 3D surface plot showing the kernel density estimate of 2D data.

    This function fits a Gaussian Kernel Density Estimation (KDE) to the provided x and y data,
    then creates a 3D surface plot where the height represents the density of points.
    This visualization helps identify regions of high concentration in 2D data.

    Parameters
    ----------
    x : Union[pd.Series, List[float], np.ndarray]
        Data for x-axis. Will be converted to numpy array internally.
    y : Union[pd.Series, List[float], np.ndarray]
        Data for y-axis. Will be converted to numpy array internally.
    xlabel : str, default="X"
        Label for the x-axis.
    ylabel : str, default="Y"
        Label for the y-axis.
    title : Optional[str], default=None
        Title for the plot. If None, no title is displayed.
    grid_size : int, default=100
        Resolution of the grid for density evaluation. Higher values give smoother results but increase computation time.
    cmap : str, default="viridis"
        Colormap used for the surface. Any matplotlib colormap name.
    savefig : bool, default=False
        Whether to save the figure to a file.
    filename : str, default="3d_density_plot.png"
        Name of the file to save the figure to if savefig is True.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
        This function doesn't return any value but shows or saves the plot.
    """
    # Stack x and y for input to the KDE
    xy_data = np.vstack([x, y])

    # Fit a Gaussian KDE to the data
    kde = gaussian_kde(xy_data)

    # Define grid limits based on data range (you can customize these)
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1

    # Create a regular grid to evaluate the KDE upon
    X, Y = np.mgrid[x_min : x_max : grid_size * 1j, y_min : y_max : grid_size * 1j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Evaluate the density on the grid
    Z = np.reshape(kde(positions).T, X.shape)

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none", alpha=0.9)

    # Add a color bar to show density scale
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Density")

    # Set labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_zlabel("Density")

    # Adjust the view angle if desired
    ax.view_init(elev=30, azim=-60)
    if title:
        plt.title(title)

    plt.tight_layout()
    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


def plot_2d_scatter(
    x: Union[pd.Series, List[float], np.ndarray],
    y: Union[pd.Series, List[float], np.ndarray],
    xlabel: str = "X",
    ylabel: str = "Y",
    title: Optional[str] = None,
    hue: Optional[Union[pd.Series, List[float], np.ndarray]] = None,
    hue_label: str = "Group",
    marker: str = "o",
    alpha: float = 0.7,
    figsize: Tuple[float, float] = (10, 8),
    savefig: bool = False,
    filename: str = "scatter_plot.png",
    overwrite: bool = False,
) -> None:
    """
    Create a basic 2D scatter plot with optional coloring by a third variable.

    Parameters
    ----------
    x : Union[pd.Series, List[float], np.ndarray]
        Data for x-axis. Will be converted to numpy array internally.
    y : Union[pd.Series, List[float], np.ndarray]
        Data for y-axis. Will be converted to numpy array internally.
    xlabel : str, default="X"
        Label for the x-axis.
    ylabel : str, default="Y"
        Label for the y-axis.
    title : Optional[str], default=None
        Title for the plot. If None, no title is displayed.
    hue : Optional[Union[pd.Series, List[float], np.ndarray]], default=None
        Values to map to colors for points. If None, all points have the same color.
    hue_label : str, default="Group"
        Label for the colorbar when hue is provided.
    marker : str, default="o"
        Marker style for scatter points. Any matplotlib marker symbol.
    alpha : float, default=0.7
        Transparency of points (0.0 is completely transparent, 1.0 is opaque).
    figsize : Tuple[float, float], default=(10, 8)
        Figure size (width, height) in inches.
    savefig : bool, default=False
        Whether to save the figure to a file.
    filename : str, default="scatter_plot.png"
        Name of the file to save the figure to if savefig is True.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
        This function doesn't return any value but shows or saves the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if hue is not None:
        scatter = ax.scatter(x, y, c=hue, marker=marker, alpha=alpha, cmap="viridis")
        plt.colorbar(scatter, ax=ax, label=hue_label)
    else:
        ax.scatter(x, y, marker=marker, alpha=alpha)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    fig.tight_layout()

    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


def plot_2d_scatter_with_marginals(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    hue: Optional[str] = None,
    kind: str = "hist",
    savefig: bool = False,
    filename: str = "scatter_with_marginals.png",
    overwrite: bool = False,
) -> None:
    """
    Create a scatter plot with marginal distributions on the sides.

    This function creates a joint plot that combines a central scatter plot with
    marginal distribution plots along the x and y axes. This visualization helps
    to understand both the relationship between two variables and their individual
    distributions simultaneously.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    x_col : str
        Column name for x-axis data.
    y_col : str
        Column name for y-axis data.
    xlabel : Optional[str], default=None
        Custom label for x-axis. If None, uses x_col name.
    ylabel : Optional[str], default=None
        Custom label for y-axis. If None, uses y_col name.
    title : Optional[str], default=None
        Title for the plot. If None, no title is displayed.
    hue : Optional[str], default=None
        Column name to use for grouping/coloring data points. If None, all points have the same color.
    kind : str, default="hist"
        Kind of marginal plot to draw. Options include 'hist', 'kde', 'box', etc.
    savefig : bool, default=False
        Whether to save the figure to a file.
    filename : str, default="scatter_with_marginals.png"
        Name of the file to save the figure to if savefig is True.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
        This function doesn't return any value but shows or saves the plot.
    """
    # Create the JointGrid
    g = sns.JointGrid(data=df, x=x_col, y=y_col, hue=hue)

    # Add the plot elements
    g.plot_joint(sns.scatterplot)
    g.plot_marginals(getattr(sns, f"{kind}plot"))

    # Customize labels
    g.set_axis_labels(xlabel or x_col, ylabel or y_col)

    if title:
        g.figure.suptitle(title, y=1.02)

    g.figure.tight_layout()

    if savefig:
        _savefig_helper(g.figure, filename, overwrite)
    else:
        plt.show()
    plt.close()


def plot_counts(
    df: pd.DataFrame,
    x_col: str,
    hue: Optional[str] = None,
    order: Optional[List[Any]] = None,
    hue_order: Optional[List[Any]] = None,
    orientation: str = "vertical",
    figsize: Tuple[float, float] = (10, 6),
    xlabel: Optional[str] = None,
    ylabel: str = "Count",
    title: Optional[str] = None,
    palette: Union[str, List[str]] = "viridis",
    savefig: bool = False,
    filename: str = "count_plot.png",
    overwrite: bool = False,
) -> None:
    """
    Create a count plot for categorical variables.

    This function creates a bar plot showing counts of observations in each categorical bin.
    It can be used to visualize the distribution of categorical data, with optional grouping
    by another categorical variable (hue).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    x_col : str
        Column name for categories to count.
    hue : Optional[str], default=None
        Column name for grouping/splitting the counts. If None, all counts are shown in one color.
    order : Optional[List[Any]], default=None
        List specifying the order of categorical levels for x. If None, levels are inferred from data.
    hue_order : Optional[List[Any]], default=None
        List specifying the order of categorical levels for hue. If None, levels are inferred from data.
    orientation : str, default="vertical"
        Orientation of the plot. Either "vertical" (categories on x-axis) or "horizontal" (categories on y-axis).
    figsize : Tuple[float, float], default=(10, 6)
        Figure size (width, height) in inches.
    xlabel : Optional[str], default=None
        Label for the category axis. If None, uses x_col name.
    ylabel : str, default="Count"
        Label for the count axis.
    title : Optional[str], default=None
        Title for the plot. If None, no title is displayed.
    palette : Union[str, List[str]], default="viridis"
        Color palette used for the plot. Can be a seaborn/matplotlib colormap name or list of colors.
    savefig : bool, default=False
        Whether to save the figure to a file.
    filename : str, default="count_plot.png"
        Name of the file to save the figure to if savefig is True.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
        This function doesn't return any value but shows or saves the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if orientation == "vertical":
        sns.countplot(
            data=df,
            x=x_col,
            hue=hue,
            order=order,
            hue_order=hue_order,
            palette=palette,
            ax=ax,
        )
        ax.set_xlabel(xlabel or x_col)
        ax.set_ylabel(ylabel)
    else:
        sns.countplot(
            data=df,
            y=x_col,
            hue=hue,
            order=order,
            hue_order=hue_order,
            palette=palette,
            ax=ax,
        )
        ax.set_ylabel(xlabel or x_col)
        ax.set_xlabel(ylabel)

    if title:
        ax.set_title(title)

    # Rotate x-tick labels if they might overlap
    if orientation == "vertical" and len(df[x_col].unique()) > 5:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


def plot_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: Optional[str] = None,
    aggfunc: Union[str, Callable] = "count",
    cmap: str = "YlGnBu",
    figsize: Tuple[float, float] = (12, 10),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    annot: bool = True,
    fmt: str = ".0f",
    cbar_label: Optional[str] = None,
    savefig: bool = False,
    filename: str = "heatmap.png",
    overwrite: bool = False,
) -> None:
    """
    Create a heatmap from two categorical variables.

    This function creates a heatmap visualization by first generating a pivot table from two categorical
    columns in the provided DataFrame. The cells in the heatmap represent counts or aggregated values
    of an optional third column. This is useful for visualizing the relationship and frequency distribution
    between two categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    x_col : str
        Column name for categories to appear on the x-axis (columns of the heatmap).
    y_col : str
        Column name for categories to appear on the y-axis (rows of the heatmap).
    value_col : Optional[str], default=None
        Column name containing values to aggregate. If None, counts the occurrences in each x_col/y_col combination.
    aggfunc : Union[str, Callable], default="count"
        Aggregation function to apply when creating the pivot table. Can be a function or a string
        like 'count', 'sum', 'mean', 'median', etc.
    cmap : str, default="YlGnBu"
        Colormap used for the heatmap. Any matplotlib colormap name.
    figsize : Tuple[float, float], default=(12, 10)
        Figure size (width, height) in inches.
    title : Optional[str], default=None
        Title for the plot. If None, no title is displayed.
    xlabel : Optional[str], default=None
        Label for the x-axis. If None, uses x_col name.
    ylabel : Optional[str], default=None
        Label for the y-axis. If None, uses y_col name.
    annot : bool, default=True
        Whether to annotate each cell with its numerical value.
    fmt : str, default=".0f"
        Format string for cell annotations.
    cbar_label : Optional[str], default=None
        Label for the color bar. If None, uses the aggregation function name.
    savefig : bool, default=False
        Whether to save the figure to a file.
    filename : str, default="heatmap.png"
        Name of the file to save the figure to if savefig is True.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
        This function doesn't return any value but shows or saves the plot.
    """
    # Create pivot table
    if value_col:
        pivot_data = df.pivot_table(
            index=y_col, columns=x_col, values=value_col, aggfunc=aggfunc
        )
    else:
        pivot_data = df.pivot_table(index=y_col, columns=x_col, aggfunc="size")

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    heatmap = sns.heatmap(
        pivot_data,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        cbar_kws={"label": cbar_label or aggfunc},
    )

    # Set labels
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


def box_plots(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    plot_type: str = "box",
    hue: Optional[str] = None,
    order: Optional[List[Any]] = None,
    hue_order: Optional[List[Any]] = None,
    figsize: Tuple[float, float] = (10, 6),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    palette: Union[str, List[str]] = "viridis",
    orient: str = "vertical",
    savefig: bool = False,
    filename: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Create box, violin, or boxen plots for distribution visualization.

    This function creates statistical plots to visualize the distribution of a numerical variable
    across different categories. It supports several plot types (box, violin, boxen) that show
    different aspects of the distribution. These plots are useful for comparing distributions
    across groups and identifying outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    x_col : str
        Column name for categories (typically on x-axis). Usually categorical data.
    y_col : str
        Column name for values (typically on y-axis). Usually numerical data to show distribution of.
    plot_type : str, default="box"
        Type of distribution plot to create. Options include:
        - 'box': Traditional box plot showing quartiles and outliers
        - 'violin': Violin plot showing full distribution with kernel density estimate
        - 'boxen': Enhanced box plot that shows more quantiles for larger datasets
    hue : Optional[str], default=None
        Column name for an additional categorical split within each x-axis category.
        If None, no additional grouping is applied.
    order : Optional[List[Any]], default=None
        List specifying the order of categories on the category axis. If None, order is inferred from data.
    hue_order : Optional[List[Any]], default=None
        List specifying the order of categories for the hue variable. If None, order is inferred from data.
    figsize : Tuple[float, float], default=(10, 6)
        Figure size (width, height) in inches.
    xlabel : Optional[str], default=None
        Label for the x-axis. If None, uses x_col or y_col name depending on orientation.
    ylabel : Optional[str], default=None
        Label for the y-axis. If None, uses y_col or x_col name depending on orientation.
    title : Optional[str], default=None
        Title for the plot. If None, no title is displayed.
    palette : Union[str, List[str]], default="viridis"
        Color palette used for the plot. Can be a seaborn/matplotlib colormap name or list of colors.
    orient : str, default="vertical"
        Orientation of the plot. Either "vertical" (categories on x-axis) or
        "horizontal" (categories on y-axis).
    savefig : bool, default=False
        Whether to save the figure to a file.
    filename : Optional[str], default=None
        Name of the file to save the figure to. If None, defaults to "{plot_type}_plot.png".
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
        This function doesn't return any value but shows or saves the plot.
    """
    if filename is None:
        filename = f"{plot_type}_plot.png"

    fig, ax = plt.subplots(figsize=figsize)

    plot_func = {
        "box": sns.boxplot,
        "violin": sns.violinplot,
        "boxen": sns.boxenplot,
    }.get(plot_type.lower(), sns.boxplot)

    if orient == "vertical":
        plot_func(
            data=df,
            x=x_col,
            y=y_col,
            hue=hue,
            order=order,
            hue_order=hue_order,
            palette=palette,
            ax=ax,
        )
        ax.set_xlabel(xlabel or x_col)
        ax.set_ylabel(ylabel or y_col)
    else:
        plot_func(
            data=df,
            x=y_col,
            y=x_col,
            hue=hue,
            order=order,
            hue_order=hue_order,
            palette=palette,
            ax=ax,
        )
        ax.set_xlabel(ylabel or y_col)
        ax.set_ylabel(xlabel or x_col)

    if title:
        ax.set_title(title)

    if x_col and orient == "vertical":
        if len(df[x_col].unique()) > 5:
            plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


def plot_time_series(
    df: pd.DataFrame,
    time_col: str,
    value_cols: Union[str, List[str]],
    group_col: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
    xlabel: str = "Time",
    ylabel: str = "Value",
    title: Optional[str] = None,
    markers: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    palette: Optional[Union[str, List[str]]] = None,
    legend_loc: str = "best",
    savefig: bool = False,
    filename: str = "time_series.png",
    overwrite: bool = False,
) -> None:
    """
    Create a time series line plot for one or more data series.

    This function creates line plots for time series data, with options for multiple series,
    grouped data, and various customization options. It's useful for visualizing trends,
    patterns, and comparisons in temporal data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data.
    time_col : str
        Column name for time values (x-axis). Should contain datetime-like values.
    value_cols : Union[str, List[str]]
        Column name(s) for values to plot (y-axis). Can be a single column name or a list of column names.
        If multiple columns are provided, each will be plotted as a separate line.
    group_col : Optional[str], default=None
        Column name for grouping data. If provided, the data will be split by this column,
        and each group will be plotted as a separate line with different colors.
    figsize : Tuple[float, float], default=(12, 6)
        Figure size (width, height) in inches.
    xlabel : str, default="Time"
        Label for the x-axis.
    ylabel : str, default="Value"
        Label for the y-axis.
    title : Optional[str], default=None
        Title for the plot. If None, no title is displayed.
    markers : Optional[List[str]], default=None
        List of marker styles for each series. If None, no markers are used.
        If provided, must have at least as many elements as value_cols.
    linestyles : Optional[List[str]], default=None
        List of line styles for each series. If None, solid lines are used.
        If provided, must have at least as many elements as value_cols.
    palette : Optional[Union[str, List[str]]], default=None
        Color palette for the plot. Can be a seaborn/matplotlib colormap name or list of colors.
        If None, the default seaborn color palette is used.
    legend_loc : str, default="best"
        Location of the legend. Common values include "best", "upper left", "upper right",
        "lower left", "lower right", etc.
    savefig : bool, default=False
        Whether to save the figure to a file.
    filename : str, default="time_series.png"
        Name of the file to save the figure to if savefig is True.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
        This function doesn't return any value but shows or saves the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to list if a single column name is provided
    if isinstance(value_cols, str):
        value_cols = [value_cols]

    # If no group column is specified, plot each value column as a separate line
    if group_col is None:
        for i, col in enumerate(value_cols):
            marker = markers[i] if markers and i < len(markers) else None
            ls = linestyles[i] if linestyles and i < len(linestyles) else "-"
            sns.lineplot(
                data=df,
                x=time_col,
                y=col,
                marker=marker,
                linestyle=ls,
                ax=ax,
                label=col,
            )

    # Otherwise, plot each group as a separate line for each value column
    else:
        for col in value_cols:
            sns.lineplot(
                data=df,
                x=time_col,
                y=col,
                hue=group_col,
                marker=markers,
                palette=palette,
                ax=ax,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    if len(df[time_col].unique()) > 10:
        # Show fewer x-ticks for clarity
        plt.locator_params(axis="x", nbins=10)
        plt.xticks(rotation=45)

    plt.legend(loc=legend_loc)
    plt.tight_layout()

    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


def plot_aggregated_ts_frequency(
    df: pd.DataFrame,
    time_col: str = "Time",
    resample_rate: str = "ME",
    figsize: Tuple[float, float] = (10, 6),
    xlabel: str = "Time",
    ylabel: str = "Frequency",
    title: str = "Frequency Over Time",
    grid: bool = True,
    marker: Optional[str] = None,
    linestyle: str = "-",
    color: Optional[str] = None,
    legend_loc: str = "best",
    savefig: bool = False,
    filename: str = "ts_frequency_plot.png",
    overwrite: bool = False,
) -> None:
    """
    Create a plot showing the frequency of events over time by resampling.

    This function takes a DataFrame with time data, resamples it according to the specified
    frequency rule, and plots the count of events in each time period. It's useful for visualizing
    temporal patterns, trends, and seasonality in time series data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data to analyze.
    time_col : str, default="Time"
        Column name for time values. Must contain datetime-convertible values.
    resample_rate : str, default="ME"
        Pandas resample rule defining the frequency of aggregation. Common values include:
        - 'D': Daily
        - 'W': Weekly
        - 'ME' or 'M': Month end
        - 'Q': Quarter end
        - 'YE' or 'A': Year end
        - 'H': Hourly
        - 'T' or 'min': Minute
    figsize : Tuple[float, float], default=(10, 6)
        Figure size (width, height) in inches.
    xlabel : str, default="Time"
        Label for the x-axis (time).
    ylabel : str, default="Frequency"
        Label for the y-axis (count).
    title : str, default="Frequency Over Time"
        Title for the plot.
    grid : bool, default=True
        Whether to display grid lines on the plot.
    marker : Optional[str], default=None
        Marker style for data points. If None, no markers are shown.
    linestyle : str, default="-"
        Line style for the plot. Common values include '-', '--', ':', '-.'.
    color : Optional[str], default=None
        Color for the line. If None, uses the default color cycle.
    legend_loc : str, default="best"
        Location of the legend. Common values include "best", "upper left", "upper right", etc.
    savefig : bool, default=False
        Whether to save the figure to a file.
    filename : str, default="ts_frequency_plot.png"
        Name of the file to save the figure to if savefig is True.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
        This function doesn't return any value but shows or saves the plot.
    """
    # Create a copy to avoid modifying the original dataframe
    temp_df = df.copy()

    # Ensure the time column is datetime type
    temp_df[time_col] = pd.to_datetime(temp_df[time_col])

    # Calculate frequencies by resampling
    frequency = temp_df.resample(resample_rate, on=time_col).size()

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    frequency.plot(ax=ax, marker=marker, linestyle=linestyle, color=color)

    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(grid)

    if len(frequency) > 10:
        plt.xticks(rotation=45)

    plt.legend(["Count"], loc=legend_loc)
    plt.tight_layout()

    # Save or show the figure
    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


def plot_aggregated_ts_frequency_percentage(
    df: pd.DataFrame,
    scale_df: pd.DataFrame,
    time_col: str = "Time",
    resample_rate: str = "ME",
    figsize: Tuple[float, float] = (10, 6),
    xlabel: str = "Time",
    ylabel: str = "(%) Outliers",
    title: str = "Frequency Over Time",
    grid: bool = True,
    marker: Optional[str] = None,
    linestyle: str = "-",
    color: Optional[str] = None,
    legend_loc: str = "best",
    savefig: bool = False,
    filename: str = "ts_frequency_plot.png",
    overwrite: bool = False,
) -> None:
    """
    Create a plot showing the frequency of events over time by resampling and scale by percentage.

    This function takes a DataFrame with time data, resamples it according to the specified
    frequency rule, and plots the count of events in each time period. It's useful for visualizing
    temporal patterns, trends, and seasonality in time series data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data to analyze.
    time_col : str, default="Time"
        Column name for time values. Must contain datetime-convertible values.
    resample_rate : str, default="ME"
        Pandas resample rule defining the frequency of aggregation. Common values include:
        - 'D': Daily
        - 'W': Weekly
        - 'ME' or 'M': Month end
        - 'Q': Quarter end
        - 'YE' or 'A': Year end
        - 'H': Hourly
        - 'T' or 'min': Minute
    figsize : Tuple[float, float], default=(10, 6)
        Figure size (width, height) in inches.
    xlabel : str, default="Time"
        Label for the x-axis (time).
    ylabel : str, default="Frequency"
        Label for the y-axis (count).
    title : str, default="Frequency Over Time"
        Title for the plot.
    grid : bool, default=True
        Whether to display grid lines on the plot.
    marker : Optional[str], default=None
        Marker style for data points. If None, no markers are shown.
    linestyle : str, default="-"
        Line style for the plot. Common values include '-', '--', ':', '-.'.
    color : Optional[str], default=None
        Color for the line. If None, uses the default color cycle.
    legend_loc : str, default="best"
        Location of the legend. Common values include "best", "upper left", "upper right", etc.
    savefig : bool, default=False
        Whether to save the figure to a file.
    filename : str, default="ts_frequency_plot.png"
        Name of the file to save the figure to if savefig is True.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
        This function doesn't return any value but shows or saves the plot.
    """
    # Create a copy to avoid modifying the original dataframe
    temp_df = df.copy()
    temp_scale = scale_df.copy()

    # Ensure the time column is datetime type
    temp_df[time_col] = pd.to_datetime(temp_df[time_col])
    temp_scale[time_col] = pd.to_datetime(temp_scale[time_col])

    # Calculate frequencies by resampling
    frequency = temp_df.resample(resample_rate, on=time_col).size()
    scale = temp_scale.resample(resample_rate, on=time_col).size()

    # Divide frequency by scale and multiply by 100 to get percentage
    frequency = (frequency / scale) * 100

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    frequency.plot(ax=ax, marker=marker, linestyle=linestyle, color=color)

    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(grid)

    if len(frequency) > 10:
        plt.xticks(rotation=45)

    plt.legend(["Count"], loc=legend_loc)
    plt.tight_layout()

    # Save or show the figure
    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


def plot_K_SCORE_comparison_TS(
    df: pd.DataFrame,
    time_col: str,
    score_col: str,
    figsize: Tuple[float, float] = (12, 6),
    xlabel: str = "Time",
    ylabel: str = None,
    title: str = "Score Over Time",
    grid: bool = True,
    marker: Optional[str] = None,
    linestyle: str = "-",
    color: Optional[str] = None,
    legend_loc: str = "best",
    limit_yaxis: bool = False,
    savefig: bool = False,
    filename: str = "pps_time_series.png",
    overwrite: bool = False,
) -> None:
    """
    Create a plot showing the Predictive Power Score (PPS) over time.

    This function takes a DataFrame with time data, and plots the PPS of the target variable
    over time. It's useful for visualizing how the predictability of the target variable changes
    over time, and identifying periods with high or low predictability.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data to analyze.
    time_col : str
        Column name for time values. Must contain datetime-convertible values.
    target_col : str
        Column name for the target variable to predict.
    pps_col : str
        Column name for the Predictive Power Score (PPS) values.
    figsize : Tuple[float, float], default=(10, 6)
        Figure size (width, height) in inches.
    xlabel : str, default="Time"
        Label for the x-axis (time).
    ylabel : str, default="PPS"
        Label for the y-axis (PPS).
    title : str, default="PPS Over Time"
        Title for the plot.
    grid : bool, default=True
        Whether to display grid lines on the plot.
    marker : Optional[str], default=None
        Marker style for data points. If None, no markers are shown.
    linestyle : str, default="-"
        Line style for the plot. Common values include '-', '--', ':', '-.'.
    color : Optional[str], default=None
        Color for the line. If None, uses the default color cycle.
    legend_loc : str, default="best"
        Location of the legend. Common values include "best", "upper left", "upper right", etc.
    savefig : bool, default=False
        Whether to save the figure to a file.
    filename : str, default="pps_time_series.png"
        Name of the file to save the figure to if savefig is True.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
        This function doesn't return any value but shows or saves the plot.
    """
    # Load in df
    temp_df = df.copy()

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    for turbine in temp_df["Turbine"].unique():
        turbine_df = temp_df[temp_df["Turbine"] == turbine]
        for k in turbine_df["K"].unique():
            df_k = turbine_df[turbine_df["K"] == k]
            plt.plot(
                df_k[time_col],
                df_k[score_col],
                label=f"K={k}",
                marker=marker,
                linestyle=linestyle,
                color=color,
            )

    # Customize the plot
    if ylabel == None:
        ylabel = score_col
    # set x-axis tickangle 90 degrees
    plt.xticks(rotation=45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(grid)

    if limit_yaxis:
        # Set y-axis limits between 0 and 1
        ax.set_ylim(0, 1)

    if title:
        ax.set_title(title)

    plt.legend(loc=legend_loc)
    plt.tight_layout()

    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


def plot_weight_for_K_selection(
    x: str,
    y: str,
    x_weight_range: Tuple[float, float],
    y_weight_range: Tuple[float, float],
    result_xlsx_path: str,
    turbine_id: int,
    year: int,
    granularity: Literal["Year", "Quarter", "Month"],
    gran_value: Optional[int] = None,
    ref_point: Tuple[float, float] = [0, 1],
    grid_detail: int = 100,
    tick_skip: int = 2,
    figsize: Tuple[float, float] = (12, 6),
    xlabel: str = None,
    ylabel: str = None,
    title: Optional[str] = "Weight-space for K Selection",
    savefig: bool = False,
    filename: str = "weight_for_K_selection.png",
    overwrite: bool = False,
):
    """

    Args:
        x (str): Name of selected score in K result file
        y (str): Name of selected score in K result file
        x_weight_range (Tuple[float, float]): The weight range to explore for the x variable
        y_weight_range (Tuple[float, float]): The weight range to explore for the y variable
        result_xlsx_path (str): Path to the K result file
        sheet_name (str): Name of the sheet in the K result file (ie "Turbine 9")
        year (int): Selected year
        granularity (Literal[&quot;Year&quot;, &quot;Quarter&quot;, &quot;Month&quot;]):
            Granularity of the K result file. Can be "Year", "Quarter", or "Month".
            If granularity less than year is selected, a gran_value must be provided
        gran_value (Optional[int], optional):
            If granularity is less than year, this value will be used to select the granularity.
            For example, if granularity is "Quarter" and year is 2020, and gran_value is 1,
            then the first quarter of 2020 will be selected.
        ref_point (Tuple[float, float]):
            Reference point for the objective function. This point is used to calculate the distance
            from the reference point to the objective function. Defaults to [0, 1].
        grid_detail (int):
            Number of points to plot in the grid. This number is squared to create a grid. Defaults to 100.
            For example, if grid_detail is 100, then 100x100 points will be plotted.
        tick_skip (int):
            Number of ticks to skip on the x and y axes. This number is used to create a grid of ticks.
            For example, if tick_skip is 2, then every second tick will be shown on the x and y axes.
            Defaults to 2.
        figsize (Tuple[float, float], optional): Adjust figure size for the plot. Defaults to (12, 6).
        xlabel (str, optional):
            Set xlabel on the plot. If no str is provided, the score of the x variabel will be used.
            Defaults to None.
        ylabel (str, optional):
            Set ylabel on the plot. If no str is provided, the score of the y variabel will be used.
            Defaults to None.
        title (Optional[str], optional): Set the figure title. Defaults to "Weight-space for K Selection".
        savefig (bool, optional): Save the figure. Defaults to False.
        filename (str, optional): Set filename for figure. Defaults to "weight_for_K_selection.png".
        overwrite (bool, optional): Overwrite figure if it exists. Defaults to False.
    """

    if granularity != "Year":
        if gran_value == None:
            raise ValueError(
                "If granularity is not Year, gran_value must be provided to select the granularity."
            )
        if granularity == "Quarter":
            if gran_value < 1 or gran_value > 4:
                raise ValueError(
                    "If granularity is Quarter, gran_value must be between 1 and 4."
                )
        elif granularity == "Month":
            if gran_value < 1 or gran_value > 12:
                raise ValueError(
                    "If granularity is Month, gran_value must be between 1 and 12."
                )
    else:
        gran_value = None

    k_finder = KFinder(result_path=result_xlsx_path, turbine_id=turbine_id)
    k_df = k_finder.load_sheet_df()
    k_df = k_df[k_df["Year"] == year]
    if granularity == "Quarter":
        k_df = k_df[k_df["Quarter"] == gran_value]
    elif granularity == "Month":
        k_df = k_df[k_df["Month"] == gran_value]

    # Check if the x and y columns exist in the DataFrame
    if x not in k_df.columns or y not in k_df.columns:
        raise ValueError(f"Columns {x} and {y} must exist in the DataFrame.")

    # Create a grid of weights, based on the specified ranges of the grid_detail
    x_weights = np.linspace(x_weight_range[0], x_weight_range[1], grid_detail)
    y_weights = np.linspace(y_weight_range[0], y_weight_range[1], grid_detail)
    X, Y = np.meshgrid(x_weights, y_weights)
    # Z is the matrix used to store the K values for each point in the grid
    Z = np.zeros(X.shape)

    # Travers all points in the combined grid, and update the Z matrix with the K value
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_weight = X[i, j]
            y_weight = Y[i, j]

            k = k_finder.find_best_gran_k(
                gran_sheet_df=k_df,
                objectives=[x, y],
                weights=[x_weight, y_weight],
                ref_point=ref_point,
            )
            Z[i, j] = k

    fig, ax = plt.subplots(figsize=figsize)

    # Discrete colormap
    num_classes = len(np.unique(Z))
    cmap = plt.get_cmap("Pastel1", num_classes)
    class_values = np.unique(Z).astype(int)
    class_to_color_idx = {val: idx for idx, val in enumerate(class_values)}

    # Cell sizes, we need to calculate the size of each cell in the grid
    # based on the weight ranges and the number of cells in the grid
    y_cells, x_cells = Z.shape
    x_step = (x_weight_range[1] - x_weight_range[0]) / x_cells
    y_step = (y_weight_range[1] - y_weight_range[0]) / y_cells

    # Bottom-left corner
    x_start = x_weight_range[0]
    y_start = y_weight_range[0]

    # Draw patches on the plot for every entry in the Z matrix
    # Each patch is a rectangle with the color corresponding to the K value
    for i in range(y_cells):
        for j in range(x_cells):
            val = int(Z[i, j])
            color = cmap(class_to_color_idx[val])
            rect = patches.Rectangle(
                (x_start + j * x_step, y_start + i * y_step),
                x_step,
                y_step,
                linewidth=1,
                edgecolor="black",
                facecolor=color,
            )
            ax.add_patch(rect)

    # We add the legend, directly mapped from the values in the Z matrix
    legend_handles = [
        patches.Patch(facecolor=cmap(idx), edgecolor="black", label=f"Class {val}")
        for idx, val in enumerate(class_values)
    ]

    # Legend
    ax.legend(
        handles=legend_handles,
        title="K",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
    )

    # Offset tick positions by half a cell to land in the center of each cell. We use
    # this for placing the ticks on the correct position on the x and y axes
    x_centers = np.linspace(
        x_weight_range[0] + x_step / 2,
        x_weight_range[1] - x_step / 2,
        grid_detail,
    )
    y_centers = np.linspace(
        y_weight_range[0] + y_step / 2,
        y_weight_range[1] - y_step / 2,
        grid_detail,
    )

    # Set the ticks on the x and y axes, we skip ticks based on the tick_skip parameter
    # to avoid cluttering the plot
    ax.set_xticks(x_centers[::tick_skip])
    ax.set_xticklabels(
        [f"{val:.2f}" for val in x_weights[::tick_skip]], fontsize=8, rotation=90
    )
    ax.set_yticks(y_centers[::tick_skip])
    ax.set_yticklabels(
        [f"{val:.2f}" for val in y_weights[::tick_skip]], fontsize=8, rotation=0
    )

    # Labels and title
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)

    plt.tight_layout()
    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


def plot_k_selection_history(
    turbine_id: int,
    granularity: Literal["Year", "Quarter", "Month"],
    history_file_path: str = "data/k_filtered_data/Selected_Ks.xlsx",
    figsize: Tuple[float, float] = (12, 6),
    xlabel: str = None,
    ylabel: str = None,
    title: Optional[str] = None,
    savefig: bool = False,
    filename: str = "K selection history",
    overwrite: bool = False,
):
    """Plot a plt step plot for the K selection history of a turbine.

    Args:
        turbine_id (int): The turbine id
        granularity (Literal[&quot;Year&quot;, &quot;Quarter&quot;, &quot;Month&quot;]): granularity in the result file
        history_file_path (str, optional): Path to selection file. Defaults to "data/k_filtered_data/Selected_Ks.xlsx".
        figsize (Tuple[float, float], optional): Fig size. Defaults to (12, 6).
        xlabel (str, optional): Label, if None, the label will be set to "History". Defaults to None.
        ylabel (str, optional): Label, if None, the label will be set to "Selected K". Defaults to None.
        title (Optional[str], optional): Set the figure title. Defaults to None.
        savefig (bool, optional): Save the figure. Defaults to False.
        filename (str, optional): Set filename for figure. Defaults to "K selection history".
        overwrite (bool, optional): Overwrite figure if it exists. Defaults to False.
    """

    sheet_name = f"Turbine {turbine_id}"
    df = pd.read_excel(history_file_path, sheet_name=sheet_name)
    sheet_cols = df.columns

    df = df.drop(sheet_cols[0], axis=1)
    df = df.reset_index(drop=True)
    df["Year"] = df["Year"].astype(int)

    # create cols for plotting
    if granularity == "Year":
        time_col = "Year"
        df = df.sort_values(by=["Year"])
        cols = [col for col in df.columns if col not in ["Year", "K", "Turbine"]]
    elif granularity == "Quarter":
        df["Quarter"] = df["Quarter"].astype(int)
        time_col = "Year_Quarter"
        df[time_col] = df["Year"].astype(str) + " : Q" + df["Quarter"].astype(str)
        df = df.sort_values(by=["Year", "Quarter"])
        # drop month and year
        cols = [col for col in df.columns if col in ["K", "Year_Quarter"]]

    elif granularity == "Month":
        df["Month"] = df["Month"].astype(int)
        time_col = "Year_Month"
        df[time_col] = df["Year"].astype(str) + " : M" + df["Month"].astype(str)
        df = df.sort_values(by=["Year", "Month"])
        cols = [col for col in df.columns if col not in ["K", "Year_Month"]]

    else:
        raise ValueError("Invalid granularity. Choose from 'year', 'quarter', 'month'")

    df = df[cols]

    # Get numeric x positions
    x_vals = np.arange(len(df))
    labels = df[time_col]

    fig, ax = plt.subplots(figsize=figsize)

    # Step plot and fill, using numeric x
    ax.step(x_vals, df["K"], where="post", label="K", color="steelblue")
    ax.fill_between(
        x_vals, df["K"], step="post", alpha=0.4, color="skyblue", label="Filled Area"
    )

    # Set the x-ticks at correct positions (center of each step)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # Labels
    if xlabel is None:
        ax.set_xlabel("Time")
    if ylabel is None:
        ax.set_ylabel("K")
    ax.set_title(title)
    ax.grid(True)
    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


def stable_period_plots(
    df: pd.DataFrame,
    turbine_id: int,
    file_name: str = None,
    save_plot: bool = True,
    overwrite: bool = True,
):
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    years = df["Time"].dt.year.unique()
    years.sort()
    colors = plt.cm.viridis(
        np.linspace(0, 1, len(years))
    )  # Create color map based on number of years

    # Plot 1: GridPower vs PitchAngles
    axs[0].scatter(
        df["PitchAngleA"], df["GridPower"], alpha=0.1, label="PitchAngleA", color="blue"
    )
    axs[0].scatter(
        df["PitchAngleB"],
        df["GridPower"],
        alpha=0.1,
        label="PitchAngleB",
        color="green",
    )
    axs[0].scatter(
        df["PitchAngleC"], df["GridPower"], alpha=0.1, label="PitchAngleC", color="red"
    )
    axs[0].set_xlabel("Pitch Angles")
    axs[0].set_ylabel("Grid Power")
    axs[0].set_title(f"Turbine {turbine_id} - Grid Power vs Pitch Angles")
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: GridPower vs BladeLoads
    # Dividing BladeLoad values by 1e6 for better visualization (values in millions)
    axs[1].scatter(
        df["BladeLoadA"] / 1e6 * -1,
        df["GridPower"],
        alpha=0.1,
        label="BladeLoadA",
        color="blue",
    )
    axs[1].scatter(
        df["BladeLoadB"] / 1e6 * -1,
        df["GridPower"],
        alpha=0.1,
        label="BladeLoadB",
        color="green",
    )
    axs[1].scatter(
        df["BladeLoadC"] / 1e6 * -1,
        df["GridPower"],
        alpha=0.1,
        label="BladeLoadC",
        color="red",
    )
    axs[1].set_xlabel("Blade Loads (millions)")
    axs[1].set_ylabel("Grid Power")
    axs[1].set_title(f"Turbine {turbine_id} - Grid Power vs Blade Loads")
    axs[1].legend()
    axs[1].grid(True)

    # Plot 3: WindSpeed vs yearly GridPower
    for i, year in enumerate(years):
        year_data = df[df["Time"].dt.year == year]
        axs[2].scatter(
            year_data["WindSpeed"],
            year_data["GridPower"],
            alpha=0.3,
            color=colors[i],
            label=str(year),
        )
    # axs[2].scatter(df["WindSpeed"], df["GridPower"], alpha=0.5, color="yellow")
    axs[2].set_xlabel("Wind Speed")
    axs[2].set_ylabel("Grid Power")
    axs[2].set_title(f"Turbine {turbine_id} - Wind Speed vs Grid Power")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    if save_plot:
        if file_name is None:
            file_name = f"turbine_{turbine_id}_grid_vs_pitch.png"
            _savefig_helper(fig, file_name, overwrite)
            file_name = None
        else:
            _savefig_helper(fig, file_name, overwrite)
    else:
        plt.show()
    plt.close(fig)

    # Create a figure with 6 subplots arranged in a 3x3 grid
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))

    # Plot PitchAngles in left column
    axs[0, 0].scatter(df["PitchAngleA"], df["GridPower"], alpha=0.1, color="blue")
    axs[0, 0].set_xlabel("Pitch Angle A")
    axs[0, 0].set_ylabel("Grid Power")
    axs[0, 0].set_title(f"Turbine {turbine_id} - Grid Power vs Pitch Angle A")
    axs[0, 0].grid(True)

    axs[1, 0].scatter(df["PitchAngleB"], df["GridPower"], alpha=0.1, color="green")
    axs[1, 0].set_xlabel("Pitch Angle B")
    axs[1, 0].set_ylabel("Grid Power")
    axs[1, 0].set_title(f"Turbine {turbine_id} - Grid Power vs Pitch Angle B")
    axs[1, 0].grid(True)

    axs[2, 0].scatter(df["PitchAngleC"], df["GridPower"], alpha=0.1, color="red")
    axs[2, 0].set_xlabel("Pitch Angle C")
    axs[2, 0].set_ylabel("Grid Power")
    axs[2, 0].set_title(f"Turbine {turbine_id} - Grid Power vs Pitch Angle C")
    axs[2, 0].grid(True)

    # Plot BladeLoads in center column
    # Dividing BladeLoad values by 1e6 for better visualization (values in millions)
    axs[0, 1].scatter(
        df["BladeLoadA"] / 1e6 * -1, df["GridPower"], alpha=0.1, color="blue"
    )
    axs[0, 1].set_xlabel("Blade Load A (millions)")
    axs[0, 1].set_ylabel("Grid Power")
    axs[0, 1].set_title(f"Turbine {turbine_id} - Grid Power vs Blade Load A")
    axs[0, 1].grid(True)

    axs[1, 1].scatter(
        df["BladeLoadB"] / 1e6 * -1, df["GridPower"], alpha=0.1, color="green"
    )
    axs[1, 1].set_xlabel("Blade Load B (millions)")
    axs[1, 1].set_ylabel("Grid Power")
    axs[1, 1].set_title(f"Turbine {turbine_id} - Grid Power vs Blade Load B")
    axs[1, 1].grid(True)

    axs[2, 1].scatter(
        df["BladeLoadC"] / 1e6 * -1, df["GridPower"], alpha=0.1, color="red"
    )
    axs[2, 1].set_xlabel("Blade Load C (millions)")
    axs[2, 1].set_ylabel("Grid Power")
    axs[2, 1].set_title(f"Turbine {turbine_id} - Grid Power vs Blade Load C")
    axs[2, 1].grid(True)

    # Plot BladeLoads in right column
    # Plot data by year in right column
    # Extract unique years from the Time column

    # Plot BladeLoadA vs GridPower by year
    for i, year in enumerate(years):
        year_data = df[df["Time"].dt.year == year]
        axs[0, 2].scatter(
            year_data["BladeLoadA"] / 1e6 * -1,
            year_data["GridPower"],
            alpha=0.3,
            color=colors[i],
            label=str(year),
        )
    axs[0, 2].set_xlabel("Blade Load A (millions)")
    axs[0, 2].set_ylabel("Grid Power")
    axs[0, 2].set_title(f"Turbine {turbine_id} - Grid Power vs Blade Load A by Year")
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # Plot BladeLoadB vs GridPower by year
    for i, year in enumerate(years):
        year_data = df[df["Time"].dt.year == year]
        axs[1, 2].scatter(
            year_data["BladeLoadB"] / 1e6 * -1,
            year_data["GridPower"],
            alpha=0.3,
            color=colors[i],
            label=str(year),
        )
    axs[1, 2].set_xlabel("Blade Load B (millions)")
    axs[1, 2].set_ylabel("Grid Power")
    axs[1, 2].set_title(f"Turbine {turbine_id} - Grid Power vs Blade Load B by Year")
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    # Plot BladeLoadC vs GridPower by year
    for i, year in enumerate(years):
        year_data = df[df["Time"].dt.year == year]
        axs[2, 2].scatter(
            year_data["BladeLoadC"] / 1e6 * -1,
            year_data["GridPower"],
            alpha=0.3,
            color=colors[i],
            label=str(year),
        )
    axs[2, 2].set_xlabel("Blade Load C (millions)")
    axs[2, 2].set_ylabel("Grid Power")
    axs[2, 2].set_title(f"Turbine {turbine_id} - Grid Power vs Blade Load C by Year")
    axs[2, 2].legend()
    axs[2, 2].grid(True)

    plt.tight_layout()
    if save_plot:
        if file_name is None:
            file_name = f"turbine_{turbine_id}_stable_period_plots_3x3.png"
        _savefig_helper(fig, file_name, overwrite)
    else:
        plt.show()

    plt.close(fig)


def qq_plot(
    data: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    actuals: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    predictions: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    theoretical_dist: Union[str, Callable] = "norm",
    figsize: Tuple[float, float] = (10, 6),
    xlabel: str = "Theoretical Quantiles",
    ylabel: str = "Sample Quantiles",
    title: Optional[str] = None,
    line: bool = True,
    line_color: str = "r",
    marker: str = "o",
    marker_color: str = "blue",
    alpha: float = 0.7,
    grid: bool = True,
    savefig: bool = False,
    filename: str = "qq_plot.png",
    overwrite: bool = False,
) -> None:
    """
    Create a Quantile-Quantile (QQ) plot to compare sample data against a theoretical distribution
    or to compare actual values against predicted values.

    Parameters
    ----------
    data : Optional[Union[pd.Series, np.ndarray, List[float]]], default=None
        The sample data to be plotted on the y-axis when comparing against a theoretical distribution.
        Not used when both actuals and predictions are provided.
    actuals : Optional[Union[pd.Series, np.ndarray, List[float]]], default=None
        The actual values to be plotted on the y-axis when comparing against predictions.
    predictions : Optional[Union[pd.Series, np.ndarray, List[float]]], default=None
        The predicted values to be plotted on the x-axis when comparing against actuals.
    theoretical_dist : Union[str, Callable], default="norm"
        The distribution to compare against when using the 'data' parameter. Can be:
        - "norm": Normal distribution (default)
        - "uniform": Uniform distribution
        - "exp": Exponential distribution
        - Callable: A custom distribution function that takes a single parameter (the quantiles)
    figsize : Tuple[float, float], default=(10, 6)
        Figure size (width, height) in inches.
    xlabel : str, default="Theoretical Quantiles"
        Label for the x-axis.
    ylabel : str, default="Sample Quantiles"
        Label for the y-axis.
    title : Optional[str], default=None
        Title for the plot. If None, a default title will be generated.
    line : bool, default=True
        Whether to plot a reference line. If comparing with a theoretical distribution,
        this is the line y=x. If comparing actuals and predictions, this is a line through their quantiles.
    line_color : str, default="r"
        Color of the reference line.
    marker : str, default="o"
        Marker style for data points.
    marker_color : str, default="blue"
        Color of the markers.
    alpha : float, default=0.7
        Transparency of markers (0.0 is completely transparent, 1.0 is opaque).
    grid : bool, default=True
        Whether to display grid lines on the plot.
    savefig : bool, default=False
        Whether to save the figure to a file.
    filename : str, default="qq_plot.png"
        Name of the file to save the figure to if savefig is True.
    overwrite : bool, default=False
        Whether to overwrite the file if it already exists.

    Returns
    -------
    None
        This function doesn't return any value but shows or saves the plot.
    """
    # Check input parameters
    if data is None and (actuals is None or predictions is None):
        raise ValueError(
            "Either 'data' or both 'actuals' and 'predictions' must be provided"
        )

    if data is not None and (actuals is not None or predictions is not None):
        raise ValueError(
            "Either provide 'data' or both 'actuals' and 'predictions', but not both"
        )

    fig, ax = plt.subplots(figsize=figsize)

    # Case 1: Comparing actuals and predictions
    if actuals is not None and predictions is not None:
        # Convert input data to numpy arrays
        if isinstance(actuals, pd.Series):
            actuals = actuals.values
        elif isinstance(actuals, list):
            actuals = np.array(actuals)

        if isinstance(predictions, pd.Series):
            predictions = predictions.values
        elif isinstance(predictions, list):
            predictions = np.array(predictions)

        # Sort the data
        actuals_sorted = np.sort(actuals)
        predictions_sorted = np.sort(predictions)

        # Plot the QQ points
        ax.scatter(
            predictions_sorted,
            actuals_sorted,
            color=marker_color,
            marker=marker,
            alpha=alpha,
        )

        # Add the reference line if requested
        if line:
            # For two samples, fit a line through their quantiles
            slope, intercept, _, _, _ = linregress(predictions_sorted, actuals_sorted)
            min_val = min(np.min(predictions_sorted), np.min(actuals_sorted))
            max_val = max(np.max(predictions_sorted), np.max(actuals_sorted))
            ax.plot(
                [min_val, max_val],
                [intercept + slope * min_val, intercept + slope * max_val],
                color=line_color,
                linestyle="-",
            )

        # Set default labels if not provided
        if xlabel == "Theoretical Quantiles":
            xlabel = "Predicted Quantiles"
        if ylabel == "Sample Quantiles":
            ylabel = "Actual Quantiles"
        if title is None:
            title = "Q-Q Plot: Actuals vs Predictions"

    # Case 2: Comparing data with a theoretical distribution
    else:
        # Convert input data to numpy array
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)

        # Sort the sample data
        data = np.sort(data)
        n = len(data)
        print(f"The length of data is {n}")

        # Handle different types of theoretical distributions
        if isinstance(theoretical_dist, str):
            # For standard theoretical distributions
            if theoretical_dist.lower() == "norm":
                theoretical_quantiles = norm.ppf(np.arange(1, n + 1) / (n + 1))
                if title is None:
                    title = "Normal Q-Q Plot"
            elif theoretical_dist.lower() == "uniform":
                theoretical_quantiles = np.linspace(0, 1, n)
                if title is None:
                    title = "Uniform Q-Q Plot"
            elif theoretical_dist.lower() == "exp":
                theoretical_quantiles = expon.ppf(np.arange(1, n + 1) / (n + 1))
                if title is None:
                    title = "Exponential Q-Q Plot"
            else:
                raise ValueError(
                    f"Unsupported theoretical distribution: {theoretical_dist}"
                )
        elif callable(theoretical_dist):
            # For custom distribution functions
            theoretical_quantiles = theoretical_dist(np.arange(1, n + 1) / (n + 1))
            if title is None:
                title = "Custom Distribution Q-Q Plot"

        # Plot the QQ points
        ax.scatter(
            theoretical_quantiles, data, color=marker_color, marker=marker, alpha=alpha
        )

        # Add the reference line if requested
        if line:
            # For theoretical distributions, plot y=x line
            min_val = min(np.min(theoretical_quantiles), np.min(data))
            max_val = max(np.max(theoretical_quantiles), np.max(data))
            ax.plot(
                [min_val, max_val], [min_val, max_val], color=line_color, linestyle="-"
            )

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Add grid if requested
    ax.grid(grid)

    plt.tight_layout()

    # Save or display the figure
    if savefig:
        _savefig_helper(fig, filename, overwrite)
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
   pass