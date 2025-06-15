"""
Cut Values Analyzer and Visualizer

This module provides a comprehensive class for analyzing and visualizing
cut values data generated from hyperparameter testing in Simulated Bifurcation algorithms.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure

import visualize
from visualize import to_latex


class CutValuesAnalyzer:
    """
    A comprehensive analyzer for cut values data from Simulated Bifurcation experiments.

    This class provides methods to load, process, and visualize cut values data
    generated from hyperparameter testing experiments.
    """

    def __init__(self, data_path: str, methods: list[str] = ["bSB", "dSB", "sSB"]):
        """
        Initialize the CutValuesAnalyzer.

        Args:
            data_path: path to the cut values pickle file
        """
        self.methods = methods

        self.num_seeds = 128  # Default number of seeds

        with open(data_path, "rb") as f:
            self.cut_values = pickle.load(f)

        self.cut_values_df = pd.DataFrame(self.cut_values).map(lambda x: [np.max(x[i]) for i in range(self.num_seeds)])
        self.cut_values_df.index.name = "method"
        self.cut_values_df.columns.names = ("beta", "eta")

        self.cut_values_mean_df = self.cut_values_df.map(np.mean)
        self.cut_values_std_df = self.cut_values_df.map(np.std)

    @property
    def num_methods(self) -> int:
        """
        Get the number of methods used in the analysis.

        Returns:
            int: Number of methods
        """
        return len(self.methods)

    def get_cut_histories(self, beta: float, eta: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Get mean and standard deviation of cut histories for specific beta and eta values.

        Args:
            beta: Annealing rate parameter value
            eta: Time step value

        Returns:
            tuple of (mean_histories, std_histories) dictionaries
        """
        if self.cut_values is None:
            raise ValueError("No data loaded. Call load_data() first.")

        cut_histories_means = {}
        cut_histories_std = {}

        for method in self.methods:
            h = self.cut_values[(beta, eta)][method]
            h_ = np.empty((len(h), max(map(len, h.values()))), dtype=np.int32)

            for i, (k, v) in enumerate(h.items()):
                h_[i, : len(v)] = v
                h_[i, len(v) :] = v[-1]  # Pad with last value

            cut_histories_means[method] = h_.mean(0)
            cut_histories_std[method] = h_.std(0)

        return cut_histories_means, cut_histories_std

    def plot_average_cut_history(
        self,
        beta: float,
        eta: float,
        best_cut: int | None = None,
        save_path: str | None = None,
    ) -> tuple[Axes, Axes | None]:
        """
        Plot average cut value history for a specific beta and eta combination.

        Args:
            beta: Annealing rate parameter value
            eta: Time step value
            best_cut: Optional best cut value to display as reference line
            save_path: Optional path to save the figure
            figsize: Figure size tuple

        Returns:
            tuple of (main_axis, inset_axis)
        """
        cut_histories_means, cut_histories_std = self.get_cut_histories(beta, eta)

        tmax = max(map(len, cut_histories_means.values())) * eta
        tmin = tmax * 0.8

        # Determine y-axis limits for inset
        cut_max = max((cut_histories_means[m][-1] + cut_histories_std[m][-1]) for m in self.methods)
        cut_min = min((cut_histories_means[m][-1] - cut_histories_std[m][-1]) for m in self.methods)
        if best_cut is not None:
            cut_max = max(cut_max, best_cut)
        cut_rng = cut_max - cut_min
        inset_y = (cut_min - cut_rng * 0.2, cut_max + cut_rng * 0.2)

        # Plot using visualize module
        ax, axin = visualize.plot_history_compare(
            eta=eta,
            histories=cut_histories_means,
            histories_std=cut_histories_std,
            alpha=0.6,
            xlim=(0.1, None),
            inset_zoom=((tmin, tmax), inset_y),
            show_max=True,
            max_text_offset={"bSB": -10, "dSB": 0, "sSB": 10},
        )

        # Customize the plot
        ax.set_ylabel("Cut Value", fontsize=13)
        if axin is not None:
            axin.set_ylabel("")
        ax.set_title(rf"Average Cut Value History ($\beta={to_latex(beta)}$, $\eta={to_latex(eta)}$)", fontsize=16)

        # Add best cut reference line if provided
        if best_cut is not None and axin is not None:
            axin.axhline(best_cut, label="Best cut", color="black", linestyle="-", alpha=0.8, linewidth=1)
            axin.text(tmax, best_cut, f" Best cut={best_cut}", color="black", fontsize=10, ha="left", va="center", alpha=0.8)

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return ax, axin

    def plot_hyperparameter_heatmap(
        self,
        save_path: str | None = None,
        figsize: tuple[int, int] = (13, 9),
        cmap: Colormap = visualize.default_cmap,
    ) -> Figure:
        """
        Plot comprehensive hyperparameter heatmap showing method comparison.

        Args:
            save_path: Optional path to save the figure
            figsize: Figure size tuple

        Returns:
            matplotlib Figure object
        """
        if self.cut_values_mean_df is None:
            raise ValueError("No processed data available. Call load_data() first.")

        # Get best performing method for each parameter combination
        best_method = self.cut_values_mean_df.idxmax(axis=0).unstack(level="eta").map(self.methods.index).values

        vmin = self.cut_values_mean_df.values.min()
        vmax = self.cut_values_mean_df.values.max()

        # Get parameter values
        midx = self.cut_values_df.columns
        beta_values = midx.get_level_values("beta").unique()
        eta_values = midx.get_level_values("eta").unique()

        # Set up figure with grid layout
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, width_ratios=[3, 1.2])

        bgcolor = fig.get_facecolor()
        textcolor = "black" if np.array(bgcolor).mean() > 0.5 else "white"

        if isinstance(cmap, LinearSegmentedColormap):
            colors = cmap(np.arange(self.num_methods) / self.num_methods)
            colors[:, 3] = 0.6
            cmap = ListedColormap(colors)
        assert isinstance(cmap, ListedColormap)

        cmaps = [
            LinearSegmentedColormap.from_list(
                f"cm_{method}",
                [bgcolor, cmap(i / self.num_methods)],
            )
            for i, method in enumerate(self.methods)
        ]

        # Create the best method plot on the left (spans all rows)
        ax_best = fig.add_subplot(gs[:, 0])

        # Create three smaller plots on the right for individual methods
        axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]

        # Plot heatmaps for each method on the right
        for i, (ax, method, cm) in enumerate(zip(axes[::-1], self.methods, cmaps)):
            im = ax.imshow(
                self.cut_values_mean_df.loc[method].unstack(level="eta"),
                cmap=cm,
                aspect="auto",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )

            # Set ticks and labels
            if i == 0:  # Only show x-axis labels on bottom plot
                ax.set_xticks(range(len(eta_values)))
                ax.set_xticklabels([f"${to_latex(eta)}$" for eta in eta_values])
            else:
                ax.set_xticks([])

            ax.set_yticks(range(len(beta_values)))
            ax.set_yticklabels([f"${to_latex(beta)}$" for beta in beta_values])

            # Add colorbar
            fig.colorbar(im, ax=ax)

            # Add value annotations
            mean_value_matrix = self.cut_values_mean_df.loc[method].unstack(level="eta")
            for ii in range(len(beta_values)):
                for jj in range(len(eta_values)):
                    ax.text(
                        x=jj,
                        y=ii,
                        s=f"{mean_value_matrix.iat[ii, jj]:.0f}",
                        ha="center",
                        va="center",
                        color=textcolor,
                        fontsize=9,
                    )

        # Plot the best method heatmap on the left
        im_best = ax_best.imshow(
            best_method,
            cmap=cmap,
            aspect="equal",
            origin="lower",
        )

        ax_best.set_title("Best Performing Method", fontsize=18)
        ax_best.set_xticks(range(len(eta_values)))
        ax_best.set_yticks(range(len(beta_values)))
        ax_best.set_xticklabels([f"${to_latex(eta)}$" for eta in eta_values])
        ax_best.set_yticklabels([f"${to_latex(beta)}$" for beta in beta_values])

        # Add value annotations for best method plot
        for ii in range(len(beta_values)):
            for jj in range(len(eta_values)):
                best_idx = best_method[ii, jj]
                best_value = self.cut_values_mean_df.iloc[best_idx].unstack(level="eta").iloc[ii, jj]
                ax_best.text(
                    jj,
                    ii,
                    f"{best_value:.0f}",
                    ha="center",
                    va="center",
                    color=textcolor,
                    fontsize=12,
                )

        # Add colorbar with method names for the best method plot
        cbar_best = fig.colorbar(
            im_best,
            ax=ax_best,
            ticks=(np.arange(self.num_methods) + 0.5) * (1 - 1 / self.num_methods),
        )
        cbar_best.set_ticklabels(self.methods, fontsize=14)

        # Add labels
        ax_best.set_xlabel(r"$\eta$ (delta time)", fontsize=16)
        ax_best.set_ylabel(r"$\beta$ (annealing rate)", fontsize=16)

        # Overall title
        fig.suptitle("Max Cut Optimization Performance Comparison", fontsize=20, y=0.98)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def get_best_method_summary(self) -> pd.DataFrame:
        """
        Get a summary of the best performing method for each parameter combination.

        Returns:
            DataFrame showing best method for each (beta, eta) combination
        """
        if self.cut_values_mean_df is None:
            raise ValueError("No processed data available. Call load_data() first.")

        return self.cut_values_mean_df.idxmax(axis=0).unstack(level="eta")

    def get_performance_statistics(self) -> dict[str, pd.DataFrame]:
        """
        Get performance statistics (mean and std) for all methods and parameters.

        Returns:
            dictionary with 'mean' and 'std' DataFrames
        """
        if self.cut_values_mean_df is None or self.cut_values_std_df is None:
            raise ValueError("No processed data available. Call load_data() first.")

        return {"mean": self.cut_values_mean_df, "std": self.cut_values_std_df}

    def print_method_performance(self, beta: float, eta: float) -> None:
        """
        Print performance summary for a specific parameter combination.

        Args:
            beta: Annealing rate parameter value
            eta: Time step value
        """
        if self.cut_values is None:
            raise ValueError("No data loaded. Call load_data() first.")

        print(f"Performance for beta={beta}, eta={eta}:")
        print("-" * 40)

        for method in self.methods:
            h = self.cut_values[(beta, eta)][method]
            max_cuts = [max(h[i]) for i in range(len(h))]
            mean_performance = np.mean(max_cuts)
            std_performance = np.std(max_cuts)

            print(f"{method}: {mean_performance:.2f} ± {std_performance:.2f}")


# Example usage functions
def create_standard_plots(
    data_path: str,
    beta: float,
    eta: float,
    best_cut: int,
    save_prefix: str | None,
) -> None:
    """
    Create standard analysis plots for cut values data.

    Args:
        data_path: Path to the cut values pickle file
        beta: Annealing rate parameter value
        eta: Time step value
        output_dir: Directory to save output figures
    """
    # Create output directory if it doesn't exist

    # Load and analyze data
    analyzer = CutValuesAnalyzer(data_path)

    # Create history plot
    analyzer.plot_average_cut_history(
        beta=beta,
        eta=eta,
        best_cut=best_cut,  # Known best cut for the K2000 problem
        save_path=f"{save_prefix}average_cut_history.png" if save_prefix else None,
    )

    # Create hyperparameter heatmap
    analyzer.plot_hyperparameter_heatmap(save_path=f"{save_prefix}hyperparameter_heatmap.png" if save_prefix else None)

    # Print performance summary
    analyzer.print_method_performance(beta=beta, eta=eta)

    # Print best method summary
    best_methods = analyzer.get_best_method_summary()
    print("\nBest method for each parameter combination:")
    print(best_methods)


if __name__ == "__main__":
    # Example usage
    data_file = "cut_values/betas=[0.00048828 0.00097656 0.00195312 0.00390625 0.0078125 ]_etas=[0.0625 0.125  0.25   0.5    1.    ].pkl"

    if Path(data_file).exists():
        create_standard_plots(
            data_file,
            beta=2**-11,
            eta=2**-1,
            best_cut=33337,
            save_prefix="figures/",
        )
    else:
        print(f"Data file {data_file} not found. Please check the path.")
