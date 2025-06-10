"""
Simulated Bifurcation Solver
Encapsulates the run function from core.py, allowing storage of parameters and results.
Provides convenient plotting methods for visualization.
"""

from typing import Any, Callable, Self

import numpy as np
import torch
from matplotlib.axes import Axes

import visualize
from core import MethodType, SBHistoryArray, run


class Solver:
    """
    Simulated Bifurcation Solver class
    Encapsulates the run function from core.py, allowing storage of parameters and results.
    Provides convenient plotting methods for trajectory and time histogram visualization.
    """

    def __init__(self):
        self.params: dict[str, Any] = {}
        self._result: SBHistoryArray | None = None
        self._J: np.ndarray | None = None

    def solve(
        self,
        J: torch.Tensor | np.ndarray,
        method: MethodType = "bSB",
        *,
        beta: float = 2**-11,
        xi: float | None = None,
        eta: float = 2**-3,
        max_steps: int | None = None,
        seed: int = 42,
        progress_bar: bool = False,
    ) -> SBHistoryArray:
        """
        Run the Simulated Bifurcation algorithm to solve the Ising model.

        Parameters
        ----------
        J : torch.Tensor or np.ndarray
            Input Ising matrix (negative of the coupling matrix).
        method : str
            The method to use. Options are "aSB", "bSB", "dSB", "sSB", "sSB_clip".
        beta : float
            Growth rate of p(t).
        xi : float
            Coupling strength. Defaults to 1 / (2 * sqrt(N)) if None.
        eta : float
            Time step size.
        max_steps : int
            Maximum number of steps. Defaults to int(2 / beta / eta) if None.
        seed : int
            Random seed.
        progress_bar : bool
            Whether to display a progress bar.

        Returns
        -------
        SBHistoryArray
            The result of the solve.
        """
        # Store original J matrix for plotting
        if isinstance(J, np.ndarray):
            self._J = J.copy()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            J_torch = torch.tensor(J, dtype=torch.float32, device=device)
        else:
            self._J = J.cpu().numpy()
            J_torch = J

        # Apply defaults for None values (same as core.py)
        if xi is None:
            xi = 1 / 2 / J_torch.shape[0] ** 0.5
        if max_steps is None:
            max_steps = int(2 / beta / eta)

        # Store parameters with actual values (not None)
        self.params = {
            "method": method,
            "beta": beta,
            "xi": xi,
            "eta": eta,
            "max_steps": max_steps,
            "seed": seed,
            "progress_bar": progress_bar,
        }

        result = run(
            J_torch,
            method=method,
            beta=beta,
            xi=xi,
            eta=eta,
            max_steps=max_steps,
            seed=seed,
            progress_bar=progress_bar,
        ).numpy()

        self._result = result

        return result

    @property
    def result(self) -> SBHistoryArray:
        """
        Get the result of the last solve.

        Returns
        -------
        SBHistoryArray
            The result of the last solve, or None if no result exists.
        """
        if self._result is None:
            raise ValueError("No result exists. Please run the solver first.")
        return self._result

    def get_best_solution(self) -> np.ndarray:
        """
        Get the current best solution.

        Returns
        -------
        np.ndarray
            The best solution's sign values (+1 or -1), or None if no result exists.
        """
        return self.result.best_x

    def get_best_cut(self) -> float:
        """
        Get the current best cut value.

        Returns
        -------
        float
            The best cut value.
        """
        if self.result.cut is None:
            raise ValueError("cut is None")
        return self.result.cut.max()

    def _get_energy_fn(self, dim0: int, dim1: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        Create energy function for the specific method and dimensions.

        Parameters
        ----------
        dim0, dim1 : int
            The dimensions to plot

        Returns
        -------
        Callable
            Energy function that takes (x0, x1) and returns energy
        """
        if self._J is None:
            raise ValueError("No J matrix stored. Please run solve() first.")

        J = self._J
        N = J.shape[0]
        x_other_dim = [i for i in range(N) if i not in [dim0, dim1]]
        coeff01: float = J[dim0, dim1]
        xi: float = self.params["xi"]  # xi should never be None now after solve() processing
        beta: float = self.params["beta"]
        eta: float = self.params["eta"]
        method: MethodType = self.params["method"]

        def energy_fn_aSB(x0, x1):
            V = 1 / 4 * (x0**4 + x1**4)
            t = len(self.result.x) * eta
            V += 1 / 2 * (1 - beta * t) * (x0**2 + x1**2)
            coeff0, coeff1 = J[[dim0, dim1], :][:, x_other_dim] @ self.result.x[-1, x_other_dim]
            V -= xi * (coeff01 * x0 * x1 + coeff0 * x0 + coeff1 * x1)
            return V

        def energy_fn_bSB(x0, x1):
            t = len(self.result.x) * eta
            V = 1 / 2 * (1 - beta * t) * (x0**2 + x1**2)
            coeff0, coeff1 = J[[dim0, dim1], :][:, x_other_dim] @ self.result.x[-1, x_other_dim]
            V -= xi * (coeff01 * x0 * x1 + coeff0 * x0 + coeff1 * x1)
            return V

        def energy_fn_dSB(x0, x1):
            t = len(self.result.x) * eta
            V = 1 / 2 * (1 - beta * t) * (x0**2 + x1**2)
            V -= xi * (coeff01 * x0 * x1)
            coeff0, coeff1 = J[[dim0, dim1], :][:, x_other_dim] @ self.result.x[-1, x_other_dim]
            V -= (coeff0 * np.sign(x0) + coeff1 * np.sign(x1)) / 2
            coeff0, coeff1 = J[[dim0, dim1], :][:, x_other_dim] @ np.sign(self.result.x[-1, x_other_dim])
            V -= (coeff0 * x0 + coeff1 * x1) / 2
            return V

        match method:
            case "aSB":
                return energy_fn_aSB
            case "bSB" | "sSB" | "sSB_clip":
                return energy_fn_bSB
            case "dSB":
                return energy_fn_dSB
            case _:
                raise ValueError(f"Unknown method: {method}. Supported methods are 'aSB', 'bSB', 'dSB', 'sSB'.")

    def plot_trajectory(self, ax: Axes | None = None, dim0: int = 0, dim1: int = 1, n_arrows: int = 100, show_energy: bool = True) -> Axes:
        """
        Plot the trajectory in 2D phase space.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses current axes.
        dim0, dim1 : int
            Dimensions to plot (default: 0, 1)
        n_arrows : int
            Number of arrows to show direction (default: 100)
        show_energy : bool
            Whether to show energy contours (default: True)

        Returns
        -------
        Axes
            The matplotlib axes with the plot
        """
        if self._result is None:
            raise ValueError("No result exists. Please run solve() first.")

        eta = self.params["eta"]
        energy_fn = self._get_energy_fn(dim0, dim1) if show_energy else None

        return visualize.plot_trajectory(
            ax=ax,
            eta=eta,
            x_history=self.result.x,
            dim0=dim0,
            dim1=dim1,
            energy_fn=energy_fn,
            n_arrows=n_arrows,
        )

    def plot_time_hist(self, ax: Axes | None = None, n_slice: int = 2000, n_bins: int = 80, vmax: float = 0.05, xlim_factor: float = 5.0) -> Axes:
        """
        Plot time histogram of the solution trajectory.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, uses current axes.
        n_slice : int
            Number of time slices (default: 2000)
        n_bins : int
            Number of histogram bins (default: 80)
        vmax : float
            Maximum value for colormap (default: 0.05)
        xlim_factor : float
            Factor to multiply time range for x-axis limit (default: 5.0)

        Returns
        -------
        Axes
            The matplotlib axes with the plot
        """
        if self._result is None:
            raise ValueError("No result exists. Please run solve() first.")

        eta = self.params["eta"]

        ax = visualize.plot_time_hist(
            ax=ax,
            eta=eta,
            history=self.result.x,
            n_slice=n_slice,
            n_bins=n_bins,
            vmax=vmax,
        )

        ax.set_ylabel("x")
        ax.set_xlim(0, len(self.result.x) * eta * xlim_factor)

        return ax

    def plot_history(
        self,
        axes: list[Axes] | None = None,
        title: str | None = None,
        color: str = "purple",
        alpha: float = 0.9,
        t_range: slice | None = None,
        dim_range: slice | range | None = None,
        ylabel: bool = True,
    ) -> list[Axes]:
        """
        Plot complete history of x, y, g, V, H over time.

        Parameters
        ----------
        axes : list[Axes], optional
            List of 4 axes for plotting. If None, creates new subplots.
        title : str, optional
            Title for the plots. Defaults to method name.
        color : str
            Color for the plots (default: "purple")
        alpha : float
            Alpha transparency (default: 0.9)
        t_range : slice, optional
            Time range to plot
        dim_range : slice or range, optional
            Dimension range to plot (default: first 20 dimensions)
        ylabel : bool
            Whether to add y-axis labels (default: True)

        Returns
        -------
        list[Axes]
            List of matplotlib axes with the plots
        """
        if self._result is None:
            raise ValueError("No result exists. Please run solve() first.")

        if title is None:
            title = self.params["method"]

        # Type assertion: title is guaranteed to be a string after the check above
        assert isinstance(title, str), "title must be a string"

        # Ensure H is not None
        H_history = self.result.H
        if H_history is None:
            raise ValueError("H_history is None. This may indicate an issue with the solve process.")

        return visualize.plot_history(
            title=title,
            eta=self.params["eta"],
            x_history=self.result.x,
            y_history=self.result.y,
            g_history=self.result.g,
            V_history=self.result.V,
            H_history=H_history,
            color=color,
            alpha=alpha,
            axes=axes,
            t_range=t_range,
            dim_range=dim_range,
            ylabel=ylabel,
        )

    def get_cut_history(self) -> np.ndarray:
        """
        Get the cut value history.

        Returns
        -------
        np.ndarray
            Array of cut values over time
        """
        if self.result.cut is None:
            raise ValueError("cut is None")
        return self.result.cut

    def get_convergence_time(self) -> float:
        """
        Estimate convergence time based on when the solution stops changing significantly.

        Returns
        -------
        float
            Estimated convergence time
        """
        return len(self.result.x) * self.params["eta"]

    def get_final_energy(self) -> float:
        """
        Get the final energy value.

        Returns
        -------
        float
            Final energy value
        """
        if self._result is None:
            raise ValueError("No result exists. Please run solve() first.")
        return float(self.result.V[-1])

    def summary(self) -> dict:
        """
        Get a summary of the solver results.

        Returns
        -------
        dict
            Dictionary containing key metrics and parameters
        """
        if self._result is None:
            raise ValueError("No result exists. Please run solve() first.")

        return {
            "method": self.params["method"],
            "parameters": {k: v for k, v in self.params.items() if k != "method"},
            "best_cut": self.get_best_cut(),
            "final_energy": self.get_final_energy(),
            "convergence_time": self.get_convergence_time(),
            "total_steps": len(self.result.x),
            "total_time": len(self.result.x) * self.params["eta"],
        }

    def compare_with(self, other_solver: Self) -> dict:
        """
        Compare this solver's results with another solver.

        Parameters
        ----------
        other_solver : Solver
            Another Solver instance to compare with

        Returns
        -------
        dict
            Comparison results
        """
        if self._result is None or other_solver._result is None:
            raise ValueError("Both solvers must have results to compare.")

        self_summary = self.summary()
        other_summary = other_solver.summary()

        return {
            "methods": (self_summary["method"], other_summary["method"]),
            "cut_difference": self_summary["best_cut"] - other_summary["best_cut"],
            "energy_difference": self_summary["final_energy"] - other_summary["final_energy"],
            "convergence_time_difference": self_summary["convergence_time"] - other_summary["convergence_time"],
            "better_solver": self_summary["method"] if self_summary["best_cut"] > other_summary["best_cut"] else other_summary["method"],
        }
