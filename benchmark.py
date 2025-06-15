import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure

import visualize
from core import MethodType
from solver import Solver
from visualize import BoundsType, KeyType


class Benchmark:
    def __init__(
        self,
        methods: list[MethodType],
    ):
        self.methods = methods
        self.solvers: dict[str, Solver] = {}
        self.J: np.ndarray | None = None
        self.params: dict[str, float | int] = {}

    def run(
        self,
        J: np.ndarray,
        beta: float,
        eta: float,
        xi: float | None = None,
        seed: int = 0,
        verbose: bool = False,
        device: str | None = None,
    ) -> dict[str, Solver]:
        self.J = J.copy()
        self.params = {"beta": beta, "eta": eta, "seed": seed, "N": J.shape[0]}
        if verbose:
            print("Starting SB algorithms comparison...")
            print("=" * 50)
            print(f"Problem size: {J.shape[0]}×{J.shape[0]}")
            print(f"Parameters: β={beta}, η={eta}, seed={seed}")
            print(f"Methods: {', '.join(self.methods)}")
            print("=" * 50)
        for m in self.methods:
            s = Solver()
            s.solve(J, method=m, beta=beta, eta=eta, xi=xi, seed=seed, progress_bar=verbose, device=device)
            self.solvers[m] = s
        if verbose:
            print("=" * 50)
            print("🎉 All algorithms completed!")
        return self.solvers

    def summary(self) -> pd.DataFrame:
        if not self.solvers:
            raise ValueError("No algorithms have been run yet. Call run() first.")
        df = pd.DataFrame([
            {
                "Alg": s["method"],
                "Best": s["best_cut"],
                "Energy": s["final_energy"],
                "Steps": s["total_steps"],
            }
            for s in (sol.summary() for sol in self.solvers.values())
        ])
        df.sort_values(by="Best", ascending=False, inplace=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
        df.set_index("Rank", inplace=True)
        return df

    def plot_traj(
        self,
        figsize: tuple[int, int] | None = None,
        interval: tuple[float, float] | None = None,
        dims: tuple[int, int] = (0, 1),
        n_arrows: int = 100,
        n_slice: int = 2000,
        n_bins: int = 80,
        vmax: float = 0.05,
        save: str | None = None,
    ) -> Figure:
        if not self.solvers:
            raise ValueError("No algorithms have been run yet. Call run() first.")
        if figsize is None:
            figsize = (14, 4 * len(self.methods))
        fig, axes = plt.subplots(len(self.methods), 2, figsize=figsize)
        beta = self.params["beta"]
        eta = self.params["eta"]

        if interval is None:
            # get the interval from for the maximum and minimum values of the first two dimensions
            x_min = min(s.result.x[:, dims[0]].min() for s in self.solvers.values())
            x_max = max(s.result.x[:, dims[0]].max() for s in self.solvers.values())
            x_abs = max(abs(x_min), abs(x_max)) * 1.1
            interval = (-x_abs, x_abs)

        for i, m in enumerate(self.methods):
            s = self.solvers[m]
            s.plot_trajectory(
                ax=axes[i, 0],
                dim0=dims[0],
                dim1=dims[1],
                xlim=interval,
                ylim=interval,
                n_arrows=n_arrows,
                show_bound=m != "aSB",
                show_energy=True,
            )
            axes[i, 0].set_title(f"{m} Trajectory")
            s.plot_time_hist(
                ax=axes[i, 1],
                n_slice=n_slice,
                n_bins=n_bins,
                vmax=vmax,
                ylim=interval,
            )
            axes[i, 1].set_ylabel("x")
            axes[i, 1].set_xlim(0, len(s.result.x) * eta)
            axes[i, 1].set_ylim(*interval)
            axes[i, 1].set_title(f"{m} Time Distribution")
        plt.suptitle(f"SB Algorithms Trajectory & Time Distribution Comparison (β={beta}, η={eta})", fontsize=16)
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches="tight")
        return fig

    def plot_history(
        self,
        figsize: tuple[int, int] | None = None,
        t_range: KeyType = slice(1000, None),
        dim_range: KeyType = slice(0, 20),
        save: str | None = None,
    ) -> Figure:
        if not self.solvers:
            raise ValueError("No algorithms have been run yet. Call run() first.")
        if figsize is None:
            figsize = (4 * len(self.methods), 12)
        fig, axes = plt.subplots(4, len(self.methods), figsize=figsize)
        for i, m in enumerate(self.methods):
            s = self.solvers[m]
            s.plot_history(
                axes=axes[:, i],
                title=m,
                color=visualize.default_cmap(i / len(self.methods)),
                t_range=t_range,
                dim_range=dim_range,
                ylabel=(i == 0),
            )

        for i in range(1, 4):
            ylims = [ax.get_ylim() for ax in axes[i, :]]
            ymin = min((y[0] for y in ylims))
            ymax = max((y[1] for y in ylims))
            for ax in axes[i, :]:
                ax.set_ylim(ymin, ymax)

        plt.suptitle("SB Algorithms History Variables Comparison", fontsize=16)
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches="tight")
        return fig

    def plot_cut(
        self,
        figsize: tuple[int, int] = (6, 5),
        cmap: str | Colormap = visualize.default_cmap,
        alpha: float = 0.6,
        xlim: BoundsType = (0.1, None),
        inset_zoom: float = 0.8,
        show_max: bool = True,
        max_offset: dict[str, float] = {},
        best_cut: int | None = None,
        save: str | None = None,
    ) -> Figure:
        if not self.solvers:
            raise ValueError("No algorithms have been run yet. Call run() first.")
        cuts = {}
        eta = self.params["eta"]
        for m, s in self.solvers.items():
            cuts[m] = s.get_cut_history()
        tmax = max(len(h) for h in cuts.values()) * eta
        tmin = tmax * inset_zoom
        fig, ax = plt.subplots(figsize=figsize)

        # Determine y-axis limits for inset
        cut_vals = [h.max() for h in cuts.values()]
        cut_min, cut_max = min(cut_vals), max(cut_vals)
        if best_cut is not None:
            cut_max = best_cut
        cut_rng = cut_max - cut_min
        inset_y = (cut_min - cut_rng * 0.2, cut_max + cut_rng * 0.2)

        ax, axins = visualize.plot_history_compare(
            eta=eta,
            ylabel="Cut Value",
            histories=cuts,
            cmap=cmap,
            alpha=alpha,
            xlim=xlim,
            inset_zoom=((tmin, tmax), inset_y),
            show_max=show_max,
            max_text_offset=max_offset,
            ax=ax,
        )
        if best_cut is not None and axins is not None:
            bgcolor = axins.get_facecolor()
            color = np.array([1, 1, 1, 1]) - bgcolor
            axins.axhline(best_cut, label="Best cut", color=color, linestyle="-", alpha=0.8, linewidth=1)
            axins.text(tmax, best_cut, f" Best cut={best_cut}", color=color, fontsize=10, ha="left", va="center", alpha=0.8)
        plt.title("SB Algorithms Cut Value History Comparison")
        if save:
            plt.savefig(save, dpi=300, bbox_inches="tight")
        return fig

    def plot_all(
        self,
        save_prefix: str | None = None,
        best_cut: int | None = None,
    ) -> dict[str, Figure]:
        if not self.solvers:
            raise ValueError("No algorithms have been run yet. Call run() first.")
        figs = {}
        print("📊 Generating trajectory and time distribution comparison...")
        figs["traj"] = self.plot_traj(save=f"{save_prefix}_traj_time.png" if save_prefix else None)
        print("📊 Generating history variables comparison...")
        figs["history"] = self.plot_history(save=f"{save_prefix}_history.png" if save_prefix else None)
        print("📊 Generating cut value history comparison...")
        figs["cut"] = self.plot_cut(save=f"{save_prefix}_cut.png" if save_prefix else None, best_cut=best_cut)
        if save_prefix:
            print(f"✅ All plots generated and saved with prefix '{save_prefix}'")
        return figs

    def set_methods(self, methods: list[MethodType]) -> None:
        self.methods = methods
        self.solvers.clear()

    def best(self) -> tuple[str, Solver]:
        if not self.solvers:
            raise ValueError("No algorithms have been run yet. Call run() first.")
        best = max(self.solvers.keys(), key=lambda m: self.solvers[m].get_best_cut())
        return best, self.solvers[best]

    def export(self, path: str) -> None:
        if not self.solvers:
            raise ValueError("No algorithms have been run yet. Call run() first.")
        results = {
            "params": self.params,
            "methods": self.methods,
            "results": {
                m: {
                    "best_cut": s.get_best_cut(),
                    "cut": s.get_cut_history(),
                    "summary": s.summary(),
                }
                for m, s in self.solvers.items()
            },
        }
        import pickle

        with open(path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results exported to {path}")


def benchmark_plot(
    J: np.ndarray,
    methods: list[MethodType],
    beta: float,
    eta: float,
    xi: float | None = None,
    save_prefix: str | None = None,
    verbose: bool = True,
    seed: int = 0,
    device: str | None = None,
    best_cut: int | None = None,
) -> Benchmark:
    cmp = Benchmark(methods)
    cmp.run(J, beta=beta, eta=eta, xi=xi, verbose=verbose, seed=seed, device=device)
    cmp.plot_all(save_prefix=save_prefix, best_cut=best_cut)
    return cmp
