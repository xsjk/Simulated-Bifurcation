from types import EllipsisType
from typing import Any, Callable, SupportsIndex

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from numpy import bool_, integer
from numpy.typing import NDArray

type KeyType = slice | EllipsisType | SupportsIndex | NDArray[integer[Any]] | NDArray[bool_]
type BoundsType = tuple[float, float] | tuple[float, None] | tuple[None, float] | None


def generate_alpha(num_colors: int = 9) -> np.ndarray:
    return np.tanh(np.linspace(0, 2, num_colors))


def transparentise(cmap: str | Colormap, alphas: np.ndarray | None = None) -> Colormap:
    if isinstance(cmap, str):
        cmap = colormaps[cmap]

    alphas = alphas if alphas is not None else generate_alpha()
    num_colors = len(alphas)
    colors = cmap(np.linspace(0, 1, num_colors))
    if colors.shape[1] == 3:
        colors = np.hstack((colors, alphas))
    elif colors.shape[1] == 4:
        colors[:, 3] = alphas
    else:
        raise ValueError(f"Unsupported colormap format: {colors.shape[1]} channels")

    # Adjust RGB channels based on alpha
    colors[1:, :3] -= 1 - colors[1:, [3]]
    colors[1:, :3] /= colors[1:, [3]]
    colors = np.clip(colors, 0, 1)
    return LinearSegmentedColormap.from_list(f"{cmap.name}_alpha", colors)


def plot_history(
    title: str,
    eta: float,
    x_history: np.ndarray,
    y_history: np.ndarray,
    g_history: np.ndarray,
    V_history: np.ndarray,
    H_history: np.ndarray,
    color: str = "purple",
    alpha: float = 0.9,
    axes: list[Axes] | None = None,
    t_range: KeyType | None = None,
    dim_range: KeyType | None = None,
    ylabel: bool = True,
) -> list[Axes]:
    T, _N = x_history.shape
    t = np.arange(T, dtype=float)[:, None] * eta

    if t_range is not None:
        x_history = x_history[t_range]
        y_history = y_history[t_range]
        g_history = g_history[t_range]
        V_history = V_history[t_range]
        H_history = H_history[t_range]
        t = t[t_range]

    if dim_range is None:
        dim_range = slice(0, min(20, _N))

    if axes is None:
        _fig, axes = plt.subplots(4, 1, figsize=(7, 8), sharex=True)

    # Type assertion: axes is guaranteed to be not None after the check above
    assert axes is not None

    # Now axes is guaranteed to be not None
    plot_V_H(axes, t, V_history, H_history, title, color)
    plot_xyg(axes, t, x_history, y_history, g_history, title, color, alpha, dim_range)

    if ylabel:
        axes[0].set_ylabel("Energy")
        axes[1].set_ylabel("Position")
        axes[2].set_ylabel("Velocity")
        axes[3].set_ylabel("Acceleration")

    return axes


def plot_V_H(axes: list[Axes], t: np.ndarray, V_history: np.ndarray, H_history: np.ndarray, title: str, color: str) -> None:
    axes[0].plot(t, V_history, label=rf"$V_\text{{{title}}}$", color=color, alpha=0.5)
    axes[0].plot(t, H_history, label=rf"$H_\text{{{title}}}$", color="black", alpha=0.5)
    axes[0].grid(alpha=0.5)
    axes[0].legend(loc="upper right")


def plot_xyg(
    axes: list[Axes],
    t: np.ndarray,
    x_history: np.ndarray,
    y_history: np.ndarray,
    g_history: np.ndarray,
    title: str,
    color: str,
    alpha: float,
    dim_range: KeyType,
) -> None:
    # plot x, y, g
    if isinstance(dim_range, int):
        # No specific behavior implemented for integer dim_range.
        # Consider adding functionality if needed or leave as is intentionally.
        alpha *= 1  # Placeholder for potential future implementation
    else:
        alpha *= 2 / x_history[:, dim_range].shape[1]

    if alpha > 1:
        alpha = 1

    # plot x[0], x[1]
    axes[1].plot(t, x_history[:, dim_range], color=color, alpha=alpha)
    axes[1].grid(alpha=0.5)
    axes[1].legend([rf"$\mathbf{{x}}_\text{{{title}}}$"], loc="upper right")

    # plot y[0], y[1]
    axes[2].plot(t, y_history[:, dim_range], color=color, alpha=alpha)
    axes[2].grid(alpha=0.5)
    axes[2].legend([rf"$\mathbf{{y}}_\text{{{title}}}$"], loc="upper right")

    # plot g[0], g[1]
    axes[3].plot(t, g_history[:, dim_range], color=color, alpha=alpha)
    axes[3].grid(alpha=0.5)
    axes[3].set_xlabel("Time (s)")
    axes[3].legend([rf"$\mathbf{{g}}_\text{{{title}}}$"], loc="upper right")


def plot_trajectory(
    eta: float,
    x_history: np.ndarray,
    dim0: int | None = None,
    dim1: int | None = None,
    n_arrows: int | None = None,
    trajectory_point_size: float = 1,
    start_point_size: float = 50,
    end_point_size: float = 50,
    xlim: tuple[float, float] = (-1, 1),
    ylim: tuple[float, float] = (-1, 1),
    show_bound: bool = False,
    energy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        ax = plt.gca()

    x_var = np.var(x_history, axis=0)
    x_var = np.argsort(x_var)[::-1]

    if energy_fn is not None and (dim0 is None or dim1 is None):
        raise ValueError("dim0 and dim1 must be specified if energy_fn is provided")

    if dim0 is None:
        dim0 = x_var[0]
    if dim1 is None:
        dim1 = x_var[1]

    assert dim0 is not None and dim1 is not None  # Type assertion after None checks

    T, _N = x_history.shape
    t = np.arange(T, dtype=float) * eta

    # the time that (x_dim0, x_dim1) is going to leave the initial position
    tmin = np.min(np.where(~np.all(np.isclose(x_history[:, [dim0, dim1]], x_history[0, [dim0, dim1]], atol=1e-2), axis=1))[0]) * eta
    # the time that (x_dim0, x_dim1) starts staying at the final position
    tmax = np.max(np.where(~np.all(np.isclose(x_history[:, [dim0, dim1]], x_history[-1, [dim0, dim1]], atol=1e-2), axis=1))[0]) * eta

    if energy_fn is not None:
        X, Y = np.meshgrid(np.linspace(*xlim, 100), np.linspace(*ylim, 100))
        Z = energy_fn(X, Y)
        ct = ax.contourf(X, Y, Z, levels=40, cmap="Blues_r", alpha=0.5)
        plt.colorbar(ct, label="Energy", ax=ax)

    # show the trajectory
    points = ax.scatter(
        x_history[:, dim0],
        x_history[:, dim1],
        # norm=LogNorm(vmin=eta, vmax=tmax),
        vmin=float(tmin),
        vmax=float(tmax),
        c=t,
        alpha=1,
        s=trajectory_point_size,
        linewidths=0,
        cmap="Greys",
        marker="o",
    )

    # show the start and end points
    ax.scatter(x_history[0, dim0], x_history[0, dim1], c="white", s=start_point_size, label="Start", zorder=100)
    ax.scatter(x_history[-1, dim0], x_history[-1, dim1], c="black", s=end_point_size, label="End", zorder=100)

    ax.set_xlabel(f"$x_{{{dim0}}}$")
    ax.set_ylabel(f"$x_{{{dim1}}}$")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.colorbar(points, label="Time", ax=ax)

    if show_bound:
        color = "black" if energy_fn is None else "grey"
        linestyle = "solid" if energy_fn is None else "dashed"
        alpha = 0.9 if energy_fn is None else 0.5
        ax.plot([-1, 1], [-1, -1], color=color, linestyle=linestyle, alpha=alpha)
        ax.plot([-1, 1], [1, 1], color=color, linestyle=linestyle, alpha=alpha)
        ax.plot([-1, -1], [-1, 1], color=color, linestyle=linestyle, alpha=alpha)
        ax.plot([1, 1], [-1, 1], color=color, linestyle=linestyle, alpha=alpha)

    # Add arrows to show direction
    if n_arrows is not None:
        dt = T / n_arrows
        for i in range(n_arrows):
            idx = int(i * dt)
            if idx >= T - 1:
                break
            ax.arrow(
                float(x_history[idx, dim0]),
                float(x_history[idx, dim1]),
                float(x_history[idx + 1, dim0] - x_history[idx, dim0]),
                float(x_history[idx + 1, dim1] - x_history[idx, dim1]),
                head_width=0.05,
                head_length=0.05,
                fc="white",
                ec="white",
                alpha=0.5,
            )

    return ax


def plot_time_hist(
    eta: float,
    history: np.ndarray,
    n_slice: int = 20,
    n_bins: int = 40,
    vmax: float = 0.1,
    xlim: tuple[float, float | None] = (0, None),
    ylim: tuple[float, float] = (-1, 1),
    ax: Axes | None = None,
    colorbar: bool = True,
    cmap: str | Colormap | None = None,
) -> Axes:
    T, _N = history.shape
    dt = T * eta / n_slice

    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 4))

    if cmap is None:
        # Make it more visually appealing on any background
        cmap = transparentise("Blues")
    elif isinstance(cmap, str):
        cmap = colormaps[cmap]

    bins = np.linspace(*ylim, n_bins)
    history = np.clip(history, *ylim)
    heatmap = np.zeros((len(bins) - 1, n_slice))

    # For each time slice, compute histogram
    for i in range(n_slice):
        time_slice = history[T * i // n_slice : T * (i + 1) // n_slice].flatten()
        if len(time_slice) == 0:
            continue
        hist, _ = np.histogram(time_slice, bins=bins, density=True)
        heatmap[:, i] = hist

    # Normalize
    heatmap = heatmap / heatmap[np.isfinite(heatmap)].max()

    im = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        extent=(0, dt * n_slice, float(bins[0]), float(bins[-1])),
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )

    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks([])
        cbar.set_label("Density")

    ax.set_xlabel("Time")
    ax.grid(False)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    return ax


def plot_history_compare(
    eta: float,
    histories: dict[str, np.ndarray],
    histories_std: dict[str, np.ndarray] | None = None,
    ylabel: str = "Value",
    colors: dict[str, str] = {},
    cmap: str | Colormap = "Darks",
    alpha: float = 0.8,
    xlim: BoundsType = None,
    ylim: BoundsType = None,
    show_max: bool = False,
    max_text_offset: dict[str, float] = {},
    inset_zoom: tuple[tuple[float, float], tuple[float, float]] | None = None,  # ((x1,x2), (y1,y2))
    ax: Axes | None = None,
) -> tuple[Axes, Axes | None]:
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 5))

    if isinstance(cmap, str):
        cmap = colormaps[cmap]

    assert isinstance(cmap, Colormap), f"Invalid colormap: {cmap}"

    for i, (label, history) in enumerate(histories.items()):
        t = np.arange(history.shape[0], dtype=float) * eta
        color = colors.get(label, cmap(i / len(histories)))
        ax.plot(t, history, label=label, color=color, alpha=alpha)
        if histories_std is not None:
            ax.fill_between(
                t,
                history - histories_std[label],
                history + histories_std[label],
                color=color,
                alpha=0.1,
            )

    ax.legend()
    ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)

    # Add inset axes if zoom region is specified
    if inset_zoom is not None:
        (x1, x2), (y1, y2) = inset_zoom
        axins: Axes = inset_axes(
            ax,
            width="80%",
            height="90%",
            loc="lower right",
            bbox_to_anchor=(1.1, 0.03, 1, 1),
            bbox_transform=ax.transAxes,
        )

        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="grey", lw=1)

        max_values = {k: v.max() for k, v in histories.items()}

        for i, (label, history) in enumerate(histories.items()):
            t = np.arange(history.shape[0], dtype=float) * eta
            color = colors.get(label, cmap(i / len(histories)))
            axins.plot(t, history, color=color, alpha=alpha)
            if histories_std is not None:
                axins.fill_between(
                    t,
                    history - histories_std[label],
                    history + histories_std[label],
                    color=color,
                    alpha=0.1,
                )

            if show_max:
                max_value = max_values[label]
                if max_value > y1 and max_value < y2:
                    axins.axhline(
                        max_value,
                        color=color,
                        linestyle="--",
                        linewidth=1,
                    )
                    axins.text(
                        x=x2,
                        y=max_value + max_text_offset.get(label, 0),
                        s=f" {label} max={max_value:.0f}",
                        color=color,
                        fontsize=10,
                        ha="left",
                        va="center_baseline",
                    )

        axins.set_xscale("linear")
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xlabel("Time")
        axins.set_ylabel(ylabel)

        axins.grid()

    return ax, axins if inset_zoom is not None else None
