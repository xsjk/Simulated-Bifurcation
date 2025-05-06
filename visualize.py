from typing import Callable, SupportsIndex

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Colormap, LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def plot_history(
    title: str,
    eta: float,
    x_history: np.ndarray,
    y_history: np.ndarray,
    g_history: np.ndarray,
    V_history: np.ndarray,
    H_history: np.ndarray,
    color: str = "purple",
    axes: list[plt.Axes] = None,
    t_range: SupportsIndex = None,
    dim_range: SupportsIndex = None,
    ylabel: bool = True,
) -> list[plt.Axes]:
    T, N = x_history.shape
    t = np.arange(T)[:, None] * eta

    if t_range is not None:
        x_history = x_history[t_range]
        y_history = y_history[t_range]
        g_history = g_history[t_range]
        V_history = V_history[t_range]
        H_history = H_history[t_range]
        t = t[t_range]

    if dim_range is None:
        dim_range = slice(0, min(20, N))

    if axes is None:
        fig, axes = plt.subplots(4, 1, figsize=(7, 8), sharex=True)

    # plot V, H
    axes[0].plot(t, V_history, label=rf"$V_\text{{{title}}}$", color=color, alpha=0.5)
    axes[0].plot(t, H_history, label=rf"$H_\text{{{title}}}$", color="black", alpha=0.5)
    axes[0].grid(alpha=0.5)
    axes[0].legend(loc="upper right")

    # plot x, y, g
    if isinstance(dim_range, int):
        alpha = 1
    elif isinstance(dim_range, slice):
        alpha = 5 * (dim_range.step or 1) / (dim_range.stop - dim_range.start)
    else:
        alpha = 5 / len(dim_range)

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

    if ylabel:
        axes[0].set_ylabel("Energy")
        axes[1].set_ylabel("x")
        axes[2].set_ylabel("y")
        axes[3].set_ylabel("g")

    return axes


def plot_trajectory(
    eta: float,
    x_history: np.ndarray,
    dim0: int = None,
    dim1: int = None,
    n_arrows: int = None,
    point_size: float = 50,
    energy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    x_var = np.var(x_history, axis=0)
    x_var = np.argsort(x_var)[::-1]
    if dim0 is None:
        dim0 = x_var[0]
    if dim1 is None:
        dim1 = x_var[1]

    T, N = x_history.shape
    t = np.arange(T) * eta

    # the time that (x_dim0, x_dim1) is going to leave the initial position
    tmin = np.min(np.where(~np.all(np.isclose(x_history[:, [dim0, dim1]], x_history[0, [dim0, dim1]], atol=1e-2), axis=1))[0]) * eta
    # the time that (x_dim0, x_dim1) starts staying at the final position
    tmax = np.max(np.where(~np.all(np.isclose(x_history[:, [dim0, dim1]], x_history[-1, [dim0, dim1]], atol=1e-2), axis=1))[0]) * eta

    if energy_fn is not None:
        X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        Z = energy_fn(X, Y)
        ct = ax.contourf(X, Y, Z, levels=40, cmap="Blues_r", alpha=0.5)
        plt.colorbar(ct, label="Energy", ax=ax)

    # show the trajectory
    points = ax.scatter(
        x_history[:, dim0],
        x_history[:, dim1],
        # norm=LogNorm(vmin=eta, vmax=tmax),
        vmin=tmin,
        vmax=tmax,
        c=t,
        alpha=1,
        s=1,
        cmap="Greys",
        linewidth=0.5,
        marker="o",
    )

    # show the start and end points
    ax.scatter(x_history[0, dim0], x_history[0, dim1], c="white", s=point_size, label="Start", zorder=100)
    ax.scatter(x_history[-1, dim0], x_history[-1, dim1], c="black", s=point_size, label="End", zorder=100)

    ax.set_xlabel(f"$x_{{{dim0}}}$")
    ax.set_ylabel(f"$x_{{{dim1}}}$")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.colorbar(points, label="Time", ax=ax)

    # Add arrows to show direction
    if n_arrows is not None:
        dt = T / n_arrows
        for i in range(n_arrows):
            idx = int(i * dt)
            if idx >= T - 1:
                break
            ax.arrow(x_history[idx, dim0], x_history[idx, dim1], x_history[idx + 1, dim0] - x_history[idx, dim0], x_history[idx + 1, dim1] - x_history[idx, dim1], head_width=0.05, head_length=0.05, fc="white", ec="white", alpha=0.5)

    return ax


def plot_time_hist(
    eta: float,
    history: np.ndarray,
    n_slice: int = 20,
    n_bins: int = 40,
    vmax: float = 0.1,
    ax: plt.Axes = None,
) -> plt.Axes:
    T, N = history.shape
    dt = T * eta / n_slice

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    bins = np.linspace(-1, 1, n_bins)
    heatmap = np.zeros((len(bins) - 1, n_slice))

    # For each time slice, compute histogram
    for i in range(n_slice):
        time_slice = history[(T // n_slice) * i : (T // n_slice) * (i + 1)].flatten()
        hist, _ = np.histogram(time_slice, bins=bins, density=True)
        heatmap[:, i] = hist

    # Normalize
    heatmap = heatmap / np.max(heatmap)

    im = ax.imshow(heatmap, aspect="auto", origin="lower", extent=[0, dt * n_slice, bins[0], bins[-1]], cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([])
    cbar.set_label("Density")

    ax.set_xlabel("Time")
    ax.grid(False)

    return ax


def plot_history_compare(
    eta: float,
    histories: dict[str, np.ndarray],
    ylabel: str = "Value",
    cmap: str | Colormap = "Darks",
    alpha: float = 0.8,
    xlim: tuple[float, float] = (None, None),
    ylim: tuple[float, float] = (None, None),
    show_max: bool = False,
    max_text_offset: dict[str, float] = {},
    inset_zoom: tuple[tuple[float, float], tuple[float, float]] = None,  # ((x1,x2), (y1,y2))
    ax: plt.Axes = None,
) -> tuple[plt.Axes, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    if isinstance(cmap, str):
        cmap = colormaps[cmap]

    assert isinstance(cmap, Colormap), f"Invalid colormap: {cmap}"

    for i, (label, history) in enumerate(histories.items()):
        t = np.arange(history.shape[0]) * eta
        ax.plot(t, history, label=label, color=cmap(i / len(histories)), alpha=alpha)

    ax.legend()
    ax.set_xscale("log")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)

    # Add inset axes if zoom region is specified
    if inset_zoom is not None:
        (x1, x2), (y1, y2) = inset_zoom
        axins = inset_axes(
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
            t = np.arange(history.shape[0]) * eta
            axins.plot(t, history, color=cmap(i / len(histories)), alpha=alpha)

            if show_max:
                max_value = max_values[label]
                if max_value > y1 and max_value < y2:
                    axins.axhline(
                        max_value,
                        color=cmap(i / len(histories)),
                        linestyle="--",
                        linewidth=1,
                    )
                    axins.text(
                        x=x2,
                        y=max_value + max_text_offset.get(label, 0),
                        s=f" {label} max={max_value:.0f}",
                        color=cmap(i / len(histories)),
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
