from dataclasses import dataclass
from typing import Literal

import cupy as cp
from tqdm import tqdm


@dataclass
class SBResult:
    x: cp.ndarray
    y: cp.ndarray
    g: cp.ndarray
    V: cp.ndarray
    H: cp.ndarray = None
    cut: cp.ndarray = None

    @property
    def best_x(self):
        return cp.sign(self.x[self.cut.argmax()])


def _aSB(J, k_p, xi, eta, max_steps, seed, progress_bar) -> SBResult:
    N = J.shape[0]
    if xi is None:
        xi = 1 / 2 / N**0.5
    if max_steps is None:
        max_steps = int(2 / k_p / eta)

    cp.random.seed(seed)
    x0 = 0.02 * cp.random.uniform(-1, 1, N)
    y0 = cp.zeros(N)

    x_history = cp.zeros((max_steps, N))
    y_history = cp.zeros((max_steps, N))
    x_history[0] = x = x0.copy()
    y_history[0] = y = y0.copy()

    iterator = tqdm(range(max_steps), desc="aSB Progress", disable=not progress_bar)

    for i in iterator:
        t = i * eta

        g = -(x**2 + (1 - k_p * t)) * x + xi * J @ x
        x += y * eta
        y += g * eta

        x_history[i] = x
        y_history[i] = y

        if cp.any(cp.linalg.norm(g) > 20):
            break

    x_history = x_history[: i + 1]
    y_history = y_history[: i + 1]

    t = cp.arange(len(x_history))[:, None] * eta
    g_history = -(x_history**2 + (1 - k_p * t)) * x_history + xi * x_history @ J
    V_history = 1 / 4 * (x_history**4).sum(-1) + (1 - k_p * t.flatten()) / 2 * (x_history**2).sum(-1) - xi * cp.einsum("tn,tm,nm->t", x_history, x_history, J)

    return SBResult(x_history, y_history, g_history, V_history)


def _bSB(J, k_p, xi, eta, max_steps, seed, progress_bar):
    N = J.shape[0]
    if xi is None:
        xi = 1 / 2 / N**0.5
    if max_steps is None:
        max_steps = int(2 / k_p / eta)

    cp.random.seed(seed)
    x0 = 0.02 * cp.random.uniform(-1, 1, N)
    y0 = cp.zeros(N)

    x_history = cp.zeros((max_steps, N))
    y_history = cp.zeros((max_steps, N))
    x_history[0] = x = x0.copy()
    y_history[0] = y = y0.copy()

    iterator = tqdm(range(max_steps), desc="bSB Progress", disable=not progress_bar)
    for i in iterator:
        t = i * eta

        g = -(1 - k_p * t) * x + xi * J @ x
        x += y * eta
        y += g * eta

        x_history[i] = x
        y_history[i] = y

        y[cp.abs(x) > 1] = 0
        x = cp.clip(x, -1, 1)

        if (cp.abs(x) == 1).all():
            break

    x_history = x_history[: i + 1]
    y_history = y_history[: i + 1]

    t = cp.arange(len(x_history))[:, None] * eta
    g_history = -(1 - k_p * t) * x_history + xi * x_history @ J
    V_history = (1 - k_p * t.flatten()) / 2 * (x_history**2).sum(-1) - xi * cp.einsum("tn,tm,nm->t", x_history, x_history, J)

    return SBResult(x_history, y_history, g_history, V_history)


def _dSB(J, k_p, xi, eta, max_steps, seed, progress_bar):
    N = J.shape[0]

    cp.random.seed(seed)
    x0 = 0.02 * cp.random.uniform(-1, 1, N)
    y0 = cp.zeros(N)

    x_history = cp.zeros((max_steps, N))
    y_history = cp.zeros((max_steps, N))
    x_history[0] = x = x0.copy()
    y_history[0] = y = y0.copy()

    iterator = tqdm(range(max_steps), desc="dSB Progress", disable=not progress_bar)
    for i in iterator:
        t = i * eta

        g = -(1 - k_p * t) * x + xi * J @ cp.sign(x)
        x += y * eta
        y += g * eta

        x_history[i] = x
        y_history[i] = y

        y[cp.abs(x) > 1] = 0
        x = cp.clip(x, -1, 1)

        if (cp.abs(x) == 1).all():
            break

    x_history = x_history[: i + 1]
    y_history = y_history[: i + 1]

    t = cp.arange(len(x_history))[:, None] * eta
    g_history = -(1 - k_p * t) * x_history + xi * cp.sign(x_history) @ J
    V_history = (1 - k_p * t.flatten()) / 2 * (x_history**2).sum(-1) - xi * cp.einsum("tn,tm,nm->t", x_history, cp.sign(x_history), J)

    return SBResult(x_history, y_history, g_history, V_history)


def R(x, r):
    x_floor = cp.floor(x)
    x_ceil = cp.ceil(x)
    x_decimal = x - x_floor

    x_out = x_floor
    x_out[r < x_decimal] = x_ceil[r < x_decimal]
    return x_out


def _sSB(J, k_p, xi, eta, max_steps, seed, progress_bar, clip=True):
    N = J.shape[0]

    cp.random.seed(seed)
    x0 = cp.random.uniform(-1, 1, N) / N
    y0 = cp.zeros(N)

    x_history = cp.zeros((max_steps, N))
    X_history = cp.zeros((max_steps, N))
    y_history = cp.zeros((max_steps, N))
    g_history = cp.zeros((max_steps, N))
    x_history[0] = x = x0.copy()
    y_history[0] = y = y0.copy()

    iterator = tqdm(range(max_steps), desc="sSB Progress", disable=not progress_bar)
    for i in iterator:
        t = i * eta

        X = R(x, cp.random.rand(N))
        g = -(1 - k_p * t) * x + xi * J @ X
        if clip:
            g = cp.clip(g, -1, 1)

        x += R(y, cp.random.rand(N)) * eta
        y += R(g, cp.random.rand(N)) * eta

        x_history[i] = x
        X_history[i] = X
        y_history[i] = y
        g_history[i] = g

        y[cp.abs(x) > 1] = 0
        x = cp.clip(x, -1, 1)

        if clip:
            y = cp.clip(y, -1, 1)

        if (cp.abs(x) == 1).all():
            break

    x_history = x_history[: i + 1]
    X_history = X_history[: i + 1]
    y_history = y_history[: i + 1]
    g_history = g_history[: i + 1]

    t = cp.arange(len(x_history))[:, None] * eta
    V_history = (1 - k_p * t.flatten()) / 2 * (x_history**2).sum(-1) - xi * cp.einsum("tn,tm,nm->t", x_history, X_history, J)

    return SBResult(x_history, y_history, g_history, V_history, None, None)


def run(
    J,
    method: Literal["aSB", "bSB", "dSB", "sSB"],
    *,
    k_p=2**-11,
    xi=None,
    eta=2**-3,
    max_steps=None,
    seed=42,
    progress_bar=False,
    **kwargs,
) -> SBResult:
    """
    Run the SB algorithm.

    Parameters
    ----------
    J : cupy.ndarray
        The input ising matrix (neg of the coupling matrix).
    method : str
        The method to use. One of "aSB", "bSB", "dSB", "sSB".
    k_p : float
        The increasing rate of the p(t)
    xi : float
        The coupling strength.
    eta : float
        The time step.
    max_steps : int
        The maximum number of steps to run.
    seed : int
        The random seed.
    progress_bar : bool
        Whether to show a progress bar.
    """
    if not isinstance(J, cp.ndarray):
        raise TypeError("J must be a cupy array.")
    if J.ndim != 2:
        raise ValueError("J must be a 2D array.")
    if J.shape[0] != J.shape[1]:
        raise ValueError("J must be a square matrix.")

    if xi is None:
        xi = 1 / 2 / J.shape[0] ** 0.5

    if max_steps is None:
        max_steps = int(2 / k_p / eta)

    match method:
        case "aSB":
            r = _aSB(J, k_p, xi, eta, max_steps, seed, progress_bar)
        case "bSB":
            r = _bSB(J, k_p, xi, eta, max_steps, seed, progress_bar)
        case "dSB":
            r = _dSB(J, k_p, xi, eta, max_steps, seed, progress_bar)
        case "sSB":
            r = _sSB(J, k_p, xi, eta, max_steps, seed, progress_bar, **kwargs)
        case _:
            raise ValueError(f"Unknown method: {method}. Supported methods are: aSB, bSB, dSB, sSB.")

    r.H = 1 / 2 * (r.y**2).sum(-1) + r.V

    x = cp.sign(r.x)
    r.cut = (-J.sum() + cp.einsum("tn,tm,nm->t", x, x, J)) / 4

    return r
