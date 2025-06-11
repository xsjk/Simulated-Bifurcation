from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class SBHistoryArray:
    x: np.ndarray
    y: np.ndarray
    g: np.ndarray
    V: np.ndarray
    X: np.ndarray | None = None
    H: np.ndarray | None = None
    cut: np.ndarray | None = None

    @property
    def best_x(self):
        assert self.cut is not None, "cut is None"
        return np.sign(self.x[self.cut.argmax()])


@dataclass
class SBHistoryTensor:
    x: torch.Tensor
    y: torch.Tensor
    g: torch.Tensor
    V: torch.Tensor
    X: torch.Tensor | None = None
    H: torch.Tensor | None = None
    cut: torch.Tensor | None = None

    @property
    def best_x(self):
        assert self.cut is not None, "cut is None"
        return torch.sign(self.x[self.cut.argmax()])

    def to(self, device):
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(device))
        return self

    def numpy(self):
        return SBHistoryArray(**{attr_name: (attr.cpu().numpy() if isinstance(attr := getattr(self, attr_name), torch.Tensor) else attr) for attr_name in self.__dict__})

    @property
    def device(self):
        return list({a.device for attr_name in self.__dict__ if isinstance(a := getattr(self, attr_name), torch.Tensor)})

    def __getitem__(self, item):
        return SBHistoryTensor(**{attr_name: getattr(self, attr_name)[item] for attr_name in self.__dict__ if isinstance(getattr(self, attr_name), torch.Tensor)})


torch.serialization.add_safe_globals([SBHistoryTensor])


def qform(J, x1, x2=None):
    """
    Given a batch of vectors x (shape: [batch_size, n]) and a matrix J (shape: [n, n]),
    this function computes the quadratic form x^T J x for each vector in the batch.

    Parameters
    ----------
    J : torch.Tensor
        A matrix of shape [n, n].
    x1 : torch.Tensor
        A batch of vectors of shape [batch_size, n].
    x2 : torch.Tensor
        A batch of vectors of shape [batch_size, n].
        If not provided, x1 is used for the quadratic form.

    Returns
    -------
    torch.Tensor
        A tensor of shape [batch_size] containing the quadratic form for each vector in the batch.

    """
    x2 = x1 if x2 is None else x2
    assert J.shape[0] == J.shape[1] == x1.shape[1] == x2.shape[1], "J must be a square matrix of shape [n, n] and x1, x2 must have the same second dimension as J."
    return torch.einsum("bn,nm,bm->b", x1, J, x2)
    # When using torch.einsum, although 'bi,bj,ij->b' and 'bi,ij,bj->b' are mathematically equivalent,
    # the former broadcasts x to shape [batch_size, n, batch_size], which can cause excessive memory usage.
    # The latter ('bi,ij,bj->b'), as used here, avoids this issue and is more memory efficient.


def _aSB(J: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor, beta: float, xi: float, eta: float, max_steps: int, progress_bar: bool) -> SBHistoryTensor:
    N = J.shape[0]

    x_history = torch.zeros((max_steps, N), device=J.device)
    y_history = torch.zeros((max_steps, N), device=J.device)
    x_history[0] = x = torch.clone(x0)
    y_history[0] = y = torch.clone(y0)

    iterator = tqdm(range(max_steps), desc="aSB Progress", disable=not progress_bar)

    for i in iterator:
        t = i * eta

        g = -(x**2 + (1 - beta * t)) * x + xi * J @ x
        x += y * eta
        y += g * eta

        x_history[i] = x
        y_history[i] = y

        if torch.any(torch.linalg.norm(g) > 20):
            break

    x_history = x_history[: i + 1]
    y_history = y_history[: i + 1]

    t = torch.arange(len(x_history), device=J.device)[:, None] * eta
    g_history = -(x_history**2 + (1 - beta * t)) * x_history + xi * x_history @ J
    V_history = 1 / 4 * (x_history**4).sum(-1) + (1 - beta * t.flatten()) / 2 * (x_history**2).sum(-1) - xi * qform(J, x_history)

    return SBHistoryTensor(x_history, y_history, g_history, V_history)


def _bSB(J: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor, beta: float, xi: float, eta: float, max_steps: int, progress_bar: bool) -> SBHistoryTensor:
    N = J.shape[0]

    x_history = torch.zeros((max_steps, N), device=J.device)
    y_history = torch.zeros((max_steps, N), device=J.device)
    x_history[0] = x = torch.clone(x0)
    y_history[0] = y = torch.clone(y0)

    iterator = tqdm(range(max_steps), desc="bSB Progress", disable=not progress_bar)
    for i in iterator:
        t = i * eta

        g = -(1 - beta * t) * x + xi * J @ x
        x += y * eta
        y += g * eta

        y[torch.abs(x) > 1] = 0
        x = torch.clip(x, -1, 1)

        x_history[i] = x
        y_history[i] = y

        if (torch.abs(x) == 1).all():
            break

    x_history = x_history[: i + 1]
    y_history = y_history[: i + 1]

    t = torch.arange(len(x_history), device=J.device)[:, None] * eta
    g_history = -(1 - beta * t) * x_history + xi * x_history @ J
    V_history = (1 - beta * t.flatten()) / 2 * (x_history**2).sum(-1) - xi * qform(J, x_history)

    return SBHistoryTensor(x_history, y_history, g_history, V_history)


def _dSB(J: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor, beta: float, xi: float, eta: float, max_steps: int, progress_bar: bool) -> SBHistoryTensor:
    N = J.shape[0]

    x_history = torch.zeros((max_steps, N), device=J.device)
    y_history = torch.zeros((max_steps, N), device=J.device)
    x_history[0] = x = torch.clone(x0)
    y_history[0] = y = torch.clone(y0)

    iterator = tqdm(range(max_steps), desc="dSB Progress", disable=not progress_bar)
    for i in iterator:
        t = i * eta

        g = -(1 - beta * t) * x + xi * J @ torch.sign(x)
        x += y * eta
        y += g * eta

        y[torch.abs(x) > 1] = 0
        x = torch.clip(x, -1, 1)

        x_history[i] = x
        y_history[i] = y

        if (torch.abs(x) == 1).all():
            break

    x_history = x_history[: i + 1]
    y_history = y_history[: i + 1]

    t = torch.arange(len(x_history), device=J.device)[:, None] * eta
    g_history = -(1 - beta * t) * x_history + xi * torch.sign(x_history) @ J
    V_history = (1 - beta * t.flatten()) / 2 * (x_history**2).sum(-1) - xi * qform(J, x_history, torch.sign(x_history))

    return SBHistoryTensor(x_history, y_history, g_history, V_history)


def R(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    x_floor = torch.floor(x)
    x_ceil = torch.ceil(x)
    x_decimal = x - x_floor

    x_out = x_floor
    x_out[r < x_decimal] = x_ceil[r < x_decimal]
    return x_out


def _sSB(J: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor, beta: float, xi: float, eta: float, max_steps: int, progress_bar: bool, rng: torch.Generator, clip: bool = True) -> SBHistoryTensor:
    N = J.shape[0]

    x_history = torch.zeros((max_steps, N), device=J.device)
    X_history = torch.zeros((max_steps, N), device=J.device)
    y_history = torch.zeros((max_steps, N), device=J.device)
    g_history = torch.zeros((max_steps, N), device=J.device)
    x_history[0] = x = torch.clone(x0)
    y_history[0] = y = torch.clone(y0)

    iterator = tqdm(range(max_steps), desc="sSB Progress", disable=not progress_bar)
    for i in iterator:
        t = i * eta

        X = R(x, torch.rand(N, device=J.device, generator=rng))
        g = -(1 - beta * t) * x + xi * J @ X
        if clip:
            g = torch.clip(g, -1, 1)

        x += R(y, torch.rand(N, device=J.device, generator=rng)) * eta
        y += R(g, torch.rand(N, device=J.device, generator=rng)) * eta

        y[torch.abs(x) > 1] = 0
        x = torch.clip(x, -1, 1)

        if clip:
            y = torch.clip(y, -1, 1)

        x_history[i] = x
        X_history[i] = X
        y_history[i] = y
        g_history[i] = g

        if (torch.abs(x) == 1).all():
            break

    x_history = x_history[: i + 1]
    X_history = X_history[: i + 1]
    y_history = y_history[: i + 1]
    g_history = g_history[: i + 1]

    t = torch.arange(len(x_history), device=J.device)[:, None] * eta
    V_history = (1 - beta * t.flatten()) / 2 * (x_history**2).sum(-1) - xi * qform(J, x_history, X_history)

    return SBHistoryTensor(x_history, y_history, g_history, V_history, X_history)


type MethodType = Literal["aSB", "bSB", "dSB", "sSB", "sSB_clip"]


def run(
    J: torch.Tensor,
    method: MethodType,
    *,
    beta: float = 2**-11,
    xi: float | None = None,
    eta: float = 2**-3,
    max_steps: int | None = None,
    seed: int = 0,
    progress_bar: bool = False,
) -> SBHistoryTensor:
    """
    Run the SB algorithm.

    Parameters
    ----------
    J : torch.Tensor
        The input ising matrix (neg of the coupling matrix).
    method : str
        The method to use. One of "aSB", "bSB", "dSB", "sSB", "sSB_clip".
    beta : float
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

    if J.ndim != 2:
        raise ValueError("J must be a 2D array.")
    if J.shape[0] != J.shape[1]:
        raise ValueError("J must be a square matrix.")

    if xi is None:
        xi = 1 / 2 / J.shape[0] ** 0.5

    if max_steps is None:
        max_steps = int(2 / beta / eta)

    # Type assertions after None checks
    assert xi is not None
    assert max_steps is not None

    N = J.shape[0]

    rng = torch.Generator(device=J.device)
    rng.manual_seed(seed)

    x0 = (2 * torch.rand(N, device=J.device, generator=rng) - 1) / N
    y0 = torch.zeros(N, device=J.device)

    match method:
        case "aSB":
            r = _aSB(J, x0, y0, beta, xi, eta, max_steps, progress_bar)
        case "bSB":
            r = _bSB(J, x0, y0, beta, xi, eta, max_steps, progress_bar)
        case "dSB":
            r = _dSB(J, x0, y0, beta, xi, eta, max_steps, progress_bar)
        case "sSB":
            r = _sSB(J, x0, y0, beta, xi, eta, max_steps, progress_bar, rng, clip=False)
        case "sSB_clip":
            r = _sSB(J, x0, y0, beta, xi, eta, max_steps, progress_bar, rng, clip=True)
        case _:
            raise ValueError(f"Unknown method: {method}. Supported methods are: aSB, bSB, dSB, sSB.")

    r.H = 1 / 2 * (r.y**2).sum(-1) + r.V

    x = torch.sign(r.x)
    x[x == 0] = 1
    assert ((x == 1) | (x == -1)).all(), "x must be a valid configuration of the Ising model."
    cut = (-J.sum() + qform(J, x)) / 4
    r.cut = cut.to(dtype=torch.int32)
    assert (cut == r.cut).all(), "cut must be an integer value."
    return r


def run_numpy(
    J: np.ndarray,
    method: MethodType,
    *,
    beta: float = 2**-11,
    xi: float | None = None,
    eta: float = 2**-3,
    max_steps: int | None = None,
    seed: int = 0,
    progress_bar: bool = False,
) -> SBHistoryArray:
    return run(
        torch.tensor(J, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu"),
        method=method,
        beta=beta,
        xi=xi,
        eta=eta,
        max_steps=max_steps,
        seed=seed,
        progress_bar=progress_bar,
    ).numpy()
