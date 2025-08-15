from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Literal

import numpy as np
import torch
from tqdm import trange


@dataclass
class StateArray:
    x: np.ndarray
    y: np.ndarray
    g: np.ndarray
    V: np.ndarray
    X: np.ndarray
    H: np.ndarray
    cut: np.ndarray
    steps: int

    @property
    def best_x(self):
        if self.cut.ndim == 0:
            return np.sign(self.x)
        assert self.cut is not None, "cut is None"
        return np.sign(self.x[self.cut.argmax()])


@dataclass
class StateTensor:
    x: torch.Tensor
    y: torch.Tensor
    g: torch.Tensor
    V: torch.Tensor
    X: torch.Tensor
    H: torch.Tensor
    cut: torch.Tensor
    steps: int

    @property
    def best_x(self):
        if self.cut.ndim == 0:
            return self.x.sign()
        assert self.cut is not None, "cut is None"
        return self.x[self.cut.argmax()].sign()

    def to(self, device):
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(device))
        return self

    def numpy(self):
        return StateArray(**{attr_name: (attr.cpu().numpy() if isinstance(attr := getattr(self, attr_name), torch.Tensor) else attr) for attr_name in self.__dict__})  # type: ignore

    @property
    def device(self):
        return list({a.device for attr_name in self.__dict__ if isinstance(a := getattr(self, attr_name), torch.Tensor)})

    def __getitem__(self, item):
        return StateTensor(**{attr_name: getattr(self, attr_name)[item] for attr_name in self.__dict__ if isinstance(getattr(self, attr_name), torch.Tensor)})


torch.serialization.add_safe_globals([StateTensor])


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


class Field(IntEnum):
    A = auto()  # Use quadratic term in the potential
    B = auto()  # No quadratic term in the potential but add a boundary at |x_i|=1 for all i


class InitMode(IntEnum):
    CENTER = auto()  # Initialize x close to the origin
    BOUNDARY = auto()  # Initialize x at the boundaries (|x_i|=1 for all i)
    ONES = auto()  # Initialize x as 1 (x_i=1 for all i)


class CoupleMode(IntEnum):
    DEFAULT = auto()  # Use J @ x as the coupling term
    SIGN = auto()  # Use J @ sign x as the coupling term
    RANDOM = auto()  # Use J @ random x as the coupling term


class UpdateMode(IntEnum):
    DEFAULT = auto()
    SIGN = auto()  # Update x with the sign of y
    RANDOM = auto()  # Update x with random quantization


def run_impl(
    J: torch.Tensor,
    field: Field,
    init_mode: InitMode,
    couple_mode: CoupleMode,
    update_mode: UpdateMode,
    beta: float,
    eta: float,
    xi: float | None = None,
    max_steps: int | None = None,
    progress_bar: bool = False,
    seed: int = 0,
    return_history: bool = True,
    fix_point_x: int | None = None,
    fix_point_y: int | None = None,
    fix_point_g: int | None = None,
) -> StateTensor:
    N = J.shape[0]

    if xi is None:
        xi = float(1 / 2 / J.shape[0] ** 0.5)
    if J.ndim != 2:
        raise ValueError("J must be a 2D array.")
    if J.shape[0] != J.shape[1]:
        raise ValueError("J must be a square matrix.")

    if max_steps is None:
        max_steps = int(2 / beta / eta)

    rng = torch.Generator(device=J.device)
    rng.manual_seed(seed)

    def R():
        return torch.rand(N, device=J.device, generator=rng)

    match init_mode:
        case InitMode.CENTER:
            x0 = (2 * torch.rand(J.shape[0], device=J.device, generator=rng) - 1) / J.shape[0]
        case InitMode.BOUNDARY:
            x0 = torch.randint(2, (J.shape[0],), generator=rng, dtype=torch.float32, device=J.device) * 2 - 1
        case InitMode.ONES:
            x0 = torch.ones(J.shape[0], dtype=torch.float32, device=J.device)

    y0 = torch.zeros(J.shape[0], device=J.device)

    x: torch.Tensor = torch.clone(x0)
    y: torch.Tensor = torch.clone(y0)
    g: torch.Tensor

    if return_history:
        x_history = torch.empty((max_steps, N), device=J.device)
        X_history = torch.empty((max_steps, N), device=J.device)
        y_history = torch.empty((max_steps, N), device=J.device)
        g_history = torch.empty((max_steps, N), device=J.device)
        x_history[0] = torch.clone(x)
        y_history[0] = torch.clone(y)

    for i in trange(max_steps, disable=not progress_bar):
        t = i * eta

        match couple_mode:
            case CoupleMode.DEFAULT:
                X = x
            case CoupleMode.SIGN:
                X = x.sign()
            case CoupleMode.RANDOM:
                X = (x + R()).floor()

        match field:
            case Field.A:
                g = -(x**2 + (1 - beta * t)) * x + xi * J @ X
            case Field.B:
                g = -(1 - beta * t) * x + xi * J @ X

        if fix_point_g:
            scale = 2**fix_point_g
            g = (g * scale).round() / scale

        y += g * eta

        if fix_point_y:
            scale = 2**fix_point_y
            y = (y * scale).round() / scale

        match update_mode:
            case UpdateMode.DEFAULT:
                x += y * eta
            case UpdateMode.SIGN:
                x += y.sign() * eta
            case UpdateMode.RANDOM:
                x += (y + R()).floor() * eta

        if fix_point_x:
            x = (x * 2**fix_point_x).round() / 2**fix_point_x

        if field == Field.B:
            y[x.abs() > 1] = 0
            x = x.clip(-1, 1)

        if return_history:
            x_history[i] = x
            X_history[i] = X
            y_history[i] = y
            g_history[i] = g

        if (x.abs() == 1).all():
            break

    if return_history:
        x_history = x_history[: i + 1]
        X_history = X_history[: i + 1]
        y_history = y_history[: i + 1]
        g_history = g_history[: i + 1]
        t = torch.arange(len(x_history), device=J.device) * eta
        match field:
            case Field.A:
                V_history = 1 / 4 * (x_history**4).sum(-1) + (1 - beta * t) / 2 * (x_history**2).sum(-1) - xi * qform(J, x_history, X_history)
            case Field.B:
                V_history = (1 - beta * t) / 2 * (x_history**2).sum(-1) - xi * qform(J, x_history, X_history)

        H_history = 1 / 2 * (y_history**2).sum(-1) + V_history
        x_history[x_history.sign() == 0] = 1
        x_sign = x_history.sign()
        cut_history = ((-J.sum() + qform(J, x_sign)) / 4).to(dtype=torch.int32)
        return StateTensor(x_history, y_history, g_history, V_history, X_history, H_history, cut_history, i + 1)

    else:
        V = (1 - beta * t) / 2 * (x**2).sum(-1) - xi * x @ J @ X
        H = 1 / 2 * (y**2).sum(-1) + V
        x[x.sign() == 0] = 1
        cut = ((-J.sum() + x @ J @ x) / 4).to(dtype=torch.int32)
        return StateTensor(x, y, g, V, X, H, cut, i + 1)


MethodType = Literal["aSB", "bSB", "dSB", "sSB", "sSB_sgn"]
InitType = Literal["center", "boundary", "ones"]

def run(
    J: torch.Tensor,
    method: MethodType,
    *,
    beta: float = 2**-11,
    xi: float | None = None,
    eta: float = 2**-3,
    max_steps: int | None = None,
    init: InitType = "center",
    seed: int = 0,
    progress_bar: bool = False,
    return_history: bool = True,
    fix_point_x: int | None = None,
    fix_point_y: int | None = None,
    fix_point_g: int | None = None,
) -> StateTensor:
    """
    Run the SB algorithm.

    Parameters
    ----------
    J : torch.Tensor
        The input ising matrix (neg of the coupling matrix).
    method : MethodType
        The method to use. One of MethodType.
    beta : float
        The increasing rate of the p(t)
    xi : float
        The coupling strength.
    eta : float
        The time step.
    max_steps : int
        The maximum number of steps to run.
    init : InitType
        The initial state of x. One of InitType.
    seed : int
        The random seed.
    progress_bar : bool
        Whether to show a progress bar.
    return_history : bool
        Whether to record the history of the optimization process.
    fix_point_x : int | None
        The number of decimal places to round x to.
    fix_point_y : int | None
        The number of decimal places to round y to.
    fix_point_g : int | None
        The number of decimal places to round g to.
    """

    match method:
        case "aSB":
            # Adaptive Simulated Bifurcation
            field = Field.A
            couple_mode = CoupleMode.DEFAULT
            update_mode = UpdateMode.DEFAULT
        case "bSB":
            # Ballistic Simulated Bifurcation
            field = Field.B
            couple_mode = CoupleMode.DEFAULT
            update_mode = UpdateMode.DEFAULT
        case "dSB":
            # Discrete Simulated Bifurcation
            field = Field.B
            couple_mode = CoupleMode.SIGN
            update_mode = UpdateMode.DEFAULT
        case "sSB":
            # Stochastic Simulated Bifurcation
            field = Field.B
            couple_mode = CoupleMode.RANDOM
            update_mode = UpdateMode.DEFAULT
        case "sSB_sgn":
            # Stochastic Simulated Bifurcation that updates x with the sign of y
            field = Field.B
            couple_mode = CoupleMode.RANDOM
            update_mode = UpdateMode.SIGN
        case _:
            raise ValueError(f"Unknown method: {method}")

    init_mode = InitMode[init.upper()]

    r = run_impl(
        J=J,
        beta=beta,
        xi=xi,
        eta=eta,
        max_steps=max_steps,
        progress_bar=progress_bar,
        seed=seed,
        field=field,
        init_mode=init_mode,
        couple_mode=couple_mode,
        update_mode=update_mode,
        return_history=return_history,
        fix_point_x=fix_point_x,
        fix_point_y=fix_point_y,
        fix_point_g=fix_point_g,
    )

    return r
