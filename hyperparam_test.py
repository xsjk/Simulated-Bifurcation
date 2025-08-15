import pickle
from typing import Literal

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

import core
from core import InitType, MethodType, StateTensor

methods: list[MethodType] = ["aSB", "bSB", "dSB", "sSB", "sSB_sgn"]
seeds = list(range(128))

k_beta_eta_max = 15
k_beta_min = 5
k_eta_min = 0
k_xi = 6
init: InitType = "boundary"

xi = 2**-k_xi


def run_method(queue, J, device, seeds, params):
    try:
        J = J.to(device)
        for method in methods:
            for seed in seeds:
                for beta, eta in params:
                    result: StateTensor = core.run(
                        J,
                        method=method,
                        beta=beta,
                        eta=eta,
                        init=init,
                        seed=seed,
                        xi=xi,
                        progress_bar=False,
                        return_history=False,
                    )
                    queue.put((method, seed, beta, eta, result.cut.item(), result.steps))
    except KeyboardInterrupt:
        pass


type HyperparamResultType = dict[tuple[float, float], dict[MethodType, dict[int, dict[Literal["best_cut", "n_step"], int]]]]

def plot_result(result_path: str, save_prefix: str, **kwargs) -> None:
    '''
    Plot the results of hyperparameter exploration.

    Parameters
    ----------
    result_path : str
        Path to the pickled results file.
    save_prefix : str
        Prefix for saving the figures.
    kwargs : dict
        Additional keyword arguments to pass to the plotting function. See `visualize.plot_hyperparam_characterization` for details.
    '''
    import visualize
    import os.path
    if os.path.exists(result_path):
        visualize.plot_hyperparam_characterization(max_cut_results=pickle.load(open(result_path, "rb")), **kwargs).savefig(f"{save_prefix}.png", bbox_inches="tight", dpi=300)
    else:
        print(f"Results file not found: {result_path}. Please run hyperparameter_test.py first to generate it or check the file path.")

if __name__ == "__main__":
    results_save_path = f"max_cut_values_{init}_init_k_xi_{k_xi}.pkl"

    params = {(2 ** (-k_beta), 2 ** (-(k_eta := k_beta_eta - k_beta))) for k_beta_eta in range(k_beta_min + k_eta_min, k_beta_eta_max + 1) for k_beta in range(k_beta_min, k_beta_eta + 1 - k_eta_min)}

    print(f"Running {len(params)} parameter combinations...")

    J: torch.Tensor = -torch.tensor(np.load("data/k2000.npy"), dtype=torch.float32)
    N = J.shape[0]

    results: HyperparamResultType = {}
    for beta, eta in sorted(params):
        if (beta, eta) not in results:
            results[(beta, eta)] = {}
            for method in methods:
                results[(beta, eta)][method] = {}

    num_devices = torch.cuda.device_count()
    process_per_device = 2

    processes = []
    queue = mp.Queue()
    chunk_size = len(seeds) // num_devices // process_per_device
    if chunk_size == 0:
        chunk_size = 1
    num_chunks = len(seeds) // chunk_size
    for i in range(num_chunks):
        p = mp.Process(
            target=run_method,
            args=(
                queue,
                J,
                f"cuda:{i % num_devices}",
                seeds[i * chunk_size : (i + 1) * chunk_size],
                params,
            ),
        )
        p.start()
        processes.append(p)

    try:
        for _ in tqdm(range(len(methods) * len(seeds) * len(params)), desc="Collecting results"):
            (method, seed, beta, eta, best_cut, n_step) = queue.get()
            results[(beta, eta)][method][seed] = {
                "best_cut": best_cut,
                "n_step": n_step,
            }
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Terminating processes...")
    else:
        for p in processes:
            p.join()
    finally:
        # clear unfinished param
        for beta, eta in params:
            is_finished = True
            for method in methods:
                if method not in results[(beta, eta)]:
                    is_finished = False
                else:
                    is_finished = all(seed in results[(beta, eta)][method] for seed in seeds)
            if not is_finished and (beta, eta) in results:
                del results[(beta, eta)]

        print(f"Saving results to {results_save_path}...")
        pickle.dump(results, open(results_save_path, "wb"))

