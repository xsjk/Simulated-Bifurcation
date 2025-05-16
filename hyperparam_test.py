import os
import pickle

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

import core

methods = ["bSB", "dSB", "sSB"]
seeds = list(range(128))

k_ps = 2.0 ** np.arange(-11, -6, 1)
etas = 2.0 ** np.arange(-4, 1, 1)


def run_method(queue, J, device, seeds, k_ps, etas):
    J = J.to(device)
    for method in methods:
        for seed in seeds:
            for k_p in k_ps:
                for eta in etas:
                    result: core.SBHistoryTensor = core.run(
                        J,
                        method=method,
                        k_p=k_p,
                        eta=eta,
                        seed=seed,
                        progress_bar=False,
                    )
                    queue.put((method, seed, k_p, eta, result.cut.cpu().numpy()))


if __name__ == "__main__":
    results_save_path = f"cut_values/k_ps={k_ps}_etas={etas}.pkl"
    if os.path.exists(results_save_path):
        print(f"Result file {results_save_path} already exists. Exiting.")
        exit(0)

    mp.set_sharing_strategy("file_system")  # otherwise large data coming back will be rejected

    J: torch.Tensor = -torch.tensor(np.load("data/k2000.npy"), dtype=torch.float32)
    N = J.shape[0]

    results = {}
    for k_p in k_ps:
        for eta in etas:
            results[(k_p, eta)] = {}
            for method in methods:
                # results[(k_p, eta)][method] = np.full(len(seeds), np.nan)
                results[(k_p, eta)][method] = {}

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
                k_ps,
                etas,
            ),
        )
        p.start()
        processes.append(p)

    for _ in tqdm(range(len(methods) * len(seeds) * len(k_ps) * len(etas))):
        (method, seed, k_p, eta, result) = queue.get()
        results[(k_p, eta)][method][seed] = result

    for p in processes:
        p.join()

    pickle.dump(results, open(results_save_path, "wb"))
