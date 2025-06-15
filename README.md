# Simulated Bifurcation Algorithms

A comprehensive implementation and comparison of Simulated Bifurcation (SB) algorithms for solving Ising models and optimization problems.

## Background

### Ising Model Definition

The Ising model is a mathematical model of ferromagnetism in statistical mechanics. In our optimization context, we consider the Ising model without external magnetic field, where the system consists of discrete variables (spins) that can take values of +1 or -1.

For a system of N spins, the **Ising Hamiltonian** (energy function) is defined as:

$$H(\mathbf{s}) = -\sum_{i=1}^{N} \sum_{j=1}^{N} J_{ij} s_i s_j$$

where:

- $\mathbf{s} = (s_1, s_2, \ldots, s_N)$ is the spin configuration with $s_i \in \{-1, +1\}$
- $J_{ij}$ is the coupling strength between spins $i$ and $j$
- $J_{ii} = 0$ (no self-interaction)
- For symmetric problems: $J_{ij} = J_{ji}$

### Optimization Objective

The **optimization problem** we solve is:

$$\min_{\mathbf{s} \in \{-1,+1\}^N} H(\mathbf{s}) = \min_{\mathbf{s} \in \{-1,+1\}^N} \left(-\sum_{i=1}^{N} \sum_{j=1}^{N} J_{ij} s_i s_j\right)$$

This can be equivalently written as a **maximization problem**:

$$\max_{\mathbf{s} \in \{-1,+1\}^N} \sum_{i=1}^{N} \sum_{j=1}^{N} J_{ij} s_i s_j$$

### Connection to Max-Cut Problem

For graph problems like Max-Cut, the Ising model provides a natural formulation:

- Graph vertices correspond to spins
- Edge weights correspond to coupling strengths $J_{ij}$
- The cut value is maximized when connected vertices have opposite spins

The **cut value** for a given spin configuration is:

$$\text{Cut}(\mathbf{s}) = \sum_{(i,j) \in E} J_{ij} \frac{1-s_i s_j}{2}$$

where $E$ is the set of edges in the graph.

### Simulated Bifurcation Approach

Simulated Bifurcation algorithms solve the discrete optimization problem by:

1. **Continuous Relaxation**: Map discrete spins $s_i \in \{-1,+1\}$ to continuous variables $x_i \in \mathbb{R}$
2. **Dynamical System**: Evolve the system using differential equations that naturally bifurcate toward $\pm 1$ solutions
3. **Energy Minimization**: The dynamics are designed to minimize the Ising Hamiltonian while driving variables toward binary values

## Overview

This project implements four variants of Simulated Bifurcation algorithms:

- **aSB** (Adiabatic Simulated Bifurcation)
- **bSB** (Ballistic Simulated Bifurcation)
- **dSB** (Discrete Simulated Bifurcation)
- **sSB** (Simplified Simulated Bifurcation)

These algorithms are designed to solve combinatorial optimization problems by mapping them to Ising models and using bifurcation dynamics to find optimal solutions.

## Features

- **Multiple Algorithm Variants**: Compare performance across different SB implementations
- **GPU Acceleration**: CUDA support for large-scale problems (automatically detected)
- **Comprehensive Benchmarking**: Built-in tools for performance analysis and visualization
- **Flexible Parameters**: Customizable hyperparameters (β, η, ξ) for different problem scales
- **Rich Visualization**: Generate plots for trajectory analysis and performance comparison
- **Interactive Demo**: Jupyter notebook with step-by-step examples
- **Efficient Implementation**: Optimized PyTorch backend with batch processing

## Installation

### Requirements

- Python ≥ 3.12
- PyTorch (with CUDA support recommended for large problems)
- NumPy, Matplotlib, Pandas, tqdm

### Using uv (recommended)

```bash
git clone https://github.com/xsjk/Simulated-Bifurcation.git
cd Simulated-Bifurcation
uv sync
```

## Quick Start

### Run the Interactive Demo

The demo notebook `demo.ipynb` includes:

- Basic algorithm comparison on small problems
- Large-scale benchmarking examples
- Hyperparameter analysis
- Performance visualization

### Basic Usage

```python
import numpy as np
from solver import Solver

# Load a 2000×2000 Ising matrix (negative for Max-Cut formulation)
J = -np.load("data/k2000.npy")

# Initialize the solver
solver = Solver()

# Solve using ballistic Simulated Bifurcation (bSB) algorithm
# beta: controls the growth rate of bifurcation parameter
# eta: time step size for numerical integration
result = solver.solve(J, method="bSB", beta=0.01, eta=0.001)

# Access the results:
# result.x: trajectory of position variables over time
# result.y: trajectory of velocity variables over time
# result.g: trajectory of acceleration variables over time
# result.V: potential energy evolution over time
# result.H: Hamiltonian (total energy) evolution over time
# result.cut: cut values (objective function) over time

# Get the best solution found
best_solution = result.best_x
print(f"Best solution: {best_solution}")
print(f"Best cut value: {result.cut.max()}")
```

#### Parameters

- **β**: Growth rate of the bifurcation parameter p(t)

  - Smaller values: More stable convergence, slower speed
  - Larger values: Faster convergence, potential instability

- **η**: Time step size

  - Smaller values: Higher precision, slower computation
  - Larger values: Faster computation, potential numerical errors

- **ξ**: Coupling strength
  - Default: $1 / (2\sqrt{N})$ where N is the problem size
  - Controls the strength of interactions between variables

### Performance Comparison

Use the built-in benchmarking tools to compare different algorithms:

```python
from benchmark import benchmark_plot

# Compare all four SB methods on a small problem
# This will generate performance plots showing convergence behavior
benchmark_plot(
    J=np.array([[0, 1], [1, 0]]),  # Simple 2x2 coupling matrix
    beta=0.01,                     # Bifurcation parameter growth rate
    eta=0.001,                     # Time step size
    methods=["aSB", "bSB", "dSB", "sSB"],  # All available methods
    verbose=True,                  # Print detailed progress information
)
```

## Demo Example

The algorithms show different performance characteristics. The demo notebook provides a visual comparison of the trajectories and cut values over time for each method.

### Small Scale Example (2×2)

![Small Scale - Trajectory](figures/benchmark_2x2_beta_0.01_eta_0.001_traj_time.png)

![Small Scale - History](figures/benchmark_2x2_beta_0.01_eta_0.001_history.png)

![Small Scale - Cut](figures/benchmark_2x2_beta_0.01_eta_0.001_cut.png)

### Large Scale Example (2000×2000)

#### Small Time Step Configuration

![Large Scale - Small Time Step - Trajectory](figures/benchmark_2000x2000_eta_2^-6_beta_2^-8_traj_time.png)

![Large Scale - Small Time Step - History](figures/benchmark_2000x2000_eta_2^-6_beta_2^-8_history.png)

![Large Scale - Small Time Step - Cut](figures/benchmark_2000x2000_eta_2^-6_beta_2^-8_cut.png)

#### Big Time Step Configuration

![Large Scale - Small Time Step - Trajectory](figures/benchmark_2000x2000_eta_2^-1_beta_2^-11_traj_time.png)

![Large Scale - Small Time Step - History](figures/benchmark_2000x2000_eta_2^-1_beta_2^-11_history.png)

![Large Scale - Small Time Step - Cut](figures/benchmark_2000x2000_eta_2^-1_beta_2^-11_cut.png)

### Hyperparameter Analysis

![Hyperparameter Heatmap](figures/hyperparameter_heatmap.png)

## References

1. Goto, H., Endo, K., Suzuki, M., Sakai, Y., Kanao, T., Hamakawa, Y., Hidaka, R., Yamasaki, M., & Tatsumura, K. (2021). High-performance combinatorial optimization based on classical mechanics. _Science Advances_, 7(6), eabe7953.

2. Zhang, T., Zhang, H., Yu, Z., Liu, S., & Han, J. (2024). A high-performance stochastic simulated bifurcation Ising machine. In _Proceedings of the 61st ACM/IEEE Design Automation Conference_ (pp. 1-6).
