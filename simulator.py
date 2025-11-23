"""
N-Pendulum Simulation
Numerically integrate the equations of motion
"""

import multiprocessing as mp
import time
from typing import Iterable, Tuple

import dill
import numpy as np
from scipy.integrate import solve_ivp

_worker_equations = None
_worker_t = None
_worker_T = None
_worker_solver_kwargs = None
_worker_N = None
_worker_M = None
_worker_perturbation = None


def _worker_init(equations_blob: bytes, t: np.ndarray, T: float, solver_kwargs: dict, N: int, M: int, perturbation: float) -> None:
    """Initializer for worker processes; restores shared context."""
    global _worker_equations, _worker_t, _worker_T, _worker_solver_kwargs, _worker_N, _worker_M, _worker_perturbation
    _worker_equations = dill.loads(equations_blob)
    _worker_t = t
    _worker_T = T
    _worker_solver_kwargs = solver_kwargs
    _worker_N = N
    _worker_M = M
    _worker_perturbation = perturbation


def _worker_simulate_single(index: int) -> Tuple[int, np.ndarray]:
    """Integrate a single pendulum instance inside a worker process."""
    if _worker_equations is None:
        raise RuntimeError("Worker equations not initialized")

    initial_angles = np.ones(_worker_N) * np.pi / 2 - index / _worker_M * _worker_perturbation
    u0 = np.concatenate([initial_angles, np.zeros(_worker_N)])
    sol = solve_ivp(
        _worker_equations,
        [0, _worker_T],
        u0,
        t_eval=_worker_t,
        **_worker_solver_kwargs,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed for instance {index}: {sol.message}")
    theta = sol.y[:_worker_N].T  # (Frame, N)
    return index, theta


def _accumulate_positions(theta: np.ndarray, x: np.ndarray, y: np.ndarray, idx: int) -> None:
    """Convert angular positions to Cartesian coordinates for a single instance."""
    for k in range(theta.shape[1]):
        x[:, k + 1, idx] = x[:, k, idx] + np.sin(theta[:, k])
        y[:, k + 1, idx] = y[:, k, idx] - np.cos(theta[:, k])

def simulate_pendulum(
    N: int = 3,
    T: float = 100,
    M: int = 100,
    perturbation: float = 1e-8,
    processes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate M instances of N-pendulum with slightly different initial conditions
    
    Parameters:
    -----------
    N : int
        Number of pendulum segments
    T : float
        Total simulation time
    M : int
        Number of pendulum instances (with slightly different initial conditions)
    perturbation : float
        Small perturbation to initial conditions to show chaos
    processes : int | None
        Number of worker processes to use (default: cpu_count, falls back to sequential when <=1)
    
    Returns:
    --------
    t : array
        Time points
    x : array
        X positions of all masses (shape: Frame x N+1 x M)
    y : array
        Y positions of all masses (shape: Frame x N+1 x M)
    """
    
    print(f"Loading equations for N={N} pendulum...")
    try:
        with open(f'func_N{N}k.pkl', 'rb') as f:
            data = dill.load(f)
        equations_of_motion = data['equations_of_motion']
    except FileNotFoundError:
        print(f"Error: func_N{N}k.pkl not found. Please run function_generator.py first.")
        return None, None, None
    
    fps = 60  # Fixed frames per second
    Frame = int(T * fps)
    t = np.linspace(0, T, Frame)
    
    # Initialize position arrays
    x = np.zeros((Frame, N+1, M))
    y = np.zeros((Frame, N+1, M))
    
    print(f"Simulating {M} pendulum instances...")
    tic = time.time()

    solver_kwargs = dict(
        method='DOP853',  # High-order Runge-Kutta method (similar to ode89)
        rtol=1e-13,
        atol=1e-16,
        max_step=5e-3,
        first_step=1e-5,
    )

    cpu_total = mp.cpu_count() or 1
    processes = processes or min(M, cpu_total)
    processes = max(1, min(processes, M))
    
    if processes == 1:
        for ii in range(M):
            if (ii + 1) % 10 == 0 or ii + 1 == M:
                print(f"Progress: {ii+1}/{M}")

            initial_angles = np.ones(N) * np.pi / 2 - ii / M * perturbation
            u0 = np.concatenate([initial_angles, np.zeros(N)])
            sol = solve_ivp(
                equations_of_motion,
                [0, T],
                u0,
                t_eval=t,
                **solver_kwargs,
            )
            if not sol.success:
                print(f"Warning: Integration failed for instance {ii}: {sol.message}")
                continue

            theta = sol.y[:N].T
            _accumulate_positions(theta, x, y, ii)
    else:
        print(f"Using {processes} parallel workers...")
        equations_blob = dill.dumps(equations_of_motion)
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=processes,
            initializer=_worker_init,
            initargs=(equations_blob, t, T, solver_kwargs, N, M, perturbation),
        ) as pool:
            chunk_iter: Iterable[Tuple[int, np.ndarray]] = pool.imap_unordered(_worker_simulate_single, range(M))
            for completed, (idx, theta) in enumerate(chunk_iter, start=1):
                _accumulate_positions(theta, x, y, idx)
                if (completed % 10 == 0) or completed == M:
                    print(f"Progress: {completed}/{M}")
    
    toc = time.time()
    print(f"Simulation completed in {toc-tic:.1f} seconds")
    
    # Save results
    np.savez('simulation_results.npz', t=t, x=x, y=y, N=N, M=M)
    print("Results saved to simulation_results.npz")
    
    return t, x, y


if __name__ == '__main__':
    # Run simulation
    t, x, y = simulate_pendulum(N=3, T=100, M=100)

