from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict

import dill
import numpy as np


def _build_equations_of_motion(N: int, gravity: float) -> Callable:
    """
    equations of motion for N-link pendulum.
    """

    # Coupling matrix counts how many masses lie beyond joint i/j
    idx = np.arange(N, dtype=float)
    coupling = (N - np.maximum.outer(idx, idx)).astype(float)
    gravity_multipliers = (N - idx).astype(float)

    def equations_of_motion(t: float, u: np.ndarray) -> np.ndarray:
        """Return time derivative of state [theta, omega]."""

        theta = u[:N]
        omega = u[N:]

        delta = theta[:, None] - theta[None, :]
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)

        mass_matrix = coupling * cos_delta
        omega_sq = omega**2

        centrifugal = (coupling * sin_delta) @ omega_sq
        gravity_term = gravity * gravity_multipliers * np.sin(theta)

        rhs = -centrifugal - gravity_term

        try:
            theta_ddot = np.linalg.solve(mass_matrix, rhs)
        except np.linalg.LinAlgError:
            theta_ddot, *_ = np.linalg.lstsq(mass_matrix, rhs, rcond=None)

        return np.concatenate([omega, theta_ddot])

    return equations_of_motion


def generate_pendulum_equations(
    N: int = 3,
    gravity: float = 9.81,
    output_dir: str | Path = ".",
) -> Dict[str, Callable]:
    """
    Build and persist the equations of motion for an N-link pendulum.

    Returns the dictionary that is written to disk for convenience.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"func_N{N}k.pkl"

    print(f"Building equations of motion for N={N}...")
    tic = time.time()
    equations_of_motion = _build_equations_of_motion(N, gravity)
    payload = {
        "N": N,
        "gravity": gravity,
        "equations_of_motion": equations_of_motion,
    }

    with open(filename, "wb") as f:
        dill.dump(payload, f)

    toc = time.time()
    print(f"Saved equations to {filename} in {toc - tic:.2f} s")
    return payload


if __name__ == "__main__":
    generate_pendulum_equations()
