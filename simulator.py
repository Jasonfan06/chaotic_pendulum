"""
N-Pendulum Simulation
Numerically integrate the equations of motion
"""

import numpy as np
from scipy.integrate import solve_ivp
import dill
import time

def simulate_pendulum(N=3, T=100, M=100, perturbation=1e-8):
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
    Frame = T * fps
    t = np.linspace(0, T, Frame)
    
    # Initialize position arrays
    x = np.zeros((Frame, N+1, M))
    y = np.zeros((Frame, N+1, M))
    
    print(f"Simulating {M} pendulum instances...")
    tic = time.time()
    
    for ii in range(M):
        if (ii + 1) % 10 == 0:
            print(f"Progress: {ii+1}/{M}")
        
        # Initial conditions: all angles at pi/2 with small perturbation
        u0 = np.concatenate([
            np.ones(N) * np.pi/2 - ii/M * perturbation,  # Initial angles
            np.zeros(N)  # Initial angular velocities
        ])
        
        # Solve ODE with high precision
        sol = solve_ivp(
            equations_of_motion,
            [0, T],
            u0,
            t_eval=t,
            method='DOP853',  # High-order Runge-Kutta method (similar to ode89)
            rtol=1e-13,
            atol=1e-16,
            max_step=5e-3,
            first_step=1e-5
        )
        
        if not sol.success:
            print(f"Warning: Integration failed for instance {ii}")
            continue
        
        u = sol.y.T  # Transpose to get (time, state) shape
        
        # Calculate positions of each mass
        for k in range(N):
            x[:, k+1, ii] = x[:, k, ii] + np.sin(u[:, k])
            y[:, k+1, ii] = y[:, k, ii] - np.cos(u[:, k])
    
    toc = time.time()
    print(f"Simulation completed in {toc-tic:.1f} seconds")
    
    # Save results
    np.savez('simulation_results.npz', t=t, x=x, y=y, N=N, M=M)
    print("Results saved to simulation_results.npz")
    
    return t, x, y


if __name__ == '__main__':
    # Run simulation
    t, x, y = simulate_pendulum(N=3, T=100, M=100)

