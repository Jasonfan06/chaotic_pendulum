"""
N-Pendulum Lagrangian Equation Generator
Uses symbolic math to derive equations of motion
"""

import numpy as np
import sympy as sp
from sympy import symbols, Function, diff, simplify, solve, cos, sin
import dill
import time

def generate_pendulum_equations(N=3):
    """
    Generate symbolic equations of motion for N-pendulum using Lagrange's equations
    
    Parameters:
    -----------
    N : int
        Number of pendulum segments
    """
    print(f"Generating equations for N={N} pendulum...")
    
    # Create symbolic variables
    t = symbols('t', real=True)
    theta = [Function(f'theta{i+1}')(t) for i in range(N)]
    
    # Position coordinates for each mass
    x = []
    y = []
    
    for k in range(N):
        if k == 0:
            x.append(sin(theta[k]))
            y.append(-cos(theta[k]))
        else:
            x.append(x[k-1] + sin(theta[k]))
            y.append(y[k-1] - cos(theta[k]))
    
    # Kinetic energy for each mass
    E = []
    for k in range(N):
        dx_dt = diff(x[k], t)
        dy_dt = diff(y[k], t)
        Ek = sp.Rational(1, 2) * (dx_dt**2 + dy_dt**2)
        E.append(Ek)
    
    # Potential energy, kinetic energy, and Lagrangian
    V = sum(y)  # Using unit mass and g=1
    T = sum(E)
    L = T - V
    
    print("Deriving Lagrangian equations...")
    
    # Lagrange's equations
    equations = []
    for k in range(N):
        dL_dtheta_dot = diff(L, diff(theta[k], t))
        d_dt_dL_dtheta_dot = diff(dL_dtheta_dot, t)
        dL_dtheta = diff(L, theta[k])
        eq = d_dt_dL_dtheta_dot - dL_dtheta
        equations.append(eq)
    
    # Create symbols for substitution
    b = symbols(f'b1:{N+1}', real=True)  # theta values
    c = symbols(f'c1:{N+1}', real=True)  # theta dot values
    a = symbols(f'a1:{N+1}', real=True)  # theta double dot values
    
    # Substitute symbolic functions with algebraic symbols
    subs_dict = {}
    for k in range(N):
        subs_dict[theta[k]] = b[k]
        subs_dict[diff(theta[k], t)] = c[k]
        subs_dict[diff(theta[k], t, 2)] = a[k]
    
    equations_sub = [eq.subs(subs_dict) for eq in equations]
    equations_sub = [simplify(eq) for eq in equations_sub]
    
    print("Solving for accelerations...")
    tic = time.time()
    solution = solve(equations_sub, a)
    toc = time.time()
    print(f"Solved in {toc-tic:.2f} seconds")
    
    # Extract acceleration expressions
    accelerations = [solution[a[k]] for k in range(N)]
    
    print("Simplifying...")
    tic = time.time()
    accelerations = [simplify(acc) for acc in accelerations]
    toc = time.time()
    print(f"Simplified in {toc-tic:.2f} seconds")
    
    # Convert to numerical functions
    print("Converting to numerical functions...")
    
    # Create function that takes state vector u = [theta1, ..., thetaN, omega1, ..., omegaN]
    u_syms = list(b) + list(c)
    
    # Convert accelerations to lambda functions
    accel_funcs = [sp.lambdify(u_syms, acc, 'numpy') for acc in accelerations]
    
    def equations_of_motion(t, u):
        """
        Equations of motion for N-pendulum
        
        Parameters:
        -----------
        t : float
            Time
        u : array_like
            State vector [theta1, ..., thetaN, omega1, ..., omegaN]
        
        Returns:
        --------
        du : array
            Derivative of state vector
        """
        theta_vals = u[:N]
        omega_vals = u[N:]
        
        # Calculate accelerations
        alpha_vals = np.array([func(*u) for func in accel_funcs])
        
        return np.concatenate([omega_vals, alpha_vals])
    
    # Kinetic energy function for energy conservation check
    T_sub = T.subs({theta[k]: b[k] for k in range(N)})
    T_sub = T_sub.subs({diff(theta[k], t): c[k] for k in range(N)})
    T_sub = simplify(T_sub)
    T_func = sp.lambdify(u_syms, T_sub, 'numpy')
    
    def kinetic_energy(u):
        """Calculate kinetic energy from state vector"""
        return T_func(*u)
    
    # Save to file
    data = {
        'N': N,
        'equations_of_motion': equations_of_motion,
        'kinetic_energy': kinetic_energy,
        'accel_funcs': accel_funcs
    }
    
    with open(f'func_N{N}k.pkl', 'wb') as f:
        dill.dump(data, f)
    
    print(f"Saved to func_N{N}k.pkl")
    
    return data


if __name__ == '__main__':
    # Generate equations for N=3 pendulum
    generate_pendulum_equations(N=3)

