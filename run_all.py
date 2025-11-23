"""
Run complete N-Pendulum simulation pipeline
"""

import function_generator
import simulator
import animator

def main():
    """
    Complete pipeline:
    1. Generate equations of motion (symbolic)
    2. Run numerical simulation
    3. Create animation/video
    """
    
    print("=" * 60)
    print("N-PENDULUM CHAOTIC SIMULATION")
    print("=" * 60)
    print()
    
    # Configuration
    N = 5             # Number of pendulum segments
    T = 100            # Simulation time (seconds)
    M = 100             # Number of pendulum instances
    perturbation = 1e-6  # Initial condition perturbation
    use_cpp_renderer = True  # Set True to use the C++ video renderer
    
    print(f"Configuration:")
    print(f"  N (segments): {N}")
    print(f"  Duration: {T} seconds")
    print(f"  Number of instances: {M}")
    print(f"  Perturbation: {perturbation:.2e}")
    print(f"  Video backend: {'cpp' if use_cpp_renderer else 'matplotlib'}")
    print()
    
    # Step 1: Generate equations
    print("STEP 1: Generating symbolic equations of motion...")
    print("-" * 60)
    function_generator.generate_pendulum_equations(N=N)
    print()
    
    # Step 2: Run simulation
    print("STEP 2: Running numerical simulation...")
    print("-" * 60)
    simulator.simulate_pendulum(N=N, T=T, M=M, perturbation=perturbation)
    print()
    
    # Step 3: Create animation
    print("STEP 3: Creating animation...")
    print("-" * 60)
    animator.animate_pendulum(
        save_video=True,
        video_filename=f'{N}_pendulum_{M}_instances.mp4',
        backend='cpp' if use_cpp_renderer else 'matplotlib',
    )
    print()
    
    print("=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()

