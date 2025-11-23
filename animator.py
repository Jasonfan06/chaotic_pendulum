"""
N-Pendulum Animation
Create visualization and video of the simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import time

def animate_pendulum(save_video=True, video_filename='pendulum_animation.mp4',
                     playback_speed=1.0):
    """
    Create animation of N-pendulum simulation
    
    Parameters:
    -----------
    save_video : bool
        Whether to save animation as video
    video_filename : str
        Output video filename
    playback_speed : float
        Relative playback multiplier (>1 faster, <1 slower)
    """
    
    print("Loading simulation results...")
    try:
        data = np.load('simulation_results.npz')
        t = data['t']
        x = data['x']
        y = data['y']
        N = int(data['N'])
        M = int(data['M'])
    except FileNotFoundError:
        print("Error: simulation_results.npz not found. Please run simulator.py first.")
        return
    
    Frame = len(t)
    print(f"Loaded {Frame} frames for {M} pendulums with {N} segments")
    
    # Set up the figure with dark background
    dpi = 100
    fig = plt.figure(figsize=(16, 9), dpi=dpi, facecolor='black')
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Determine axis limits
    axis_limit = N * 1.2
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, 0.25 * axis_limit)
    
    # Initialize plot objects
    origin_point, = ax.plot([0], [0], 'o', markersize=6,
                            color='red', zorder=3)
    pendulum_points = []  # Points for the masses
    pendulum_strings = []  # Lines for the strings
    
    for ii in range(M):
        # Yellow dots for masses
        point, = ax.plot([], [], 'o', markersize=8, color='yellow', zorder=2)
        pendulum_points.append(point)
        
        # White lines for strings
        string, = ax.plot([], [], '-', linewidth=1, color='white', zorder=1)
        pendulum_strings.append(string)
    
    def init():
        """Initialize animation"""
        origin_point.set_data([0], [0])
        for point in pendulum_points:
            point.set_data([], [])
        for string in pendulum_strings:
            string.set_data([], [])
        return [origin_point] + pendulum_points + pendulum_strings
    
    def update(frame):
        """Update animation frame"""
        for k in range(M):
            # Update mass positions (skip first point which is origin)
            pendulum_points[k].set_data(x[frame, 1:, k], y[frame, 1:, k])
            
            # Update string positions (include origin)
            pendulum_strings[k].set_data(x[frame, :, k], y[frame, :, k])
        
        # Terminal progress (not shown in video)
        if (frame + 1) % 30 == 0:
            progress = 100 * (frame + 1) / Frame
            print(f'Animating: {progress:.1f}%', end='\r')
        
        return [origin_point] + pendulum_points + pendulum_strings
    
    # Create animation (allowing custom playback speed)
    playback_speed = max(playback_speed, 1e-3)
    base_fps = 60
    render_fps = max(1, int(round(base_fps * playback_speed)))
    actual_speed = render_fps / base_fps
    print(f"Creating animation at {render_fps} fps (~{actual_speed:.2f}x speed)...")
    anim = FuncAnimation(
        fig,
        update,
        frames=Frame,
        init_func=init,
        blit=True,
        interval=1000 / render_fps,
    )
    
    if save_video:
        print(f"Saving video to {video_filename}...")
        writer = FFMpegWriter(fps=render_fps, bitrate=5000, 
                             extra_args=['-vcodec', 'libx264'])
        anim.save(video_filename, writer=writer, dpi=dpi)
        print(f"Video saved successfully!")
    else:
        print("Displaying animation (close window to exit)...")
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    # Create animation and save as video
    animate_pendulum(save_video=True, video_filename='triple_pendulum_100.mp4')

