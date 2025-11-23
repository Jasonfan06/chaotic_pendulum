"""
N-Pendulum Animation
Create visualization and video of the simulation results
"""

from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

CPP_MAGIC = 0x43485044
CPP_VERSION = 1

def animate_pendulum(
    save_video: bool = True,
    video_filename: str = 'pendulum_animation.mp4',
    playback_speed: float = 1.0,
    backend: str = 'matplotlib',
    cpp_binary: str = 'cpp_renderer/build/pendulum_renderer',
    render_width: int = 1920,
    render_height: int = 1080,
):
    """
    Create animation of N-pendulum simulation.

    Parameters
    ----------
    save_video : bool
        Whether to save animation as video (required for backend='cpp').
    video_filename : str
        Output video filename.
    playback_speed : float
        Relative playback multiplier (>1 faster, <1 slower).
    backend : str
        'matplotlib' for the legacy Python renderer or 'cpp' for the high-performance renderer.
    cpp_binary : str
        Path to the compiled C++ renderer (used when backend='cpp').
    render_width, render_height : int
        Output resolution for the C++ renderer.
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
    axis_limit = max(1.0, N * 1.2)
    backend = (backend or 'matplotlib').lower()

    playback_speed = max(playback_speed, 1e-3)
    base_fps = 60
    render_fps = max(1, int(round(base_fps * playback_speed)))
    actual_speed = render_fps / base_fps
    print(f"Loaded {Frame} frames for {M} pendulums with {N} segments")

    if backend == 'cpp':
        if not save_video:
            print("C++ renderer only supports video export; forcing save_video=True.")
        try:
            _render_with_cpp(
                x=x,
                y=y,
                video_filename=video_filename,
                cpp_binary=cpp_binary,
                fps=base_fps,
                playback_speed=playback_speed,
                width=render_width,
                height=render_height,
                axis_limit=axis_limit,
            )
        except FileNotFoundError as err:
            print(f"Error: {err}")
        return

    if backend != 'matplotlib':
        raise ValueError(f"Unknown backend '{backend}'. Use 'matplotlib' or 'cpp'.")

    # Set up the figure with dark background
    dpi = 100
    fig = plt.figure(figsize=(16, 9), dpi=dpi, facecolor='black')
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, 0.25 * axis_limit)

    # Initialize plot objects
    origin_point, = ax.plot([0], [0], 'o', markersize=6, color='red', zorder=3)
    pendulum_points = []  # Points for the masses
    pendulum_strings = []  # Lines for the strings

    for _ in range(M):
        point, = ax.plot([], [], 'o', markersize=8, color='yellow', zorder=2)
        string, = ax.plot([], [], '-', linewidth=1, color='white', zorder=1)
        pendulum_points.append(point)
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
            pendulum_points[k].set_data(x[frame, 1:, k], y[frame, 1:, k])
            pendulum_strings[k].set_data(x[frame, :, k], y[frame, :, k])

        if (frame + 1) % 30 == 0:
            progress = 100 * (frame + 1) / Frame
            print(f'Animating: {progress:.1f}%', end='\r')

        return [origin_point] + pendulum_points + pendulum_strings

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
        writer = FFMpegWriter(fps=render_fps, bitrate=5000, extra_args=['-vcodec', 'libx264'])
        anim.save(video_filename, writer=writer, dpi=dpi)
        print("Video saved successfully!")
    else:
        print("Displaying animation (close window to exit)...")
        plt.show()

    plt.close()


def _render_with_cpp(
    x: np.ndarray,
    y: np.ndarray,
    video_filename: str,
    cpp_binary: str,
    fps: int,
    playback_speed: float,
    width: int,
    height: int,
    axis_limit: float,
) -> None:
    cpp_path = Path(cpp_binary)
    if not cpp_path.is_file():
        raise FileNotFoundError(
            f"C++ renderer not found at {cpp_path}. Build it via cpp_renderer/CMakeLists.txt."
        )

    target_path = Path(video_filename).resolve()
    with tempfile.TemporaryDirectory(prefix="cpp_renderer_") as tmp_dir:
        payload_path = Path(tmp_dir) / "payload.bin"
        _write_cpp_payload(payload_path, x, y)
        cmd = [
            str(cpp_path.resolve()),
            "--input",
            str(payload_path),
            "--output",
            str(target_path),
            "--width",
            str(width),
            "--height",
            str(height),
            "--fps",
            f"{fps}",
            "--speed",
            f"{playback_speed}",
            "--axis-limit",
            f"{axis_limit}",
        ]
        print(f"Running C++ renderer ({cpp_path})...")
        subprocess.run(cmd, check=True)
        print(f"Video saved successfully to {target_path}")


def _write_cpp_payload(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    if x.shape != y.shape:
        raise ValueError("X and Y arrays must share the same shape")

    frames, nodes, instances = x.shape
    header = struct.pack(
        "<5I",
        CPP_MAGIC,
        CPP_VERSION,
        frames,
        nodes,
        instances,
    )

    x32 = np.ascontiguousarray(x, dtype=np.float32)
    y32 = np.ascontiguousarray(y, dtype=np.float32)

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(x32.tobytes(order="C"))
        handle.write(y32.tobytes(order="C"))


if __name__ == '__main__':
    # Create animation and save as video
    animate_pendulum(save_video=True, video_filename='triple_pendulum_100.mp4')

