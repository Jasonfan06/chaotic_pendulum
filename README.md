# Chaotic N-Pendulum Simulation

A Python implementation of a chaotic N-pendulum simulation using Lagrangian mechanics. Extreme sensitivity of chaotic systems to initial conditions.


https://github.com/user-attachments/assets/e6367f40-4ac1-4c5f-8bbf-db284bfeb968



## Installation

```bash
pip install -r requirements.txt
```

**Note**: For video export, you'll need FFmpeg installed:
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/)

## Usage
```bash
python run_all.py
```
**Config**: Edit the configuration in `run_all.py`:

```python
N = int             # Number of pendulum segments (2, 3, 4, ...)
T = int            # Simulation duration in seconds
M = int            # Number of pendulum instances
pertubation = float # Positional deviation from each other
```

## Output

- `func_N{N}k.pkl`: Saved equations of motion
- `simulation_results.npz`: Simulation data (positions over time)
- `{N}_pendulum_{M}_instances.mp4`: Final animation video


## C++ Renderer

A C++ renderer is available for high-resolution, high-instance videos.
It reads the simulated trajectories from Python and streams raw RGB frames directly into FFmpeg.

### Build

```bash
cd cpp_renderer
cmake -S . -B build
cmake --build build --config Release
```

Requirements:

- CMake ≥ 3.16 (`brew install cmake` on macOS, `sudo apt install cmake` on Debian/Ubuntu)
- A C++17-capable compiler such as clang++ or g++ (Xcode command-line tools or `build-essential`)
- FFmpeg available on your PATH (`brew install ffmpeg`, `sudo apt install ffmpeg`)

### Use

- Toggle `use_cpp_renderer = True` inside `run_all.py`.
