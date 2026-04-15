# Fractal Explorer

A real-time interactive fractal viewer built with Python, Numba JIT, and Pygame. Renders Mandelbrot, Julia, Burning Ship, Tricorn, and Newton fractals at full resolution in real time with smooth zoom and pan.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Pygame](https://img.shields.io/badge/Pygame-2.6-green) ![Numba](https://img.shields.io/badge/Numba-JIT-orange)

---

## Setup & Run (All Methods)

### Method 1 -- Direct run with Python 3.12 (recommended)
```bash
py -3.12 -m pip install numpy numba pygame matplotlib
py -3.12 fractal_explorer.py
```

### Method 2 -- Virtual environment
```bash
py -3.12 -m venv venv
venv\Scriptsctivate
pip install numpy numba pygame matplotlib
python fractal_explorer.py
```

### Method 3 -- If python3.12 is your default python
```bash
pip install numpy numba pygame matplotlib
python fractal_explorer.py
```

> First launch takes 10-20 seconds to JIT-compile the kernels. Subsequent launches are instant due to `cache=True`.


## Features

- 5 fractals: Mandelbrot, Julia, Burning Ship, Tricorn, Newton (z^3 - 1)
- Real-time zoom toward cursor and left-click pan
- 8 colormaps, cycling with a single key
- Smooth (continuous) coloring via the normalized iteration algorithm
- Parallel CPU rendering using Numba `prange`
- HUD overlay with FPS, zoom level, and controls
- PNG screenshot export

---

## Requirements

```
pip install numpy numba pygame matplotlib
```

> **Important:** Use Python 3.12. Python 3.14+ is not yet supported by Numba or Pygame.

---

## Running

```bash
py -3.12 fractal_explorer.py
```

---

## Controls

| Key / Input       | Action                              |
|-------------------|-------------------------------------|
| Mouse wheel       | Zoom in / out toward cursor         |
| Left drag         | Pan                                 |
| 1 2 3 4 5         | Switch fractal                      |
| C                 | Cycle colormap                      |
| Up / Down         | Increase / decrease max iterations  |
| R                 | Reset view                          |
| S                 | Save PNG screenshot                 |
| H                 | Toggle HUD                          |
| Esc               | Quit                                |

---

## Code Walkthrough

### Imports (lines 28-34)

```python
import sys
import os
import math
import numpy as np
import pygame
from numba import njit, prange
import matplotlib.cm as cm
```

- `sys`, `os`, `math` - standard library: system calls, file paths, and math functions like `log` and `sqrt`.
- `numpy` - provides the multi-dimensional float32 arrays that store each pixel's iteration value.
- `pygame` - handles the window, event loop, keyboard/mouse input, and drawing pixels to the screen.
- `numba.njit` - the "no-Python JIT" decorator; compiles Python functions to native machine code at first call.
- `numba.prange` - a parallel version of `range`; distributes loop iterations across CPU cores automatically.
- `matplotlib.cm` - used only to sample colormaps (e.g. inferno, viridis) into lookup tables.

---

### Global constant (line 42)

```python
LOG2 = math.log(2.0)
```

Precomputed at module load time so it is not recalculated inside the JIT kernels on every pixel.

---

### Mandelbrot kernel (lines 45-71)

```python
@njit(parallel=True, fastmath=True, cache=True)
def k_mandelbrot(W, H, xmin, xmax, ymin, ymax, max_iter):
```

The `@njit` decorator tells Numba to:
- `parallel=True` - run the outer `prange` loop across all CPU cores.
- `fastmath=True` - allow floating-point reassociation for speed (slightly less IEEE-754 strict).
- `cache=True` - save the compiled binary to disk so the first run of subsequent sessions is fast.

```python
    out = np.empty((H, W), dtype=np.float32)
    dx = (xmax - xmin) / W
    dy = (ymax - ymin) / H
```

`out` is the H x W output array. `dx` and `dy` are the step sizes in the complex plane per pixel -- how much the real and imaginary parts increase as you move one pixel right or down.

```python
    for j in prange(H):
        cy = ymin + j * dy
        for i in range(W):
            cx = xmin + i * dx
```

The outer loop (rows) is parallelized across cores. For each pixel (i, j), (cx, cy) is the corresponding point c in the complex plane.

```python
            zx = 0.0
            zy = 0.0
            n = 0
            while n < max_iter:
                zx2 = zx * zx
                zy2 = zy * zy
                if zx2 + zy2 > 256.0:
                    break
                zy = 2.0 * zx * zy + cy
                zx = zx2 - zy2 + cx
                n += 1
```

The Mandelbrot iteration: z -> z^2 + c, starting from z = 0. In real arithmetic:
- `zx` = Re(z), `zy` = Im(z)
- z^2 = (zx^2 - zy^2) + i*(2*zx*zy)
- Escape condition is |z|^2 > 256 (radius 16) instead of the usual 4; the larger radius gives more accurate smooth coloring.

```python
            if n >= max_iter:
                out[j, i] = -1.0
            else:
                mag = math.sqrt(zx * zx + zy * zy)
                nu = math.log(math.log(mag) / LOG2) / LOG2
                out[j, i] = n + 1.0 - nu
```

- Points that never escape get `-1.0` (rendered black).
- All other points use the **normalized iteration count** formula. The raw count `n` produces hard color bands. Subtracting `nu` (derived from how far past the escape radius the orbit flew) interpolates smoothly between levels, eliminating banding.

---

### Julia kernel (lines 74-99)

```python
def k_julia(W, H, xmin, xmax, ymin, ymax, max_iter, cre, cim):
```

Identical structure to Mandelbrot, but the roles of z and c are swapped:
- Each pixel's (x, y) coordinate is the **starting point** z, not the constant c.
- c is fixed to `cre + i*cim` (set to -0.7 + 0.27015i), giving the classic Julia set shape.
- The iteration is the same: z -> z^2 + c.

---

### Burning Ship kernel (lines 102-129)

```python
def k_burning_ship(W, H, xmin, xmax, ymin, ymax, max_iter):
```

The formula is z -> (|Re(z)| + i*|Im(z)|)^2 + c. The key difference from Mandelbrot:

```python
    zy = 2.0 * abs(zx * zy) + cy   # abs() on the imaginary product
    zx = zx2 - zy2 + cx
```

The `abs()` folds the orbit into the first quadrant at every step, producing the ship-like structure.

```python
    cy = -(ymin + j * dy)
```

The y-axis is flipped so the "ship" appears right-side up, matching the conventional orientation.

---

### Tricorn kernel (lines 132-159)

```python
def k_tricorn(W, H, xmin, xmax, ymin, ymax, max_iter):
```

Also called the Mandelbar. The formula is z -> conj(z)^2 + c. Conjugation negates the imaginary part, so the only change from Mandelbrot is a minus sign:

```python
    zy = -2.0 * zx * zy + cy   # minus sign here
    zx = zx2 - zy2 + cx
```

---

### Newton kernel (lines 162-214)

```python
def k_newton(W, H, xmin, xmax, ymin, ymax, max_iter):
```

Applies Newton's root-finding method to f(z) = z^3 - 1, which has three roots (cube roots of unity):

```python
    r0x, r0y =  1.0,  0.0                   # root 0:  1
    r1x, r1y = -0.5,  0.8660254037844386    # root 1: -1/2 + i*sqrt(3)/2
    r2x, r2y = -0.5, -0.8660254037844386    # root 2: -1/2 - i*sqrt(3)/2
```

Each iteration applies z = z - f(z)/f'(z) using complex arithmetic:

```python
    # f = z^3 - 1
    fx = zx2 * zx - zy2 * zy - 1.0
    fy = zx2 * zy + zy2 * zx

    # f' = 3z^2
    dfx = 3.0 * (zx * zx - zy * zy)
    dfy = 3.0 * 2.0 * zx * zy

    # z -= f / f' (complex division)
    denom = dfx * dfx + dfy * dfy
    zx -= (fx * dfx + fy * dfy) / denom
    zy -= (fy * dfx - fx * dfy) / denom
```

After convergence, each pixel is encoded as `root * 64 + iteration_count`, producing three distinct hue regions with smooth shading inside each.

---

### Colorize kernel (lines 221-244)

```python
@njit(parallel=True, fastmath=True, cache=True)
def colorize(values, lut, vmax):
```

Maps the float32 iteration array to RGB using a prebuilt lookup table:

```python
    t = math.sqrt(v / vmax)        # gamma 0.5
    idx = int(t * (n_lut - 1))
    out[j, i, 0] = lut[idx, 0]
    out[j, i, 1] = lut[idx, 1]
    out[j, i, 2] = lut[idx, 2]
```

A square-root gamma (0.5) is applied before indexing. This perceptually linearizes the gradient -- without it, deep-iteration bands near the set boundary look too compressed.

---

### make_lut (lines 247-250)

```python
def make_lut(name, n=512):
    cmap = cm.get_cmap(name)
    arr = (cmap(np.linspace(0.0, 1.0, n))[:, :3] * 255).astype(np.uint8)
    return arr
```

Samples a matplotlib colormap at 512 evenly-spaced points and converts to a (512, 3) uint8 array. All 8 colormaps are built once at startup; switching is then just a pointer swap, not a recomputation.

---

### Fractal registry (lines 257-263)

```python
FRACTALS = [
    ("Mandelbrot",    k_mandelbrot,    (-2.5,  1.0, -1.25, 1.25)),
    ("Julia",         k_julia,         (-1.6,  1.6, -1.2,  1.2)),
    ("Burning Ship",  k_burning_ship,  (-2.0,  1.5, -2.0,  1.0)),
    ("Tricorn",       k_tricorn,       (-2.0,  2.0, -2.0,  2.0)),
    ("Newton z^3-1",  k_newton,        (-2.0,  2.0, -2.0,  2.0)),
]
```

Each entry is a (name, kernel_function, initial_view) tuple. The initial view (xmin, xmax, ymin, ymax) is tuned to show the most interesting region of each fractal.

---

### run() -- Initialization (lines 275-310)

```python
pygame.init()
W, H = 900, 650
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas,monospace", 14)
```

Sets up the Pygame window (900x650), a frame-rate clock, and a monospace font for the HUD.

```python
luts = [make_lut(n) for n in CMAP_NAMES]
```

Builds all 8 lookup tables at startup so colormap switching has zero cost at runtime.

```python
cy = (ymin + ymax) / 2
half_h = (xmax - xmin) * H / W / 2
ymin, ymax = cy - half_h, cy + half_h
```

Adjusts the initial y-range to exactly match the window's aspect ratio, preventing the fractal from appearing stretched.

```python
for _, kfn, _ in FRACTALS:
    kfn(32, 32, -1.0, 1.0, -1.0, 1.0, 30, ...)
print("Ready.")
```

Runs each kernel once on a tiny 32x32 image to trigger Numba's JIT compilation at startup. Without this warmup, the first interactive frame would freeze for 1-2 seconds.

---

### run() -- Keyboard events (lines 327-348)

```python
elif event.key == pygame.K_UP:
    max_iter = min(4000, int(max_iter * 1.25) + 1)
elif event.key == pygame.K_DOWN:
    max_iter = max(20, int(max_iter / 1.25))
```

Iterations scale multiplicatively (x1.25 / /1.25) so each keypress is a consistent perceptual step. Clamped to [20, 4000].

```python
elif event.key in (pygame.K_1, ..., pygame.K_5):
    fractal_idx = event.key - pygame.K_1
```

`pygame.K_1` through `pygame.K_5` are consecutive integers, so subtracting `pygame.K_1` converts the key directly to a 0-based index.

---

### run() -- Mouse wheel zoom (lines 350-359)

```python
cx = xmin + (mx / W) * (xmax - xmin)
cy = ymin + (my / H) * (ymax - ymin)
factor = 0.85 if event.y > 0 else 1.0 / 0.85
xmin = cx + (xmin - cx) * factor
xmax = cx + (xmax - cx) * factor
ymin = cy + (ymin - cy) * factor
ymax = cy + (ymax - cy) * factor
```

Converts the mouse pixel position to complex-plane coordinates, then scales all four viewport edges toward or away from that point. factor < 1 shrinks the viewport (zoom in); factor > 1 expands it (zoom out). The point under the cursor stays stationary.

---

### run() -- Mouse drag pan (lines 361-377)

```python
elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
    drag_anchor = (mx, my, xmin, xmax, ymin, ymax)
```

Records the mouse position and the full viewport state at button-down.

```python
elif event.type == pygame.MOUSEMOTION and dragging:
    dxd = (mx - ax_mx) / W * (ax_xmax - ax_xmin)
    xmin = ax_xmin - dxd
```

On every motion event, recalculates the offset from the anchor in complex-plane units. Panning from the anchor (not cumulatively) prevents floating-point drift.

---

### run() -- Render (lines 379-390)

```python
data = kernel(W, H, xmin, xmax, ymin, ymax, max_iter)
rgb  = colorize(data, luts[cmap_idx], float(max_iter))
surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
screen.blit(surf, (0, 0))
```

Every frame:
1. Call the active kernel -> H x W float32 array of iteration values.
2. Colorize it -> H x W x 3 uint8 RGB array.
3. `swapaxes(0, 1)` transposes from (H, W, 3) to (W, H, 3) because Pygame's surfarray uses column-major (x, y) order.
4. Blit the surface onto the screen.

---

### run() -- HUD (lines 392-408)

```python
shadow = font.render(line, True, (0, 0, 0))
text   = font.render(line, True, (255, 255, 255))
screen.blit(shadow, (11, 11 + i * 17))
screen.blit(text,   (10, 10 + i * 17))
```

Each line is rendered twice -- black offset by 1 pixel (shadow), then white on top -- making text readable over any fractal color.

```python
zoom = 3.5 / max(xmax - xmin, 1e-300)
```

Zoom is expressed relative to the full Mandelbrot width (3.5 units). At 1x you see the whole set; at 1000x you are 1000 times deeper.

---

### run() -- Frame end (lines 410-411)

```python
pygame.display.flip()
clock.tick(120)
```

`display.flip()` pushes the completed frame to the screen. `clock.tick(120)` caps at 120 FPS, but the render time (CPU-bound) is the real limiter.

---

### Entry point (lines 414-415)

```python
if __name__ == "__main__":
    run()
```

Standard Python guard -- ensures `run()` is only called when the script is executed directly, not when imported as a module.

---

## How the Math Works

### Escape-time fractals (Mandelbrot, Julia, Burning Ship, Tricorn)

Each pixel represents a point in the complex plane. The iteration formula is applied repeatedly until the orbit escapes (|z| > 16) or max_iter is reached. Points that never escape are colored black. Escaped points are colored by how quickly they left, smoothed with:

```
smooth_n = n + 1 - log(log(|z|) / log(2)) / log(2)
```

### Newton fractal

Newton's method is applied to f(z) = z^3 - 1. The plane splits into three basins of attraction, one per root. Points are colored by which root they converge to and how many iterations it took.

---

## Project Structure

```
Fractal Explorer/
|-- fractal_explorer.py   # All source code
|-- README.md
```
