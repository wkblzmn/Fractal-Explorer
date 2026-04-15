"""
Fractal Explorer (Real-Time)
============================
A game-loop fractal viewer using Numba JIT + Pygame for true real-time
zoom and pan. Renders every frame at full resolution; aims for 30-60 FPS.

Fractals: Mandelbrot, Julia, Burning Ship, Tricorn, Newton (z^3 - 1).

Requirements:
    pip install numpy numba pygame matplotlib

Run:
    python fractal_explorer.py

Controls:
    Mouse wheel    zoom in/out toward cursor (continuous)
    Left drag      pan
    1 / 2 / 3 / 4 / 5    switch fractal
    C              cycle colormap
    Up / Down      iterations +/-
    [ / ]          shrink / grow window (re-create)
    R              reset view
    S              save PNG screenshot
    H              toggle HUD
    Esc            quit
"""

import sys
import os
import math
import numpy as np
import pygame
from numba import njit, prange
import matplotlib.cm as cm


# ----------------------------------------------------------------------
#  Numba JIT fractal kernels
#  Each kernel writes a float32 "smooth iteration count" array.
# ----------------------------------------------------------------------

LOG2 = math.log(2.0)


@njit(parallel=True, fastmath=True, cache=True)
def k_mandelbrot(W, H, xmin, xmax, ymin, ymax, max_iter):
    out = np.empty((H, W), dtype=np.float32)
    dx = (xmax - xmin) / W
    dy = (ymax - ymin) / H
    for j in prange(H):
        cy = ymin + j * dy
        for i in range(W):
            cx = xmin + i * dx
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
            if n >= max_iter:
                out[j, i] = -1.0  # inside the set
            else:
                mag = math.sqrt(zx * zx + zy * zy)
                nu = math.log(math.log(mag) / LOG2) / LOG2
                out[j, i] = n + 1.0 - nu
    return out


@njit(parallel=True, fastmath=True, cache=True)
def k_julia(W, H, xmin, xmax, ymin, ymax, max_iter, cre, cim):
    out = np.empty((H, W), dtype=np.float32)
    dx = (xmax - xmin) / W
    dy = (ymax - ymin) / H
    for j in prange(H):
        zy0 = ymin + j * dy
        for i in range(W):
            zx = xmin + i * dx
            zy = zy0
            n = 0
            while n < max_iter:
                zx2 = zx * zx
                zy2 = zy * zy
                if zx2 + zy2 > 256.0:
                    break
                zy = 2.0 * zx * zy + cim
                zx = zx2 - zy2 + cre
                n += 1
            if n >= max_iter:
                out[j, i] = -1.0
            else:
                mag = math.sqrt(zx * zx + zy * zy)
                nu = math.log(math.log(mag) / LOG2) / LOG2
                out[j, i] = n + 1.0 - nu
    return out


@njit(parallel=True, fastmath=True, cache=True)
def k_burning_ship(W, H, xmin, xmax, ymin, ymax, max_iter):
    out = np.empty((H, W), dtype=np.float32)
    dx = (xmax - xmin) / W
    dy = (ymax - ymin) / H
    for j in prange(H):
        # Conventional orientation: flip y so the "ship" sits upright
        cy = -(ymin + j * dy)
        for i in range(W):
            cx = xmin + i * dx
            zx = 0.0
            zy = 0.0
            n = 0
            while n < max_iter:
                zx2 = zx * zx
                zy2 = zy * zy
                if zx2 + zy2 > 256.0:
                    break
                zy = 2.0 * abs(zx * zy) + cy
                zx = zx2 - zy2 + cx
                n += 1
            if n >= max_iter:
                out[j, i] = -1.0
            else:
                mag = math.sqrt(zx * zx + zy * zy)
                nu = math.log(math.log(mag) / LOG2) / LOG2
                out[j, i] = n + 1.0 - nu
    return out


@njit(parallel=True, fastmath=True, cache=True)
def k_tricorn(W, H, xmin, xmax, ymin, ymax, max_iter):
    out = np.empty((H, W), dtype=np.float32)
    dx = (xmax - xmin) / W
    dy = (ymax - ymin) / H
    for j in prange(H):
        cy = ymin + j * dy
        for i in range(W):
            cx = xmin + i * dx
            zx = 0.0
            zy = 0.0
            n = 0
            while n < max_iter:
                zx2 = zx * zx
                zy2 = zy * zy
                if zx2 + zy2 > 256.0:
                    break
                # z -> conj(z)^2 + c
                zy = -2.0 * zx * zy + cy
                zx = zx2 - zy2 + cx
                n += 1
            if n >= max_iter:
                out[j, i] = -1.0
            else:
                mag = math.sqrt(zx * zx + zy * zy)
                nu = math.log(math.log(mag) / LOG2) / LOG2
                out[j, i] = n + 1.0 - nu
    return out


@njit(parallel=True, fastmath=True, cache=True)
def k_newton(W, H, xmin, xmax, ymin, ymax, max_iter):
    """Newton's method for z^3 - 1. Stores root_index + smooth fraction."""
    out = np.empty((H, W), dtype=np.float32)
    dx = (xmax - xmin) / W
    dy = (ymax - ymin) / H
    # Three roots of unity
    r0x, r0y = 1.0, 0.0
    r1x, r1y = -0.5, 0.8660254037844386
    r2x, r2y = -0.5, -0.8660254037844386
    tol = 1e-6
    for j in prange(H):
        zy0 = ymin + j * dy
        for i in range(W):
            zx = xmin + i * dx
            zy = zy0
            n = 0
            root = -1
            while n < max_iter:
                # f  = z^3 - 1
                zx2 = zx * zx - zy * zy
                zy2 = 2.0 * zx * zy
                fx = zx2 * zx - zy2 * zy - 1.0
                fy = zx2 * zy + zy2 * zx
                # f' = 3 z^2
                dfx = 3.0 * (zx * zx - zy * zy)
                dfy = 3.0 * 2.0 * zx * zy
                denom = dfx * dfx + dfy * dfy
                if denom == 0.0:
                    break
                # z -= f / f'
                zx -= (fx * dfx + fy * dfy) / denom
                zy -= (fy * dfx - fx * dfy) / denom
                # Convergence check
                d0 = (zx - r0x) ** 2 + (zy - r0y) ** 2
                d1 = (zx - r1x) ** 2 + (zy - r1y) ** 2
                d2 = (zx - r2x) ** 2 + (zy - r2y) ** 2
                if d0 < tol:
                    root = 0
                    break
                if d1 < tol:
                    root = 1
                    break
                if d2 < tol:
                    root = 2
                    break
                n += 1
            if root < 0:
                out[j, i] = -1.0
            else:
                # Encode root + smooth shading from iteration count
                out[j, i] = root * 64.0 + min(n, 63)
    return out


# ----------------------------------------------------------------------
#  Color lookup
# ----------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def colorize(values, lut, vmax):
    H, W = values.shape
    n_lut = lut.shape[0]
    out = np.empty((H, W, 3), dtype=np.uint8)
    for j in prange(H):
        for i in range(W):
            v = values[j, i]
            if v < 0.0:
                out[j, i, 0] = 0
                out[j, i, 1] = 0
                out[j, i, 2] = 0
            else:
                # Gamma 0.5 (sqrt) for a smooth perceptual gradient
                t = math.sqrt(v / vmax) if vmax > 0 else 0.0
                if t < 0.0:
                    t = 0.0
                if t > 1.0:
                    t = 1.0
                idx = int(t * (n_lut - 1))
                out[j, i, 0] = lut[idx, 0]
                out[j, i, 1] = lut[idx, 1]
                out[j, i, 2] = lut[idx, 2]
    return out


def make_lut(name, n=512):
    cmap = cm.get_cmap(name)
    arr = (cmap(np.linspace(0.0, 1.0, n))[:, :3] * 255).astype(np.uint8)
    return arr


# ----------------------------------------------------------------------
#  Fractal registry
# ----------------------------------------------------------------------

FRACTALS = [
    ("Mandelbrot",     k_mandelbrot,    (-2.5,  1.0, -1.25, 1.25)),
    ("Julia",          k_julia,         (-1.6,  1.6, -1.2,  1.2)),
    ("Burning Ship",   k_burning_ship,  (-2.0,  1.5, -2.0,  1.0)),
    ("Tricorn",        k_tricorn,       (-2.0,  2.0, -2.0,  2.0)),
    ("Newton z^3-1",   k_newton,        (-2.0,  2.0, -2.0,  2.0)),
]

CMAP_NAMES = [
    "twilight_shifted", "inferno", "magma", "plasma",
    "viridis", "turbo", "cubehelix", "nipy_spectral",
]


# ----------------------------------------------------------------------
#  Main loop
# ----------------------------------------------------------------------

def run():
    pygame.init()
    pygame.display.set_caption("Fractal Explorer (Numba + Pygame)")

    W, H = 900, 650
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,monospace", 14)

    # State
    fractal_idx = 0
    cmap_idx = 0
    luts = [make_lut(n) for n in CMAP_NAMES]
    max_iter = 200
    julia_c = [-0.7, 0.27015]
    show_hud = True

    name, kernel, init_view = FRACTALS[fractal_idx]
    xmin, xmax, ymin, ymax = init_view
    # Lock initial aspect ratio to the window
    cy = (ymin + ymax) / 2
    half_h = (xmax - xmin) * H / W / 2
    ymin, ymax = cy - half_h, cy + half_h

    dragging = False
    drag_anchor = None  # (mx, my, xmin, xmax, ymin, ymax) at drag start

    # Warm up the JIT compilers so the first interactive frame isn't laggy
    print("Compiling kernels (first run only)...")
    for _, kfn, _ in FRACTALS:
        if kfn is k_julia:
            kfn(32, 32, -1.0, 1.0, -1.0, 1.0, 30, julia_c[0], julia_c[1])
        else:
            kfn(32, 32, -1.0, 1.0, -1.0, 1.0, 30)
    colorize(np.zeros((4, 4), dtype=np.float32), luts[0], 1.0)
    print("Ready.")

    def reset_view():
        nonlocal xmin, xmax, ymin, ymax
        a, b, c, d = FRACTALS[fractal_idx][2]
        xmin, xmax = a, b
        cy = (c + d) / 2
        half_h = (xmax - xmin) * H / W / 2
        ymin, ymax = cy - half_h, cy + half_h

    while True:
        # ---- Input ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                elif event.key == pygame.K_r:
                    reset_view()
                elif event.key == pygame.K_c:
                    cmap_idx = (cmap_idx + 1) % len(CMAP_NAMES)
                elif event.key == pygame.K_h:
                    show_hud = not show_hud
                elif event.key == pygame.K_UP:
                    max_iter = min(4000, int(max_iter * 1.25) + 1)
                elif event.key == pygame.K_DOWN:
                    max_iter = max(20, int(max_iter / 1.25))
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3,
                                   pygame.K_4, pygame.K_5):
                    fractal_idx = event.key - pygame.K_1
                    reset_view()
                elif event.key == pygame.K_s:
                    fname = f"fractal_{FRACTALS[fractal_idx][0].replace(' ', '_')}.png"
                    pygame.image.save(screen, fname)
                    print(f"Saved {os.path.abspath(fname)}")

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                # Convert pixel to data coords
                cx = xmin + (mx / W) * (xmax - xmin)
                cy = ymin + (my / H) * (ymax - ymin)
                factor = 0.85 if event.y > 0 else 1.0 / 0.85
                xmin = cx + (xmin - cx) * factor
                xmax = cx + (xmax - cx) * factor
                ymin = cy + (ymin - cy) * factor
                ymax = cy + (ymax - cy) * factor

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                mx, my = event.pos
                drag_anchor = (mx, my, xmin, xmax, ymin, ymax)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False

            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = event.pos
                ax_mx, ax_my, ax_xmin, ax_xmax, ax_ymin, ax_ymax = drag_anchor
                dxd = (mx - ax_mx) / W * (ax_xmax - ax_xmin)
                dyd = (my - ax_my) / H * (ax_ymax - ax_ymin)
                xmin = ax_xmin - dxd
                xmax = ax_xmax - dxd
                ymin = ax_ymin - dyd
                ymax = ax_ymax - dyd

        # ---- Render ----
        name, kernel, _ = FRACTALS[fractal_idx]
        if kernel is k_julia:
            data = kernel(W, H, xmin, xmax, ymin, ymax, max_iter,
                          julia_c[0], julia_c[1])
        else:
            data = kernel(W, H, xmin, xmax, ymin, ymax, max_iter)

        rgb = colorize(data, luts[cmap_idx], float(max_iter))
        # pygame surfarray wants (W, H, 3)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        screen.blit(surf, (0, 0))

        # ---- HUD ----
        if show_hud:
            fps = clock.get_fps()
            zoom = 3.5 / max(xmax - xmin, 1e-300)
            lines = [
                f"FPS: {fps:5.1f}    {W}x{H}",
                f"Fractal: {name}   [1-5 to switch]",
                f"Iterations: {max_iter}   [Up/Dn]",
                f"Colormap: {CMAP_NAMES[cmap_idx]}   [C to cycle]",
                f"Zoom: {zoom:.2e}x",
                "Wheel: zoom   Drag: pan   R: reset   S: save   H: hide   Esc: quit",
            ]
            for i, line in enumerate(lines):
                shadow = font.render(line, True, (0, 0, 0))
                text = font.render(line, True, (255, 255, 255))
                screen.blit(shadow, (11, 11 + i * 17))
                screen.blit(text,   (10, 10 + i * 17))

        pygame.display.flip()
        clock.tick(120)  # cap, but actual rate is render-bound


if __name__ == "__main__":
    run()