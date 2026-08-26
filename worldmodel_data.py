from __future__ import annotations

import numpy as np
import torch


def generate_road_profile(L=200.0, n_points=200, seed=None):
    rng = np.random.default_rng(seed)
    s = np.linspace(0, L, n_points)
    long_wave = rng.uniform(0.012, 0.028) * np.sin(np.pi * s / rng.uniform(40.0, 70.0) + rng.uniform(-np.pi, np.pi))
    short_wave = rng.uniform(0.004, 0.014) * np.sin(np.pi * s / rng.uniform(12.0, 30.0) + rng.uniform(-np.pi, np.pi))
    kappa = long_wave + short_wave
    d = 22 + 6 * np.sin(np.pi * s / rng.uniform(80.0, 140.0) + rng.uniform(-np.pi, np.pi))
    obstacle_center = rng.uniform(20.0, L - 20.0)
    obstacle_width = rng.uniform(6.0, 18.0)
    obstacle_drop = rng.uniform(6.0, 14.0)
    d -= obstacle_drop * np.exp(-((s - obstacle_center) ** 2) / (2 * obstacle_width**2))
    d = np.clip(d, 3, 30)
    return s, kappa, d


def generate_dataset(num_samples=1000, L=200.0, n_points=200, dt=0.1, speed=20.0):
    s = np.linspace(0, L, n_points)
    ds = s[1] - s[0]
    data = []
    for i in range(num_samples):
        _, kappa, d = generate_road_profile(L, n_points, seed=i)
        v = np.random.uniform(15, 25)
        shift = dt * v
        shift_idx = int(shift / ds)
        kappa_next = np.roll(kappa, -shift_idx)
        d_next = np.roll(d, -shift_idx)
        if shift_idx > 0:
            kappa_next[-shift_idx:] = 0
            d_next[-shift_idx:] = 30
        data.append((kappa, d, v, kappa_next, d_next, v))
    return data


def build_default_adjacency():
    return torch.tensor(
        [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
        dtype=torch.bool,
    )
