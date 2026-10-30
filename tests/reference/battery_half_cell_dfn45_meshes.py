"""Explicit geometry helpers for fresh half-cell DFN mesh development only.

Importing this module neither changes legacy grid validation nor installs a source
mesh. Source adapters must opt in to UserSupplied1DSubMesh with these actual edges.
"""

from collections.abc import Mapping
from math import isfinite
from numbers import Real

import numpy as np


def validate_mesh_grid(grid):
    """Return a new canonical grid; cell counts retain the legacy strict-int rule.

    The two x entries are the real separator and working-electrode domains. No
    development upper bounds are applied here. An omitted radial exponent is 1.
    """
    if not isinstance(grid, Mapping) or set(grid) not in (
        {"x", "r"},
        {"x", "r", "radial_exponent"},
    ):
        raise ValueError("mesh grid requires exactly x, r and optional radial_exponent")
    x, radial = grid["x"], grid["r"]
    if (
        not isinstance(x, (tuple, list))
        or len(x) != 2
        or any(type(n) is not int or n < 2 for n in (*x, radial))
    ):
        raise ValueError("mesh grid requires two real x domains and integer radial cells >= 2")
    exponent = grid.get("radial_exponent", 1.0)
    if isinstance(exponent, bool) or not isinstance(exponent, Real):
        raise ValueError("radial exponent must be a finite real number in [1, 3]")
    try:
        exponent = float(exponent)
    except (OverflowError, ValueError) as exc:
        raise ValueError("radial exponent must be a finite real number in [1, 3]") from exc
    if not isfinite(exponent) or not 1 <= exponent <= 3:
        raise ValueError("radial exponent must be a finite real number in [1, 3]")
    return {"x": tuple(x), "r": radial, "radial_exponent": exponent}


def radial_edges(grid, radius_m):
    """Return float64 spherical faces R*(1-(1-j/N)**p), including 0 and R.

    For p=1 the result is exactly np.linspace, preserving legacy uniform source
    geometry bit for bit. Otherwise normalized uniform physical faces follow the
    native ParticleSpec arithmetic. Shell volumes are 4*pi/3*diff(edges**3), never
    uniform-shell weights; reject geometry that loses positive finite volumes in
    float64. This is geometry construction, with no state or solver operation.
    """
    grid = validate_mesh_grid(grid)
    if isinstance(radius_m, bool) or not isinstance(radius_m, Real):
        raise ValueError("particle radius must be positive and finite")
    try:
        radius = float(radius_m)
    except (OverflowError, ValueError) as exc:
        raise ValueError("particle radius must be positive and finite") from exc
    if not isfinite(radius) or radius <= 0:
        raise ValueError("particle radius must be positive and finite")
    edges = np.linspace(0.0, radius, grid["r"] + 1, dtype=np.float64)
    if grid["radial_exponent"] != 1.0:
        edges = radius * (1 - (1 - edges / radius) ** grid["radial_exponent"])
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        volumes = (4 * np.pi / 3) * np.diff(edges**3)
        total_volume = (4 * np.pi / 3) * np.float64(radius) ** 3
        summed_volume = np.sum(volumes)
    if (
        not np.all(np.isfinite(edges))
        or edges[0] != 0.0
        or edges[-1] != radius
        or not np.all(np.diff(edges) > 0)
        or not np.all(np.isfinite(volumes))
        or not np.all(volumes > 0)
        or not np.isfinite(total_volume)
        or not np.isfinite(summed_volume)
        or not np.isclose(summed_volume, total_volume, rtol=32 * np.finfo(float).eps, atol=0)
    ):
        raise ValueError("particle edges must have positive finite conservative shell volumes")
    return edges


def development_mesh_bounds(grid):
    """Return the grid only within the explicitly bounded fresh development domain.

    This opt-in limit is not a replacement for legacy evaluator guards. It does
    not authorize any development history or change a frozen experiment block.
    """
    grid = validate_mesh_grid(grid)
    if max(grid["x"]) > 96 or grid["r"] > 512:
        raise ValueError("fresh development mesh bounds require x <= 96 and r <= 512")
    return grid
