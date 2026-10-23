"""Composition, flow and partial-molar calculus in explicit physical coordinates.

Flows use mol/s, kg/s or declared reference-liquid m3/s; molecular weights
use g/mol. These pure functions never modify an inventory or regularize one
implicitly. Formulae are derived from normalization and extensive n*F.
"""

from collections.abc import Mapping
from math import exp, expm1, fsum, isclose, isfinite, log, log1p


def _finite(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} is outside its physical domain")
    return float(value)


def _vector(values, name, **kwargs):
    result = tuple(_finite(v, name, **kwargs) for v in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _aligned(*vectors):
    if len({len(v) for v in vectors}) != 1:
        raise ValueError("component vectors must have identical lengths")


def normalized_fractions(values):
    """Normalize nonnegative amounts; zero total has no defined composition."""
    values = _vector(values, "amount", nonnegative=True)
    total = sum(values)
    if isfinite(total) and total > 0:
        # Preserve the existing mixture arithmetic (including already-normalized
        # fractions); rescale only when their direct sum overflows.
        return tuple(v / total for v in values)
    scale = max(values)
    if scale == 0:
        raise ValueError("composition requires a positive total")
    scaled = tuple(v / scale for v in values)
    total = fsum(scaled)
    return tuple(v / total for v in scaled)


def regularized_fractions(values, replacement=1e-6):
    """Explicit preprocessing of zeros, never an automatic inventory operation.

    With no zero present, preserve the input values. Otherwise replace zeros
    and normalize. Always return an immutable tuple and leave inputs alone.
    """
    values = _vector(values, "fraction", nonnegative=True)
    replacement = _finite(replacement, "zero replacement", positive=True)
    return normalized_fractions(tuple(v or replacement for v in values)) if 0 in values else values


def fraction_basis(values, component_factors, *, inverse=False):
    """Rebase fractions with positive per-component factors.

    Mole to mass uses MW; mole to reference-volume uses Vm. Inverse=True
    converts mass or reference-volume fractions back to mole fractions.
    """
    values = normalized_fractions(values)
    factors = _vector(component_factors, "component factor", positive=True)
    _aligned(values, factors)
    return normalized_fractions(
        tuple(v / f if inverse else v * f for v, f in zip(values, factors, strict=True))
    )


def convert_component_flows(
    values, *, source, target, molecular_weights=None, reference_molar_volumes=None
):
    """Convert component flows; reference-volume is not the actual mixture volume."""
    values = _vector(values, "flow", nonnegative=True)
    units = {"molar", "mass", "reference_volume"}
    if source not in units or target not in units:
        raise ValueError("flow basis must be molar, mass or reference_volume")
    scales = {"molar": (1.0,) * len(values)}
    if "mass" in {source, target}:
        if molecular_weights is None:
            raise ValueError("mass flow conversion requires molecular weights in g/mol")
        scales["mass"] = tuple(mw / 1000 for mw in _vector(molecular_weights, "MW", positive=True))
    if "reference_volume" in {source, target}:
        if reference_molar_volumes is None:
            raise ValueError("reference-volume flow requires declared molar volumes")
        scales["reference_volume"] = _vector(reference_molar_volumes, "Vm", positive=True)
    _aligned(values, scales[source], scales[target])
    return tuple(v * b / a for v, a, b in zip(values, scales[source], scales[target], strict=True))


def _fractions(values):
    values = _vector(values, "fraction", nonnegative=True)
    if not isclose(fsum(values), 1.0, abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError("fractions must sum to one; normalize explicitly first")
    return values


def weighted_property(fractions, properties, *, rule="linear", exponent=None):
    """Linear, geometric or real power mean of aligned component properties."""
    xs = _fractions(fractions)
    props = _vector(properties, "property")
    _aligned(xs, props)
    if rule == "linear":
        return fsum(x * p for x, p in zip(xs, props, strict=True))
    if rule not in {"logarithmic", "power"}:
        raise ValueError("unknown property mixing rule")
    if any(p <= 0 for p in props):
        raise ValueError("logarithmic and real power means require positive properties")
    r = 0.0 if rule == "logarithmic" else _finite(exponent, "power exponent")
    pairs = [(x, log(p)) for x, p in zip(xs, props, strict=True) if x > 0]
    mean = fsum(x * p for x, p in pairs)
    if r == 0:
        return exp(mean)
    offsets = [r * (p - mean) for _, p in pairs]
    if max(abs(v) for v in offsets) < 0.5:
        return exp(
            mean + log1p(fsum(x * expm1(v) for (x, _), v in zip(pairs, offsets, strict=True))) / r
        )
    shift = max(r * p for _, p in pairs)
    return exp((shift + log(fsum(x * exp(r * p - shift) for x, p in pairs))) / r)


def combine_component_flows(component_ids, *, partial_flows=None, totals=None, fractions=None):
    """Merge any number of streams, preserving IDs and component amount balances.

    Unknown partial streams are explicitly treated as zero contributions,
    matching the public partial-flow convention; this does not infer a flow.
    Equal ID sets retain the first order; other unions are sorted.
    """
    ids = tuple(tuple(row) for row in component_ids)
    if not ids or any(
        not row or len(set(row)) != len(row) or any(not isinstance(i, str) or not i for i in row)
        for row in ids
    ):
        raise ValueError("each stream requires nonempty, unique component IDs")
    if partial_flows is not None:
        if totals is not None or fractions is not None:
            raise ValueError("supply partial flows or total/fraction specs, not both")
        if len(partial_flows) != len(ids):
            raise ValueError("one partial-flow vector is required per stream")
        flows = tuple(
            (0.0,) * len(row)
            if values is None
            else _vector(values, "partial flow", nonnegative=True)
            for row, values in zip(ids, partial_flows, strict=True)
        )
    else:
        if (
            totals is None
            or fractions is None
            or len(totals) != len(ids)
            or len(fractions) != len(ids)
        ):
            raise ValueError("one total and fraction vector is required per stream")
        flows = tuple(
            tuple(_finite(total, "total flow", nonnegative=True) * x for x in _fractions(xs))
            for total, xs in zip(totals, fractions, strict=True)
        )
    for row, values in zip(ids, flows, strict=True):
        _aligned(row, values)
    order = (
        ids[0] if all(set(row) == set(ids[0]) for row in ids) else tuple(sorted(set().union(*ids)))
    )
    streams = [dict(zip(row, values, strict=True)) for row, values in zip(ids, flows, strict=True)]
    return order, tuple(fsum(stream.get(i, 0.0) for stream in streams) for i in order)


def solve_stream_composition(molar_flows, mole_fractions, mass_fractions, molecular_weights):
    """Solve mixed per-component specifications via the two total balances.

    Exactly one spec per component and an absolute molar-flow scale are
    required. Underdetermined, inconsistent or negative solutions fail.
    """
    mw = _vector(molecular_weights, "MW", positive=True)
    fs, zs, ws = tuple(molar_flows), tuple(mole_fractions), tuple(mass_fractions)
    _aligned(fs, zs, ws, mw)
    for f, z, w in zip(fs, zs, ws, strict=True):
        if sum(v is not None for v in (f, z, w)) != 1:
            raise ValueError("exactly one flow, mole fraction or mass fraction per component")
        for v in (f, z, w):
            if v is not None:
                _finite(v, "stream specification", nonnegative=True)
        if (z is not None and z > 1) or (w is not None and w > 1):
            raise ValueError("specified fraction exceeds one")
    known = [i for i, f in enumerate(fs) if f is not None]
    if not known or fsum(fs[i] for i in known) <= 0:
        raise ValueError("a positive absolute flow scale is required")
    # N = sum(Fknown) + N*sum(z) + M*sum(w/MW).
    # M = sum(Fknown*MW) + N*sum(z*MW) + M*sum(w).
    a = 1 - fsum(z for z in zs if z is not None)
    b = -fsum(w / m for w, m in zip(ws, mw, strict=True) if w is not None)
    c = -fsum(z * m for z, m in zip(zs, mw, strict=True) if z is not None)
    d = 1 - fsum(w for w in ws if w is not None)
    fn, fm = fsum(fs[i] for i in known), fsum(fs[i] * mw[i] for i in known)
    det = a * d - b * c
    if det <= 0 or not isfinite(det):
        raise ValueError("stream specifications have no unique positive solution")
    total_n, total_m = (fn * d - b * fm) / det, (a * fm - c * fn) / det
    flows = tuple(
        f if f is not None else z * total_n if z is not None else w * total_m / m
        for f, z, w, m in zip(fs, zs, ws, mw, strict=True)
    )
    if any(not isfinite(f) or f < 0 for f in flows):
        raise ValueError("stream solution must have finite nonnegative component flows")
    mole = normalized_fractions(flows)
    return flows, mole, fraction_basis(mole, mw)


def _matrix(values):
    rows = tuple(_vector(row, "derivative") for row in values)
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError("derivative matrix must be nonempty and square")
    return rows


def constrained_composition_gradient(gradient):
    """n-1 gradient with x_last = 1-sum(x_independent)."""
    g = _vector(gradient, "gradient")
    return tuple(v - g[-1] for v in g[:-1])


def constrained_composition_hessian(hessian):
    """Project both derivative axes onto the composition simplex."""
    h = _matrix(hessian)
    return tuple(
        tuple(h[i][j] - h[i][-1] - h[-1][j] + h[-1][-1] for j in range(len(h) - 1))
        for i in range(len(h) - 1)
    )


def composition_gradient_to_moles(gradient, fractions, *, total_moles=1.0):
    """dF/dn_i from unconstrained dF/dx_i, holding every other n fixed."""
    g, xs = _vector(gradient, "gradient"), _fractions(fractions)
    _aligned(g, xs)
    total = _finite(total_moles, "total moles", positive=True)
    center = fsum(x * v for x, v in zip(xs, g, strict=True))
    return tuple((v - center) / total for v in g)


def partial_molar_from_mole_gradient(value, mole_gradient, *, total_moles=1.0):
    """d(nF)/dn_i = F + n*dF/dn_i, in the original molar property units."""
    value = _finite(value, "molar property")
    total = _finite(total_moles, "total moles", positive=True)
    return tuple(value + total * v for v in _vector(mole_gradient, "mole gradient"))


def partial_molar_properties(value, gradient, fractions):
    return partial_molar_from_mole_gradient(
        value, composition_gradient_to_moles(gradient, fractions)
    )


def partial_molar_composition_jacobian(hessian, fractions):
    """Rows are partial molar species i; columns are unconstrained fraction j."""
    h, xs = _matrix(hessian), _fractions(fractions)
    _aligned(h, xs)
    center = tuple(fsum(xs[k] * h[k][j] for k in range(len(xs))) for j in range(len(xs)))
    return tuple(tuple(v - c for v, c in zip(row, center, strict=True)) for row in h)


def extensive_hessian_from_moles(mole_hessian, mole_gradient, *, total_moles=1.0):
    """d2(nF)/dni dnj from derivatives of the molar quantity F."""
    h, g = _matrix(mole_hessian), _vector(mole_gradient, "mole gradient")
    _aligned(h, g)
    total = _finite(total_moles, "total moles", positive=True)
    return tuple(
        tuple(total * v + g[i] + g[j] for j, v in enumerate(row)) for i, row in enumerate(h)
    )


# Mapping consumers preserve their established conversion, order and sum semantics.
def _normalize_composition(composition: Mapping[str, float]) -> dict[str, float]:
    if not composition:
        raise ValueError("composition cannot be empty")
    if any(value < 0 or not isfinite(value) for value in composition.values()):
        raise ValueError("composition values must be finite and nonnegative")
    total = sum(composition.values())
    if total <= 0:
        raise ValueError("composition must contain positive material")
    return {component_id: float(value) / total for component_id, value in composition.items()}


def _positive_feed_amounts(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        raise ValueError("feed_amounts_mol cannot be empty")
    result = {str(key): float(value) for key, value in values.items()}
    if any(value < 0.0 or not isfinite(value) for value in result.values()):
        raise ValueError("feed_amounts_mol must contain finite nonnegative values")
    if sum(result.values()) <= 0.0:
        raise ValueError("feed_amounts_mol must contain positive total material")
    return result
