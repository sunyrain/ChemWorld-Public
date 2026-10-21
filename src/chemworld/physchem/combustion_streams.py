"""Linear complete-combustion stream specifications on the shared solver.

No second integration loop, component inventory or equilibrium calculation.
Amounts have one caller-selected consistent extensive basis (mol or mol/s).
"""

from itertools import combinations
from math import fsum, isclose

import numpy as np

from chemworld.physchem.combustion import (
    COMBUSTION_PRODUCT_IDENTITIES,
    _finite,
    combustion_products_mixture,
)
from chemworld.physchem.residual_system import linear_solve
from chemworld.physchem.viscosity_kernels import _positive


def _ratios(n_air, n_fuel, *, MW_air=None, MW_fuel=None, Vm_air=None, Vm_fuel=None):
    molar = n_air / _positive(n_fuel, "fuel amount")

    def ratio(a, b, label):
        if a is None and b is None:
            return None
        if a is None or b is None:
            raise ValueError(f"both {label} inputs are required")
        return molar * _positive(a, label) / _positive(b, label)

    return molar, ratio(MW_air, MW_fuel, "molecular weight"), ratio(Vm_air, Vm_fuel, "molar volume")


def air_fuel_ratio_solver(
    *,
    ratio=None,
    basis="mass",
    n_air=None,
    n_fuel=None,
    MW_air=None,
    MW_fuel=None,
    Vm_air=None,
    Vm_fuel=None,
):
    if basis not in {"mole", "mass", "volume"}:
        raise ValueError("unknown ratio basis")
    factors = (
        (1, 1) if basis == "mole" else (MW_air, MW_fuel) if basis == "mass" else (Vm_air, Vm_fuel)
    )
    if ratio is not None:
        _positive(ratio, "air/fuel ratio")
        if any(v is None for v in factors):
            raise ValueError("selected ratio basis requires explicit weights or molar volumes")
        molar = ratio * _positive(factors[1], "fuel factor") / _positive(factors[0], "air factor")
    if n_air is None and n_fuel is None:
        raise ValueError("one extensive flow is required")
    if (n_air is None or n_fuel is None) and ratio is None:
        raise ValueError("missing flow requires an explicit ratio")
    if n_air is None:
        n_air = _positive(n_fuel, "fuel amount") * molar
    if n_fuel is None:
        n_fuel = _positive(n_air, "air amount") / molar
    _positive(n_air, "air amount")
    _positive(n_fuel, "fuel amount")
    if ratio is not None and not isclose(n_air / n_fuel, molar, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("inconsistent air/fuel ratio and flows")
    return (
        n_air,
        n_fuel,
        *_ratios(n_air, n_fuel, MW_air=MW_air, MW_fuel=MW_fuel, Vm_air=Vm_air, Vm_fuel=Vm_fuel),
    )


def _fractions(values, size):
    if len(values) != size:
        raise ValueError("one fraction per declared species is required")
    for x in values:
        _positive(x, "mole fraction", zero=True)
    if not isclose(fsum(values), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("stream mole fractions must sum to one")
    return np.asarray(values, dtype=float)


def combustion_spec_solver(
    *,
    zs_air,
    zs_fuel,
    CASs,
    atomss,
    zs_third=None,
    n_third=0,
    n_air=None,
    n_fuel=None,
    n_out=None,
    O2_excess=None,
    frac_out_O2=None,
    frac_out_O2_dry=None,
    ratio=None,
    ratio_basis="mass",
    MW_air=None,
    MW_fuel=None,
    MW_third=None,
    Vm_air=None,
    Vm_fuel=None,
    Vm_third=None,
    reactivities=None,
    combustion_stoichiometries=None,
):
    """All specifications become linear equations in air and variable fuel.

    A third stream has fixed composition and amount. Wet/dry oxygen equations
    are obtained by cross multiplication; ratios never divide by a guessed
    denominator during solve. The existing linear_solve owns factorization.
    Additional constraints are checked, not discarded by a least-squares fit.
    """
    ids = tuple(CASs)
    if not ids or len(ids) != len(atomss) or len(set(ids)) != len(ids):
        raise ValueError("unique identities must align with atom compositions")
    if "7782-44-7" not in ids:
        raise ValueError("oxygen identity must be explicit in the component basis")
    _positive(n_third, "third stream amount", zero=True)
    if zs_third is None and n_third != 0:
        raise ValueError("fixed third flow requires its composition")
    fractions = [_fractions(zs_air, len(ids)), _fractions(zs_fuel, len(ids))]
    fractions.append(np.zeros(len(ids)) if zs_third is None else _fractions(zs_third, len(ids)))
    products = []
    for zs in fractions:
        raw = combustion_products_mixture(
            atomss,
            zs,
            identities=ids,
            reactivities=reactivities,
            stoichiometries=combustion_stoichiometries,
        )
        vector = np.zeros(len(ids))
        for key, value in raw.items():
            if value == 0:
                continue
            identity = COMBUSTION_PRODUCT_IDENTITIES.get(key, key)
            if identity not in ids:
                raise ValueError(f"product {key} needs an explicit identity in the component basis")
            vector[ids.index(identity)] += value
        products.append(vector)
    p = np.column_stack(products)
    if not np.allclose(p[:, 0], fractions[0], rtol=0, atol=1e-12):
        raise ValueError("air stream contains reactive components")
    oxygen_index = ids.index("7782-44-7")
    water_index = ids.index("7732-18-5") if "7732-18-5" in ids else None
    totals = p.sum(axis=0)
    oxygen = p[oxygen_index]
    dry = totals if water_index is None else totals - p[water_index]
    required = np.array([z[oxygen_index] for z in fractions]) - oxygen
    rows, rhs, labels = [], [], []

    def add(label, coefficients, value):
        v = _finite(value, label)
        coefficients = np.asarray(coefficients, dtype=float)
        rows.append(coefficients[:2])
        rhs.append(v - n_third * coefficients[2])
        labels.append(label)

    for label, coeff, value in (
        ("n_air", (1, 0, 0), n_air),
        ("n_fuel", (0, 1, 0), n_fuel),
        ("n_out", totals, n_out),
    ):
        if value is not None:
            _positive(value, label, zero=True)
            add(label, coeff, value)
    if O2_excess is not None:
        _positive(O2_excess, "oxygen excess", zero=True)
        add("O2_excess", oxygen - O2_excess * required, 0)
    for label, target, denominator in (
        ("frac_out_O2", frac_out_O2, totals),
        ("frac_out_O2_dry", frac_out_O2_dry, dry),
    ):
        if target is not None:
            _positive(target, label, zero=True)
            if target >= 1:
                raise ValueError("oxygen outlet fraction must be below one")
            add(label, oxygen - target * denominator, 0)
    if ratio is not None:
        _positive(ratio, "air/fuel ratio")
        if ratio_basis == "mole":
            factors = (1, 1, 1)
        elif ratio_basis == "mass":
            factors = (MW_air, MW_fuel, MW_third)
        elif ratio_basis == "volume":
            factors = (Vm_air, Vm_fuel, Vm_third)
        else:
            raise ValueError("unknown air/fuel ratio basis")
        factors = list(factors)
        if n_third == 0:
            factors[2] = 1
        if any(v is None for v in factors):
            raise ValueError("ratio requires explicit basis factors for every participating stream")
        for v in factors:
            _positive(v, "ratio factor")
        add("ratio", (factors[0], -ratio * factors[1], -ratio * factors[2]), 0)
    if len(rows) < 2:
        raise ValueError("at least two independent specifications are required")
    if n_third == 0 and all(v is None for v in (n_air, n_fuel, n_out)):
        raise ValueError("two-stream specifications require an extensive scale")
    a, b = np.asarray(rows), np.asarray(rhs)
    scales = np.max(np.abs(a), axis=1)
    nonzero = np.flatnonzero(scales > 0)
    selected = None
    for i, j in combinations(nonzero, 2):
        normalized = a[[i, j]] / scales[[i, j], None]
        if abs(np.linalg.det(normalized)) > 1e-12:
            selected = (i, j)
            break
    if selected is None:
        raise ValueError("specifications are dependent or singular")
    pick = list(selected)
    solution = linear_solve(a[pick] / scales[pick, None], b[pick] / scales[pick])
    errors = a @ solution - b
    allowances = 1e-10 * (1 + np.abs(b) + np.abs(a) @ np.abs(solution))
    if np.any(np.abs(errors) > allowances):
        raise ValueError("inconsistent combustion specifications")
    air, fuel = map(float, solution)
    _positive(air, "solved air amount", zero=True)
    _positive(fuel, "solved fuel amount", zero=True)
    _positive(fuel + n_third, "total fuel amount")
    ns = p @ np.array([air, fuel, n_third])
    scale = max(air + fuel + n_third, 1)
    if np.any(ns < -1e-12 * scale):
        raise ValueError("complete combustion has insufficient oxygen/water or negative products")
    roundoff = {ids[i]: float(v) for i, v in enumerate(ns) if -1e-12 * scale <= v < 0}
    ns = np.maximum(ns, 0)  # Only recorded negative roundoff, after the physical rejection above.
    total = _positive(float(ns.sum()), "total product amount")
    o2 = float(ns[oxygen_index])
    demand = float(required @ np.array([air, fuel, n_third]))
    if O2_excess is not None and demand <= 0:
        raise ValueError("oxygen excess requires positive combustion oxygen demand")
    dry_total = total - (float(ns[water_index]) if water_index is not None else 0)
    if frac_out_O2_dry is not None and dry_total <= 0:
        raise ValueError("dry oxygen specification has no dry product amount")

    def combined(first, third):
        if first is None and third is None:
            return None
        if first is None or (n_third and third is None):
            raise ValueError("complete fuel basis factors are required")
        return (
            fuel * _positive(first, "fuel factor")
            + (n_third * _positive(third, "third factor") if n_third else 0)
        ) / (fuel + n_third)

    ratios = _ratios(
        air,
        fuel + n_third,
        MW_air=MW_air,
        MW_fuel=combined(MW_fuel, MW_third),
        Vm_air=Vm_air,
        Vm_fuel=combined(Vm_fuel, Vm_third),
    )
    return dict(  # noqa: C408 -- named scientific result fields
        n_air=air,
        n_fuel=fuel,
        n_third=n_third,
        n_out=total,
        O2_excess=o2 / demand if demand > 0 else None,
        frac_out_O2=o2 / total,
        frac_out_O2_dry=o2 / dry_total if dry_total > 0 else None,
        mole_ratio=ratios[0],
        mass_ratio=ratios[1],
        volume_ratio=ratios[2],
        ns_out=ns.tolist(),
        zs_out=(ns / total).tolist(),
        identities=list(ids),
        specification_residuals=dict(zip(labels, map(float, errors), strict=True)),
        negative_roundoff_products=roundoff,
    )


def mixture_combustion_spec(air, fuel, components, *, third=None, **specifications):
    """Original MixtureSpec declarations -> actual complete-combustion inventory.

    Components include every possible product, with genuine CAS/formula identity.
    The output has the same extensive units as the specified feed amounts.
    No product temperature/phase equilibrium is invented from this atom balance.
    """
    by_id = {c.identifier: c for c in components}
    if len(by_id) != len(components) or any(not c.cas_number or c.charge for c in components):
        raise ValueError("unique neutral components with explicit CAS identities are required")

    def vector(mixture):
        if mixture.phase_label != "gas" or set(mixture.component_ids) - by_id.keys():
            raise ValueError("declared gas stream components must be in the complete basis")
        for identity, mw in zip(
            mixture.component_ids, mixture.molecular_weights_g_mol, strict=True
        ):
            if not isclose(mw, by_id[identity].molecular_weight_g_mol, rel_tol=1e-12):
                raise ValueError("mixture/component molecular weights disagree")
        zs = dict(zip(mixture.component_ids, mixture.mole_fractions, strict=True))
        return [zs.get(k, 0) for k in by_id]

    options = dict(specifications)
    for key, mix in (("MW_air", air), ("MW_fuel", fuel), ("MW_third", third)):
        if mix is not None:
            mw = mix.average_molecular_weight_g_mol
            if key in options and not isclose(options[key], mw, rel_tol=1e-12):
                raise ValueError("specified molecular weight conflicts with actual mixture")
            options[key] = mw
    return combustion_spec_solver(
        zs_air=vector(air),
        zs_fuel=vector(fuel),
        zs_third=None if third is None else vector(third),
        CASs=[c.cas_number for c in components],
        atomss=[c.composition for c in components],
        **options,
    )
