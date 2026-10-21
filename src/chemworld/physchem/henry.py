"""Henry scale definitions and solvent mixing on native property reports.

H is p/x in Pa, not saturation pressure. Hcc uses the explicitly stated gas
reference temperature (298.15 K by default); Bunsen volume uses 273.15 K.
No implicit solvent identity, density, phase split or missing-pair estimate.
"""

from math import exp, isfinite, log

from chemworld.physchem.conductivity_mixtures import _vectors
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.property_reports import PropertyEvaluation
from chemworld.physchem.vapor_pressure_kernels import pressure_derivatives
from chemworld.physchem.viscosity_kernels import _positive

# Public names grouped by the defining relation, not by nominal unit alone.
HENRY_SCALES = {
    "Hcp": "cp",
    "mol/(m^3*Pa)": "cp",
    "M/atm": "molar_atm",
    "Hcc": "cc",
    "mol/(kg*Pa)": "bp",
    "Hbp": "bp_atm",
    "mol/(kg*atm)": "bp_atm",
    "Hxp": "xp",
    "1/atm": "xp",
    "alpha": "bunsen",
    "bunsen coefficient": "bunsen",
    "KHpx": "px",
    "atm": "px",
    "m^3*Pa/mol": "pc",
    "KHpc": "pc_atm",
    "m^3*atm/mol": "pc_atm",
    "KHcc": "kc",
    "SI": "si",
}


def henry_convert(
    value,
    old_scale,
    new_scale,
    *,
    solvent_molar_density_mol_m3=None,
    solvent_molecular_weight_g_mol=None,
    gas_reference_temperature_K=298.15,
    bunsen_reference_temperature_K=273.15,
):
    """Convert all 18 names via c_liquid/p (mol m^-3 Pa^-1).

    Concentration-basis conversion needs solvent density only if a mole- or
    mass-fraction scale is involved. MW is that of the solvent, in g/mol.
    Reference temperature dependence is a scale definition, not a fit of H(T).
    The historical aliases Hbp and KHpc use atm, not Pa; choose the explicit
    mol/(kg*Pa) or m^3*Pa/mol names for their SI counterparts.
    """
    _positive(value, "Henry constant")
    if old_scale not in HENRY_SCALES or new_scale not in HENRY_SCALES:
        raise ValueError("unknown Henry scale")
    _positive(gas_reference_temperature_K, "gas reference temperature")
    _positive(bunsen_reference_temperature_K, "Bunsen reference temperature")
    for v, label in (
        (solvent_molar_density_mol_m3, "solvent molar density"),
        (solvent_molecular_weight_g_mol, "solvent molecular weight"),
    ):
        if v is not None:
            _positive(v, label)

    def relation(scale):
        # Hcp = factor * value**power * rhom**a * (MW/1000)**b.
        # Cancel basis factors before requiring input data.
        return {
            "cp": (1, 1, 0, 0),
            "molar_atm": (1000 / 101325, 1, 0, 0),
            "cc": (1 / (R * gas_reference_temperature_K), 1, 0, 0),
            "bp": (1, 1, 1, 1),
            "bp_atm": (1 / 101325, 1, 1, 1),
            "xp": (1 / 101325, 1, 1, 0),
            "px": (1 / 101325, -1, 1, 0),
            "si": (1, -1, 1, 0),
            "pc": (1, -1, 0, 0),
            "pc_atm": (1 / 101325, -1, 0, 0),
            "kc": (1 / (R * gas_reference_temperature_K), -1, 0, 0),
            "bunsen": (1 / (R * bunsen_reference_temperature_K), 1, 0, 0),
        }[scale]

    old_factor, old_power, old_rho, old_mw = relation(HENRY_SCALES[old_scale])
    new_factor, new_power, new_rho, new_mw = relation(HENRY_SCALES[new_scale])
    factor = old_factor / new_factor
    for exponent, supplied, label, unit_factor in (
        (old_rho - new_rho, solvent_molar_density_mol_m3, "solvent molar density", 1),
        (old_mw - new_mw, solvent_molecular_weight_g_mol, "solvent molecular weight", 0.001),
    ):
        if exponent:
            if supplied is None:
                raise ValueError(f"this Henry conversion requires {label}")
            factor *= (supplied * unit_factor) ** exponent
    try:
        converted = (factor * value**old_power) ** new_power
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Henry scale conversion is outside floating-point range") from exc
    return _positive(converted, "converted Henry constant")


def henry_pressure_derivatives(T, A, B=0, C=0, D=0, E=0, F=0, *, order=1):
    """Value and partial T derivatives through third order, Pa/K**order."""
    p = {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}
    if any(
        isinstance(v, bool) or not isinstance(v, int | float) or not isfinite(v) for v in p.values()
    ):
        raise ValueError("Henry fit parameters must be finite")
    result = pressure_derivatives("henry_pressure", T, p, order=order)
    _positive(result[0], "Henry pressure")
    return result


def henry_pressure(T, A, B=0, C=0, D=0, E=0, F=0):
    return henry_pressure_derivatives(T, A, B, C, D, E, F, order=0)[0]


def henry_log_mixture(
    log_pressures, weights, *, dlog_dT=None, d2log_dT2=None, missing="raise", temperature_K=None
):
    """Log-average H and its supplied T derivatives; reports include coverage.

    Weights need not sum to one: they describe solvent amounts/fractions and
    are normalized here. Missing pairs are None. ``ignore`` renormalizes over
    known pairs; ``zero_log`` explicitly includes missing pairs as lnH=0 in
    the full denominator, reproducing the upstream skip_zero=False convention.
    A literal numeric zero is always the valid H=1 Pa unless adapted explicitly.
    """
    n = len(weights)
    if not n or len(log_pressures) != n or missing not in {"raise", "ignore", "zero_log"}:
        raise ValueError("Henry mixing needs aligned vectors and an explicit missing policy")
    for w in weights:
        _positive(w, "solvent weight", zero=True)
    total = _positive(sum(weights), "total solvent weight")
    derivatives = [log_pressures]
    if d2log_dT2 is not None and dlog_dT is None:
        raise ValueError("second Henry derivative also requires the first log derivative")
    if dlog_dT is not None:
        derivatives.append(dlog_dT)
    if d2log_dT2 is not None:
        derivatives.append(d2log_dT2)
    if any(len(row) != n for row in derivatives):
        raise ValueError("Henry derivative vectors must align")
    for j, h in enumerate(log_pressures):
        for row in derivatives:
            v = row[j]
            if h is not None and (
                v is None
                or isinstance(v, bool)
                or not isinstance(v, int | float)
                or not isfinite(v)
            ):
                raise ValueError("known Henry pair requires finite supplied log derivatives")
        if h is None and any(row[j] is not None for row in derivatives[1:]):
            raise ValueError("missing Henry pair cannot have a supplied derivative")
    absent = [j for j, h in enumerate(log_pressures) if h is None and weights[j] > 0]
    if absent and missing == "raise":
        raise ValueError(f"missing Henry solvent pairs at indices {absent}")
    available = sum(w for w, h in zip(weights, log_pressures, strict=True) if h is not None)
    if available <= 0:
        raise ValueError("no known Henry pair with nonzero solvent weight")
    denominator = available if missing == "ignore" else total
    logs = [
        sum(w * v for w, v in zip(weights, row, strict=True) if v is not None) / denominator
        for row in derivatives
    ]
    try:
        h = _positive(exp(logs[0]), "mixed Henry pressure")
    except OverflowError as exc:
        raise ValueError("mixed Henry pressure overflows") from exc
    values = [h]
    if len(logs) > 1:
        values.append(h * logs[1])
    if len(logs) > 2:
        values.append(h * (logs[2] + logs[1] ** 2))
    if any(not isfinite(v) for v in values):
        raise ValueError("mixed Henry derivative is nonfinite")
    inputs = {
        "known_solvent_fraction": available / total,
        **{f"weight:{j}": w / denominator for j, w in enumerate(weights)},
        **{f"lnH_Pa:{j}": v for j, v in enumerate(log_pressures) if v is not None},
    }
    if temperature_K is not None:
        inputs["temperature"] = _positive(temperature_K, "temperature")
    warnings = (
        (
            f"Missing Henry pairs {absent}; explicit {missing} policy; "
            f"known solvent fraction {available / total:g}",
        )
        if absent
        else ()
    )
    return tuple(
        PropertyEvaluation(
            "henry_pressure" if order == 0 else f"henry_pressure_dT{order}",
            "Henry_log_mixture",
            "solvent_normalized_log_average",
            value,
            ("Pa", "Pa/K", "Pa/K^2")[order],
            inputs,
            warnings,
        )
        for order, value in enumerate(values)
    )


def henry_pressure_mixture(
    pressures_Pa, *, mole_fractions=None, weights=None, missing="raise", temperature_K=None
):
    if (mole_fractions is None) == (weights is None):
        raise ValueError("provide exactly one of mole_fractions or weights")
    if mole_fractions is not None:
        _vectors(mole_fractions)
    logs = [None if h is None else log(_positive(h, "Henry pressure")) for h in pressures_Pa]
    return henry_log_mixture(
        logs,
        mole_fractions if weights is None else weights,
        missing=missing,
        temperature_K=temperature_K,
    )[0]


def henry_matrix(
    log_pressures,
    fractions,
    solute_mask,
    *,
    dlog_dT=None,
    d2log_dT2=None,
    missing="raise",
    zero_is_missing=False,
    temperature_K=None,
):
    """One report tuple per Henry solute, None for each non-Henry component.

    True in solute_mask means a Henry solute; only False columns are solvents.
    ``zero_is_missing`` is the explicit legacy zero-sentinel adapter. Derivative
    matrices must provide every known pair; no guessed zero derivatives.
    """
    _vectors(fractions)
    if missing not in {"raise", "ignore", "zero_log"}:
        raise ValueError("unknown Henry missing policy")
    if d2log_dT2 is not None and dlog_dT is None:
        raise ValueError("second Henry derivative also requires the first log derivative")
    if temperature_K is not None:
        _positive(temperature_K, "temperature")
    n = len(fractions)
    if len(solute_mask) != n or any(type(v) is not bool for v in solute_mask):
        raise ValueError("Henry solute mask must be an aligned boolean vector")
    if type(zero_is_missing) is not bool:
        raise ValueError("zero_is_missing must be boolean")
    matrices = [
        log_pressures,
        *([] if dlog_dT is None else [dlog_dT]),
        *([] if d2log_dT2 is None else [d2log_dT2]),
    ]
    if any(len(matrix) != n or any(len(row) != n for row in matrix) for matrix in matrices):
        raise ValueError("Henry matrices must be square and aligned")
    solvents = [j for j, is_solute in enumerate(solute_mask) if not is_solute]
    results = []
    for i, is_solute in enumerate(solute_mask):
        if not is_solute:
            results.append(None)
            continue
        logs = [
            None if zero_is_missing and log_pressures[i][j] == 0 else log_pressures[i][j]
            for j in solvents
        ]

        def derivative(matrix, i=i):
            return (
                None
                if matrix is None
                else [
                    None if zero_is_missing and log_pressures[i][j] == 0 else matrix[i][j]
                    for j in solvents
                ]
            )

        results.append(
            henry_log_mixture(
                logs,
                [fractions[j] for j in solvents],
                dlog_dT=derivative(dlog_dT),
                d2log_dT2=derivative(d2log_dT2),
                missing=missing,
                temperature_K=temperature_K,
            )
        )
    return tuple(results)


def henry_raoult_flash(mixture, *, henry_evaluations, solvent_vapor_pressures):
    """Ideal Henry-solute/Raoult-solvent K values into the original flash.

    Returns the existing FlashResult: actual equilibrium phase fractions and
    compositions, with no invented enthalpy or additional runtime. Inputs are
    already evaluated for this T; missing/overlapping species are rejected.
    """
    from chemworld.physchem.equilibrium import flash_isothermal

    _vectors(mixture.mole_fractions)
    _positive(mixture.temperature_K, "temperature")
    _positive(mixture.pressure_Pa, "pressure")
    if not henry_evaluations or not solvent_vapor_pressures:
        raise ValueError("Henry flash requires both solute and solvent properties")
    if set(henry_evaluations) & set(solvent_vapor_pressures) or set(henry_evaluations) | set(
        solvent_vapor_pressures
    ) != set(mixture.component_ids):
        raise ValueError("Henry and Raoult inputs must partition the mixture identities")
    ks = {}
    for group, kind in (
        (henry_evaluations, "henry_pressure"),
        (solvent_vapor_pressures, "vapor_pressure"),
    ):
        for identity, evaluation in group.items():
            if (
                evaluation.property_id != kind
                or not evaluation.valid
                or evaluation.inputs.get("temperature") != mixture.temperature_K
            ):
                raise ValueError("flash needs valid same-temperature Henry and solvent evaluations")
            ks[identity] = (
                _positive(evaluation.to("Pa").value, "phase pressure") / mixture.pressure_Pa
            )
    return flash_isothermal(
        dict(zip(mixture.component_ids, mixture.mole_fractions, strict=True)), ks
    )
