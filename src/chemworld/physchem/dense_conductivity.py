"""Pressure corrections on explicit liquid reference conductivity reports.

DIPPR9G and Missenard as documented in Properties of Gases and Liquids, 5e.
No EOS or saturation state is created. T/Tc K, P/Pc Pa, k W/(m K).
"""

from bisect import bisect_right
from math import exp, expm1, isfinite

from chemworld.physchem.caloric_math import FirstDerivative, derivative_values, scalar_value
from chemworld.physchem.viscosity_kernels import _positive

# Scientific Q table only, not upstream interpolation code or fitted splines.
_MISSENARD_TR = (0.5, 0.6, 0.7, 0.8)
_MISSENARD_PR = (1, 5, 10, 50, 100, 200)
_MISSENARD_Q = (
    (0.012, 0.0165, 0.017, 0.019, 0.02, 0.02),
    (0.015, 0.02, 0.022, 0.024, 0.025, 0.025),
    (0.018, 0.025, 0.027, 0.031, 0.032, 0.032),
    (0.036, 0.038, 0.038, 0.038, 0.038, 0.038),
)

# Chung et al. (1988) scientific coefficient table, a/b/c/d columns.
_CHUNG_COEFFICIENTS = (
    (2.4166, 0.74824, -0.91858, 121.72),
    (-0.50924, -1.5094, -49.991, 69.983),
    (6.6107, 5.6207, 64.760, 27.039),
    (14.543, -8.9139, -5.6379, 74.344),
    (0.79274, 0.82019, -0.69369, 6.3173),
    (-5.8634, 12.801, 9.5893, 65.529),
    (91.089, 128.11, -54.217, 523.81),
)


def chung_dense_conductivity(T, *, MW, Tc, Vc, omega, Cvm, Vm, mu, dipole, association=0):
    """Chung dense gas W/(m K), using actual Vm and low-pressure mu.

    Boundary units: MW g/mol, Vc/Vm m³/mol, Cvm J/(mol K), mu Pa s,
    dipole debye, T/Tc K. Internal M' is kg/mol and Vc is cm³/mol.
    Reduced dipole = 131.3*dipole/sqrt(Vc_cm3*Tc). No ignored pressure
    argument or inferred EOS is introduced. Packing y=Vc/(6Vm) must be <1.
    """
    from chemworld.physchem.conductivity_kernels import _chung_psi

    for value, label in (
        (T, "temperature"),
        (MW, "MW"),
        (Tc, "Tc"),
        (Vc, "Vc"),
        (Cvm, "Cvm"),
        (Vm, "actual Vm"),
        (mu, "dilute viscosity"),
    ):
        _positive(value, label)
    for value, label in ((dipole, "dipole"), (association, "association")):
        _positive(value, label, zero=True)
    if isinstance(omega, bool) or not isinstance(omega, int | float) or not isfinite(omega):
        raise ValueError("acentric factor must be finite")
    y = Vc / (6 * Vm)
    if y >= 1:
        raise ValueError("Chung packing requires Vc/(6Vm) < 1")
    vc_cm3, mass_kg = Vc * 1e6, MW / 1000
    reduced_dipole = 131.3 * dipole / (vc_cm3 * Tc) ** 0.5
    b = tuple(
        a + w * omega + c * reduced_dipole**4 + d * association
        for a, w, c, d in _CHUNG_COEFFICIENTS
    )
    g1 = (1 - 0.5 * y) / (1 - y) ** 3
    try:
        g2 = (-b[0] * expm1(-b[3] * y) / y + b[1] * g1 * exp(b[4] * y) + b[2] * g1) / (
            b[0] * b[3] + b[1] + b[2]
        )
        _positive(g2, "Chung G2")
        q = 0.003586 * (Tc / mass_kg) ** 0.5 / vc_cm3 ** (2 / 3)
        result = (
            31.2 * mu * _chung_psi(T, Tc, omega, Cvm) / mass_kg * (1 / g2 + b[5] * y)
            + q * b[6] * y * y * (T / Tc) ** 0.5 * g2
        )
        return _positive(result, "Chung dense conductivity")
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Chung dense conductivity is outside its numerical domain") from exc


def _table_position(value, grid, outside_table):
    v = scalar_value(value)
    if v < grid[0] or v > grid[-1]:
        if outside_table == "reject":
            raise ValueError("Missenard requires 0.5 <= Tr <= 0.8 and 1 <= Pr <= 200")
        value = grid[0] if v < grid[0] else grid[-1]
        v = value
    if isinstance(value, FirstDerivative) and v in grid:
        raise ValueError("Missenard derivative at a table knot requires a side")
    index = min(max(bisect_right(grid, v) - 1, 0), len(grid) - 2)
    return index, (value - grid[index]) / (grid[index + 1] - grid[index])


def compressed_liquid_conductivity(
    T,
    pressure_Pa,
    *,
    method,
    critical_temperature_K,
    critical_pressure_Pa,
    reference_conductivity_W_m_K,
    outside_table="reject",
):
    """DIPPR9G or bilinear Missenard correction of a supplied baseline.

    Missenard rejects outside its Q table by default. Explicit 'nearest'
    clamps Q coordinates only, matching the reference interpolation boundary
    convention; the actual Pr**0.7 remains unchanged. DIPPR9G is used as
    published, including its .98 intercept, without forced renormalization.
    """
    for value, label in (
        (T, "temperature"),
        (pressure_Pa, "pressure"),
        (critical_temperature_K, "critical temperature"),
        (critical_pressure_Pa, "critical pressure"),
        (reference_conductivity_W_m_K, "reference conductivity"),
    ):
        _positive(value, label)
    if outside_table not in {"reject", "nearest"}:
        raise ValueError("unknown table boundary policy")
    tr, pr = T / critical_temperature_K, pressure_Pa / critical_pressure_Pa
    if method == "DIPPR9G":
        if outside_table != "reject":
            raise ValueError("table boundary policy applies only to Missenard")
        if scalar_value(tr) > 1:
            raise ValueError("liquid DIPPR9G requires T <= Tc")
        factor = 0.98 + 0.0079 * pr * tr**1.4 + 0.63 * tr**1.2 * pr / (30 + pr)
    elif method == "Missenard":
        i, a = _table_position(tr, _MISSENARD_TR, outside_table)
        j, b = _table_position(pr, _MISSENARD_PR, outside_table)
        q = sum(
            _MISSENARD_Q[i + di][j + dj] * wa * wb
            for di, wa in ((0, 1 - a), (1, a))
            for dj, wb in ((0, 1 - b), (1, b))
        )
        factor = 1 + q * pr**0.7
    else:
        raise ValueError(f"unknown compressed-liquid conductivity method {method!r}")
    return _positive(reference_conductivity_W_m_K * factor, "compressed conductivity")


def compressed_liquid_conductivity_derivatives(
    T,
    pressure_Pa,
    *,
    variable="temperature",
    order=1,
    **parameters,
):
    """Partial T or P derivatives with the supplied baseline held fixed."""
    if variable == "temperature":
        return derivative_values(
            lambda t: compressed_liquid_conductivity(t, pressure_Pa, **parameters), T, order=order
        )
    if variable == "pressure":
        return derivative_values(
            lambda p: compressed_liquid_conductivity(T, p, **parameters), pressure_Pa, order=order
        )
    raise ValueError("derivative variable must be temperature or pressure")


def compressed_liquid_conductivity_report(
    reference_report,
    *,
    temperature_K,
    pressure_Pa,
    method,
    reference_kind,
    critical_temperature_K,
    critical_pressure_Pa,
    outside_table="reject",
):
    """Transform the original liquid report at the same T into the same type.

    Reference kind is explicitly 'atmospheric' (101325 Pa) or 'saturation'.
    The source must record its reference pressure; requested P cannot be below
    it. Reference validity and warnings survive, without invented uncertainty.
    """
    from chemworld.physchem.property_reports import PropertyEvaluation
    from chemworld.physchem.transport_properties import TransportPropertyReport

    source = reference_report.evaluation
    if reference_report.phase != "liquid" or source.property_id != "thermal_conductivity":
        raise ValueError("pressure correction requires a liquid conductivity report")
    if source.inputs.get("temperature") != temperature_K:
        raise ValueError("reference report must use the same temperature in K")
    if reference_kind not in {"atmospheric", "saturation"}:
        raise ValueError("explicit atmospheric or saturation reference kind is required")
    baseline_pressure = source.inputs.get("pressure")
    _positive(baseline_pressure, "reference pressure")
    if reference_kind == "atmospheric" and baseline_pressure != 101325:
        raise ValueError("atmospheric reference must record 101325 Pa")
    if pressure_Pa < baseline_pressure:
        raise ValueError("requested pressure is below the reference pressure")
    value = compressed_liquid_conductivity(
        temperature_K,
        pressure_Pa,
        method=method,
        critical_temperature_K=critical_temperature_K,
        critical_pressure_Pa=critical_pressure_Pa,
        reference_conductivity_W_m_K=source.to("W/(m*K)").value,
        outside_table=outside_table,
    )
    outside = method == "Missenard" and not (
        0.5 <= temperature_K / critical_temperature_K <= 0.8
        and 1 <= pressure_Pa / critical_pressure_Pa <= 200
    )
    warnings = source.warnings + (
        ("Missenard Q coordinates explicitly clamped to the table boundary.",) if outside else ()
    )
    return TransportPropertyReport(
        PropertyEvaluation(
            "thermal_conductivity",
            source.correlation_id + ":" + method,
            method,
            value,
            "W/(m*K)",
            {
                "temperature": temperature_K,
                "pressure": pressure_Pa,
                "reference_pressure": baseline_pressure,
                "reference_conductivity": source.to("W/(m*K)").value,
                "critical_temperature": critical_temperature_K,
                "critical_pressure": critical_pressure_Pa,
            },
            warnings,
        ),
        "liquid",
        method,
        "out_of_range"
        if outside or reference_report.validity_status == "out_of_range"
        else "estimated",
        uncertainty_note=f"{method} {reference_kind} pressure correction; no combined uncertainty.",
        reference_reading=(
            *reference_report.reference_reading,
            "Poling et al., Properties of Gases and Liquids, 5e",
        ),
    )
