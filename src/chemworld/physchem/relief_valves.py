"""API-style relief sizing and flare acoustics, with source-bearing user tables.

Engineering formulas only, not safety certification or a dynamic relief model.
API520 tabulated data are not bundled: their redistribution licence is not
inferred from reference software's licence. Supply ScientificTable functions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, expm1, isfinite, log, log1p, log10, pi, sqrt
from typing import Any

from chemworld.physchem.flow_meters import _engineering_root, _positive
from chemworld.physchem.tabulated_functions import TabulatedScalar

ATMOSPHERE_PA = 101325.0
RELIEF_REFERENCE_WATER_KG_M3 = 999.0107539518483
# API526 effective orifice size facts; letters I/O/S are intentionally absent.
API526_ORIFICES_IN2 = tuple(
    zip(
        ("D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "T"),
        (0.11, 0.196, 0.307, 0.503, 0.785, 1.287, 1.838, 2.853, 3.6, 4.34, 6.38, 11.05, 16.0, 26.0),
        strict=True,
    )
)
API526_ORIFICES_M2 = tuple((letter, area * 0.0254**2) for letter, area in API526_ORIFICES_IN2)


def _edition(edition: str) -> None:
    if edition not in {"7E", "10E"}:
        raise ValueError("API520 edition must be 7E or 10E")


def _factor(value: float, name: str) -> None:
    if not isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must lie in (0, 1]")


def _gamma(value: float) -> None:
    if not isfinite(value) or value < 1:
        raise ValueError("gas heat capacity ratio must be finite and at least one")


def _table_value(
    table: TabulatedScalar | None,
    inputs: dict[str, float],
    boundary_policy: str = "strict",
) -> float:
    if table is None:
        raise ValueError("this correction requires an explicit sourced parameter table")
    if table.output_unit != "dimensionless" or {axis[0] for axis in table.axes} != set(inputs):
        raise ValueError("relief table requires dimensionless output and declared input axis names")
    expected_units = {
        "temperature_K": "K",
        "pressure_Pa": "Pa",
        "gauge_backpressure_fraction": "dimensionless",
    }
    if any(unit != expected_units[name] for name, _, unit in table.axes):
        raise ValueError("relief table evaluation axes must use K, Pa and dimensionless fractions")
    if boundary_policy not in {"strict", "clip"}:
        raise ValueError("table boundary policy must be strict or explicitly clip")
    if boundary_policy == "clip":
        inputs = {
            name: min(max(value, table.domain[name][0]), table.domain[name][1])
            for name, value in inputs.items()
        }
    value = float(table.value(inputs))
    _factor(value, "tabulated correction")
    return value


def api520_critical_coefficient(heat_capacity_ratio: float) -> float:
    """API metric C (equation uses kg/h, kPa, mm² and g/mol), including gamma=1."""
    _gamma(heat_capacity_ratio)
    difference = heat_capacity_ratio - 1
    log_power = -1.0 if difference == 0 else -(2 + difference) / difference * log1p(difference / 2)
    return 0.03948 * sqrt(heat_capacity_ratio * exp(log_power))


def api520_subcritical_factor(
    *,
    heat_capacity_ratio: float,
    upstream_pressure_Pa: float,
    downstream_pressure_Pa: float,
) -> float:
    _gamma(heat_capacity_ratio)
    _positive(upstream_pressure_Pa, "upstream absolute pressure")
    _positive(downstream_pressure_Pa, "downstream absolute pressure")
    if downstream_pressure_Pa > upstream_pressure_Pa:
        raise ValueError("subcritical factor requires downstream <= upstream pressure")
    if downstream_pressure_Pa == upstream_pressure_Pa:
        return 1.0
    logarithm = log(downstream_pressure_Pa / upstream_pressure_Pa)
    a = (heat_capacity_ratio - 1) / heat_capacity_ratio
    change = -logarithm if a == 0 else -expm1(a * logarithm) / a
    return sqrt(exp(2 * logarithm / heat_capacity_ratio) * change / -expm1(logarithm))


def api520_steam_pressure_factor(upstream_pressure_Pa: float) -> float:
    _positive(upstream_pressure_Pa, "upstream absolute pressure")
    if upstream_pressure_Pa > 22063223.338138755:
        raise ValueError("steam pressure above 22.06 MPa needs a gas property model")
    pressure_kPa = upstream_pressure_Pa / 1000
    return (
        1.0
        if pressure_kPa <= 10339
        else ((0.02764 * pressure_kPa - 1000) / (0.03324 * pressure_kPa - 1061))
    )


def _steam_domain(temperature_K: float, upstream_pressure_Pa: float, edition: str) -> None:
    _edition(edition)
    _positive(temperature_K, "temperature")
    _positive(upstream_pressure_Pa, "upstream absolute pressure")
    if temperature_K > 922.15:
        raise ValueError("API520 superheat temperature exceeds 922.15 K")
    limit = 20780325.0 if edition == "7E" else 22063223.338138755
    if upstream_pressure_Pa > limit:
        raise ValueError("steam pressure exceeds this edition's gas-model boundary")


def api520_steam_superheat_factor(
    *,
    temperature_K: float,
    upstream_pressure_Pa: float,
    edition: str,
    table: TabulatedScalar | None = None,
    table_boundary_policy: str = "strict",
) -> float:
    """Declared table domain is strict unless edge clipping is explicitly requested."""
    _steam_domain(temperature_K, upstream_pressure_Pa, edition)
    if temperature_K < (422.15 if edition == "7E" else 478.15):
        return 1.0
    return _table_value(
        table,
        {"temperature_K": temperature_K, "pressure_Pa": upstream_pressure_Pa},
        table_boundary_policy,
    )


def _backpressure_fraction(set_pressure: float, backpressure: float) -> float:
    _positive(backpressure, "backpressure absolute pressure")
    if not isfinite(set_pressure) or set_pressure <= ATMOSPHERE_PA:
        raise ValueError("set pressure must be absolute and above atmospheric pressure")
    value = (backpressure - ATMOSPHERE_PA) / (set_pressure - ATMOSPHERE_PA)
    if value > 0.5 + 1e-14:
        raise ValueError("balanced-valve gauge backpressure exceeds 50 percent")
    return min(0.5, value)


def api520_gas_backpressure_factor(
    *,
    set_pressure_Pa: float,
    backpressure_Pa: float,
    overpressure: float = 0.1,
    table: TabulatedScalar | None = None,
    table_boundary_policy: str = "strict",
) -> float:
    """Balanced spring-loaded gas valve; pilot-operated unity is a separate declaration."""
    if overpressure not in {0.1, 0.16, 0.21}:
        raise ValueError("gas overpressure must be 10, 16 or 21 percent")
    fraction = _backpressure_fraction(set_pressure_Pa, backpressure_Pa)
    if overpressure == 0.21 or fraction < (0.3 if overpressure == 0.1 else 0.38):
        return 1.0
    return _table_value(table, {"gauge_backpressure_fraction": fraction}, table_boundary_policy)


def api520_liquid_backpressure_factor(
    *,
    set_pressure_Pa: float,
    backpressure_Pa: float,
    table: TabulatedScalar | None = None,
) -> float:
    fraction = _backpressure_fraction(set_pressure_Pa, backpressure_Pa)
    if fraction < 0.15:
        return 1.0
    return _table_value(table, {"gauge_backpressure_fraction": fraction})


def api520_viscosity_factor(reynolds_number: float, *, edition: str = "10E") -> float:
    _edition(edition)
    _positive(reynolds_number, "Reynolds number")
    if edition == "10E":
        if reynolds_number <= 80:
            raise ValueError("API520 10E viscosity correlation requires Reynolds number > 80")
        return 1 / sqrt(1 + 170 / reynolds_number)
    return min(1.0, 1 / (0.9935 + 2.878 / sqrt(reynolds_number) + 342.75 / reynolds_number**1.5))


def api526_select_orifice(required_area_m2: float) -> tuple[str, float]:
    _positive(required_area_m2, "required area")
    for letter, area in API526_ORIFICES_M2:
        if required_area_m2 <= area:
            return letter, area
    raise ValueError("required area exceeds the largest API526 T orifice")


@dataclass(frozen=True)
class ReliefSizingResult:
    mass_flow_kg_s: float
    area_m2: float
    service: str
    choked: bool | None
    discharge_coefficient: float
    backpressure_factor: float
    combination_factor: float
    viscosity_factor: float = 1.0
    viscosity_basis_reynolds: float | None = None
    steam_pressure_factor: float = 1.0
    steam_superheat_factor: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _flow_or_area(mass: float | None, area: float | None) -> None:
    if (mass is None) == (area is None):
        raise ValueError("declare exactly one mass flow or area")
    if mass is not None:
        _positive(mass, "mass flow")
    if area is not None:
        _positive(area, "area")


def size_gas_relief_valve(
    *,
    temperature_K: float,
    compressibility_factor: float,
    molar_mass_g_mol: float,
    heat_capacity_ratio: float,
    upstream_pressure_Pa: float,
    downstream_pressure_Pa: float,
    mass_flow_kg_s: float | None = None,
    area_m2: float | None = None,
    discharge_coefficient: float = 0.975,
    backpressure_factor: float = 1.0,
    combination_factor: float = 1.0,
) -> ReliefSizingResult:
    _flow_or_area(mass_flow_kg_s, area_m2)
    _gamma(heat_capacity_ratio)
    for name, value in (
        ("temperature", temperature_K),
        ("Z", compressibility_factor),
        ("molar mass", molar_mass_g_mol),
        ("upstream pressure", upstream_pressure_Pa),
        ("downstream pressure", downstream_pressure_Pa),
    ):
        _positive(value, name)
    for name, value in (
        ("discharge", discharge_coefficient),
        ("backpressure", backpressure_factor),
        ("combination", combination_factor),
    ):
        _factor(value, name)
    if downstream_pressure_Pa >= upstream_pressure_Pa:
        raise ValueError("relief requires a positive pressure difference")
    delta = heat_capacity_ratio - 1
    critical_ratio = exp(-0.5 if delta == 0 else -heat_capacity_ratio / delta * log1p(delta / 2))
    choked = downstream_pressure_Pa / upstream_pressure_Pa <= critical_ratio
    if choked:
        factor = 3.6 * sqrt(temperature_K * compressibility_factor / molar_mass_g_mol)
        factor /= (
            api520_critical_coefficient(heat_capacity_ratio)
            * discharge_coefficient
            * upstream_pressure_Pa
            * backpressure_factor
            * combination_factor
        )
    else:
        f2 = api520_subcritical_factor(
            heat_capacity_ratio=heat_capacity_ratio,
            upstream_pressure_Pa=upstream_pressure_Pa,
            downstream_pressure_Pa=downstream_pressure_Pa,
        )
        factor = (
            64.44
            / (f2 * discharge_coefficient * combination_factor)
            * sqrt(
                temperature_K
                * compressibility_factor
                / (
                    molar_mass_g_mol
                    * upstream_pressure_Pa
                    * (upstream_pressure_Pa - downstream_pressure_Pa)
                )
            )
        )
    if mass_flow_kg_s is None:
        assert area_m2 is not None
        mass_flow_kg_s = area_m2 / factor
    else:
        area_m2 = mass_flow_kg_s * factor
    return ReliefSizingResult(
        mass_flow_kg_s,
        area_m2,
        "gas",
        choked,
        discharge_coefficient,
        backpressure_factor if choked else 1.0,
        combination_factor,
    )


def size_steam_relief_valve(
    *,
    temperature_K: float,
    upstream_pressure_Pa: float,
    edition: str,
    steam_phase: str,
    superheat_table: TabulatedScalar | None = None,
    table_boundary_policy: str = "strict",
    mass_flow_kg_s: float | None = None,
    area_m2: float | None = None,
    discharge_coefficient: float = 0.975,
    backpressure_factor: float = 1.0,
    combination_factor: float = 1.0,
) -> ReliefSizingResult:
    """Dry saturated/superheated steam declaration, not a T/P phase classifier."""
    _flow_or_area(mass_flow_kg_s, area_m2)
    if steam_phase not in {"dry_saturated", "superheated"}:
        raise ValueError("steam sizing requires declared dry saturated or superheated steam")
    for name, value in (
        ("discharge", discharge_coefficient),
        ("backpressure", backpressure_factor),
        ("combination", combination_factor),
    ):
        _factor(value, name)
    _steam_domain(temperature_K, upstream_pressure_Pa, edition)
    superheat = (
        1.0
        if steam_phase == "dry_saturated"
        else api520_steam_superheat_factor(
            temperature_K=temperature_K,
            upstream_pressure_Pa=upstream_pressure_Pa,
            edition=edition,
            table=superheat_table,
            table_boundary_policy=table_boundary_policy,
        )
    )
    pressure = api520_steam_pressure_factor(upstream_pressure_Pa)
    factor = 685.8 / (
        upstream_pressure_Pa
        * discharge_coefficient
        * backpressure_factor
        * combination_factor
        * pressure
        * superheat
    )
    if mass_flow_kg_s is None:
        assert area_m2 is not None
        mass_flow_kg_s = area_m2 / factor
    else:
        area_m2 = mass_flow_kg_s * factor
    return ReliefSizingResult(
        mass_flow_kg_s,
        area_m2,
        "dry_steam",
        None,
        discharge_coefficient,
        backpressure_factor,
        combination_factor,
        steam_pressure_factor=pressure,
        steam_superheat_factor=superheat,
    )


def size_liquid_relief_valve(
    *,
    density_kg_m3: float,
    upstream_pressure_Pa: float,
    downstream_pressure_Pa: float,
    overpressure: float,
    mass_flow_kg_s: float | None = None,
    area_m2: float | None = None,
    discharge_coefficient: float = 0.65,
    combination_factor: float = 1.0,
    backpressure_factor: float | None = None,
    backpressure_table: TabulatedScalar | None = None,
    viscosity_factor: float | None = None,
    viscosity_Pa_s: float | None = None,
    edition: str = "10E",
) -> ReliefSizingResult:
    """API two-step viscosity sizing: initial inviscid area -> Re -> Kv -> final area."""
    _flow_or_area(mass_flow_kg_s, area_m2)
    _edition(edition)
    for name, value in (
        ("density", density_kg_m3),
        ("upstream pressure", upstream_pressure_Pa),
        ("downstream pressure", downstream_pressure_Pa),
    ):
        _positive(value, name)
    if not isfinite(overpressure) or overpressure < 0:
        raise ValueError("overpressure fraction must be finite and nonnegative")
    if upstream_pressure_Pa <= downstream_pressure_Pa:
        raise ValueError("liquid relief requires a positive pressure difference")
    if backpressure_factor is None:
        set_pressure = (upstream_pressure_Pa - ATMOSPHERE_PA) / (1 + overpressure) + ATMOSPHERE_PA
        backpressure_factor = api520_liquid_backpressure_factor(
            set_pressure_Pa=set_pressure,
            backpressure_Pa=downstream_pressure_Pa,
            table=backpressure_table,
        )
    for name, value in (
        ("discharge", discharge_coefficient),
        ("backpressure", backpressure_factor),
        ("combination", combination_factor),
    ):
        _factor(value, name)
    if viscosity_factor is not None:
        _factor(viscosity_factor, "viscosity correction")
    if viscosity_Pa_s is not None:
        _positive(viscosity_Pa_s, "viscosity")
    if viscosity_factor is None and viscosity_Pa_s is None:
        raise ValueError("declare a viscosity correction or dynamic viscosity")
    base = (
        11.78
        * 0.06
        / density_kg_m3
        * sqrt(
            density_kg_m3
            / RELIEF_REFERENCE_WATER_KG_M3
            * 1000
            / (upstream_pressure_Pa - downstream_pressure_Pa)
        )
        / (discharge_coefficient * backpressure_factor * combination_factor)
    )

    def correction(mass: float) -> tuple[float, float | None]:
        if viscosity_factor is not None:
            return viscosity_factor, None
        assert viscosity_Pa_s is not None
        initial_area = base * mass
        reynolds = mass / initial_area * sqrt(4 * initial_area / pi) / viscosity_Pa_s
        return api520_viscosity_factor(reynolds, edition=edition), reynolds

    if mass_flow_kg_s is None:
        assert area_m2 is not None
        if viscosity_factor is not None:
            mass_flow_kg_s = area_m2 * viscosity_factor / base
        else:
            assert viscosity_Pa_s is not None
            target = area_m2
            minimum_mass = (
                (80.0000001 * viscosity_Pa_s) ** 2 * pi * base / 4 if edition == "10E" else 1e-20
            )
            upper_mass = target / base
            if upper_mass <= minimum_mass:
                raise ValueError("area inverse lies outside viscosity-correlation Reynolds domain")

            def residual(log_mass: float) -> float:
                mass = exp(log_mass)
                return log(base * mass / correction(mass)[0] / target)

            lower = log(minimum_mass)
            if residual(lower) > 0:
                raise ValueError("area inverse lies outside viscosity-correlation Reynolds domain")
            mass_flow_kg_s = exp(
                _engineering_root(
                    residual,
                    log(upper_mass) - 0.01,
                    lower,
                    log(upper_mass),
                    name="liquid_relief_log_mass",
                    unit="dimensionless",
                    scale=1,
                )
            )
    kv, reynolds = correction(mass_flow_kg_s)
    area_m2 = base * mass_flow_kg_s / kv
    return ReliefSizingResult(
        mass_flow_kg_s,
        area_m2,
        "single_liquid",
        False,
        discharge_coefficient,
        backpressure_factor,
        combination_factor,
        kv,
        reynolds,
    )


def api521_noise_curve(pressure_ratio: float) -> float:
    if not isfinite(pressure_ratio) or pressure_ratio < 1:
        raise ValueError("flare pressure ratio must be finite and at least one")
    x = log10(pressure_ratio)
    lower, upper = 87.9084 * x + 12.7647, 4.8239 * x + 51.6217
    if pressure_ratio <= 2.92:
        return lower
    if pressure_ratio >= 2.93:
        return upper
    fraction = (pressure_ratio - 2.92) / 0.01
    return lower + fraction * (upper - lower)


def api521_flare_noise(
    *,
    mass_flow_kg_s: float,
    upstream_pressure_Pa: float,
    ambient_pressure_Pa: float,
    sound_speed_m_s: float,
    distance_m: float,
) -> float:
    """API521 sound pressure level [dB] at distance, with geometric spreading."""
    for name, value in (
        ("mass flow", mass_flow_kg_s),
        ("ambient pressure", ambient_pressure_Pa),
        ("sound speed", sound_speed_m_s),
        ("distance", distance_m),
    ):
        _positive(value, name)
    return (
        api521_noise_curve(upstream_pressure_Pa / ambient_pressure_Pa)
        + 10 * log10(0.5 * mass_flow_kg_s * sound_speed_m_s**2)
        - 20 * log10(distance_m / 30)
    )


def vdi3732_flare_noise(mass_flow_kg_s: float, *, configuration: str) -> float:
    """Published VDI fit [dB] with m in tonne/hour; no distance attenuation here."""
    _positive(mass_flow_kg_s, "mass flow")
    if configuration not in {"elevated", "ground"}:
        raise ValueError("flare configuration must be elevated or ground")
    intercept, slope = (112, 17) if configuration == "elevated" else (100, 15)
    return intercept + slope * log10(mass_flow_kg_s * 3.6)
