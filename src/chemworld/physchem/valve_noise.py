"""IEC liquid/gas control-valve acoustic engineering in one-third-octave bands.

Independent spectral-power expressions of IEC 60534-8-4:2015 and -8-3:2011.
The geometry and flow must be declared; these predictions do not drive state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, log10, pi, sqrt
from typing import Any

from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.property_reports import R_J_PER_MOL_K
from chemworld.physchem.valve_characteristics import convert_valve_coefficient

# Nominal one-third-octave centre frequencies and rounded standard A weighting.
OCTAVE_CENTRES_HZ = (
    12.5,
    16.0,
    20.0,
    25.0,
    31.5,
    40.0,
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
    12500.0,
    16000.0,
    20000.0,
)
A_WEIGHT_DB = (
    -63.4,
    -56.7,
    -50.5,
    -44.7,
    -39.4,
    -34.6,
    -30.2,
    -26.2,
    -22.5,
    -19.1,
    -16.1,
    -13.4,
    -10.9,
    -8.6,
    -6.6,
    -4.8,
    -3.2,
    -1.9,
    -0.8,
    0.0,
    0.6,
    1.0,
    1.2,
    1.3,
    1.2,
    1.0,
    0.5,
    -0.1,
    -1.1,
    -2.5,
    -4.3,
    -6.6,
    -9.3,
)


@dataclass(frozen=True)
class ValveAcousticGeometry:
    kv_m3_h: float
    valve_diameter_m: float
    pipe_internal_diameter_m: float
    pipe_wall_m: float
    effective_recovery_factor: float
    style_modifier: float
    source_note: str
    pipe_density_kg_m3: float = 7800.0
    pipe_sound_speed_m_s: float = 5000.0
    air_density_kg_m3: float = 1.2
    air_sound_speed_m_s: float = 343.0
    ambient_pressure_Pa: float = 101325.0

    def __post_init__(self) -> None:
        for name in (
            "kv_m3_h",
            "valve_diameter_m",
            "pipe_internal_diameter_m",
            "pipe_wall_m",
            "pipe_density_kg_m3",
            "pipe_sound_speed_m_s",
            "air_density_kg_m3",
            "air_sound_speed_m_s",
            "ambient_pressure_Pa",
        ):
            _positive(getattr(self, name), name)
        for name in ("effective_recovery_factor", "style_modifier"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.valve_diameter_m > self.pipe_internal_diameter_m:
            raise ValueError("acoustic geometry requires valve no larger than adjoining pipe")
        if not self.source_note.strip():
            raise ValueError("declare acoustic geometry and recovery-factor provenance")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ValveAcousticGeometry:
        return cls(**payload)


@dataclass(frozen=True)
class ValveNoiseResult:
    a_weighted_level_dB: float
    band_levels_dB: tuple[float, ...]
    frequencies_Hz: tuple[float, ...]
    acoustic_power_W: float
    peak_frequency_Hz: float
    regime: str
    outlet_mach: float | None = None
    outlet_density_kg_m3: float | None = None
    expander_acoustic_power_W: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _common_inputs(mass: float, p1: float, p2: float, rho: float, acoustic_exponent: float) -> None:
    for name, value in (
        ("mass flow", mass),
        ("upstream pressure", p1),
        ("downstream pressure", p2),
        ("density", rho),
    ):
        _positive(value, name)
    if p2 >= p1:
        raise ValueError("valve noise needs forward positive pressure difference")
    if not isfinite(acoustic_exponent):
        raise ValueError("acoustic efficiency exponent must be finite")


def _spectrum_result(
    intensities: list[float],
    acoustic_power: float,
    peak: float,
    regime: str,
    *,
    outlet_mach: float | None = None,
    outlet_density: float | None = None,
    expander_power: float = 0.0,
) -> ValveNoiseResult:
    total = sum(
        power * 10 ** (weight / 10) for power, weight in zip(intensities, A_WEIGHT_DB, strict=True)
    )
    if not isfinite(total) or total <= 0:
        raise ValueError("acoustic model left its positive finite power domain")
    return ValveNoiseResult(
        10 * log10(total),
        tuple(10 * log10(p) for p in intensities),
        OCTAVE_CENTRES_HZ,
        acoustic_power,
        peak,
        regime,
        outlet_mach,
        outlet_density,
        expander_power,
    )


def liquid_control_valve_noise(
    geometry: ValveAcousticGeometry,
    *,
    mass_flow_kg_s: float,
    upstream_pressure_Pa: float,
    downstream_pressure_Pa: float,
    saturation_pressure_Pa: float,
    density_kg_m3: float,
    sound_speed_m_s: float,
    cavitation_onset_ratio: float | None = None,
    acoustic_efficiency_exponent: float = -4.6,
) -> ValveNoiseResult:
    """A-weighted level at 1 m outside pipe, 1 m downstream; turbulence/cavitation."""
    _common_inputs(
        mass_flow_kg_s,
        upstream_pressure_Pa,
        downstream_pressure_Pa,
        density_kg_m3,
        acoustic_efficiency_exponent,
    )
    _positive(sound_speed_m_s, "liquid sound speed")
    if (
        not isfinite(saturation_pressure_Pa)
        or not 0 <= saturation_pressure_Pa < downstream_pressure_Pa
    ):
        raise ValueError("flashing outlet is outside liquid acoustic model")
    g = geometry
    cv = convert_valve_coefficient(g.kv_m3_h, from_scale="Kv", to_scale="Cv")
    recovery, diameter = g.effective_recovery_factor, g.pipe_internal_diameter_m
    available = upstream_pressure_Pa - saturation_pressure_Pa
    drop = upstream_pressure_Pa - downstream_pressure_Pa
    pressure_ratio = drop / available
    critical_drop = min(drop, recovery**2 * available)
    onset = (
        0.9 / sqrt(1 + 3 * g.style_modifier * sqrt(cv / (1.17 * recovery)))
        if cavitation_onset_ratio is None
        else cavitation_onset_ratio
    )
    if not isfinite(onset) or not 0 < onset < 1:
        raise ValueError("cavitation onset ratio must lie in (0, 1)")
    onset *= (600000 / upstream_pressure_Pa) ** 0.125
    if onset >= 1:
        raise ValueError("pressure-adjusted cavitation onset is outside (0, 1)")
    jet_diameter = 0.0046 * g.style_modifier * sqrt(cv * recovery)
    jet_speed = sqrt(2 * critical_drop / density_kg_m3) / recovery
    mechanical_power = 0.5 * mass_flow_kg_s * jet_speed**2 * recovery**2
    eta_turbulent = 10**acoustic_efficiency_exponent * jet_speed / sound_speed_m_s
    eta_cavitation = 0.0
    if pressure_ratio > onset:
        eta_cavitation = (
            0.32
            * eta_turbulent
            * sqrt(drop / (critical_drop * onset))
            * exp(5 * onset)
            * sqrt((1 - onset) / (1 - pressure_ratio))
            * (pressure_ratio / onset) ** 5
            * (pressure_ratio - onset) ** 1.5
        )
    strouhal = (
        0.036
        * recovery**2
        * cv
        * g.style_modifier**0.75
        / (1.17 * onset**1.5 * g.valve_diameter_m**2)
        * available**-0.57
    )
    peak = strouhal * jet_speed / jet_diameter
    cavitation_peak = (
        6 * peak * ((1 - pressure_ratio) / (1 - onset)) ** 2 * (onset / pressure_ratio) ** 2.5
    )
    acoustic_power = mechanical_power * (eta_turbulent + eta_cavitation)
    internal_per_watt = 3.2e9 * density_kg_m3 * sound_speed_m_s / diameter**2
    ring_frequency = g.pipe_sound_speed_m_s / (pi * diameter)
    wall_transmission = (
        0.1
        * g.air_sound_speed_m_s
        * g.air_density_kg_m3
        * diameter
        / (g.pipe_sound_speed_m_s * g.pipe_density_kg_m3 * g.pipe_wall_m)
    )
    outside = diameter + 2 * g.pipe_wall_m
    spread = outside / (outside + 2)
    bands = []
    for frequency in OCTAVE_CENTRES_HZ:
        ratio = frequency / peak
        spectral_power = eta_turbulent * 10**-0.8 / (0.25 * ratio**3 + 1 / ratio)
        if eta_cavitation:
            ratio_cav = frequency / cavitation_peak
            spectral_power += eta_cavitation * 10**-0.9 / (0.25 * ratio_cav**1.5 + ratio_cav**-1.5)
        transmission = (
            wall_transmission
            / (ring_frequency / frequency + (frequency / ring_frequency) ** 1.5) ** 2
        )
        bands.append(mechanical_power * spectral_power * internal_per_watt * transmission * spread)
    return _spectrum_result(
        bands, acoustic_power, peak, "cavitation" if eta_cavitation else "turbulence"
    )


def gas_control_valve_noise(
    geometry: ValveAcousticGeometry,
    *,
    mass_flow_kg_s: float,
    upstream_pressure_Pa: float,
    downstream_pressure_Pa: float,
    inlet_temperature_K: float,
    inlet_density_kg_m3: float,
    heat_capacity_ratio: float,
    molar_mass_g_mol: float,
    outlet_temperature_K: float | None = None,
    outlet_density_kg_m3: float | None = None,
    acoustic_efficiency_exponent: float = -3.8,
    peak_strouhal: float = 0.2,
    contraction_coefficient: float = 0.93,
) -> ValveNoiseResult:
    """Five aerodynamic regimes and outlet-expander contribution above Mach 0.3.

    Default outlet density conserves a fixed-Z gas law through declared T2/T1.
    Supply actual rho2 from the property package when composition/Z changes.
    """
    _common_inputs(
        mass_flow_kg_s,
        upstream_pressure_Pa,
        downstream_pressure_Pa,
        inlet_density_kg_m3,
        acoustic_efficiency_exponent,
    )
    if not isfinite(heat_capacity_ratio) or heat_capacity_ratio <= 1:
        raise ValueError("gas heat capacity ratio must exceed one")
    _positive(inlet_temperature_K, "inlet temperature")
    _positive(molar_mass_g_mol, "molar mass")
    _positive(peak_strouhal, "peak Strouhal number")
    if not isfinite(contraction_coefficient) or not 0 < contraction_coefficient <= 1:
        raise ValueError("expander contraction coefficient must lie in (0, 1]")
    t2 = inlet_temperature_K if outlet_temperature_K is None else outlet_temperature_K
    _positive(t2, "outlet temperature")
    rho2 = (
        inlet_density_kg_m3
        * downstream_pressure_Pa
        / upstream_pressure_Pa
        * inlet_temperature_K
        / t2
        if outlet_density_kg_m3 is None
        else outlet_density_kg_m3
    )
    _positive(rho2, "outlet density")
    g, gamma = geometry, heat_capacity_ratio
    recovery, diameter = g.effective_recovery_factor, g.pipe_internal_diameter_m
    cv = convert_valve_coefficient(g.kv_m3_h, from_scale="Kv", to_scale="Cv")
    ratio = 1 - downstream_pressure_Pa / upstream_pressure_Pa
    vena_critical = 1 - (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    installed_critical = recovery**2 * vena_critical
    alpha = (1 - vena_critical) / (1 - installed_critical)
    boundary_b = 1 - gamma ** (-gamma / (gamma - 1)) / alpha
    boundary_ce = 1 - 1 / (22 * alpha)
    regime = 1 + sum(
        ratio > boundary
        for boundary in (installed_critical, vena_critical, boundary_b, boundary_ce)
    )
    jet_diameter = 0.0046 * g.style_modifier * sqrt(cv * recovery)
    maximum_mach = sqrt(2 / (gamma - 1) * (22 ** ((gamma - 1) / gamma) - 1))
    sonic_speed = sqrt(2 * gamma * upstream_pressure_Pa / ((gamma + 1) * inlet_density_kg_m3))
    eta_base = 10**acoustic_efficiency_exponent
    if regime == 1:
        vena_ratio = 1 - ratio / recovery**2
        local_speed = sqrt(
            gamma * upstream_pressure_Pa / inlet_density_kg_m3 * vena_ratio ** ((gamma - 1) / gamma)
        )
        mach = sqrt(2 / (gamma - 1) * (vena_ratio ** ((1 - gamma) / gamma) - 1))
        mechanical = 0.5 * mass_flow_kg_s * (mach * local_speed) ** 2
        peak = peak_strouhal * mach * local_speed / jet_diameter
        efficiency = eta_base * recovery**2 * mach**3
    else:
        mach = (
            maximum_mach
            if regime == 5
            else min(
                maximum_mach,
                sqrt(2 / (gamma - 1) * ((1 / (alpha * (1 - ratio))) ** ((gamma - 1) / gamma) - 1)),
            )
        )
        mechanical = 0.5 * mass_flow_kg_s * sonic_speed**2
        peak = (
            peak_strouhal * mach * sonic_speed / jet_diameter
            if regime < 4
            else 1.4 * peak_strouhal * sonic_speed / (jet_diameter * sqrt(mach**2 - 1))
        )
        efficiency = (
            eta_base * mach ** (6.6 * recovery**2)
            if regime < 4
            else 0.5 * eta_base * mach**2 * 2 ** (3.3 * recovery**2)
        )
        if regime == 2:
            efficiency *= ratio / vena_critical
    acoustic = mechanical * efficiency
    outlet_sound_speed = sqrt(gamma * R_J_PER_MOL_K * t2 / (molar_mass_g_mol / 1000))
    outlet_speed = 4 * mass_flow_kg_s / (pi * diameter**2 * rho2)
    outlet_mach = outlet_speed / outlet_sound_speed
    internal_factor = (
        3.2e9 * rho2 * outlet_sound_speed / diameter**2 / (1 - min(outlet_mach, 0.3)) ** 1.6
    )
    expander_power, expander_peak = 0.0, 1.0
    if outlet_mach > 0.3:
        expander_speed = (
            outlet_speed * diameter**2 / (contraction_coefficient * g.valve_diameter_m**2)
        )
        expander_mechanical = (
            0.5
            * mass_flow_kg_s
            * expander_speed**2
            * ((1 - g.valve_diameter_m**2 / diameter**2) ** 2 + 0.2)
        )
        expander_power = eta_base * (expander_speed / outlet_sound_speed) ** 3 * expander_mechanical
        expander_peak = peak_strouhal * expander_speed / g.valve_diameter_m
    ring = g.pipe_sound_speed_m_s / (pi * diameter)
    cutoff = 0.25 * ring * outlet_sound_speed / g.air_sound_speed_m_s
    wall_frequency = (
        sqrt(3) * g.air_sound_speed_m_s**2 / (pi * g.pipe_wall_m * g.pipe_sound_speed_m_s)
    )
    valve_d = g.valve_diameter_m
    diameter_loss = (
        0
        if valve_d > 0.15
        else 9
        if valve_d < 0.05
        else (-16660 * valve_d**3 + 6370 * valve_d**2 - 813 * valve_d + 35.8)
    )
    outside = diameter + 2 * g.pipe_wall_m
    spread = outside / (outside + 2)

    def spectrum(frequency: float, peak_frequency: float) -> float:
        r = frequency / peak_frequency
        return 10**-0.8 / ((1 + (0.5 * r) ** 2.5) * (1 + (0.5 / r) ** 1.7))

    bands = []
    for frequency in OCTAVE_CENTRES_HZ:
        if frequency < cutoff:
            gx = (cutoff / ring) ** (2 / 3) * (frequency / cutoff) ** 4
            gy = min(cutoff / wall_frequency, 1.0)
        else:
            gx = min(sqrt(frequency / ring), 1.0)
            gy = min(frequency / wall_frequency, 1.0)
        damping = sqrt(0.01 / frequency)
        denominator = (
            rho2 * outlet_sound_speed
            + 2 * pi * g.pipe_wall_m * frequency * g.pipe_density_kg_m3 * damping
        ) / (415 * gy) + 1
        transmission = (
            8.25e-7
            * (outlet_sound_speed / (g.pipe_wall_m * frequency)) ** 2
            * gx
            / denominator
            * g.ambient_pressure_Pa
            / 101325
            * 10 ** (-diameter_loss / 10)
        )
        band_power = acoustic * spectrum(frequency, peak)
        if expander_power:
            band_power += expander_power * spectrum(frequency, expander_peak)
        bands.append(band_power * internal_factor * transmission * spread)
    return _spectrum_result(
        bands,
        acoustic,
        peak,
        f"aerodynamic_{regime}",
        outlet_mach=outlet_mach,
        outlet_density=rho2,
        expander_power=expander_power,
    )
