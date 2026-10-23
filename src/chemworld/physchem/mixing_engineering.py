"""Mixing design correlations with current fluid properties and explicit SI power."""

from dataclasses import asdict, dataclass
from math import isfinite, log, pi, sqrt
from typing import Any

from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.friction_correlations import pipe_friction_factor
from chemworld.physchem.transport import GRAVITY_M_S2, FluidState


@dataclass(frozen=True)
class AgitatorMixingResult:
    time_s: float
    power_number: float
    reynolds: float
    regime_constant: float
    regime: str
    power_convention: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def agitator_mixing_time(
    fluid: FluidState,
    *,
    rotation_frequency_Hz: float,
    fluid_power_W: float,
    tank_diameter_m: float,
    liquid_height_m: float,
    impeller_diameter_m: float | None = None,
    homogeneity: float = 0.95,
    power_convention: str = "SI",
) -> AgitatorMixingResult:
    """Transition/turbulent mixing time, with Np=P/(rho*N^3*D^5) in SI.

    ``reference_g`` reproduces fluids 1.3.1's extra numerical g multiplier even
    though its declared input is W; it is not a dimensionally valid SI law.
    Laminar states are reported as outside the correlation's qualified regime.
    """
    diameter = tank_diameter_m / 2 if impeller_diameter_m is None else impeller_diameter_m
    for name, value in (
        ("rotation frequency", rotation_frequency_Hz),
        ("fluid power", fluid_power_W),
        ("tank diameter", tank_diameter_m),
        ("liquid height", liquid_height_m),
        ("impeller diameter", diameter),
    ):
        _positive(value, name)
    if diameter >= tank_diameter_m:
        raise ValueError("impeller must fit inside tank")
    if isinstance(homogeneity, bool) or not isfinite(homogeneity) or not 0 < homogeneity < 1:
        raise ValueError("homogeneity must lie in (0,1)")
    if power_convention not in {"SI", "reference_g"}:
        raise ValueError("power convention must be SI or reference_g")
    power_number = fluid_power_W / (fluid.density_kg_m3 * rotation_frequency_Hz**3 * diameter**5)
    notes = []
    if power_convention == "reference_g":
        power_number *= GRAVITY_M_S2
        notes.append("reference-only numerical g multiplier; dimensionally inconsistent for W")
    reynolds = fluid.density_kg_m3 * diameter**2 * rotation_frequency_Hz / fluid.viscosity_Pa_s
    criterion = power_number ** (1 / 3) * reynolds
    turbulent = criterion >= 6370
    fourier = 5.2 / criterion if turbulent else (183 / criterion) ** 2
    time = fluid.density_kg_m3 * tank_diameter_m**1.5 * sqrt(liquid_height_m)
    time *= fourier / fluid.viscosity_Pa_s * log(1 - homogeneity) / log(0.05)
    if reynolds < 10:
        notes.append("laminar impeller Reynolds <10; mixing time is unqualified")
    notes.append("empirical transition/turbulent estimate; no guaranteed material accuracy")
    return AgitatorMixingResult(
        time,
        power_number,
        reynolds,
        criterion,
        "turbulent" if turbulent else "transition",
        power_convention,
        tuple(notes),
    )


def ribbon_power_reynolds(
    *,
    impeller_diameter_m: float,
    ribbon_height_m: float,
    blades: int,
    pitch_m: float,
    blade_width_m: float,
    tank_diameter_m: float,
) -> float:
    """Rieger laminar Np*Re, using an explicit positive wall clearance."""
    for name, value in (
        ("impeller diameter", impeller_diameter_m),
        ("ribbon height", ribbon_height_m),
        ("pitch", pitch_m),
        ("blade width", blade_width_m),
        ("tank diameter", tank_diameter_m),
    ):
        _positive(value, name)
    if not isinstance(blades, int) or isinstance(blades, bool) or blades < 1:
        raise ValueError("blade count must be a positive integer")
    if impeller_diameter_m >= tank_diameter_m:
        raise ValueError("ribbon needs positive wall clearance")
    diameter = impeller_diameter_m
    clearance_ratio = (tank_diameter_m - diameter) / (2 * diameter)
    return (
        82.8
        * ribbon_height_m
        / diameter
        * clearance_ratio**-0.38
        * (pitch_m / diameter) ** -0.35
        * (blade_width_m / diameter) ** 0.2
        * blades**0.78
    )


def ribbon_mixing_time(power_reynolds: float, rotation_frequency_Hz: float) -> float:
    """Grenville laminar ribbon time; source does not specify homogeneity."""
    _positive(power_reynolds, "Np*Re")
    _positive(rotation_frequency_Hz, "rotation frequency")
    return 896000 * power_reynolds**-1.69 / rotation_frequency_Hz


def static_mixer_cov(
    mixing_coefficient: float,
    main_flow_m3_s: float,
    injected_flow_m3_s: float,
    length_diameters: float,
) -> float:
    """Concentration coefficient of variation with declared manufacturer Ki."""
    for name, value in (
        ("mixing coefficient", mixing_coefficient),
        ("main flow", main_flow_m3_s),
        ("injected flow", injected_flow_m3_s),
    ):
        _positive(value, name)
    if mixing_coefficient > 1:
        raise ValueError("decaying mixing coefficient must not exceed one")
    if not isfinite(length_diameters) or length_diameters < 0:
        raise ValueError("mixer length/diameter must be nonnegative")
    return sqrt(main_flow_m3_s / injected_flow_m3_s) * mixing_coefficient**length_diameters


def static_mixer_loss_coefficient(
    manufacturer_factor: float,
    length_m: float,
    diameter_m: float,
    darcy_factor: float,
) -> float:
    """K=(L/D)*f_D*Kmanufacturer on total downstream flow velocity."""
    for name, value in (
        ("manufacturer loss factor", manufacturer_factor),
        ("diameter", diameter_m),
        ("Darcy factor", darcy_factor),
    ):
        _positive(value, name)
    if not isfinite(length_m) or length_m < 0:
        raise ValueError("mixer length must be nonnegative")
    return length_m / diameter_m * darcy_factor * manufacturer_factor


@dataclass(frozen=True)
class StaticMixerResult:
    concentration_cov: float
    loss_coefficient: float
    reynolds: float
    darcy_factor: float
    pressure_drop_Pa: float
    hydraulic_power_W: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def static_mixer_performance(
    fluid: FluidState,
    *,
    main_flow_m3_s: float,
    injected_flow_m3_s: float,
    diameter_m: float,
    length_m: float,
    mixing_coefficient: float,
    manufacturer_loss_factor: float,
    roughness_m: float = 0.0,
) -> StaticMixerResult:
    """Consume the current mixed-stream state, without creating a second inventory."""
    _positive(diameter_m, "diameter")
    cov = static_mixer_cov(
        mixing_coefficient, main_flow_m3_s, injected_flow_m3_s, length_m / diameter_m
    )
    total_flow = main_flow_m3_s + injected_flow_m3_s
    velocity = total_flow / (pi * diameter_m**2 / 4)
    reynolds = fluid.density_kg_m3 * velocity * diameter_m / fluid.viscosity_Pa_s
    factor = pipe_friction_factor(reynolds, roughness_m / diameter_m)
    loss = static_mixer_loss_coefficient(manufacturer_loss_factor, length_m, diameter_m, factor)
    pressure = loss * fluid.density_kg_m3 * velocity**2 / 2
    return StaticMixerResult(cov, loss, reynolds, factor, pressure, pressure * total_flow)


@dataclass(frozen=True)
class TeeMixingResult:
    concentration_cov: float
    jet_diameter_m: float
    momentum_parameter: float
    decay_exponent: float
    branch: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tee_mixing_cov(
    *,
    main_flow_m3_s: float,
    injected_flow_m3_s: float,
    main_diameter_m: float,
    jet_diameter_m: float | None = None,
    jets: int = 1,
    length_diameters: float = 5.0,
) -> TeeMixingResult:
    """Source turbulent gas-tee estimate; no liquid or arbitrary-distance claim.

    The source one-jet upper branch contains 1/33, causing a discontinuity at
    B=.7. That published implementation behavior is retained and disclosed.
    """
    for name, value in (
        ("main flow", main_flow_m3_s),
        ("injected flow", injected_flow_m3_s),
        ("main diameter", main_diameter_m),
    ):
        _positive(value, name)
    if injected_flow_m3_s > main_flow_m3_s:
        raise ValueError("tee injection must be the smaller stream")
    if not isinstance(jets, int) or isinstance(jets, bool) or jets not in {1, 2, 3, 4}:
        raise ValueError("tee model supports one to four jets")
    if not isfinite(length_diameters) or not 0 < length_diameters <= 5:
        raise ValueError("tee distance must be in (0,5] diameters")
    ratio = injected_flow_m3_s / main_flow_m3_s
    diameter = ratio ** (2 / 3) * main_diameter_m if jet_diameter_m is None else jet_diameter_m
    _positive(diameter, "jet diameter")
    if diameter > main_diameter_m:
        raise ValueError("jet diameter exceeds main pipe")
    # This is the source momentum parameter, not a redefinition of total Q2 per jet.
    parameter = (jets * ratio * main_diameter_m / diameter) ** 2
    limit = (0.7, 0.8, 0.8, 2.0)[jets - 1]
    exponent = (1.33, 1.44, 1.75, 1.97)[jets - 1]
    notes = ["turbulent gas mixing assumed; liquid validity not established"]
    if parameter >= limit:
        if jets == 1:
            exponent = 1 / 33 + 0.95 * log(parameter / limit)
        else:
            exponent += 0.95 * log(parameter / limit) ** {2: 1.5, 3: 1.8, 4: 2.0}[jets]
    if jets == 1:
        notes.append(
            "source 1/33 upper branch is discontinuous at B=.7; unresolved source ambiguity"
        )
    cov = sqrt(0.32 * parameter**-0.86 * length_diameters**-exponent)
    return TeeMixingResult(
        cov, diameter, parameter, exponent, "lower" if parameter < limit else "upper", tuple(notes)
    )
