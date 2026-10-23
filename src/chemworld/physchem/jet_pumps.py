"""Incompressible jet-pump design using the common residual solver.

Equations follow the ESDU/Karassik model described by fluids. The published
and executable retracted-nozzle loss signs disagree; this choice is explicit.
These steady engineering reports consume current properties and own no fluid.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import exp, isfinite, log, pi, sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemworld.data.scientific_inputs import ScientificTable
from chemworld.physchem.flow_meters import _engineering_root, _positive
from chemworld.physchem.pump_engineering import _cell, _nonnegative
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)
from chemworld.physchem.transport import FluidState

JET_VARIABLE_UNITS = {
    "nozzle_diameter_m": "m",
    "mixing_diameter_m": "m",
    "motive_flow_m3_s": "m^3/s",
    "suction_flow_m3_s": "m^3/s",
    "motive_pressure_Pa": "Pa",
    "suction_pressure_Pa": "Pa",
    "discharge_pressure_Pa": "Pa",
}


@dataclass(frozen=True)
class LiquidJetPumpModel:
    motive_fluid: FluidState
    suction_fluid: FluidState
    nozzle_loss: float = 0
    suction_loss: float = 0.1
    mixing_loss: float = 0.15
    diffuser_loss: float = 0.1
    nozzle_retracted: bool = True
    retracted_loss_convention: str = "documented"

    def __post_init__(self) -> None:
        for key in ("nozzle_loss", "suction_loss", "mixing_loss", "diffuser_loss"):
            _nonnegative(getattr(self, key), key)
        if not isinstance(self.nozzle_retracted, bool):
            raise ValueError("nozzle retraction must be explicitly boolean")
        if self.retracted_loss_convention not in ("documented", "reference_code"):
            raise ValueError("retracted loss convention must be documented or reference_code")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LiquidJetPumpModel:
        return cls(
            **{
                **data,
                "motive_fluid": FluidState(**data["motive_fluid"]),
                "suction_fluid": FluidState(**data["suction_fluid"]),
            }
        )

    def inlet_pressure_difference(
        self,
        nozzle_diameter_m: float,
        mixing_diameter_m: float,
        motive_flow_m3_s: float,
        suction_flow_m3_s: float,
    ) -> float:
        for value, label in (
            (nozzle_diameter_m, "nozzle diameter"),
            (mixing_diameter_m, "mixing diameter"),
            (motive_flow_m3_s, "motive flow"),
            (suction_flow_m3_s, "suction flow"),
        ):
            _positive(value, label)
        if nozzle_diameter_m >= mixing_diameter_m:
            raise ValueError("nozzle diameter must be smaller than mixing throat")
        nozzle = pi * nozzle_diameter_m**2 / 4
        annulus = pi * (mixing_diameter_m**2 - nozzle_diameter_m**2) / 4
        return (
            self.motive_fluid.density_kg_m3
            * (motive_flow_m3_s / nozzle) ** 2
            * (1 + self.nozzle_loss)
            - self.suction_fluid.density_kg_m3
            * (suction_flow_m3_s / annulus) ** 2
            * (1 + self.suction_loss)
        ) / 2

    def pressure_ratio_terms(
        self,
        nozzle_diameter_m: float,
        mixing_diameter_m: float,
        motive_flow_m3_s: float,
        suction_flow_m3_s: float,
        *,
        diffuser_diameter_m: float | None = None,
    ) -> tuple[float, float]:
        self.inlet_pressure_difference(
            nozzle_diameter_m, mixing_diameter_m, motive_flow_m3_s, suction_flow_m3_s
        )
        if diffuser_diameter_m is not None:
            _positive(diffuser_diameter_m, "diffuser diameter")
            if diffuser_diameter_m < mixing_diameter_m:
                raise ValueError("diffuser must not contract below the mixing throat")
        r = (nozzle_diameter_m / mixing_diameter_m) ** 2
        alpha = 0 if diffuser_diameter_m is None else (mixing_diameter_m / diffuser_diameter_m) ** 2
        m = suction_flow_m3_s / motive_flow_m3_s
        c = self.suction_fluid.density_kg_m3 / self.motive_fluid.density_kg_m3
        entrainment = c * m * m * r * r / (1 - r) ** 2
        momentum = 2 * r + 2 * c * m * m * r * r / (1 - r)
        mixed = (
            r
            * r
            * (1 + c * m)
            * (1 + m)
            * (1 + self.mixing_loss + self.diffuser_loss + alpha * alpha)
        )
        numerator = momentum - mixed - entrainment * (1 + self.suction_loss)
        denominator = 1 + self.nozzle_loss - momentum + mixed
        if self.nozzle_retracted:
            sign = 1 if self.retracted_loss_convention == "documented" else -1
            denominator += entrainment * (1 + sign * self.suction_loss)
        return numerator, denominator


def solve_jet_pump_inlet(model: LiquidJetPumpModel, **values: float | None) -> float:
    """Exactly one missing inlet variable; return it on the positive physical branch."""
    keys = tuple(k for k in JET_VARIABLE_UNITS if k != "discharge_pressure_Pa")
    if set(values) != set(keys) or sum(v is None for v in values.values()) != 1:
        raise ValueError("supply all six inlet variables with exactly one None")
    missing = next(k for k, v in values.items() if v is None)
    known = {k: float(v) for k, v in values.items() if v is not None}
    for key, value in known.items():
        _positive(value, key)
    if missing in ("motive_pressure_Pa", "suction_pressure_Pa"):
        difference = model.inlet_pressure_difference(*(known[k] for k in keys[:4]))
        answer = (
            known["suction_pressure_Pa"] + difference
            if missing == "motive_pressure_Pa"
            else known["motive_pressure_Pa"] - difference
        )
    elif missing == "nozzle_diameter_m":
        dm = known["mixing_diameter_m"]
        target = known["motive_pressure_Pa"] - known["suction_pressure_Pa"]
        scale = max(abs(target), known["motive_pressure_Pa"], 1)

        def residual(ratio: float) -> float:
            return (
                model.inlet_pressure_difference(
                    dm * ratio, dm, known["motive_flow_m3_s"], known["suction_flow_m3_s"]
                )
                - target
            ) / scale

        lo, hi = 1e-8, 1 - 1e-8
        for _ in range(20):
            midpoint = (lo + hi) / 2
            if residual(midpoint) > 0:
                lo = midpoint
            else:
                hi = midpoint
        answer = dm * _engineering_root(
            residual, (lo + hi) / 2, lo, hi, name="jet_nozzle_ratio", unit="dimensionless", scale=1
        )
    else:
        dn = known["nozzle_diameter_m"]
        an = pi * dn * dn / 4
        dp = known["motive_pressure_Pa"] - known["suction_pressure_Pa"]
        primary = model.motive_fluid.density_kg_m3 * (1 + model.nozzle_loss) / 2
        secondary = model.suction_fluid.density_kg_m3 * (1 + model.suction_loss) / 2
        if missing == "mixing_diameter_m":
            suction_head = primary * (known["motive_flow_m3_s"] / an) ** 2 - dp
            _positive(suction_head, "available suction velocity head")
            annulus = known["suction_flow_m3_s"] * sqrt(secondary / suction_head)
            answer = sqrt(4 * (an + annulus) / pi)
        else:
            dm = known["mixing_diameter_m"]
            if dn >= dm:
                raise ValueError("nozzle must be smaller than mixing throat")
            annulus = pi * (dm * dm - dn * dn) / 4
            if missing == "motive_flow_m3_s":
                head = (dp + secondary * (known["suction_flow_m3_s"] / annulus) ** 2) / primary
                _positive(head, "available motive velocity head")
                answer = an * sqrt(head)
            else:
                head = (primary * (known["motive_flow_m3_s"] / an) ** 2 - dp) / secondary
                _positive(head, "available suction velocity head")
                answer = annulus * sqrt(head)
    _positive(answer, missing)
    return answer


def jet_pump_report(
    model: LiquidJetPumpModel,
    values: Mapping[str, float],
    *,
    diffuser_diameter_m: float | None = None,
) -> dict[str, Any]:
    """Read-only steady flow/pressure/continuity report, with both residuals."""
    if set(values) != set(JET_VARIABLE_UNITS):
        raise ValueError("all seven jet design values required")
    for key, value in values.items():
        _positive(value, key)
    v = values
    p1, p2, p5 = (
        v[k] for k in ("motive_pressure_Pa", "suction_pressure_Pa", "discharge_pressure_Pa")
    )
    if not p2 < p5 < p1:
        raise ValueError("pumping branch requires suction < discharge < motive pressure")
    q1, q2 = v["motive_flow_m3_s"], v["suction_flow_m3_s"]
    args = [v[k] for k in tuple(JET_VARIABLE_UNITS)[:4]]
    num, den = model.pressure_ratio_terms(*args, diffuser_diameter_m=diffuser_diameter_m)
    ratio = (p5 - p2) / (p1 - p5)
    efficiency = q2 / q1 * ratio
    if not 0 < efficiency <= 1:
        raise ValueError("jet pump efficiency must be in (0,1]")
    mass1, mass2 = model.motive_fluid.density_kg_m3 * q1, model.suction_fluid.density_kg_m3 * q2
    warnings = []
    if model.nozzle_retracted:
        warnings.append("source prose uses +(1+Ks), executable uses +(1-Ks); convention explicit")
    return {
        **v,
        "diffuser_diameter_m": diffuser_diameter_m,
        "flow_ratio": q2 / q1,
        "pressure_ratio": ratio,
        "efficiency": efficiency,
        "motive_mass_flow_kg_s": mass1,
        "suction_mass_flow_kg_s": mass2,
        "outlet_mass_flow_kg_s": mass1 + mass2,
        "outlet_flow_m3_s": q1 + q2,
        "outlet_density_kg_m3": (mass1 + mass2) / (q1 + q2),
        "motive_pressure_power_W": q1 * (p1 - p5),
        "suction_pressure_power_W": q2 * (p5 - p2),
        "inlet_residual_Pa": p1 - p2 - model.inlet_pressure_difference(*args),
        "pressure_ratio_residual": ratio * den - num,
        "retracted_loss_convention": model.retracted_loss_convention,
        "warnings": warnings,
    }


def solve_liquid_jet_pump(
    model: LiquidJetPumpModel,
    known: Mapping[str, float],
    *,
    initial: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
    diffuser_diameter_m: float | None = None,
) -> dict[str, Any]:
    """Solve any two missing variables with explicit local branch guesses/bounds.

    No random restart or absolute-value reflection of trial variables. The
    common root solver records iterations; failure does not create a result.
    A local solution is not a uniqueness certificate over all possible designs.
    """
    missing = tuple(k for k in JET_VARIABLE_UNITS if k not in known)
    if (
        len(known) != 5
        or len(missing) != 2
        or set(initial) != set(missing)
        or set(bounds) != set(missing)
    ):
        raise ValueError("exactly five known and two unknown values with guesses/bounds required")
    for key, value in known.items():
        _positive(value, key)
    variables = []
    for key in missing:
        lo, hi = bounds[key]
        if not (isfinite(lo) and isfinite(hi) and 0 < lo < initial[key] < hi):
            raise ValueError("finite positive bounds must strictly contain initial guesses")
        variables.append(
            ResidualVariable(key, JET_VARIABLE_UNITS[key], initial[key], "algebraic", lo, hi)
        )
    pressure_scale = max(v for k, v in {**known, **initial}.items() if k.endswith("pressure_Pa"))

    def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
        v = {**known, **dict(zip(missing, x, strict=True))}
        args = [v[k] for k in tuple(JET_VARIABLE_UNITS)[:4]]
        try:
            inlet = model.inlet_pressure_difference(*args)
            num, den = model.pressure_ratio_terms(*args, diffuser_diameter_m=diffuser_diameter_m)
            p1, p2, p5 = (
                v[k] for k in ("motive_pressure_Pa", "suction_pressure_Pa", "discharge_pressure_Pa")
            )
            return np.array(
                [
                    (p1 - p2 - inlet) / pressure_scale,
                    ((p5 - p2) * den - (p1 - p5) * num) / pressure_scale,
                ]
            )
        except ValueError as error:
            raise ResidualDomainError(str(error)) from error

    scales = [v.scale for v in variables]
    result = solve_root(
        residual,
        [initial[k] for k in missing],
        jacobian=lambda x: finite_jacobian(
            residual,
            x,
            scales,
            lower=np.array([bounds[k][0] for k in missing]),
            upper=np.array([bounds[k][1] for k in missing]),
        ),
        variables=tuple(variables),
        residual_scales=[1, 1],
        policy=RootPolicy(tolerance=2e-13, maximum_iterations=100),
    )
    if not result.success:
        pressures = [k for k in missing if k.endswith("pressure_Pa")]
        if len(pressures) == 1:
            return _jet_pressure_elimination(
                model, known, initial, bounds, diffuser_diameter_m, result.failure_reason
            )
        raise ValueError(f"jet pump local inverse failed: {result.failure_reason}")
    values = {**known, **dict(zip(missing, map(float, result.value), strict=True))}
    return {
        **jet_pump_report(model, values, diffuser_diameter_m=diffuser_diameter_m),
        "solver": result.diagnostics,
    }


def _jet_pressure_elimination(
    model: LiquidJetPumpModel,
    known: Mapping[str, float],
    initial: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
    diffuser: float | None,
    initial_failure: str,
) -> dict[str, Any]:
    """Deterministic scalar reduction when the local coupled solve fails.

    One pressure is eliminated exactly. A finite 257-point search supplies
    sign-changing brackets; the closest bracket to the user's guess is used.
    Tangent roots and arbitrarily narrow pairs are not certified by this scan.
    """
    pressure = next(k for k in initial if k.endswith("pressure_Pa"))
    variable = next(k for k in initial if k != pressure)
    scale = initial[variable]
    lower, upper = (v / scale for v in bounds[variable])

    def evaluate(x: float) -> tuple[float, dict[str, float]]:
        v = {**known, variable: x * scale}
        args = [v[k] for k in tuple(JET_VARIABLE_UNITS)[:4]]
        inlet = model.inlet_pressure_difference(*args)
        num, den = model.pressure_ratio_terms(*args, diffuser_diameter_m=diffuser)
        if pressure == "motive_pressure_Pa":
            v[pressure] = v["suction_pressure_Pa"] + inlet
        elif pressure == "suction_pressure_Pa":
            v[pressure] = v["motive_pressure_Pa"] - inlet
        else:
            v[pressure] = (num * v["motive_pressure_Pa"] + den * v["suction_pressure_Pa"]) / (
                num + den
            )
        if not bounds[pressure][0] <= v[pressure] <= bounds[pressure][1]:
            raise ValueError("eliminated pressure outside caller branch bounds")
        p1, p2, p5 = (
            v[k] for k in ("motive_pressure_Pa", "suction_pressure_Pa", "discharge_pressure_Pa")
        )
        error = (
            p1 - p2 - inlet
            if pressure == "discharge_pressure_Pa"
            else ((p5 - p2) * den - (p1 - p5) * num)
        )
        return error / max(p1, p2, p5), v

    brackets = []
    previous: tuple[float, float] | None = None
    for x in np.linspace(lower, upper, 257):
        try:
            value, _ = evaluate(float(x))
        except ValueError:
            previous = None
            continue
        if value == 0:
            report = jet_pump_report(model, evaluate(float(x))[1], diffuser_diameter_m=diffuser)
            return {
                **report,
                "solver": {
                    "strategy": "pressure_elimination",
                    "scan_points": 257,
                    "coupled_failure": initial_failure,
                },
            }
        if previous is not None and value * previous[1] < 0:
            brackets.append((previous[0], float(x)))
        previous = (float(x), value)
    if not brackets:
        raise ValueError(f"jet pump inverse failed: {initial_failure}; no bounded sign change")
    lo, hi = min(brackets, key=lambda pair: abs((pair[0] + pair[1]) / 2 - 1))
    for _ in range(16):
        midpoint = (lo + hi) / 2
        if evaluate(lo)[0] * evaluate(midpoint)[0] <= 0:
            hi = midpoint
        else:
            lo = midpoint
    root = _engineering_root(
        lambda x: evaluate(x)[0],
        (lo + hi) / 2,
        lo,
        hi,
        name="jet_reduced_variable",
        unit="dimensionless",
        scale=1,
    )
    return {
        **jet_pump_report(model, evaluate(root)[1], diffuser_diameter_m=diffuser),
        "solver": {
            "strategy": "pressure_elimination",
            "scan_points": 257,
            "brackets_found": len(brackets),
            "coupled_failure": initial_failure,
        },
    }


def vacuum_air_leakage(
    method: str,
    *,
    pressure_Pa: float,
    atmospheric_pressure_Pa: float = 101325,
    volume_m3: float | None = None,
    parameters: ScientificTable | None = None,
    conservative: bool = True,
    outside: str = "reject",
) -> dict[str, Any]:
    """Four empirical vacuum leak estimates; no dynamic gas mass is created.

    HEI table columns upper_drop(Pa), exponent, intercept; final bin extends
    to atmospheric pressure. Coker columns upper_drop(Pa), low/high(kg/s).
    Default HEI volume domain is 10-10000 ft3. Explicit reference policy clips
    below 10 and extrapolates above 10000, reporting the evaluated volume.
    """
    _positive(pressure_Pa, "vessel absolute pressure")
    _positive(atmospheric_pressure_Pa, "atmospheric pressure")
    if pressure_Pa >= atmospheric_pressure_Pa:
        raise ValueError("vacuum pressure must be below surrounding absolute pressure")
    if outside not in ("reject", "reference"):
        raise ValueError("outside policy must be reject or reference")
    drop = atmospheric_pressure_Pa - pressure_Pa
    evaluated_volume = volume_m3
    warnings: list[str] = []
    sources: tuple[str, ...] = ()
    if method in ("Ryans-Croll", "Seider", "HEI2633"):
        if volume_m3 is None:
            raise ValueError("vessel volume required")
        _positive(volume_m3, "vessel volume")
        cubic_feet = volume_m3 / 0.3048**3
    torr = drop * 760 / 101325
    if method == "Ryans-Croll":
        coefficient = (
            0.026 * torr**0.34 if torr < 10 else (0.032 * torr**0.26 if torr < 100 else 0.106)
        )
        mass = coefficient * cubic_feet**0.6 * 0.45359237 / 3600
    elif method == "Seider":
        x = log(torr)
        mass = (
            (5 + (0.0289 + 0.03088 * x - 0.0005733 * x * x) * cubic_feet**0.66) * 0.45359237 / 3600
        )
        warnings.append("reference executable coefficient 0.0289; source prose says 0.0298")
    elif method in ("HEI2633", "Coker-Worthington"):
        if parameters is None or parameters.kind != "parameters" or not parameters.rows:
            raise ValueError("explicit nonempty source parameter table required")
        rows = sorted(parameters.rows, key=lambda r: _cell(r, "upper_drop", "Pa"))
        if len({_cell(r, "upper_drop", "Pa") for r in rows}) != len(rows):
            raise ValueError("distinct source vacuum thresholds required")
        row = next((r for r in rows if drop <= _cell(r, "upper_drop", "Pa")), rows[-1])
        sources = (row.source_id,)
        if method == "HEI2633":
            if not 10 <= cubic_feet <= 10000:
                if outside == "reject":
                    raise ValueError("HEI volume outside 10-10000 cubic feet source domain")
                warnings.append("source lower-volume clipping or upper extrapolation requested")
            cubic_feet = max(10, cubic_feet)
            evaluated_volume = cubic_feet * 0.3048**3
            mass = (
                exp(_cell(row, "intercept") + _cell(row, "exponent") * log(cubic_feet))
                * 0.45359237
                / 3600
            )
        else:
            mass = _cell(row, "high" if conservative else "low", "kg/s")
    else:
        raise ValueError("unknown vacuum leakage method")
    _nonnegative(mass, "predicted air mass flow")
    return {
        "method": method,
        "mass_flow_kg_s": mass,
        "vacuum_pressure_difference_Pa": drop,
        "volume_m3": volume_m3,
        "evaluated_volume_m3": evaluated_volume,
        "source_ids": sources,
        "warnings": warnings,
    }
