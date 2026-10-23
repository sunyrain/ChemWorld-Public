"""Historical gas-pipeline correlations with explicit reference-volume semantics.

All flows are at the declared reference T/P, never local line volume.
Model selection and its applicability are caller declarations; these equations
do not simulate inventories, changing composition, shocks or multiphase flow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, log, sqrt
from typing import Any

from chemworld.physchem.flow_meters import _engineering_root, _positive

# coefficient, reference exponent, pressure/length exponent, diameter exponent,
# specific-gravity exponent inside the pressure group, viscosity exponent.
_POWER_MODELS = {
    "IGT": (24.62412451461407, 1.0, 5 / 9, 8 / 3, 0.8, 1 / 9),
    "Muller": (15.774399086420774, 1.0, 0.575, 2.725, 0.425 / 0.575, 0.15),
    "Fritzsche": (93.50009798751128, 1.0, 0.538, 2.69, 0.8587, 0.0),
    "Panhandle_A": (158.02053287069572, 1.0788, 0.5394, 2.6182, 0.8539, 0.0),
    "Panhandle_B": (152.88116342980555, 1.02, 0.51, 2.53, 0.961, 0.0),
    "Weymouth": (137.32958099425125, 1.0, 0.5, 2.667, 1.0, 0.0),
}
GAS_PIPELINE_METHODS = (*_POWER_MODELS, "Oliphant", "Spitzglass_high", "Spitzglass_low")


@dataclass(frozen=True)
class EmpiricalGasPipelineSpec:
    method: str
    specific_gravity: float
    average_temperature_K: float
    applicability_note: str
    reference_temperature_K: float = 288.7
    reference_pressure_Pa: float = 101325.0
    compressibility_factor: float = 1.0
    efficiency: float | None = None
    viscosity_Pa_s: float | None = None
    pressure_basis: str = "absolute"

    def __post_init__(self) -> None:
        if self.method not in GAS_PIPELINE_METHODS:
            raise ValueError(f"unknown gas-pipeline model {self.method!r}")
        if not self.applicability_note.strip():
            raise ValueError("declare empirical model applicability and engineering source")
        for name in (
            "specific_gravity",
            "average_temperature_K",
            "reference_temperature_K",
            "reference_pressure_Pa",
            "compressibility_factor",
        ):
            _positive(getattr(self, name), name)
        if self.efficiency is not None and (
            not isfinite(self.efficiency) or not 0 < self.efficiency <= 1
        ):
            raise ValueError("pipeline efficiency must lie in (0, 1]")
        if self.method in {"IGT", "Muller"} and self.viscosity_Pa_s is None:
            raise ValueError("IGT and Muller require dynamic viscosity in Pa s")
        if self.viscosity_Pa_s is not None:
            _positive(self.viscosity_Pa_s, "dynamic viscosity")
            if self.method not in {"IGT", "Muller"}:
                raise ValueError("selected model has no viscosity parameter")
        if self.pressure_basis not in {"absolute", "gauge"}:
            raise ValueError("pressure basis must be absolute or gauge")
        if self.pressure_basis == "gauge" and self.method != "Spitzglass_low":
            raise ValueError("pressure-squared models require absolute pressure")
        if self.method == "Oliphant" and (
            self.compressibility_factor != 1 or self.efficiency not in {None, 1}
        ):
            raise ValueError(
                "Oliphant has no E or Z correction; declare unity rather than ignored parameters"
            )

    @property
    def effective_efficiency(self) -> float:
        if self.efficiency is not None:
            return self.efficiency
        return 0.92 if self.method in {"Panhandle_A", "Panhandle_B", "Weymouth"} else 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EmpiricalGasPipelineSpec:
        return cls(**payload)


@dataclass(frozen=True)
class EmpiricalGasPipelineResult:
    reference_flow_m3_s: float
    upstream_pressure_Pa: float
    downstream_pressure_Pa: float
    diameter_m: float
    length_m: float
    reference_temperature_K: float
    reference_pressure_Pa: float
    method: str
    pressure_basis: str
    solved_quantity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_terms(spec: EmpiricalGasPipelineSpec, diameter: float) -> tuple[float, float]:
    """Return C(D) and a such that Q_ref = C(D) * (pressure_drive/L)**a."""
    reference = spec.reference_temperature_K / spec.reference_pressure_Pa
    sg, temperature = spec.specific_gravity, spec.average_temperature_K
    if spec.method in _POWER_MODELS:
        coefficient, ref_power, power, d_power, sg_power, mu_power = _POWER_MODELS[spec.method]
        factor = coefficient * spec.effective_efficiency * reference**ref_power * diameter**d_power
        factor /= (sg**sg_power * temperature * spec.compressibility_factor) ** power
        if mu_power:
            assert spec.viscosity_Pa_s is not None
            factor /= spec.viscosity_Pa_s**mu_power
        return factor, power
    if spec.method == "Oliphant":
        shape = diameter**2.5 + 0.20915193504605287 * diameter**3
        return 84.58717613991857 * reference * shape / sqrt(sg * temperature), 0.5
    shape = 1 + 0.09144 / diameter + (150 / 127) * diameter
    factor = 125.106 * spec.effective_efficiency * reference * diameter**2.5
    factor /= sqrt(sg * temperature * spec.compressibility_factor * shape)
    if spec.method == "Spitzglass_low":
        factor *= sqrt(2 * (spec.reference_pressure_Pa + 1210))
    return factor, 0.5


def _drive(spec: EmpiricalGasPipelineSpec, p1: float, p2: float) -> float:
    if p1 < p2:
        raise ValueError("gas-pipeline equations require forward pressure-driven flow")
    return p1 - p2 if spec.method == "Spitzglass_low" else (p1 - p2) * (p1 + p2)


def solve_empirical_gas_pipeline(
    spec: EmpiricalGasPipelineSpec,
    *,
    reference_flow_m3_s: float | None = None,
    upstream_pressure_Pa: float | None = None,
    downstream_pressure_Pa: float | None = None,
    diameter_m: float | None = None,
    length_m: float | None = None,
) -> EmpiricalGasPipelineResult:
    """Solve one unknown Q_ref/P1/P2/D/L; report declared reference and pressure basis."""
    data = {
        "reference_flow_m3_s": reference_flow_m3_s,
        "upstream_pressure_Pa": upstream_pressure_Pa,
        "downstream_pressure_Pa": downstream_pressure_Pa,
        "diameter_m": diameter_m,
        "length_m": length_m,
    }
    missing = [key for key, value in data.items() if value is None]
    if len(missing) != 1:
        raise ValueError("declare exactly one unknown Q_ref/P1/P2/D/L")
    for key, value in data.items():
        if value is None:
            continue
        if not isfinite(value):
            raise ValueError(f"{key} must be finite")
        if key.endswith("pressure_Pa"):
            if spec.pressure_basis == "absolute":
                _positive(value, key)
        elif key == "reference_flow_m3_s":
            if value < 0:
                raise ValueError("reference flow must be nonnegative")
        else:
            _positive(value, key)
    if reference_flow_m3_s is None:
        assert diameter_m is not None and length_m is not None
        assert upstream_pressure_Pa is not None and downstream_pressure_Pa is not None
        factor, power = _model_terms(spec, diameter_m)
        reference_flow_m3_s = (
            factor
            * (_drive(spec, upstream_pressure_Pa, downstream_pressure_Pa) / length_m) ** power
        )
    elif diameter_m is None or length_m is None:
        assert upstream_pressure_Pa is not None and downstream_pressure_Pa is not None
        driving = _drive(spec, upstream_pressure_Pa, downstream_pressure_Pa)
        if driving <= 0 or reference_flow_m3_s <= 0:
            raise ValueError(
                "dimensions are indeterminate without positive flow and pressure difference"
            )
        if length_m is None:
            assert diameter_m is not None
            factor, power = _model_terms(spec, diameter_m)
            length_m = driving / (reference_flow_m3_s / factor) ** (1 / power)
        else:
            fixed_length, target = length_m, reference_flow_m3_s

            def residual(log_diameter: float) -> float:
                factor, power = _model_terms(spec, exp(log_diameter))
                return log(factor) + power * log(driving / fixed_length) - log(target)

            diameter_m = exp(
                _engineering_root(
                    residual,
                    log(0.1),
                    log(1e-8),
                    log(1e4),
                    name="empirical_pipe_log_diameter",
                    unit="dimensionless",
                    scale=1,
                )
            )
    else:
        assert diameter_m is not None and length_m is not None
        factor, power = _model_terms(spec, diameter_m)
        driving = length_m * (reference_flow_m3_s / factor) ** (1 / power)
        if upstream_pressure_Pa is None:
            assert downstream_pressure_Pa is not None
            upstream_pressure_Pa = (
                downstream_pressure_Pa + driving
                if spec.method == "Spitzglass_low"
                else sqrt(downstream_pressure_Pa**2 + driving)
            )
        else:
            if spec.method == "Spitzglass_low":
                downstream_pressure_Pa = upstream_pressure_Pa - driving
            else:
                squared = upstream_pressure_Pa**2 - driving
                if squared <= 0:
                    raise ValueError("requested flow has no positive absolute outlet pressure")
                downstream_pressure_Pa = sqrt(squared)
    assert all(
        x is not None
        for x in (
            reference_flow_m3_s,
            upstream_pressure_Pa,
            downstream_pressure_Pa,
            diameter_m,
            length_m,
        )
    )
    assert upstream_pressure_Pa is not None and downstream_pressure_Pa is not None
    assert diameter_m is not None and length_m is not None
    if spec.pressure_basis == "absolute":
        _positive(upstream_pressure_Pa, "solved upstream absolute pressure")
        _positive(downstream_pressure_Pa, "solved downstream absolute pressure")
    return EmpiricalGasPipelineResult(
        reference_flow_m3_s,
        upstream_pressure_Pa,
        downstream_pressure_Pa,
        diameter_m,
        length_m,
        spec.reference_temperature_K,
        spec.reference_pressure_Pa,
        spec.method,
        spec.pressure_basis,
        missing[0],
    )
