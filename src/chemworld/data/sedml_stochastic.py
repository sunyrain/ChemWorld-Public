"""KiSAO integer policies for the existing SBML process, never a simulator.

Bindings follow SED-ML/KiSAO's original OWL. The implemented nonnegative
Poisson/critical strategy is 0000084; basic Poisson 0000040 is not substituted.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from chemworld.data.sedml_declarations import finite_number, integer
from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy

INTEGER_METHODS = {
    "KISAO:0000029": "gillespie-direct-v1",
    "KISAO:0000027": "gibson-bruck-next-reaction-v1",
    "KISAO:0000084": "adaptive-explicit-tau-leap-v1",
}
INTEGER_PARAMETERS = {
    "KISAO:0000488": "seed",
    "KISAO:0000415": "maximum_steps",
    "KISAO:0000467": "maximum_step_source",
    "KISAO:0000228": "epsilon",
    "KISAO:0000230": "minimum_reactions_per_leap",
    "KISAO:0000249": "critical_threshold",
}
_COUNTS = {"seed", "maximum_steps", "critical_threshold"}


def parameter_ids(kisao_id: str) -> frozenset[str]:
    if kisao_id not in INTEGER_METHODS:
        raise ValueError("unsupported integer stochastic KiSAO algorithm")
    return frozenset(
        INTEGER_PARAMETERS if kisao_id == "KISAO:0000084" else ("KISAO:0000488", "KISAO:0000415")
    )


@dataclass(frozen=True)
class IntegerExperimentPolicy:
    method: str
    seed: int
    maximum_events: int
    tau_policy: TauLeapPolicy | None = None


@dataclass(frozen=True)
class SEDMLStochasticAlgorithm:
    kisao_id: str
    seed: int = 0
    maximum_steps: int | None = None
    maximum_step_source: float | None = None
    epsilon: float | None = None
    minimum_reactions_per_leap: float | None = None
    critical_threshold: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kisao_id, str) or self.kisao_id not in INTEGER_METHODS:
            raise ValueError("unsupported stochastic algorithm; no KiSAO substitution is enabled")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("SED-ML process seed must be an unsigned 32-bit integer")
        if self.maximum_steps is not None and (
            type(self.maximum_steps) is not int or not 1 <= self.maximum_steps <= 1000000
        ):
            raise ValueError("SED-ML internal-step budget must be an integer from 1 to 1000000")
        tau_values = (
            self.maximum_step_source,
            self.epsilon,
            self.minimum_reactions_per_leap,
            self.critical_threshold,
        )
        if self.kisao_id != "KISAO:0000084" and any(v is not None for v in tau_values):
            raise ValueError("tau parameters require the nonnegative Poisson algorithm")
        self.native_policy(seconds_per_source_time=1.0)

    @property
    def method(self) -> str:
        return INTEGER_METHODS[self.kisao_id]

    def native_policy(self, *, seconds_per_source_time: float) -> IntegerExperimentPolicy:
        scale = seconds_per_source_time
        if type(scale) not in {int, float} or not isfinite(scale) or scale <= 0:
            raise ValueError("algorithm time scaling must use explicit positive source units")
        if self.kisao_id != "KISAO:0000084":
            return IntegerExperimentPolicy(self.method, self.seed, self.maximum_steps or 1000000)
        if self.maximum_step_source is not None and (
            type(self.maximum_step_source) not in {int, float}
            or not isfinite(self.maximum_step_source)
            or self.maximum_step_source <= 0
        ):
            raise ValueError("tau maximum step must be a positive finite source-time value")
        values: dict[str, Any] = {}
        for field, value in (
            ("epsilon", self.epsilon),
            ("ssa_threshold", self.minimum_reactions_per_leap),
            ("critical_threshold", self.critical_threshold),
            ("max_steps_per_advance", self.maximum_steps),
        ):
            if value is not None:
                values[field] = value
        if self.maximum_step_source is not None:
            values["max_tau_s"] = self.maximum_step_source * scale
        # The separate physical firing budget is the declared native default;
        # KiSAO 415 controls numerical leap/SSA attempts, not Poisson multiplicity.
        return IntegerExperimentPolicy(self.method, self.seed, 1000000, TauLeapPolicy(**values))


def stochastic_algorithm(kisao_id: str, values: Mapping[str, str]) -> SEDMLStochasticAlgorithm:
    if set(values) - parameter_ids(kisao_id):
        raise ValueError("unsupported parameter for the requested stochastic algorithm")
    options: dict[str, Any] = {}
    for key, raw in values.items():
        name = INTEGER_PARAMETERS[key]
        options[name] = (
            integer(raw, minimum=0 if name == "seed" else 1)
            if name in _COUNTS
            else finite_number(raw)
        )
    return SEDMLStochasticAlgorithm(kisao_id, **options)
