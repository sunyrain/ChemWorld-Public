"""Declared algorithm policies and source-time output grids, without substitution."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite

import numpy as np

from chemworld.data.sedml_declarations import SEDMLDocument, finite_number, records, step_count
from chemworld.data.sedml_stochastic import (
    INTEGER_METHODS,
    INTEGER_PARAMETERS,
    SEDMLStochasticAlgorithm,
    parameter_ids,
    stochastic_algorithm,
)
from chemworld.data.sedml_xml import ExperimentXMLNode
from chemworld.physchem.solver_backend import ODESolverPolicy

_METHODS = {"KISAO:0000288": "BDF", "KISAO:0000088": "LSODA"}
_OPTIONS = {"KISAO:0000209": "rtol", "KISAO:0000211": "atol",
            "KISAO:0000467": "maximum_step_source"}


@dataclass(frozen=True)
class SEDMLAlgorithm:
    kisao_id: str
    rtol: float = 1e-9
    atol: float = 1e-12
    maximum_step_source: float | None = None

    def __post_init__(self) -> None:
        values = (self.rtol, self.atol, *((self.maximum_step_source,)
                  if self.maximum_step_source is not None else ()))
        if self.kisao_id not in _METHODS or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v <= 0
                for v in values):
            raise ValueError("unsupported SED-ML deterministic algorithm or numerical policy")

    @property
    def method(self) -> str:
        return _METHODS[self.kisao_id]

    def native_policy(self, *, seconds_per_source_time: float) -> ODESolverPolicy:
        scale = seconds_per_source_time
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not isfinite(scale) \
                or scale <= 0:
            raise ValueError("algorithm time scaling must use explicit positive source units")
        maximum = None if self.maximum_step_source is None else self.maximum_step_source*scale
        return ODESolverPolicy("chemworld_sedml_"+self.method.lower(), self.method,
                               rtol=self.rtol, atol=self.atol, max_step_s=maximum)


@dataclass(frozen=True)
class UniformTimeCourse:
    simulation_id: str
    initial_time: float
    output_start: float
    output_end: float
    number_of_steps: int
    algorithm: SEDMLAlgorithm | SEDMLStochasticAlgorithm

    def __post_init__(self) -> None:
        values = (self.initial_time, self.output_start, self.output_end)
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
                for v in values) or not self.initial_time <= self.output_start <= self.output_end
                or type(self.number_of_steps) is not int or not 1 <= self.number_of_steps <= 10000
                or not isinstance(self.algorithm, (SEDMLAlgorithm, SEDMLStochasticAlgorithm))
                or not isinstance(self.simulation_id, str) or not self.simulation_id):
            raise ValueError("uniform experiment must declare a finite ordered source-time grid")

    @property
    def source_times(self) -> tuple[float, ...]:
        values = np.linspace(self.output_start, self.output_end, self.number_of_steps+1)
        if not np.all(np.isfinite(values)):
            raise ValueError("uniform experiment grid exceeds numeric range")
        return tuple(float(value) for value in values)


def _raw_parameters(nodes: Iterable[ExperimentXMLNode]) -> dict[str, str]:
    result = {}
    for node in nodes:
        key = node.attributes["kisaoID"]
        if key not in _OPTIONS.keys() | INTEGER_PARAMETERS.keys() or key in result:
            raise ValueError("unsupported or duplicate algorithm parameter")
        result[key] = node.attributes["value"]
    return result


def _algorithm(
    kisao_id: str, values: Mapping[str, str],
) -> SEDMLAlgorithm | SEDMLStochasticAlgorithm:
    if kisao_id in INTEGER_METHODS:
        return stochastic_algorithm(kisao_id, values)
    numbers = {key: finite_number(value) for key, value in values.items()}
    if set(numbers)-_OPTIONS.keys() or any(v <= 0 for v in numbers.values()):
        raise ValueError("unsupported or nonpositive deterministic numerical option")
    return SEDMLAlgorithm(kisao_id, **{_OPTIONS[k]: v for k, v in numbers.items()})


def resolve_simulations(document: SEDMLDocument) -> dict[str, UniformTimeCourse]:
    global_parameters = _raw_parameters(records(document.declaration, "listOfAlgorithmParameters"))
    used_global: set[str] = set()
    result = {}
    for node in records(document.declaration, "listOfSimulations"):
        algorithm = next(child for child in node.children if child.local_name == "algorithm")
        kisao_id = algorithm.attributes["kisaoID"]
        if kisao_id not in _METHODS and kisao_id not in INTEGER_METHODS:
            raise ValueError("requested algorithm is unavailable; no KiSAO substitution is enabled")
        # L1V4 local Simulation options override global values with the same KiSAO ID.
        eligible = parameter_ids(kisao_id) if kisao_id in INTEGER_METHODS else _OPTIONS.keys()
        globals_here = {k: v for k, v in global_parameters.items() if k in eligible}
        used_global.update(globals_here)
        _algorithm(kisao_id, globals_here)  # Validate globals before local overrides, as before.
        local_parameters = _raw_parameters(records(algorithm, "listOfAlgorithmParameters"))
        values = {**globals_here, **local_parameters}
        policy = _algorithm(kisao_id, values)
        attributes = node.attributes
        simulation = UniformTimeCourse(attributes["id"],
            finite_number(attributes["initialTime"]), finite_number(attributes["outputStartTime"]),
            finite_number(attributes["outputEndTime"]), step_count(attributes), policy)
        result[simulation.simulation_id] = simulation
    if not result:
        raise ValueError("experiment needs at least one supported simulation")
    if set(global_parameters)-used_global:
        raise ValueError("global algorithm parameters have no supported applicable simulation")
    return result
