"""Declared deterministic algorithm policies and source-time output grids."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from chemworld.data.sedml_declarations import finite_number, records, step_count
from chemworld.physchem.solver_backend import ODESolverPolicy

_METHODS = {"KISAO:0000288": "BDF", "KISAO:0000088": "LSODA"}
_OPTIONS = {"KISAO:0000209": "rtol", "KISAO:0000211": "atol",
            "KISAO:0000467": "maximum_step_source"}


def _parameters(nodes):
    result = {}
    for node in nodes:
        key = node.attributes["kisaoID"]
        if key not in _OPTIONS or key in result:
            raise ValueError("unsupported or duplicate deterministic algorithm parameter")
        value = finite_number(node.attributes["value"])
        if value <= 0:
            raise ValueError("deterministic numerical options must be positive")
        result[key] = value
    return result


@dataclass(frozen=True)
class SEDMLAlgorithm:
    kisao_id: str
    rtol: float = 1e-9
    atol: float = 1e-12
    maximum_step_source: float | None = None

    def __post_init__(self):
        values = (self.rtol, self.atol, *((self.maximum_step_source,)
                  if self.maximum_step_source is not None else ()))
        if self.kisao_id not in _METHODS or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v <= 0
                for v in values):
            raise ValueError("unsupported SED-ML deterministic algorithm or numerical policy")

    @property
    def method(self):
        return _METHODS[self.kisao_id]

    def native_policy(self, *, seconds_per_source_time):
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
    algorithm: SEDMLAlgorithm

    def __post_init__(self):
        values = (self.initial_time, self.output_start, self.output_end)
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v)
                for v in values) or not self.initial_time <= self.output_start <= self.output_end
                or type(self.number_of_steps) is not int or not 1 <= self.number_of_steps <= 10000
                or not isinstance(self.algorithm, SEDMLAlgorithm)
                or not isinstance(self.simulation_id, str) or not self.simulation_id):
            raise ValueError("uniform experiment must declare a finite ordered source-time grid")

    @property
    def source_times(self):
        values = np.linspace(self.output_start, self.output_end, self.number_of_steps+1)
        if not np.all(np.isfinite(values)):
            raise ValueError("uniform experiment grid exceeds numeric range")
        return tuple(float(value) for value in values)


def resolve_simulations(document):
    global_parameters = _parameters(records(document.declaration, "listOfAlgorithmParameters"))
    result = {}
    for node in records(document.declaration, "listOfSimulations"):
        algorithm = next(child for child in node.children if child.local_name == "algorithm")
        kisao_id = algorithm.attributes["kisaoID"]
        if kisao_id not in _METHODS:
            raise ValueError("requested algorithm is unavailable; no KiSAO substitution is enabled")
        # L1V4 local Simulation options override global values with the same KiSAO ID.
        values = {**global_parameters,
                  **_parameters(records(algorithm, "listOfAlgorithmParameters"))}
        policy = SEDMLAlgorithm(kisao_id, **{_OPTIONS[key]: value for key, value in values.items()})
        attributes = node.attributes
        simulation = UniformTimeCourse(attributes["id"],
            finite_number(attributes["initialTime"]), finite_number(attributes["outputStartTime"]),
            finite_number(attributes["outputEndTime"]), step_count(attributes), policy)
        result[simulation.simulation_id] = simulation
    if not result:
        raise ValueError("experiment needs at least one supported deterministic simulation")
    return result
