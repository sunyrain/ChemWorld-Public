"""Versioned configuration for a formal SBML entity on the native runtime.

This declaration contains initialization recipes, numerical and readout policy,
never evolving coordinates or a clock. Live data belong to EquipmentRecord.
"""

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any

from chemworld.data.sbml import SBMLModel
from chemworld.foundation.dimensions import dimension_definition
from chemworld.foundation.units import COHERENT_SI_UNITS, unit_spec
from chemworld.physchem.caloric_math import scalar_value
from chemworld.physchem.coordinate_events import _finite_real, _integer
from chemworld.physchem.sbml_events import SBMLTriggerRoots, sbml_coordinate_events
from chemworld.physchem.sbml_kinetics import SBMLKinetics, sbml_continuous_system
from chemworld.physchem.solver_backend import ODESolverPolicy

SBML_ENTITY_TYPE = "sbml_formal_kinetics_v1"


def _parameter_signature(model, identity):
    parameters = {p.get("id"): p for p in model.declaration.items("listOfParameters")}
    if identity not in parameters or parameters[identity].get("units") is None:
        raise ValueError("parameter readouts require a declared global parameter and source units")
    return model.unit_signature(parameters[identity].get("units"))


def parameter_observable(model, identity):
    signature = _parameter_signature(model, identity)
    for unit in COHERENT_SI_UNITS:
        if dimension_definition(unit_spec(unit).dimension).vector == signature.dimension:
            return SBMLObservable("parameter", identity, unit)
    raise ValueError("parameter units have no supported coherent native readout unit")


@dataclass(frozen=True)
class SBMLObservable:
    kind: str
    source_id: str
    unit_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in {
                "amount", "concentration", "volume", "parameter", "reaction_rate"}:
            raise ValueError("unknown formal readout kind")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("formal readout requires an original source identity")
        if (self.kind == "parameter" and self.unit_name not in COHERENT_SI_UNITS) or (
                self.kind != "parameter" and self.unit_name is not None):
            raise ValueError("parameter readouts need an explicit supported coherent SI unit")

    @property
    def channel_id(self):
        return self.kind+"_"+self.source_id

    @property
    def unit(self):
        return (self.unit_name if self.kind == "parameter" else
            {"amount": "mol", "concentration": "mol/L", "volume": "L",
             "reaction_rate": "mol/s"}[self.kind])

    def to_dict(self):
        payload = {"kind": self.kind, "source_id": self.source_id}
        if self.kind == "parameter":
            payload["unit"] = self.unit_name
        return payload

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise ValueError("unknown formal readout fields")
        expected = {"kind", "source_id"} | (
            {"unit"} if payload.get("kind") == "parameter" else set())
        if set(payload) != expected:
            raise ValueError("unknown formal readout fields")
        return cls(payload["kind"], payload["source_id"], payload.get("unit"))


@dataclass(frozen=True)
class SBMLSystemSpec:
    model: SBMLModel
    observables: tuple[SBMLObservable, ...] = ()
    initial_model_time: float = 0.
    numerical_form: str = "ode"
    rtol: float = 1e-9
    atol: float = 1e-12
    maximum_step_s: float | None = None
    maximum_events: int = 1000
    maximum_segments: int = 10000
    maximum_pending_events: int = 1024
    maximum_coordinates: int = 256
    coordinate_scales: object = field(default_factory=dict)
    readout_relative_sd: float = .01
    controlled_parameters: tuple[str, ...] = ()
    solver_method: str = "BDF"
    tau_policy: Any = None
    _frame: object = field(init=False, repr=False, compare=False)
    _events: tuple = field(init=False, repr=False, compare=False)
    _roots: object = field(init=False, repr=False, compare=False)
    _integer: object = field(init=False, repr=False, compare=False, default=None)
    _guard_critical_reactions: frozenset = field(init=False, repr=False, compare=False,
                                               default=frozenset())

    def __post_init__(self) -> None:
        for name in ("maximum_events", "maximum_segments", "maximum_pending_events",
                     "maximum_coordinates"):
            if not _integer(getattr(self, name), 1):
                raise ValueError("formal numerical budgets must be positive integers")
        if (not _finite_real(self.readout_relative_sd) or self.readout_relative_sd < 0
                or any(not _finite_real(v) or v <= 0 for v in (self.rtol, self.atol))):
            raise ValueError("formal tolerances and detector noise require finite declared values")
        from chemworld.physchem.sbml_stochastic import ALGORITHMS

        integer_mode = self.solver_method in ALGORITHMS
        if ((integer_mode and self.numerical_form != "stochastic")
                or (not integer_mode and self.solver_method not in ("BDF", "LSODA"))) or (
                self.numerical_form == "residual" and self.solver_method != "BDF"):
            raise ValueError("formal model solver/form must name an implemented common method")
        if not integer_mode and self.tau_policy is not None:
            raise ValueError("tau_policy requires a stochastic formal model")
        source_boundaries = integer_mode and any(self.model.declaration.items(name)
            for name in ("listOfRules", "listOfEvents", "listOfConstraints"))
        frame = SBMLKinetics(self.model, self.initial_model_time, self.controlled_parameters,
                            integer_quantities=source_boundaries)
        if not frame._compartments:
            raise ValueError("native formal worlds require at least one declared 3D compartment")
        if len(frame.initial_values) > self.maximum_coordinates:
            raise ValueError("formal model exceeds its declared coordinate budget")
        events = sbml_coordinate_events(frame)
        if events and not integer_mode and self.maximum_step_s is None:
            raise ValueError("source events require an authored finite root maximum step")
        if self.maximum_step_s is not None and (
                not _finite_real(self.maximum_step_s) or self.maximum_step_s <= 0):
            raise ValueError("formal root maximum step must be finite positive seconds")
        if integer_mode:
            from chemworld.physchem.sbml_stochastic import SBMLIntegerKinetics

            integer = SBMLIntegerKinetics(frame)
        else:
            integer = None
        if integer is not None and integer.requires_boundaries:
            from chemworld.physchem.sbml_integer_boundary_policy import guard_critical_reactions
            from chemworld.physchem.sbml_jump_triggers import SBMLJumpTriggers

            roots = SBMLJumpTriggers(frame, integer.rules)
            object.__setattr__(self, "_guard_critical_reactions", guard_critical_reactions(
                roots, integer.compile(frame.initial_values, self.initial_model_time)))
        else:
            roots = SBMLTriggerRoots(frame)
        observables = tuple(self.observables)
        if (any(not isinstance(o, SBMLObservable) for o in observables)
                or len({o.channel_id for o in observables}) != len(observables)):
            raise ValueError("formal readout declarations must be known and unique")
        for observable in observables:
            known = (frame._compartments if observable.kind == "volume" else frame._parameters
                     if observable.kind == "parameter" else frame._reaction_ids
                     if observable.kind == "reaction_rate" else frame._species)
            if observable.source_id not in known:
                raise ValueError("formal readout references an undeclared source quantity")
            if observable.kind == "parameter":
                signature = _parameter_signature(self.model, observable.source_id)
                if signature.dimension != dimension_definition(unit_spec(observable.unit).dimension
                                                               ).vector:
                    raise ValueError("parameter readout unit differs from its source dimensions")
        if not isinstance(self.coordinate_scales, (dict, MappingProxyType)):
            raise ValueError("formal coordinate scales require a named mapping")
        scales = dict(self.coordinate_scales)
        if (set(scales)-frame.initial_values.keys()
                or any(not _finite_real(v) or v <= 0 for v in scales.values())):
            raise ValueError("formal scales must name declared coordinates with positive values")
        object.__setattr__(self, "observables", observables)
        object.__setattr__(self, "coordinate_scales", MappingProxyType(scales))
        object.__setattr__(self, "controlled_parameters", frame.controlled_parameters)
        object.__setattr__(self, "_frame", frame)
        object.__setattr__(self, "_events", events)
        object.__setattr__(self, "_roots", roots)
        if integer_mode:
            policy = integer.policy(frame.initial_values, self.solver_method,
                                    self.maximum_events, self.tau_policy)
            object.__setattr__(self, "tau_policy", policy.tau_policy)
            object.__setattr__(self, "_integer", integer)
        else:
            self.continuous_system()  # Validate the common numerical declaration now.

    @property
    def frame(self):
        return self._frame

    @property
    def events(self):
        return self._events

    @property
    def roots(self):
        return self._roots

    @property
    def integer_kinetics(self):
        return self._integer

    @property
    def guard_critical_reactions(self):
        return self._guard_critical_reactions

    def continuous_system(self):
        if self.integer_kinetics is not None:
            raise ValueError("an integer SBML declaration is not a continuous ODE system")
        policy = ODESolverPolicy("chemworld_formal_sbml_"+self.solver_method.lower()+"_v1",
            self.solver_method,
            rtol=self.rtol, atol=self.atol, max_step_s=self.maximum_step_s)
        return sbml_continuous_system(self.frame, coordinate_scales=self.coordinate_scales,
                                      numerical_form=self.numerical_form, policy=policy)

    def readout(self, values, time_s):
        self.frame.validate_values(values, time_s)
        amounts, volumes = self.frame.quantities(values, time_s)
        context = self.frame.context(values, time_s)
        result = {}
        for observable in self.observables:
            name = observable.source_id
            if observable.kind == "parameter":
                value = context[name]*_parameter_signature(self.model, name).si_factor
            elif observable.kind == "reaction_rate":
                value = context[name]*self.frame._extent_scale/self.frame._time_scale
            else:
                value = (volumes[name] if observable.kind == "volume" else amounts[name]
                         if observable.kind == "amount" else
                         amounts[name]/volumes[self.frame._species[name].get("compartment")])
            if not isfinite(scalar_value(value)):
                raise ValueError("nonfinite formal readout")
            result[observable.channel_id] = value
        return result

    def to_dict(self):
        payload = {"format": "chemworld_sbml_system_v1", "model": self.model.to_dict(),
            "observables": [o.to_dict() for o in self.observables],
            **{name: getattr(self, name) for name in ("initial_model_time", "numerical_form",
                "rtol", "atol", "maximum_step_s", "maximum_events", "maximum_segments",
                "maximum_pending_events", "maximum_coordinates", "readout_relative_sd")},
            "coordinate_scales": dict(self.coordinate_scales)}
        if (self.controlled_parameters or self.solver_method != "BDF"
                or any(o.kind in {"parameter", "reaction_rate"} for o in self.observables)):
            payload.update(format="chemworld_sbml_system_v2",
                controlled_parameters=list(self.controlled_parameters),
                solver_method=self.solver_method)
        if self.integer_kinetics is not None:
            payload.update(format="chemworld_sbml_system_v3",
                tau_policy=None if self.tau_policy is None else self.tau_policy.to_dict())
        return payload

    @classmethod
    def from_dict(cls, payload):
        expected = {"format", "model", "observables", "initial_model_time", "numerical_form",
            "rtol", "atol", "maximum_step_s", "maximum_events", "maximum_segments",
            "maximum_pending_events", "maximum_coordinates", "readout_relative_sd",
            "coordinate_scales"}
        if not isinstance(payload, dict):
            raise ValueError("formal model declaration must be an object")
        if payload.get("format") in {"chemworld_sbml_system_v2", "chemworld_sbml_system_v3"}:
            expected |= {"controlled_parameters", "solver_method"}
            if not isinstance(payload.get("controlled_parameters"), list):
                raise ValueError("external model controls must be an explicit JSON list")
            if payload["format"] == "chemworld_sbml_system_v3":
                expected |= {"tau_policy"}
        elif payload.get("format") != "chemworld_sbml_system_v1":
            raise ValueError("unknown formal model declaration format")
        if set(payload) != expected or not isinstance(payload["observables"], list):
            raise ValueError("unknown formal model declaration fields/format")
        result = cls(**{**{k: v for k, v in payload.items()
                         if k not in {"format", "model", "observables"}},
            "model": SBMLModel.from_dict(payload["model"]),
            "observables": tuple(SBMLObservable.from_dict(p) for p in payload["observables"])})
        if ((payload["format"] == "chemworld_sbml_system_v3")
                != (result.integer_kinetics is not None)):
            raise ValueError("integer SBML declarations require the v3 codec")
        return result
