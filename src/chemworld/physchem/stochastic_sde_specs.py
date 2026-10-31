"""Immutable reaction SDE declarations and SI-typed scalar noise expressions."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from math import isfinite
from numbers import Real
from typing import Any, Self, cast
from xml.etree.ElementTree import Element

from chemworld.data.xml_inputs import bounded_xml
from chemworld.foundation.dimensions import DimensionVector, dimension_definition
from chemworld.foundation.discrete_inventory import CONTINUOUS_SDE_ALGORITHM
from chemworld.foundation.units import convert_value, unit_spec
from chemworld.physchem.mathml import MATHML_NS, MathExpression, _identifier

SDE_ALGORITHM = CONTINUOUS_SDE_ALGORITHM
SDE_MODEL_SCHEMA = "native-reaction-sde-v1"
POPULATION_BASIS = "continuous-particle-count"
NOISE_UNIT = "particle/sqrt(s)"


def finite_real(value: object, label: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(label + " must be a finite " + ("positive " if positive else "") + "real")
    return float(value)


@dataclass(frozen=True)
class SDEParameter:
    """A scalar in coherent SI; particle populations themselves are dimensionless."""

    name: str
    value: float
    dimension: DimensionVector = field(default_factory=DimensionVector)

    def __post_init__(self) -> None:
        if _identifier(self.name) != self.name or not isinstance(self.dimension, DimensionVector):
            raise ValueError("SDE parameters require an exact symbol and common dimension")
        object.__setattr__(self, "value", finite_real(self.value, "SDE parameter"))

    @classmethod
    def from_unit(cls, name: str, value: float, unit: str) -> Self:
        definition = dimension_definition(unit_spec(unit).dimension)
        return cls(
            name,
            convert_value(finite_real(value, "SDE parameter"), unit, definition.canonical_unit),
            definition.vector,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "dimension": self.dimension.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if (
            not isinstance(value, Mapping)
            or set(value) != {"name", "value", "dimension"}
            or not isinstance(value["dimension"], Mapping)
            or set(value["dimension"]) != set(DimensionVector().to_dict())
        ):
            raise ValueError("unknown SDE parameter fields or physical axes")
        return cls(value["name"], value["value"], DimensionVector(**value["dimension"]))


@dataclass(frozen=True)
class ReactionNoise:
    """One independent noise column on one declared stoichiometric channel.

    None means sqrt(abs(particle_flux)). A MathML expression returns particle/sqrt(s)
    from typed SI parameters, populations n0..., amounts a0..., concentrations c0...,
    fluxes f0..., the channel's flux, time_s, volume_m3 and avogadro (1/mol).
    """

    reaction_id: str
    expression: MathExpression | str | None = None
    parameters: tuple[SDEParameter, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.reaction_id, str) or not self.reaction_id:
            raise ValueError("reaction noise requires a declared channel identity")
        parameters = tuple(self.parameters)
        if any(not isinstance(value, SDEParameter) for value in parameters) or len(
            {value.name for value in parameters}
        ) != len(parameters):
            raise ValueError("noise parameters must be unique typed declarations")
        expression = self.expression
        if isinstance(expression, str):
            expression = MathExpression(expression)
        if expression is not None and not isinstance(expression, MathExpression):
            raise ValueError("reaction noise requires bounded MathML or the declared default")
        if expression is None and parameters:
            raise ValueError("the default reaction noise has no free parameters")
        if expression is not None:
            # The Core Avogadro csymbol has its own historical numeric convention.
            # The SDE language uses the explicitly typed native SI avogadro symbol.
            def check(node: Element) -> None:
                if (
                    node.tag == "{" + MATHML_NS + "}csymbol"
                    and node.get("definitionURL") != "http://www.sbml.org/sbml/symbols/time"
                ):
                    raise ValueError("use the typed native avogadro parameter in SDE noise")
                children = list(node)
                if node.tag == "{" + MATHML_NS + "}semantics":
                    children = children[:1]
                for child in children:
                    check(child)

            check(bounded_xml(expression.xml))
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "parameters", parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "expression": (
                None if self.expression is None else cast(MathExpression, self.expression).xml
            ),
            "parameters": [value.to_dict() for value in self.parameters],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if (
            not isinstance(value, Mapping)
            or set(value) != {"reaction_id", "expression", "parameters"}
            or not isinstance(value["parameters"], list)
        ):
            raise ValueError("unknown reaction noise declaration")
        return cls(
            value["reaction_id"],
            value["expression"],
            tuple(SDEParameter.from_dict(p) for p in value["parameters"]),
        )


@dataclass(frozen=True)
class ReactionFlux:
    """Explicit signed particle/s flux, evaluated on the current stage only.

    Flux expressions may use time, populations, amounts and SI concentrations.
    Other reaction fluxes are not inputs: algebraic reaction cycles are not
    silently solved by expression evaluation.
    """

    reaction_id: str
    expression: MathExpression | str
    parameters: tuple[SDEParameter, ...] = ()

    def __post_init__(self) -> None:
        declaration = ReactionNoise(self.reaction_id, self.expression, self.parameters)
        if declaration.expression is None:
            raise ValueError("an explicit reaction flux requires bounded MathML")
        object.__setattr__(self, "expression", declaration.expression)
        object.__setattr__(self, "parameters", declaration.parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "expression": cast(MathExpression, self.expression).xml,
            "parameters": [value.to_dict() for value in self.parameters],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        declaration = ReactionNoise.from_dict(value)
        # The flux constructor retains its original rejection of null expressions.
        return cls(
            declaration.reaction_id, cast(MathExpression, declaration.expression),
            declaration.parameters,
        )


@dataclass(frozen=True)
class SDEControl:
    """A bounded public input bound to one immutable reaction declaration."""

    name: str
    reaction_id: str
    target: str
    lower: float
    upper: float
    parameter: str | None = None

    def __post_init__(self) -> None:
        if (
            _identifier(self.name) != self.name
            or not isinstance(self.reaction_id, str)
            or not self.reaction_id
            or self.target not in {"rate_multiplier", "flux_parameter", "noise_parameter"}
        ):
            raise ValueError("SDE control requires an exact name and declared reaction target")
        if self.target == "rate_multiplier":
            if self.parameter is not None:
                raise ValueError("reaction multiplier control has no named parameter")
        elif not isinstance(self.parameter, str) or _identifier(self.parameter) != self.parameter:
            raise ValueError("SDE parameter control requires a declared parameter name")
        for key in ("lower", "upper"):
            object.__setattr__(self, key, finite_real(getattr(self, key), "SDE control bound"))
        if self.lower > self.upper or (self.target == "rate_multiplier" and self.lower < 0):
            raise ValueError("SDE control bounds must be ordered and multipliers nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("unknown SDE control fields")
        return cls(**value)


@dataclass(frozen=True)
class StochasticSDESpec:
    volume_L: float
    temperature_K: float = 298.15
    step_s: float = 0.01
    min_step_s: float = 1e-12
    max_steps_per_advance: int = 100000
    max_evaluations_per_advance: int = 1000000
    max_rejections_per_step: int = 256
    max_root_iterations: int = 64
    boundary_policy: str = "physical"
    noises: tuple[ReactionNoise, ...] | None = None
    fluxes: tuple[ReactionFlux, ...] = ()
    controls: tuple[SDEControl, ...] = ()
    minimum_volume_L: float | None = None
    output_policy: str = "linear"

    def __post_init__(self) -> None:
        for name in ("volume_L", "temperature_K", "step_s", "min_step_s"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name, positive=True))
        if self.min_step_s > self.step_s:
            raise ValueError("SDE minimum physical step exceeds its declared step")
        for name in (
            "max_steps_per_advance",
            "max_evaluations_per_advance",
            "max_rejections_per_step",
            "max_root_iterations",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("SDE numerical budgets must be positive integers")
        if self.boundary_policy not in {"physical", "reject"}:
            raise ValueError("unknown SDE physical boundary policy")
        if not isinstance(self.output_policy, str) or self.output_policy not in {
            "linear",
            "ri5_fractional",
            "ri5_truncated",
        }:
            raise ValueError("unknown SDE numerical output policy")
        if self.noises is not None:
            noises = tuple(self.noises)
            if any(not isinstance(value, ReactionNoise) for value in noises) or len(
                {value.reaction_id for value in noises}
            ) != len(noises):
                raise ValueError("SDE independent noise channels must be unique declarations")
            object.__setattr__(self, "noises", noises)
        fluxes = tuple(self.fluxes)
        if any(not isinstance(value, ReactionFlux) for value in fluxes) or len(
            {value.reaction_id for value in fluxes}
        ) != len(fluxes):
            raise ValueError("SDE explicit fluxes must be unique declarations")
        object.__setattr__(self, "fluxes", fluxes)
        controls = tuple(self.controls)
        if (
            len(controls) > 256
            or any(not isinstance(c, SDEControl) for c in controls)
            or len({c.name for c in controls}) != len(controls)
            or len({(c.reaction_id, c.target, c.parameter) for c in controls}) != len(controls)
        ):
            raise ValueError("SDE controls require unique bounded names and physical targets")
        object.__setattr__(self, "controls", controls)
        if self.minimum_volume_L is not None:
            minimum = finite_real(self.minimum_volume_L, "SDE sampling volume floor", positive=True)
            if minimum > self.volume_L:
                raise ValueError("SDE volume floor exceeds its initial volume")
            object.__setattr__(self, "minimum_volume_L", minimum)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SDE_MODEL_SCHEMA,
            "population_basis": POPULATION_BASIS,
            "noise_unit": NOISE_UNIT,
            **{
                f.name: getattr(self, f.name)
                for f in fields(self)
                if f.name not in {"noises", "fluxes", "controls", "output_policy"}
            },
            "noises": None if self.noises is None else [n.to_dict() for n in self.noises],
            "fluxes": [f.to_dict() for f in self.fluxes],
            "controls": [c.to_dict() for c in self.controls],
            **({"output_policy": self.output_policy} if self.output_policy != "linear" else {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise ValueError("SDE model requires a declared mapping")
        values = dict(payload)
        # The historical exact codec omitted an output-policy field. Preserve
        # that declaration and its execution signature as linear output.
        values.setdefault("output_policy", "linear")
        if (
            values.pop("schema", None) != SDE_MODEL_SCHEMA
            or values.pop("population_basis", None) != POPULATION_BASIS
            or values.pop("noise_unit", None) != NOISE_UNIT
            or set(values) != {f.name for f in fields(cls)}
        ):
            raise ValueError("unknown SDE schema, numerical basis, units or fields")
        noises = values["noises"]
        if noises is not None:
            if not isinstance(noises, list):
                raise ValueError("SDE noise channels must be a declared list")
            values["noises"] = tuple(ReactionNoise.from_dict(n) for n in noises)
        if not isinstance(values["fluxes"], list):
            raise ValueError("SDE fluxes must be a declared list")
        values["fluxes"] = tuple(ReactionFlux.from_dict(f) for f in values["fluxes"])
        if not isinstance(values["controls"], list):
            raise ValueError("SDE controls must be a declared list")
        values["controls"] = tuple(SDEControl.from_dict(c) for c in values["controls"])
        return cls(**values)
