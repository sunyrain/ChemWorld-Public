"""Resolve source quantities to declared native Observation channels and units."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import cast

from chemworld.data.sbml import SBMLModel
from chemworld.data.sedml_models import _model_target
from chemworld.data.sedml_xml import ExperimentXMLNode
from chemworld.data.xml_inputs import bounded_xml
from chemworld.physchem.sbml_kinetics import SBMLKinetics
from chemworld.physchem.sbml_system import SBMLObservable, parameter_observable

TIME_SYMBOLS = frozenset(("urn:sedml:symbol:time", "KISAO:0000832"))
_SPECIES_SYMBOLS = {"KISAO:0000836": "amount", "KISAO:0000838": "concentration"}


@dataclass(frozen=True)
class SEDMLQuantity:
    observable: SBMLObservable | None
    native_to_source: float
    source_unit: str

    def __post_init__(self) -> None:
        if ((self.observable is not None and not isinstance(self.observable, SBMLObservable))
                or isinstance(self.native_to_source, bool)
                or not isinstance(self.native_to_source, (int, float))
                or not isfinite(self.native_to_source) or self.native_to_source <= 0
                or not isinstance(self.source_unit, str) or not self.source_unit):
            raise ValueError("quantity mapping needs an explicit unit and positive finite scale")

    def value(
        self, channels: Mapping[str, float], *, time_s: float, initial_model_time: float = 0.,
    ) -> float:
        result = (initial_model_time+time_s*self.native_to_source if self.observable is None
                  else channels[self.observable.channel_id]*self.native_to_source)
        if isinstance(result, bool) or not isinstance(result, (int, float)) or not isfinite(result):
            raise ValueError("experiment observation must be a finite source quantity")
        return float(result)


def _unit(model: SBMLModel, key: str) -> str:
    name = model.declaration.get(key, model.unit_defaults.get(key))
    if name is None:
        raise ValueError("source quantity has no explicit unit or author default")
    return name


def model_quantity(model: SBMLModel, variable: ExperimentXMLNode) -> SEDMLQuantity:
    """Select a mathematical model entity; XML literals remain preprocessing inputs.

Time is derived from the existing public laboratory clock. Species selectors
explicitly distinguish amount from concentration even when SBML uses the other
basis for that species' mathematical symbol.
"""
    symbol = variable.attributes.get("symbol")
    target = variable.attributes.get("target")
    if target is None:
        if symbol not in TIME_SYMBOLS:
            raise ValueError("unknown experiment model symbol")
        return SEDMLQuantity(None, 1/model.model_unit("timeUnits").time_s(1.),
                             _unit(model, "timeUnits"))
    node, attribute = _model_target(bounded_xml(model.to_xml()), target, variable.namespaces)
    if attribute is not None:
        raise ValueError("current/output model variables must target mathematical elements")
    identity, kind = cast(str, node.get("id")), node.tag.rsplit("}", 1)[-1]
    frame = SBMLKinetics(model)
    if kind == "species":
        record = frame._species[identity]
        selected = ("amount" if record.get("hasOnlySubstanceUnits") else "concentration")
        if symbol is not None:
            if symbol not in _SPECIES_SYMBOLS:
                raise ValueError("unknown species amount/concentration selector")
            selected = _SPECIES_SYMBOLS[symbol]
        amount_unit = record.get("substanceUnits") or _unit(model, "substanceUnits")
        factor = 1/frame._amount_scale[identity]
        source_unit = amount_unit
        if selected == "concentration":
            compartment = frame._compartments[record.get("compartment")]
            source_unit += "/"+(compartment.get("units") or _unit(model, "volumeUnits"))
            factor *= frame._volume_scale[record.get("compartment")]
        return SEDMLQuantity(SBMLObservable(selected, identity), factor, source_unit)
    if symbol is not None:
        raise ValueError("quantity selector symbols apply only to species")
    if kind == "compartment":
        record = frame._compartments[identity]
        return SEDMLQuantity(SBMLObservable("volume", identity), 1/frame._volume_scale[identity],
                             record.get("units") or _unit(model, "volumeUnits"))
    if kind == "parameter":
        observable = parameter_observable(model, identity)
        source_unit = frame._parameters[identity].get("units")
        return SEDMLQuantity(observable, 1/model.unit_signature(source_unit).si_factor, source_unit)
    if kind == "reaction":
        return SEDMLQuantity(SBMLObservable("reaction_rate", identity),
            frame._time_scale/cast(float, frame._extent_scale),
            _unit(model, "extentUnits")+"/"+_unit(model, "timeUnits"))
    raise ValueError("unsupported experiment model quantity")
