"""Explicit B3 constants to the original virial correlation/input-card contract."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from chemworld.data.constants import ConstantDatabase
from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.specs import ComponentSpec, ComponentUncertainty
from chemworld.physchem.virial_correlations import virial_correlation
from chemworld.physchem.virial_kernels import VIRIAL_METHODS


def sourced_virial_input(
    database: ConstantDatabase,
    identity: object,
    component: ComponentSpec,
    *,
    method: str,
    constant_source: str,
    temperature_range_K: Sequence[float],
    parameters: Mapping[str, float] | None = None,
    polar: Mapping[str, Any] | None = None,
) -> PropertyInputCard:
    if method not in VIRIAL_METHODS:
        raise ValueError("unknown pure virial method")
    record = database.record(identity, method=constant_source)
    if (record.cas_number or record.source_key) not in {component.cas_number, component.identifier}:
        raise ValueError("virial source identity does not match ComponentSpec")
    units = {"Tc": "K", "Pc": "Pa", "Vc": "m^3/mol", "omega": "dimensionless"}
    supplied = dict(parameters or {})
    if units.keys() & supplied.keys():
        raise ValueError("explicit parameters cannot override selected source constants")
    values, provenance = {}, {}
    for key in VIRIAL_METHODS[method].required.split():
        if key in units:
            value = record.value(key)
            values[key] = value.quantity(units[key]).value
            provenance[key] = value.to_dict()
    curve = virial_correlation(
        method,
        parameters={**values, **supplied},
        temperature_range_K=temperature_range_K,
        polar=polar,
        source_note=(f"{record.source.provenance.citation}; "
                     f"explicitly selected {record.record_id}"),
    )
    curve = replace(
        curve,
        metadata={
            **curve.metadata,
            "parameter_sources": provenance,
            "source_temperature_range": None,
            "application_temperature_range_K": temperature_range_K,
            "application_basis": (
                "caller supplied; critical constants do not establish fit validity"
            ),
        },
    )
    source = record.source.provenance
    return PropertyInputCard(
        ComponentPropertyPackage(component, (curve,)),
        (InputSource(source, "literature"),),
        {
            curve.correlation_id: ComponentUncertainty(
                curve.correlation_id,
                unit=curve.output_unit,
                source_id=source.source_id,
                coverage="unspecified",
                note="Source critical inputs plus chosen correlation; uncertainty not quantified",
            )
        },
        None,
        "Virial coefficient only; caloric model/reference must be explicitly supplied",
    )
