"""Curated reference-checked property packages for ChemWorld.

This module intentionally stores a small auditable set of common compounds
instead of vendoring large third-party property tables. Coefficients are
selected from public correlation tables mirrored by the local `chemicals`
reference repository and are checked by optional reference-backend tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chemworld.physchem.curated_property_cards import (
    _REFERENCE_NOTES,
    curated_property_model_cards,
)
from chemworld.physchem.properties import R_J_PER_MOL_K, ComponentPropertyPackage
from chemworld.physchem.specs import (
    ComponentProvenance,
    ComponentSpec,
    ComponentUncertainty,
    PropertyCorrelation,
    component_alias_index,
    resolve_component_identifier,
)

_CURATED_INCHI_IDENTITIES: dict[str, tuple[str, str]] = {
    "water": ("InChI=1S/H2O/h1H2", "XLYOFNOQVPJJNP-UHFFFAOYSA-N"),
    "ethanol": (
        "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
        "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    ),
    "acetone": (
        "InChI=1S/C3H6O/c1-3(2)4/h1-2H3",
        "CSCPPACGZOOCGX-UHFFFAOYSA-N",
    ),
    "toluene": (
        "InChI=1S/C7H8/c1-7-5-3-2-4-6-7/h2-6H,1H3",
        "YXFVVABEGXRONW-UHFFFAOYSA-N",
    ),
    "methane": ("InChI=1S/CH4/h1H4", "VNWKTOKETHGBQD-UHFFFAOYSA-N"),
    "carbon_dioxide": (
        "InChI=1S/CO2/c2-1-3",
        "CURLTUGMZLYLDI-UHFFFAOYSA-N",
    ),
}


@dataclass(frozen=True)
class CuratedPropertyCase:
    component_id: str
    casrn: str
    display_name: str
    reference_temperature_K: float
    enthalpy_initial_temperature_K: float
    enthalpy_final_temperature_K: float
    reference_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "casrn": self.casrn,
            "display_name": self.display_name,
            "reference_temperature_K": self.reference_temperature_K,
            "enthalpy_initial_temperature_K": self.enthalpy_initial_temperature_K,
            "enthalpy_final_temperature_K": self.enthalpy_final_temperature_K,
            "reference_notes": list(self.reference_notes),
        }


@dataclass(frozen=True)
class _CuratedPropertyRecord:
    component_id: str
    display_name: str
    formula: str
    default_phase: str
    casrn: str
    aliases: tuple[str, ...]
    safety_tags: tuple[str, ...]
    vapor_pressure_coefficients: dict[str, float]
    vapor_pressure_temperature_range_K: tuple[float, float]
    poling_cp_coefficients: dict[str, float]
    ideal_gas_cp_temperature_range_K: tuple[float, float]
    reference_temperature_K: float
    enthalpy_initial_temperature_K: float
    enthalpy_final_temperature_K: float


_CURATED_RECORDS: tuple[_CuratedPropertyRecord, ...] = (
    _CuratedPropertyRecord(
        component_id="water",
        display_name="Water",
        formula="H2O",
        default_phase="liquid",
        casrn="7732-18-5",
        aliases=("water", "dihydrogen_oxide"),
        safety_tags=(),
        vapor_pressure_coefficients={
            "A": 73.649,
            "B": -7258.2,
            "C": -7.3037,
            "D": 4.1653e-06,
            "E": 2.0,
        },
        vapor_pressure_temperature_range_K=(273.16, 647.096),
        poling_cp_coefficients={
            "a": R_J_PER_MOL_K * 4.395,
            "b": R_J_PER_MOL_K * -0.004186,
            "c": R_J_PER_MOL_K * 1.405e-05,
            "d": R_J_PER_MOL_K * -1.564e-08,
            "e": R_J_PER_MOL_K * 6.32e-12,
        },
        ideal_gas_cp_temperature_range_K=(50.0, 1000.0),
        reference_temperature_K=300.0,
        enthalpy_initial_temperature_K=298.15,
        enthalpy_final_temperature_K=350.0,
    ),
    _CuratedPropertyRecord(
        component_id="ethanol",
        display_name="Ethanol",
        formula="C2H6O",
        default_phase="liquid",
        casrn="64-17-5",
        aliases=("ethyl_alcohol",),
        safety_tags=("flammable", "volatile"),
        vapor_pressure_coefficients={
            "A": 73.304,
            "B": -7122.3,
            "C": -7.1424,
            "D": 2.8853e-06,
            "E": 2.0,
        },
        vapor_pressure_temperature_range_K=(159.05, 514.0),
        poling_cp_coefficients={
            "a": R_J_PER_MOL_K * 4.396,
            "b": R_J_PER_MOL_K * 0.000628,
            "c": R_J_PER_MOL_K * 5.546e-05,
            "d": R_J_PER_MOL_K * -7.024e-08,
            "e": R_J_PER_MOL_K * 2.685e-11,
        },
        ideal_gas_cp_temperature_range_K=(50.0, 1000.0),
        reference_temperature_K=298.15,
        enthalpy_initial_temperature_K=298.15,
        enthalpy_final_temperature_K=360.0,
    ),
    _CuratedPropertyRecord(
        component_id="acetone",
        display_name="Acetone",
        formula="C3H6O",
        default_phase="liquid",
        casrn="67-64-1",
        aliases=("propanone",),
        safety_tags=("flammable", "volatile"),
        vapor_pressure_coefficients={
            "A": 69.006,
            "B": -5599.6,
            "C": -7.0985,
            "D": 6.2237e-06,
            "E": 2.0,
        },
        vapor_pressure_temperature_range_K=(178.45, 508.2),
        poling_cp_coefficients={
            "a": R_J_PER_MOL_K * 5.126,
            "b": R_J_PER_MOL_K * 0.001511,
            "c": R_J_PER_MOL_K * 5.731e-05,
            "d": R_J_PER_MOL_K * -7.177e-08,
            "e": R_J_PER_MOL_K * 2.728e-11,
        },
        ideal_gas_cp_temperature_range_K=(200.0, 1000.0),
        reference_temperature_K=300.0,
        enthalpy_initial_temperature_K=298.15,
        enthalpy_final_temperature_K=340.0,
    ),
    _CuratedPropertyRecord(
        component_id="toluene",
        display_name="Toluene",
        formula="C7H8",
        default_phase="liquid",
        casrn="108-88-3",
        aliases=("methylbenzene",),
        safety_tags=("flammable", "volatile"),
        vapor_pressure_coefficients={
            "A": 76.945,
            "B": -6729.8,
            "C": -8.179,
            "D": 5.3017e-06,
            "E": 2.0,
        },
        vapor_pressure_temperature_range_K=(178.18, 591.75),
        poling_cp_coefficients={
            "a": R_J_PER_MOL_K * 3.866,
            "b": R_J_PER_MOL_K * 0.003558,
            "c": R_J_PER_MOL_K * 0.00013356,
            "d": R_J_PER_MOL_K * -1.8659e-07,
            "e": R_J_PER_MOL_K * 7.69e-11,
        },
        ideal_gas_cp_temperature_range_K=(50.0, 1000.0),
        reference_temperature_K=350.0,
        enthalpy_initial_temperature_K=298.15,
        enthalpy_final_temperature_K=400.0,
    ),
    _CuratedPropertyRecord(
        component_id="methane",
        display_name="Methane",
        formula="CH4",
        default_phase="gas",
        casrn="74-82-8",
        aliases=("natural_gas_marker",),
        safety_tags=("flammable", "compressed_gas"),
        vapor_pressure_coefficients={
            "A": 39.205,
            "B": -1324.4,
            "C": -3.4366,
            "D": 3.1019e-05,
            "E": 2.0,
        },
        vapor_pressure_temperature_range_K=(90.69, 190.56),
        poling_cp_coefficients={
            "a": R_J_PER_MOL_K * 4.568,
            "b": R_J_PER_MOL_K * -0.008975,
            "c": R_J_PER_MOL_K * 3.631e-05,
            "d": R_J_PER_MOL_K * -3.407e-08,
            "e": R_J_PER_MOL_K * 1.091e-11,
        },
        ideal_gas_cp_temperature_range_K=(50.0, 1000.0),
        reference_temperature_K=150.0,
        enthalpy_initial_temperature_K=120.0,
        enthalpy_final_temperature_K=180.0,
    ),
    _CuratedPropertyRecord(
        component_id="carbon_dioxide",
        display_name="Carbon dioxide",
        formula="CO2",
        default_phase="gas",
        casrn="124-38-9",
        aliases=("co2",),
        safety_tags=("compressed_gas", "asphyxiant"),
        vapor_pressure_coefficients={
            "A": 140.54,
            "B": -4735.0,
            "C": -21.268,
            "D": 0.040909,
            "E": 1.0,
        },
        vapor_pressure_temperature_range_K=(216.58, 304.21),
        poling_cp_coefficients={
            "a": R_J_PER_MOL_K * 3.259,
            "b": R_J_PER_MOL_K * 0.001356,
            "c": R_J_PER_MOL_K * 1.502e-05,
            "d": R_J_PER_MOL_K * -2.374e-08,
            "e": R_J_PER_MOL_K * 1.056e-11,
        },
        ideal_gas_cp_temperature_range_K=(50.0, 1000.0),
        reference_temperature_K=250.0,
        enthalpy_initial_temperature_K=230.0,
        enthalpy_final_temperature_K=300.0,
    ),
)


def curated_property_cases() -> tuple[CuratedPropertyCase, ...]:
    """Return the public compounds and temperatures used for property checks."""

    return tuple(
        CuratedPropertyCase(
            component_id=record.component_id,
            casrn=record.casrn,
            display_name=record.display_name,
            reference_temperature_K=record.reference_temperature_K,
            enthalpy_initial_temperature_K=record.enthalpy_initial_temperature_K,
            enthalpy_final_temperature_K=record.enthalpy_final_temperature_K,
            reference_notes=_REFERENCE_NOTES,
        )
        for record in _CURATED_RECORDS
    )


def curated_components() -> tuple[ComponentSpec, ...]:
    """Return curated component specs with provenance metadata."""

    components = tuple(_component_from_record(record) for record in _CURATED_RECORDS)
    component_alias_index(components)
    return components


def curated_property_correlations(component_id: str) -> tuple[PropertyCorrelation, ...]:
    """Return curated correlations for one component."""

    record = _record(component_id)
    return (
        _dippr101_vapor_pressure(record),
        _poling_ideal_gas_heat_capacity(record),
    )


def curated_property_package(component_id: str) -> ComponentPropertyPackage:
    """Return a curated `ComponentPropertyPackage` by component id or alias."""

    record = _record(component_id)
    return ComponentPropertyPackage(
        _component_from_record(record),
        curated_property_correlations(record.component_id),
    )


def list_curated_property_packages() -> tuple[ComponentPropertyPackage, ...]:
    """Return all curated property packages."""

    return tuple(curated_property_package(record.component_id) for record in _CURATED_RECORDS)


def curated_property_case_map() -> dict[str, CuratedPropertyCase]:
    return {case.component_id: case for case in curated_property_cases()}


def _component_from_record(record: _CuratedPropertyRecord) -> ComponentSpec:
    inchi, inchi_key = _CURATED_INCHI_IDENTITIES[record.component_id]
    return ComponentSpec(
        identifier=record.component_id,
        formula=record.formula,
        cas_number=record.casrn,
        inchi=inchi,
        inchi_key=inchi_key,
        default_phase=record.default_phase,
        safety_tags=record.safety_tags,
        aliases=record.aliases,
        provenance=_component_provenance(record),
        uncertainty=_component_uncertainty(record),
        allowed_property_correlations=(
            "vapor_pressure",
            "ideal_gas_heat_capacity",
        ),
        metadata={
            "display_name": record.display_name,
            "casrn": record.casrn,
            "provenance": (
                "curated ChemWorld subset checked against local chemicals reference backend"
            ),
            "provenance_source_ids": [item.source_id for item in _component_provenance(record)],
            "uncertainty_field_ids": [item.field_id for item in _component_uncertainty(record)],
        },
    )


def _component_provenance(record: _CuratedPropertyRecord) -> tuple[ComponentProvenance, ...]:
    return (
        ComponentProvenance(
            source_id="chemicals:Psat_data_Perrys2_8",
            source_name="chemicals",
            source_table="Psat_data_Perrys2_8",
            source_key=record.casrn,
            source_path="reference_repos/chemicals/chemicals/vapor_pressure.py",
            notes=(
                "Perry/DIPPR101 vapor-pressure coefficients are selected for "
                "the curated ChemWorld regression slice.",
            ),
        ),
        ComponentProvenance(
            source_id="chemicals:Cp_data_Poling",
            source_name="chemicals",
            source_table="Cp_data_Poling",
            source_key=record.casrn,
            source_path="reference_repos/chemicals/chemicals/heat_capacity.py",
            notes=(
                "Poling ideal-gas heat-capacity coefficients are R-scaled into ChemWorld SI units.",
            ),
        ),
    )


def _component_uncertainty(record: _CuratedPropertyRecord) -> tuple[ComponentUncertainty, ...]:
    return (
        ComponentUncertainty(
            field_id="molecular_weight_g_mol",
            unit="g/mol",
            relative_uncertainty=ComponentSpec.molecular_weight_rel_tolerance,
            coverage="schema_consistency_tolerance",
            source_id="formula-derived",
            note=(
                "ComponentSpec validates supplied molecular weights against the "
                "formula-derived value within this relative tolerance."
            ),
        ),
        ComponentUncertainty(
            field_id="curated_property_coefficients",
            unit="dimensionless_or_declared_correlation_units",
            coverage="table_regression_case",
            source_id=f"chemicals:{record.casrn}",
            note=(
                "The curated coefficient subset is regression-checked against "
                "optional reference backends; it is not a full uncertainty model."
            ),
        ),
    )


def _dippr101_vapor_pressure(record: _CuratedPropertyRecord) -> PropertyCorrelation:
    return PropertyCorrelation(
        correlation_id=f"{record.component_id}_perry_dippr101_psat",
        property_id="vapor_pressure",
        equation_id="dippr101_vapor_pressure",
        coefficients=record.vapor_pressure_coefficients,
        input_units={"temperature": "K"},
        output_unit="Pa",
        validity_ranges={"temperature": record.vapor_pressure_temperature_range_K},
        source_note=(
            "Selected Perry 8e/DIPPR101 vapor-pressure coefficients mirrored by "
            "chemicals.vapor_pressure.Psat_data_Perrys2_8."
        ),
        metadata=_metadata(record, source_table="Psat_data_Perrys2_8"),
    )


def _poling_ideal_gas_heat_capacity(record: _CuratedPropertyRecord) -> PropertyCorrelation:
    return PropertyCorrelation(
        correlation_id=f"{record.component_id}_poling_ideal_gas_cp",
        property_id="ideal_gas_heat_capacity",
        equation_id="cp_polynomial",
        coefficients=record.poling_cp_coefficients,
        input_units={"temperature": "K"},
        output_unit="J/(mol*K)",
        validity_ranges={"temperature": record.ideal_gas_cp_temperature_range_K},
        source_note=(
            "Selected Poling ideal-gas heat-capacity coefficients mirrored by "
            "chemicals.heat_capacity.Cp_data_Poling; coefficients are R-scaled "
            "into J/(mol*K)."
        ),
        metadata=_metadata(record, source_table="Cp_data_Poling"),
    )


def _metadata(record: _CuratedPropertyRecord, *, source_table: str) -> dict[str, Any]:
    return {
        "component_id": record.component_id,
        "display_name": record.display_name,
        "casrn": record.casrn,
        "source_table": source_table,
        "reference_notes": list(_REFERENCE_NOTES),
    }


def _record(component_id_or_alias: str) -> _CuratedPropertyRecord:
    try:
        canonical = resolve_component_identifier(curated_components(), component_id_or_alias)
    except KeyError as exc:
        allowed = ", ".join(record.component_id for record in _CURATED_RECORDS)
        raise KeyError(
            f"unknown curated component {component_id_or_alias!r}; allowed={allowed}"
        ) from exc
    for record in _CURATED_RECORDS:
        if canonical == record.component_id:
            return record
    raise KeyError(f"unknown curated component {component_id_or_alias!r}")


__all__ = [
    "CuratedPropertyCase",
    "curated_components",
    "curated_property_case_map",
    "curated_property_cases",
    "curated_property_correlations",
    "curated_property_model_cards",
    "curated_property_package",
    "list_curated_property_packages",
]
