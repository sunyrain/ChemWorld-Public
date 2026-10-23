"""Native property estimates from explicit, source-bearing molecular groups.

The molecular grouping/structural descriptors are authored inputs. No RDKit or
reference package is loaded, and missing coefficients never become zeros.
Predicted constants and curves feed existing native declarations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, isfinite
from types import MappingProxyType
from typing import Any

from chemworld.data.scientific_inputs import InputValue
from chemworld.physchem.critical_properties import CriticalEstimate
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.specs import PropertyCorrelation

JOBACK_FIELDS = (
    "Tc",
    "Pc",
    "Vc",
    "Tb",
    "Tm",
    "Hform",
    "Gform",
    "Cpa",
    "Cpb",
    "Cpc",
    "Cpd",
    "Hfus",
    "Hvap",
    "mua",
    "mub",
)


@dataclass(frozen=True)
class GroupContributionTable:
    """Selected coefficient rows, with original missing values and provenance."""

    method: str
    coefficients: Mapping[str, Mapping[str, float | None]]
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        expected = {
            "Joback": set(JOBACK_FIELDS),
            "Fedors": {"Vc"},
            "Wilson-Jasperson": {"Tc", "Pc"},
            "Bondi": {"Vw", "Aw"},
        }
        if self.method not in expected or not self.coefficients:
            raise ValueError("group contribution table requires a method and coefficient rows")
        data = {}
        for key, fields in self.coefficients.items():
            if not isinstance(key, str) or not key or not fields:
                raise ValueError("group ids and coefficient fields must be explicit")
            if set(fields) != expected[self.method]:
                raise ValueError(
                    "group coefficient fields must match the selected method; "
                    "use explicit None for unavailable source increments"
                )
            if any(
                v is not None and (isinstance(v, bool) or not isfinite(v)) for v in fields.values()
            ):
                raise ValueError("group coefficients must be finite or explicitly missing")
            data[key] = MappingProxyType(dict(fields))
        object.__setattr__(self, "coefficients", MappingProxyType(data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "coefficients": {k: dict(v) for k, v in self.coefficients.items()},
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GroupContributionTable:
        return cls(
            payload["method"],
            payload["coefficients"],
            DataSourceProvenance.from_dict(payload["source"]),
        )


def _counts(counts: Mapping[str, int], table: GroupContributionTable) -> None:
    if not counts or any(not isinstance(k, str) or not k for k in counts):
        raise ValueError("molecular groups require explicit nonempty identities")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in counts.values()):
        raise ValueError("group counts must be positive integers")
    if not counts.keys() <= table.coefficients.keys():
        raise ValueError(
            f"missing group parameter rows: {sorted(counts.keys() - table.coefficients.keys())}"
        )


def _sum(counts: Mapping[str, int], table: GroupContributionTable, field: str) -> float | None:
    values = [table.coefficients[key].get(field) for key in counts]
    if any(v is None for v in values):
        return None
    return sum(
        count * float(value)
        for count, value in zip(counts.values(), values, strict=True)
        if value is not None
    )


def _positive(value: float, label: str) -> float:
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return value


def joback_group_values(
    counts: Mapping[str, int],
    table: GroupContributionTable,
    *,
    atom_count: int,
    boiling_temperature_K: float | None = None,
) -> dict[str, float | tuple[float, ...] | None]:
    """All eleven Joback outputs, retaining unavailable original increments."""
    if table.method != "Joback":
        raise ValueError("Joback estimates require a Joback coefficient table")
    _counts(counts, table)
    if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count <= 0:
        raise ValueError("Joback atom count includes all hydrogen and must be positive")
    sums = {field: _sum(counts, table, field) for field in JOBACK_FIELDS}
    result: dict[str, float | tuple[float, ...] | None] = {}
    for output, field, offset, scale in (
        ("Tb", "Tb", 198.2, 1.0),
        ("Tm", "Tm", 122.5, 1.0),
        ("Vc", "Vc", 17.5, 1e-6),
        ("Hf", "Hform", 68.29, 1000.0),
        ("Gf", "Gform", 53.88, 1000.0),
        ("Hfus", "Hfus", -0.88, 1000.0),
        ("Hvap", "Hvap", 15.3, 1000.0),
    ):
        value = sums[field]
        result[output] = None if value is None else scale * (offset + value)
    tb = (
        (None if sums["Tb"] is None else 198.2 + sums["Tb"])
        if boiling_temperature_K is None
        else _positive(boiling_temperature_K, "known boiling temperature")
    )
    tc, pc = sums["Tc"], sums["Pc"]
    denominator = None if tc is None else 0.584 + 0.965 * tc - tc * tc
    result["Tc"] = (
        None if tb is None or denominator is None or denominator <= 0 else float(tb) / denominator
    )
    pressure_denominator = None if pc is None else 0.113 + 0.0032 * atom_count - pc
    result["Pc"] = (
        None
        if pressure_denominator is None or pressure_denominator <= 0
        else 1e5 / pressure_denominator**2
    )
    cp = [sums[field] for field in ("Cpa", "Cpb", "Cpc", "Cpd")]
    mu = [sums[field] for field in ("mua", "mub")]
    result["Cpig_coefficients"] = (
        None
        if any(v is None for v in cp)
        else tuple(
            float(v) + offset
            for v, offset in zip(cp, (-37.93, 0.210, -3.91e-4, 2.06e-7), strict=True)
            if v is not None
        )
    )
    result["viscosity_coefficients"] = (
        None
        if any(v is None for v in mu)
        else tuple(
            float(v) + offset
            for v, offset in zip(mu, (-597.82, -11.202), strict=True)
            if v is not None
        )
    )
    return result


def fedors_critical_volume(
    atom_counts: Mapping[str, int],
    table: GroupContributionTable,
    *,
    alcohol_oxygens: int = 0,
    amine_nitrogens: int = 0,
    rings_by_size: Mapping[int, int] | None = None,
    double_bonds: int = 0,
    triple_bonds: int = 0,
    ring_ring_bonds: int = 0,
) -> float:
    """Fedors volume from explicit atoms, bond counts and ring descriptors.

    Alcohol O and amine N replace (rather than double count) generic atoms.
    Aromatic bonds do not enter the explicit double-bond correction.
    """
    if table.method != "Fedors":
        raise ValueError("Fedors requires the selected Fedors table")
    descriptors = {
        "O_alcohol": alcohol_oxygens,
        "N_amine": amine_nitrogens,
        "double_bond": double_bonds,
        "triple_bond": triple_bonds,
        "ring_ring_bonds": ring_ring_bonds,
    }
    for size, count in (rings_by_size or {}).items():
        if isinstance(size, bool) or size not in {3, 4, 5, 6}:
            raise ValueError("Fedors only supports declared ring sizes 3..6")
        descriptors[f"{size}_ring"] = count
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in descriptors.values()):
        raise ValueError("Fedors descriptors must be nonnegative integers")
    if any(key not in {"C", "H", "O", "N", "F", "Cl", "Br", "I", "S"} for key in atom_counts):
        raise ValueError("Fedors unsupported element")
    _counts(atom_counts, table)
    counts = dict(atom_counts)
    for atom, special in (("O", "O_alcohol"), ("N", "N_amine")):
        if descriptors[special] > counts.get(atom, 0):
            raise ValueError("special Fedors atoms exceed total element counts")
        counts[atom] = counts.get(atom, 0) - descriptors[special]
    counts.update(descriptors)
    counts = {k: v for k, v in counts.items() if v}
    _counts(counts, table)
    total = _sum(counts, table, "Vc")
    if total is None:
        raise ValueError("missing Fedors increment")
    return _positive((26.6 + total) * 1e-6, "predicted critical volume")


def wilson_jasperson_critical(
    atom_counts: Mapping[str, int],
    table: GroupContributionTable,
    *,
    boiling_temperature_K: float,
    ring_count: int = 0,
    functional_groups: Mapping[str, int] | None = None,
    second_order: bool = True,
) -> dict[str, float | None]:
    """First/second-order Wilson-Jasperson; unknown increments stay unknown."""
    if table.method != "Wilson-Jasperson":
        raise ValueError("Wilson-Jasperson requires its selected coefficient table")
    _counts(atom_counts, table)
    _positive(boiling_temperature_K, "boiling temperature")
    if isinstance(ring_count, bool) or not isinstance(ring_count, int) or ring_count < 0:
        raise ValueError("ring count must be a nonnegative integer")
    if not isinstance(second_order, bool):
        raise ValueError("second_order must be a boolean")
    groups = {"group:" + key: value for key, value in (functional_groups or {}).items()}
    if groups:
        _counts(groups, table)
    sums = {field: _sum(atom_counts, table, field) for field in ("Tc", "Pc")}
    for field in sums:
        correction = _sum(groups, table, field) if groups and second_order else 0.0
        base = sums[field]
        sums[field] = None if base is None or correction is None else base + correction
    denominator = None if sums["Tc"] is None else 0.048271 - 0.019846 * ring_count + sums["Tc"]
    tc = (
        None
        if denominator is None or denominator <= 0
        else boiling_temperature_K / denominator**0.2
    )
    pc = None
    if tc is not None and sums["Pc"] is not None:
        pressure_denominator = (
            exp(-0.00922295 - 0.0290403 * ring_count + 0.041 * sums["Pc"]) - 0.96601
        )
        if pressure_denominator > 0:
            pc = 1e5 * 0.0186233 * tc / pressure_denominator
    return {"Tc": tc, "Pc": pc}


def bondi_group_properties(
    counts: Mapping[str, int], table: GroupContributionTable
) -> dict[str, float | None]:
    """Bondi primary and explicit secondary counts in original coefficient units."""
    if table.method != "Bondi":
        raise ValueError("Bondi requires its selected coefficient table")
    _counts(counts, table)
    volume, area = (_sum(counts, table, key) for key in ("Vw", "Aw"))
    return {
        "V_vdw": None if volume is None else volume * 1e-6,
        "A_vdw": None if area is None else area * 1e5,
        "R": None if volume is None else volume / 15.17,
        "Q": None if area is None else area / 2.5,
    }


@dataclass(frozen=True)
class GroupEstimateReport:
    component_id: str
    method: str
    values: Mapping[str, Any]
    inputs: Mapping[str, Any]
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        if not self.component_id or not self.method:
            raise ValueError("group estimate needs explicit component and method")
        # Nested configuration is detached from the author's mutable objects.
        from chemworld.physchem.activity_thermodynamics import _freeze

        object.__setattr__(
            self, "values", _freeze(json.loads(json.dumps(dict(self.values), allow_nan=False)))
        )
        object.__setattr__(
            self, "inputs", _freeze(json.loads(json.dumps(dict(self.inputs), allow_nan=False)))
        )

    def to_dict(self) -> dict[str, Any]:
        from chemworld.physchem.activity_thermodynamics import _json

        return {
            "component_id": self.component_id,
            "method": self.method,
            "values": _json(self.values),
            "inputs": _json(self.inputs),
            "source": self.source.to_dict(),
            "basis": "group prediction; material accuracy and uncertainty not established",
        }

    def critical_estimate(self, property_id: str) -> CriticalEstimate:
        value = self.values.get(property_id)
        if value is None:
            raise ValueError(f"group estimate {property_id} is unavailable")
        return CriticalEstimate(
            (self.component_id,),
            property_id,
            float(value),
            self.method,
            self.to_dict(),
            self.source.citation,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GroupEstimateReport:
        return cls(
            payload["component_id"],
            payload["method"],
            payload["values"],
            payload["inputs"],
            DataSourceProvenance.from_dict(payload["source"]),
        )


def joback_estimate(
    component_id: str,
    counts: Mapping[str, int],
    table: GroupContributionTable,
    *,
    atom_count: int,
    molecular_weight_g_mol: float,
    boiling_temperature: InputValue | None = None,
) -> GroupEstimateReport:
    _positive(molecular_weight_g_mol, "molecular weight")
    if boiling_temperature is not None and boiling_temperature.quality != "valid":
        raise ValueError("known boiling temperature input must be valid")
    tb = None if boiling_temperature is None else boiling_temperature.to("K").value
    values = joback_group_values(counts, table, atom_count=atom_count, boiling_temperature_K=tb)
    return GroupEstimateReport(
        component_id,
        "Joback",
        values,
        {
            "counts": dict(counts),
            "atom_count": atom_count,
            "molecular_weight_g_mol": molecular_weight_g_mol,
            "boiling_temperature": None
            if boiling_temperature is None
            else boiling_temperature.to_dict(),
            "coefficients": {key: dict(table.coefficients[key]) for key in counts},
        },
        table.source,
    )


def structural_group_estimate(
    component_id: str, table: GroupContributionTable, *, features: Mapping[str, Any]
) -> GroupEstimateReport:
    """Report explicit structural estimates with the original critical consumer."""
    values: dict[str, float | None]
    if table.method == "Fedors":
        values = {"Vc": fedors_critical_volume(table=table, **features)}
    elif table.method == "Wilson-Jasperson":
        values = wilson_jasperson_critical(table=table, **features)
    elif table.method == "Bondi":
        values = bondi_group_properties(table=table, **features)
    else:
        raise ValueError("structural group estimate requires Fedors, Wilson-Jasperson or Bondi")
    return GroupEstimateReport(
        component_id,
        table.method,
        values,
        {"features": dict(features), "table": table.to_dict()},
        table.source,
    )


def joback_property_correlations(
    report: GroupEstimateReport,
    *,
    temperature_range_K: tuple[float, float],
) -> tuple[PropertyCorrelation, ...]:
    """Compile predicted Cp/viscosity into the original correlation catalogue.

    The author supplies the application range; group prediction alone does not
    establish a liquid phase or an absolute caloric reference.
    """
    if report.method != "Joback":
        raise ValueError("Joback curve compilation requires a Joback report")
    curves = []
    common: dict[str, Any] = {
        "input_units": {"temperature": "K"},
        "validity_ranges": {"temperature": temperature_range_K},
        "source_note": report.source.citation,
        "metadata": {
            "group_estimate": report.to_dict(),
            "basis": "predicted; caller-selected application range",
        },
    }
    cp = report.values.get("Cpig_coefficients")
    if cp is not None:
        curves.append(
            PropertyCorrelation(
                report.component_id + "-joback-cpig",
                "cp_polynomial",
                dict(zip("abcd", cp, strict=True)),
                output_unit="J/(mol*K)",
                property_id="ideal_gas_heat_capacity",
                **common,
            )
        )
    mu = report.values.get("viscosity_coefficients")
    if mu is not None:
        mw = report.inputs["molecular_weight_g_mol"]
        curves.append(
            PropertyCorrelation(
                report.component_id + "-joback-viscosity",
                "andrade_viscosity",
                {"A": mw * exp(mu[1]), "B": mu[0]},
                output_unit="Pa*s",
                property_id="liquid_viscosity",
                **common,
            )
        )
    return tuple(curves)
