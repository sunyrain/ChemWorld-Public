"""Finite energy-grain declarations; no populations or clocks live in these cards."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from typing import Any

from chemworld.physchem.elements import parse_formula

R = 8.31446261815324


def finite(value: float, label: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))


def identity(value: str) -> None:
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValueError("energy-grain identities must be nonempty without whitespace")
    if any(c in value for c in "+<=>"):
        raise ValueError("energy-grain identities cannot contain equation separators")


@dataclass(frozen=True)
class InternalModeBaseline:
    """Common ideal-gas Cp/H/S excluding the explicitly grained internal modes.

    The author must identify the excluded modes. A full molecular NASA curve is
    not a valid baseline: adding grain energies to it would count those modes twice.
    """

    cp_J_mol_K: float
    h_ref_J_mol: float
    s_ref_J_mol_K: float
    excluded_internal_modes: tuple[str, ...]
    reference_temperature_K: float = 298.15
    min_temperature_K: float = 200.0
    max_temperature_K: float = 3000.0

    def __post_init__(self) -> None:
        modes = tuple(self.excluded_internal_modes)
        object.__setattr__(self, "excluded_internal_modes", modes)
        if not modes or any(not isinstance(x, str) or not x.strip() for x in modes):
            raise ValueError("declare which internal modes the caloric baseline excludes")
        for name in ("cp_J_mol_K", "h_ref_J_mol", "s_ref_J_mol_K"):
            finite(getattr(self, name), name)
        for name in ("reference_temperature_K", "min_temperature_K", "max_temperature_K"):
            finite(getattr(self, name), name, positive=True)
        if self.cp_J_mol_K <= R or not (
            self.min_temperature_K <= self.reference_temperature_K <= self.max_temperature_K
            and self.min_temperature_K < self.max_temperature_K
        ):
            raise ValueError("baseline needs positive Cv and a valid temperature interval")


@dataclass(frozen=True)
class EnergyGrainState:
    """One original molecular species at a declared absolute internal energy.

    statistical_weight is a dimensionless state count (or DOS times grain width),
    on the same normalization across all wells. It is not a population.
    """

    species_id: str
    energy_J_mol: float
    statistical_weight: float = 1.0

    def __post_init__(self) -> None:
        identity(self.species_id)
        finite(self.energy_J_mol, "energy_J_mol")
        finite(self.statistical_weight, "statistical_weight", positive=True)


@dataclass(frozen=True)
class EnergyGrainBathGas:
    species_id: str
    formula: str
    cp_J_mol_K: float
    h_ref_J_mol: float = 0.0
    s_ref_J_mol_K: float = 0.0

    def __post_init__(self) -> None:
        identity(self.species_id)
        parse_formula(self.formula)
        for name in ("cp_J_mol_K", "h_ref_J_mol", "s_ref_J_mol_K"):
            finite(getattr(self, name), name)
        if self.cp_J_mol_K <= R:
            raise ValueError("bath gas requires positive ideal-gas Cv")


@dataclass(frozen=True)
class EnergyTransferBath:
    """Per-well bath collision law, with k in L/(mol s) and alpha in J/mol.

    k(T)=k_ref*(T/Tref)**collision_exponent, alpha(T) likewise. Alpha is
    the exponential decay scale; truncation and DOS weighting mean it is not
    necessarily the realized mean downward energy change.
    """

    bath_species_id: str
    collision_coefficient_L_mol_s: float
    alpha_J_mol: float
    reference_temperature_K: float = 300.0
    collision_exponent: float = 0.5
    alpha_exponent: float = 0.0

    def __post_init__(self) -> None:
        identity(self.bath_species_id)
        for name in ("collision_coefficient_L_mol_s", "alpha_J_mol", "reference_temperature_K"):
            finite(getattr(self, name), name, positive=True)
        for name in ("collision_exponent", "alpha_exponent"):
            finite(getattr(self, name), name)


@dataclass(frozen=True)
class EnergyGrainWellSpec:
    well_id: str
    grains: tuple[EnergyGrainState, ...]
    baths: tuple[EnergyTransferBath, ...]

    def __post_init__(self) -> None:
        identity(self.well_id)
        object.__setattr__(self, "grains", tuple(sorted(self.grains, key=lambda g: g.energy_J_mol)))
        object.__setattr__(
            self, "baths", tuple(sorted(self.baths, key=lambda b: b.bath_species_id))
        )
        if not self.grains or len({g.species_id for g in self.grains}) != len(self.grains):
            raise ValueError("a well needs unique nonempty grain identities")
        if len({g.energy_J_mol for g in self.grains}) != len(self.grains):
            raise ValueError("combine equal-energy states into one statistical weight per well")
        if len({b.bath_species_id for b in self.baths}) != len(self.baths):
            raise ValueError("a bath species can occur only once per well")

    def to_dict(self) -> dict[str, Any]:
        return {
            "well_id": self.well_id,
            "grains": [asdict(g) for g in self.grains],
            "baths": [asdict(b) for b in self.baths],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EnergyGrainWellSpec:
        return cls(
            str(value["well_id"]),
            tuple(EnergyGrainState(**g) for g in value["grains"]),
            tuple(EnergyTransferBath(**b) for b in value["baths"]),
        )


@dataclass(frozen=True)
class MicrocanonicalGrainChannel:
    """Isoenergetic interwell transfer, or irreversible same-formula product sink.

    The forward microcanonical rate is supplied, never estimated. For another
    well's isoenergetic grain the reverse rate follows g_source*k/g_target.
    A sink instantly thermalizes any excess product energy in the original bath.
    """

    channel_id: str
    source_grain_id: str
    target_species_id: str
    rate_s: float

    def __post_init__(self) -> None:
        for value in (self.channel_id, self.source_grain_id, self.target_species_id):
            identity(value)
        finite(self.rate_s, "microcanonical rate_s", positive=True)
        if self.source_grain_id == self.target_species_id:
            raise ValueError("a channel needs distinct source and target states")


@dataclass(frozen=True)
class EnergyGrainMasterEquationSpec:
    network_id: str
    molecular_formula: str
    baseline: InternalModeBaseline
    wells: tuple[EnergyGrainWellSpec, ...]
    bath_gases: tuple[EnergyGrainBathGas, ...]
    channels: tuple[MicrocanonicalGrainChannel, ...] = ()
    sinks: tuple[EnergyGrainState, ...] = ()
    source_note: str = "explicit finite energy-grain declaration"

    def __post_init__(self) -> None:
        identity(self.network_id)
        parse_formula(self.molecular_formula)
        for field in ("wells", "bath_gases", "channels", "sinks"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if not self.wells or len({w.well_id for w in self.wells}) != len(self.wells):
            raise ValueError("declare unique nonempty wells")
        grains = {g.species_id: g for w in self.wells for g in w.grains}
        sinks = {s.species_id: s for s in self.sinks}
        baths = {b.species_id for b in self.bath_gases}
        ids = [g.species_id for w in self.wells for g in w.grains]
        ids += [s.species_id for s in self.sinks] + [b.species_id for b in self.bath_gases]
        if len(set(ids)) != len(ids):
            raise ValueError("grain, sink and bath identities must be globally unique")
        if len({c.channel_id for c in self.channels}) != len(self.channels):
            raise ValueError("microcanonical channel identities must be unique")
        grain_well = {g.species_id: w.well_id for w in self.wells for g in w.grains}
        for well in self.wells:
            if {b.bath_species_id for b in well.baths} - baths:
                raise ValueError("collision law references an undeclared bath gas")
        pairs = set()
        for channel in self.channels:
            if (
                channel.source_grain_id not in grains
                or channel.target_species_id not in grains | sinks
            ):
                raise ValueError("channel references an undeclared grain or sink")
            source, target = (
                grains[channel.source_grain_id],
                (grains | sinks)[channel.target_species_id],
            )
            if channel.target_species_id in grains:
                if grain_well[source.species_id] == grain_well[target.species_id] or not isclose(
                    source.energy_J_mol, target.energy_J_mol, rel_tol=0, abs_tol=1e-10
                ):
                    raise ValueError("interwell microcanonical transfer must be isoenergetic")
                pair = tuple(sorted((source.species_id, target.species_id)))
                if pair in pairs:
                    raise ValueError("declare each reversible grain pair once")
                pairs.add(pair)
            elif target.energy_J_mol > source.energy_J_mol:
                raise ValueError(
                    "an irreversible product sink cannot exceed source internal energy"
                )

    @property
    def molecular_states(self) -> tuple[EnergyGrainState, ...]:
        return tuple(g for w in self.wells for g in w.grains) + self.sinks

    def to_dict(self) -> dict[str, Any]:
        baseline = asdict(self.baseline)
        baseline["excluded_internal_modes"] = list(self.baseline.excluded_internal_modes)
        return {
            "model": "finite-energy-grain-master-equation-v1",
            "network_id": self.network_id,
            "molecular_formula": self.molecular_formula,
            "baseline": baseline,
            "wells": [w.to_dict() for w in self.wells],
            "bath_gases": [asdict(b) for b in self.bath_gases],
            "channels": [asdict(c) for c in self.channels],
            "sinks": [asdict(s) for s in self.sinks],
            "source_note": self.source_note,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EnergyGrainMasterEquationSpec:
        if value.get("model") != "finite-energy-grain-master-equation-v1":
            raise ValueError("unsupported energy-grain declaration")
        return cls(
            str(value["network_id"]),
            str(value["molecular_formula"]),
            InternalModeBaseline(**value["baseline"]),
            tuple(EnergyGrainWellSpec.from_dict(w) for w in value["wells"]),
            tuple(EnergyGrainBathGas(**b) for b in value["bath_gases"]),
            tuple(MicrocanonicalGrainChannel(**c) for c in value["channels"]),
            tuple(EnergyGrainState(**s) for s in value["sinks"]),
            str(value["source_note"]),
        )
