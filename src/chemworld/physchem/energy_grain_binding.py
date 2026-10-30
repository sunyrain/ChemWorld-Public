"""Compile finite grain declarations into existing species, rates and gas worlds."""

from __future__ import annotations

from math import log
from typing import Any

from chemworld.physchem.energy_grain_specs import EnergyGrainMasterEquationSpec, R, finite
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reaction_network_specs import RateLawSpec, ReactionSpec, SpeciesSpec


def compile_energy_grain_network(spec: EnergyGrainMasterEquationSpec) -> ReactionNetworkSpec:
    """Every grain/sink/bath is one original gas SpeciesSpec and material stock.

    There is no additional well-total SpeciesSpec. Same-formula sinks are explicit
    absorbing kinetic products, not a claim of reversible thermodynamic equilibrium.
    """
    baseline = spec.baseline
    species, reactions = [], []
    thermo: dict[str, Any] = {}
    well_by_grain = {g.species_id: w.well_id for w in spec.wells for g in w.grains}

    def add_species(name: str, formula: str, cp: float, h: float, s: float, metadata: dict) -> None:
        item = SpeciesSpec(name, formula, phase="gas", metadata=metadata)
        species.append(item)
        thermo[name] = {
            "composition": item.composition,
            "units": {"quantity": "mol"},
            "thermo": {
                "model": "constant-cp",
                "T0": baseline.reference_temperature_K,
                "h0": h,
                "s0": s,
                "cp0": cp,
                "T-min": baseline.min_temperature_K,
                "T-max": baseline.max_temperature_K,
            },
        }

    for state in spec.molecular_states:
        add_species(
            state.species_id,
            spec.molecular_formula,
            baseline.cp_J_mol_K,
            baseline.h_ref_J_mol + state.energy_J_mol,
            baseline.s_ref_J_mol_K + R * log(state.statistical_weight),
            {
                "energy_grain_role": "grain" if state.species_id in well_by_grain else "sink",
                "well_id": well_by_grain.get(state.species_id),
                "internal_energy_offset_J_mol": state.energy_J_mol,
                "statistical_weight": state.statistical_weight,
            },
        )
    for bath in spec.bath_gases:
        add_species(
            bath.species_id,
            bath.formula,
            bath.cp_J_mol_K,
            bath.h_ref_J_mol,
            bath.s_ref_J_mol_K,
            {"energy_grain_role": "bath"},
        )

    def pathway(name: str, source: str, target: str, law: RateLawSpec, kind: str) -> None:
        reactions.append(
            ReactionSpec(
                name,
                f"{source} => {target}",
                {source: -1.0, target: 1.0},
                False,
                law,
                metadata={"energy_grain_pathway": kind},
            )
        )

    for well_index, well in enumerate(spec.wells):
        if not well.baths:
            continue
        for i, source in enumerate(well.grains):
            for j, target in enumerate(well.grains):
                if i == j:
                    continue
                name = f"grain-collision:{well_index}:{i}:{j}"
                pathway(
                    name,
                    source.species_id,
                    target.species_id,
                    RateLawSpec(
                        name,
                        "energy_grain_collision",
                        {
                            "model": "energy-grain-collision-v1",
                            "well": well.to_dict(),
                            "source_grain_id": source.species_id,
                            "target_grain_id": target.species_id,
                            "temperature_range_K": [
                                baseline.min_temperature_K,
                                baseline.max_temperature_K,
                            ],
                        },
                    ),
                    "collision",
                )
    states = {s.species_id: s for s in spec.molecular_states}
    for index, channel in enumerate(spec.channels):
        for direction in ("forward", "reverse"):
            if direction == "reverse" and channel.target_species_id not in well_by_grain:
                continue
            source_id, target_id, rate = (
                channel.source_grain_id,
                channel.target_species_id,
                channel.rate_s,
            )
            if direction == "reverse":
                rate *= states[source_id].statistical_weight / states[target_id].statistical_weight
                source_id, target_id = target_id, source_id
            finite(rate, "microcanonical rate", positive=True)
            name = f"grain-channel:{index}:{direction}"
            pathway(
                name,
                source_id,
                target_id,
                RateLawSpec(name, "mass_action", {"k": rate}),
                "isoenergetic-interwell" if channel.target_species_id in well_by_grain else "sink",
            )
    return ReactionNetworkSpec(
        spec.network_id,
        tuple(species),
        tuple(reactions),
        metadata={
            "gas_thermochemistry": thermo,
            "energy_grain_declaration": spec.to_dict(),
            "source_note": spec.source_note,
            "caloric_scope": "baseline excludes declared grained internal modes",
            "kinetic_scope": "finite explicit unimolecular grains and absorbing products",
        },
    )
