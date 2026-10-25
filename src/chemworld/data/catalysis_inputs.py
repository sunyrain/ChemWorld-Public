"""Bind energy-table results into the unmodified native mechanism declaration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from chemworld.physchem.catalysis_descriptors import FreeEnergyBEP, LinearEnergyScaling
from chemworld.physchem.catalysis_network import CatalyticMechanismSpec


def assemble_catalytic_mechanism(
    declaration: Mapping[str, Any],
    fits: Mapping[str, Mapping[str, Any]],
    *,
    species_fits: Mapping[str, str],
    reaction_fits: Mapping[str, str],
    descriptors: Mapping[str, float],
) -> tuple[CatalyticMechanismSpec, dict[str, Any]]:
    """Complete explicit energy holes, then use the original mechanism codec.

    The authored topology, gas data, modes, site assignments, interaction matrix
    and prefactors are not inferred from energy values. The report retains the
    input tables' source references and parameter uncertainty outside runtime.
    """
    payload = deepcopy(dict(declaration))
    if (
        not species_fits
        or not reaction_fits
        or set(fits) != {*species_fits.values(), *reaction_fits.values()}
    ):
        raise ValueError("explicit species/reaction bindings must consume every supplied fit")
    species = {s["species"]["species_id"]: s for s in payload["species"]}
    reactions = {r["reaction_id"]: r for r in payload["reactions"]}
    if set(species_fits) - set(species) or set(reaction_fits) - set(reactions):
        raise ValueError("fit binding names an undeclared species or reaction")
    bindings = []
    frames = {s["reference_frame"] for s in payload["species"]}
    if len(frames) != 1:
        raise ValueError("assembly requires one explicitly shared caloric reference frame")
    reference_frame = next(iter(frames))

    def fitted(key: str, kind: str) -> Mapping[str, Any]:
        result = fits[key]
        if (
            result.get("schema_version") != "chemworld-catalytic-energy-calibration-1"
            or result.get("status") != "fitted"
            or result.get("energy_kind") != kind
            or result.get("reference_frame") != reference_frame
            or "fitted_model" not in result
        ):
            raise ValueError("assembly needs converged energy fits with matching kind/reference")
        return result

    axes_used: set[str] = set()
    for name, fit_id in species_fits.items():
        result = fitted(fit_id, "electronic_energy")
        card = LinearEnergyScaling.from_dict(result["fitted_model"])
        axes = set(card.coefficients)
        axes_used.update(axes)
        if axes - set(descriptors):
            raise ValueError("target catalyst is missing a declared descriptor")
        point = {axis: descriptors[axis] for axis in card.coefficients}
        card.evaluate(point)
        item = species[name]
        adsorbate = item.get("adsorbate")
        if not isinstance(adsorbate, dict) or "electronic_energy" in adsorbate:
            raise ValueError("bound adsorbate must declare modes with an explicit energy hole")
        if "descriptors" in adsorbate:
            raise ValueError("bound descriptor values come only from the explicit target")
        adsorbate.update(electronic_energy=card.to_dict(), descriptors=point)
        bindings.append(
            {
                "species_id": name,
                "field": "adsorbate.electronic_energy",
                "fit_id": fit_id,
                "dataset_id": result["dataset"]["dataset_id"],
            }
        )
    if set(descriptors) != axes_used:
        raise ValueError("target descriptors must exactly cover the fitted species axes")
    for name, fit_id in reaction_fits.items():
        result = fitted(fit_id, "activation_free_energy")
        bep = FreeEnergyBEP.from_dict(result["fitted_model"])
        if "bep" in reactions[name]:
            raise ValueError("bound reaction must declare an explicit BEP hole")
        reactions[name]["bep"] = bep.to_dict()
        bindings.append(
            {
                "reaction_id": name,
                "field": "bep",
                "fit_id": fit_id,
                "dataset_id": result["dataset"]["dataset_id"],
            }
        )
    mechanism = CatalyticMechanismSpec.from_dict(payload)
    return mechanism, {
        "schema_version": "chemworld-catalytic-table-assembly-1",
        "mechanism": mechanism.to_dict(),
        "bindings": bindings,
        "descriptors_J_mol": dict(descriptors),
        "fits": deepcopy(dict(fits)),
        "boundary": (
            "Fitted laws use the original kinetics and thermochemistry. Parameter uncertainty "
            "is an analysis result, not stochastic runtime truth or measured material accuracy."
        ),
    }
