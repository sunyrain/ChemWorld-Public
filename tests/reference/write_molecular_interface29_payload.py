"""Serialize the fixed 29 consumer worlds using the already delivered 28 models."""

import argparse
import json
from math import sqrt
from pathlib import Path

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.physchem.molecular_interface import SquareGradientInfluence
from chemworld.physchem.molecular_multiphase import (
    MolecularMultiphaseSeed,
    MolecularMultiphaseSpec,
    solve_molecular_multiphase,
)
from chemworld.runtime.molecular_multiphase_inventory import MolecularMultiphaseInventorySpec
from chemworld.runtime.molecular_multiphase_thermal import MolecularMultiphaseThermalSpec
from chemworld.runtime.molecular_phase_inventory import (
    MolecularPhaseEquilibriumSpec,
    MolecularPhaseTarget,
)
from chemworld.world.molecular_multiphase_world import build_molecular_multiphase_world
from reference.molecular28_caloric_cases import caloric_spec


def unit(boundary):
    old = caloric_spec("PR")
    core = MolecularMultiphaseSpec(
        old.model,
        old.basins,
        (280.0, 330.0),
        (4e5, 8e5),
        phase_kind_by_basin={"vapor": "gas", "liquid": "liquid"},
    )
    feed = {"propane": 0.006, "butane": 0.004}
    anchor = solve_molecular_multiphase(core, feed, inputs={"T": 300.0, "P": 6e5})
    if not anchor.success:
        raise ValueError("initial consumer closure failed: " + anchor.failure_reason)
    inventory = MolecularMultiphaseInventorySpec(
        core,
        (
            MolecularPhaseTarget("vapor", "vapor", "gas", True),
            MolecularPhaseTarget("liquid", "liquid", "liquid"),
        ),
    )
    spec = MolecularMultiphaseThermalSpec(
        inventory,
        boundary,
        6e5 if boundary == "P" else anchor.volume_m3,
        285.0,
        325.0,
    )
    composition = build_molecular_multiphase_world(
        spec,
        feed,
        energy_J=anchor.enthalpy_J if boundary == "P" else anchor.internal_energy_J,
        seed=MolecularMultiphaseSeed(300.0, 6e5, anchor.phases),
        formulas={"propane": "C3H8", "butane": "C4H10"},
        device=ThermalDeviceSpec(
            conductance_W_K=1.0,
            maximum_power_W=0.02,
            maximum_cooling_power_W=0.02,
            setpoint_K=320.0,
            minimum_setpoint_K=280.0,
            maximum_setpoint_K=330.0,
        ),
        max_volume_L=0.1,
        max_pressure_Pa=8e5,
    )
    cross = 0.5 * sqrt(1e-20 * 1.5e-20)
    influence = SquareGradientInfluence(
        old.model.component_ids,
        ((1e-20, cross), (cross, 1.5e-20)),
        (280.0, 330.0),
        "explicit constant influence parameters; development consumer fixture",
    )
    return {
        "id": "C1-P" if boundary == "P" else "C2-V",
        "composition": composition.to_dict(),
        "inventory": inventory.to_dict(),
        "influence": influence.to_dict(),
    }


def inventory_unit(family, beta):
    from reference.molecular29_cases import initial_tanh, native_pair

    model, endpoints, influence, pair = native_pair(
        family,
        300.0 if family == "pure" else 320.0,
        None if family == "pure" else 6e5,
        beta=beta,
    )
    phases, basins, targets = {}, [], []
    for key, kind, endpoint in zip(("left", "right"), ("gas", "liquid"), endpoints, strict=True):
        density = endpoint.molar_density_mol_m3
        phases[key] = PhaseRecord(
            key,
            "vessel",
            kind,
            0.01 / density * 1000,
            {
                k: 0.01 * x
                for k, x in zip(model.component_ids, endpoint.mole_fractions, strict=True)
            },
            metadata={"carrier_representation": "species_inventory"},
        )
        basins.append(MolecularDensityBasin(key, (0.5 * density, 1.5 * density), density))
        targets.append(MolecularPhaseTarget(key, key, kind))
    ledger = PhaseLedger(phases)
    state = WorldState(
        ledger.total_amounts_mol(),
        sum(p.volume_L for p in phases.values()),
        pair["T"],
        pair["P"],
        "liquid",
        "vessel",
        phases=ledger,
    )
    declaration = MolecularPhaseEquilibriumSpec(model, tuple(basins), tuple(targets))
    options = {
        "area_m2": 2e-4,
        "max_iterations": 500,
        "residual_tolerance": 1e-7,
        "first_integral_tolerance": 0.05,
    }
    if family == "pure":
        lo, hi = (s.molar_density_mol_m3 for s in endpoints)
        options["interior_density_grid"] = [
            lo + x * (hi - lo) for x in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        ]
        identifier = "C3-pure"
    else:
        options.update(
            mesh={"length_m": 6e-9, "cells": 1024},
            component_density_bounds=[
                (0.01 * min(a, b), 1.5 * max(a, b))
                for a, b in zip(pair["left"], pair["right"], strict=True)
            ],
            initial_profile=initial_tanh(pair["left"], pair["right"], 1025).tolist(),
        )
        identifier = "C4-ternary-PD" if beta == 0.5 else "C5-ternary-rank1"
    return {
        "id": identifier,
        "state": state.to_dict(),
        "inventory": declaration.to_dict(),
        "influence": influence.to_dict(),
        "options": options,
        "source_endpoints": pair,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("consumer payload already exists")
    payload = {"mode": "development", "units": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.inventory_only:
        for index, (family, beta) in enumerate(
            (("pure", 0.5), ("ternary", 0.5), ("ternary", 0)), 1
        ):
            print(f"consumer inventory {index}/3 family={family} beta={beta}", flush=True)
            payload["units"].append(inventory_unit(family, beta))
            args.output.write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        return 0
    for index, boundary in enumerate(("P", "V"), 1):
        print(f"consumer world {index}/2 boundary={boundary}", flush=True)
        payload["units"].append(unit(boundary))
        args.output.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
