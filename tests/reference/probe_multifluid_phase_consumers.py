"""Installed 27B source-catalogue -> two phases -> selected live stream probe.

The caller writes its source catalogue outside Git. This executable imports no
reference package and no repository test helper.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

from chemworld.data.multifluid_catalog import read_multifluid_catalog
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.foundation.state_serialization import _phases
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.runtime.molecular_phase_inventory import (
    MolecularPhaseEquilibriumSpec,
    MolecularPhaseTarget,
    equilibrate_molecular_phase_inventory,
    molecular_phase_stream,
)


def run(catalog_path: Path):
    catalog = read_multifluid_catalog(catalog_path)
    identities = {item["name"]: item["cas"] for item in catalog.components}
    model = catalog.select(
        [identities[key] for key in ("n-Propane", "n-Butane")],
        gas_constant_J_mol_K=8.31446261815324,
        gas_constant_convention="common_constant_override",
    )
    spec = MolecularPhaseEquilibriumSpec(
        model,
        (MolecularDensityBasin("vapor", (1.0, 3000.0), 500.0),
         MolecularDensityBasin("liquid", (7000.0, 22000.0), 12000.0)),
        (MolecularPhaseTarget("gas", "vapor", "gas", selected=True),
         MolecularPhaseTarget("liquid", "liquid", "liquid")),
        model.caloric_model(),
    )
    spec = MolecularPhaseEquilibriumSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    outcomes = []
    for fraction in (0.5, 0.7):
        feed = dict(zip(model.component_ids, (fraction, 1 - fraction), strict=True))
        stock = PhaseLedger({"feed": PhaseRecord("feed", "vessel", "gas", 10.0, feed)})
        result = equilibrate_molecular_phase_inventory(
            stock, spec, vessel_id="vessel", source_phase_ids=("feed",),
            temperature_K=300.0, pressure_Pa=6e5, max_iterations=120,
        )
        assert result.success and result.flash.phase_status == "two_phase", result.to_dict()
        assert result.ledger.total_amounts_mol().keys() == feed.keys()
        assert all(math.isclose(result.ledger.total_amounts_mol()[k], n, rel_tol=5e-13)
                   for k, n in feed.items())
        ledger = _phases(json.loads(json.dumps(result.ledger.to_dict())))
        assert ledger == result.ledger
        streams = {}
        for phase_id in ("gas", "liquid"):
            # Switching selection must consume the current ledger phase,
            # not a cached feed composition or a new flash result.
            selected = PhaseLedger({key: replace(p, selected=key == phase_id)
                                    for key, p in ledger.phases.items()})
            stream = molecular_phase_stream(
                selected, spec, vessel_id="vessel", temperature_K=300.0,
                molar_rate_mol_s=0.02,
            )
            n = math.fsum(selected.phases[phase_id].species_amounts_mol.values())
            assert math.isclose(stream.enthalpy_rate_W,
                                0.02 * result.phase_enthalpies_J[phase_id] / n, rel_tol=2e-10)
            streams[phase_id] = {
                "mole_fractions": stream.state.mole_fractions,
                "density_mol_m3": stream.state.molar_density_mol_m3,
                "enthalpy_rate_W": stream.enthalpy_rate_W,
            }
        assert streams["gas"]["mole_fractions"] != streams["liquid"]["mole_fractions"]
        outcomes.append({"feed_propane_fraction": fraction,
                         "phase_status": result.flash.phase_status,
                         "streams": streams, "passed": True})
    return {"cases": outcomes, "planned": 2, "passed": sum(u["passed"] for u in outcomes)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite a previous installed probe result")
    report = run(args.catalog)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
