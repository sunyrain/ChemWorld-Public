"""Run the fixed 34A development block against optional Cantera 3.2.

No reference runtime is imported by production. Result attempts are exclusive
files; previous failures cannot be overwritten by a later run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path

import cantera as ct
import yaml
from mechanism34_cases import THERMOS, payload, rate_cases

from chemworld.physchem.mechanism_import import convert_mechanism, import_mechanism
from chemworld.physchem.mechanism_rates import mechanism_forward_rate_constant
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.thermochemistry import species_thermo_from_cantera_yaml


def run():
    records = []
    start = time.monotonic()

    def record(block, unit, actual, expected, rtol, atol):
        records.append(
            {
                "block": block,
                "unit": unit,
                "actual": actual,
                "expected": expected,
                "passed": math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
            }
        )

    for model, t in product(THERMOS, (300, 700, 1000, 1500, 2000)):
        try:
            gas = ct.Solution(yaml=yaml.safe_dump(payload(thermo=model)), name="synthetic")
            gas.TP = t, ct.one_atm
            source = payload(thermo=model)["species"][0]
            state = species_thermo_from_cantera_yaml(
                species_id="A", thermo=source["thermo"]
            ).evaluate(t)
            for name, reference in (
                ("cp_J_mol_K", gas.standard_cp_R[0] * ct.gas_constant / 1000),
                ("enthalpy_J_mol", gas.standard_enthalpies_RT[0] * ct.gas_constant * t / 1000),
                ("entropy_J_mol_K", gas.standard_entropies_R[0] * ct.gas_constant / 1000),
                ("gibbs_J_mol", gas.standard_gibbs_RT[0] * ct.gas_constant * t / 1000),
            ):
                record("thermo", [model, t, name], getattr(state, name), reference, 2e-10, 2e-7)
        except Exception as error:
            records.append(
                {"block": "thermo", "unit": [model, t], "passed": False, "error": repr(error)}
            )
    print(
        f"thermo complete: {len(records)} comparisons; elapsed={time.monotonic() - start:.2f}s",
        flush=True,
    )
    for (name, node), t, p in product(rate_cases().items(), (300, 700, 1500), (0.01, 1, 100)):
        try:
            document = payload(node)
            gas = ct.Solution(yaml=yaml.safe_dump(document), name="synthetic")
            gas.TPX = t, p * ct.one_atm, [0.2, 0.3, 0.5]
            network = convert_mechanism(document)
            concentrations = dict(zip(gas.species_names, gas.concentrations, strict=True))
            reaction = network.reactions[0]
            actual = mechanism_forward_rate_constant(
                reaction, concentrations, t, species_thermo=network.imported_gas_thermochemistry
            )
            expected = gas.forward_rate_constants[0]
            if name == "third-body":
                expected *= gas.third_body_concentrations[0]
            record("rates", [name, t, p, "forward_k"], actual, expected, 2e-8, 1e-18)
            native_net = network.reaction_rates(concentrations, volume_L=1, temperature_K=t)[
                reaction.reaction_id
            ]
            record(
                "rates",
                [name, t, p, "net_rate"],
                native_net,
                gas.net_rates_of_progress[0],
                2e-8,
                1e-18,
            )
        except Exception as error:
            records.append(
                {"block": "rates", "unit": [name, t, p], "passed": False, "error": repr(error)}
            )
    print(
        f"rates complete: {len(records)} cumulative comparisons; "
        f"elapsed={time.monotonic() - start:.2f}s",
        flush=True,
    )
    data = Path(ct.get_data_directories()[-1])
    imports = []
    for file, phase in (("h2o2.yaml", "ohmech"), ("gri30.yaml", "gri30")):
        try:
            gas = ct.Solution(str(data / file), phase)
            network = import_mechanism(data / file, phase_name=phase)
            restored = ReactionNetworkSpec.from_dict(json.loads(json.dumps(network.to_dict())))
            imports.append(
                {
                    "file": file,
                    "species": len(network.species),
                    "reactions": len(network.reactions),
                    "expected_species": gas.n_species,
                    "expected_reactions": gas.n_reactions,
                    "exact_roundtrip": network.to_dict() == restored.to_dict(),
                    "passed": len(network.species) == gas.n_species
                    and len(network.reactions) == gas.n_reactions
                    and network.to_dict() == restored.to_dict(),
                }
            )
            for t, p in product((700, 1000, 1500), (0.1, 1, 10)):
                gas.TPX = t, p * ct.one_atm, [1] * gas.n_species
                concentrations = dict(zip(gas.species_names, gas.concentrations, strict=True))
                rates = restored.reaction_rates(concentrations, volume_L=1, temperature_K=t)
                for index, (reaction, reference) in enumerate(
                    zip(network.reactions, gas.net_rates_of_progress, strict=True)
                ):
                    record(
                        "whole_mechanisms",
                        [file, t, p, index, reaction.equation],
                        rates[reaction.reaction_id],
                        float(reference),
                        2e-8,
                        1e-18,
                    )
        except Exception as error:
            imports.append({"file": file, "passed": False, "error": repr(error)})
        print(
            f"{file} complete: {len(records)} cumulative comparisons; "
            f"elapsed={time.monotonic() - start:.2f}s",
            flush=True,
        )
    return {
        "mode": "development",
        "reference": f"Cantera {ct.__version__}",
        "expected_thermo_states": 25,
        "expected_rate_states": 117,
        "expected_whole_mechanism_states": 18,
        "expected_comparisons": 3520,
        "completed_comparisons": len(records),
        "passed_comparisons": sum(r["passed"] for r in records),
        "imports": imports,
        "records": records,
        "elapsed_seconds": time.monotonic() - start,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print({key: value for key, value in result.items() if key != "records"})
