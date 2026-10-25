"""Fixed nine-unit local-source qualification against actual Cantera 3.2."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from itertools import product
from pathlib import Path

import cantera as ct
import yaml

from chemworld.physchem.mechanism_import import _MechanismLoader, import_mechanism
from chemworld.physchem.reaction_network import ReactionNetworkSpec


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2) if path.suffix == ".json"
                    else yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def fixtures(directory: Path):
    data = Path(ct.get_data_directories()[-1])
    inputs = []
    for filename, phase_name in (("h2o2.yaml", "ohmech"), ("gri30.yaml", "gri30")):
        source = yaml.load((data / filename).read_text(encoding="utf-8"), Loader=_MechanismLoader)
        phase = next(p for p in source["phases"] if p["name"] == phase_name)
        for layout in ("local", "yaml", "json"):
            target = directory / (Path(filename).stem + "-" + layout)
            target.mkdir()
            middle = len(source["reactions"]) // 2
            bank = {"units": source.get("units", {}), "molecules": source["species"],
                    "first": source["reactions"][:middle], "second": source["reactions"][middle:]}
            root = {"phases": [copy.deepcopy(phase)]}
            if layout == "local":
                root.update(bank)
                prefix = ""
            else:
                filename_bank = "bank." + layout
                write(target / filename_bank, bank)
                prefix = filename_bank + "/"
            root["phases"][0].update(species=[{prefix + "molecules": phase["species"]}],
                                     reactions=[prefix + "first", prefix + "second"])
            path = target / "main.yaml"
            write(path, root)
            inputs.append((target.name, path, phase_name))

    target = directory / "gri30-declared-hydrogen"
    target.mkdir()
    original = yaml.load((data / "gri30.yaml").read_text(encoding="utf-8"), Loader=_MechanismLoader)
    write(target / "bank.yaml", original)
    root = {"phases": [{
        "name": "selected", "thermo": "ideal-gas", "kinetics": "gas",
        "species": [{"bank.yaml/species": ["H2", "H", "O", "O2", "OH", "H2O",
                                            "HO2", "H2O2", "AR", "N2"]}],
        "reactions": [{"bank.yaml/reactions": "declared-species"}],
        "skip-undeclared-third-bodies": True,
    }]}
    write(target / "main.yaml", root)
    inputs.append((target.name, target / "main.yaml", "selected"))

    for reactive in (True, False):
        target = directory / ("synthetic-units" if reactive else "synthetic-none")
        target.mkdir()
        bank = {
            "units": {"energy": "cal", "quantity": "mol"},
            "species": [{"name": name, "composition": {"H": 2}, "thermo": {
                "model": "constant-cp", "cp0": 30, "h0": -100, "s0": 20,
                "T-min": 200, "T-max": 2000,
            }} for name in ("A", "B")],
            "reactions": [{"units": {"time": "min"}},
                          {"equation": "A => B", "rate-constant": {"A": 120, "b": 0, "Ea": 3}}],
        }
        root = {"units": {"energy": "J", "quantity": "kmol"}, "phases": [{
            "name": "gas", "thermo": "ideal-gas", "kinetics": "gas",
            "species": [{"bank.yaml/species": "all"}],
            "reactions": ["bank.yaml/reactions"] if reactive else "none",
        }]}
        write(target / "bank.yaml", bank)
        write(target / "main.yaml", root)
        inputs.append((target.name, target / "main.yaml", "gas"))
    return inputs


def run(directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    results = []
    for number, (name, path, phase) in enumerate(fixtures(directory), 1):
        checks = []
        unit = {"name": name, "input": str(path), "checks": checks}

        def numerical(label, actual, expected, rtol, atol, checks=checks):
            checks.append({"label": label, "actual": float(actual), "expected": float(expected),
                           "rtol": rtol, "atol": atol,
                           "passed": math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol)})

        def functional(label, passed, checks=checks):
            checks.append({"label": label, "passed": bool(passed)})

        try:
            network = import_mechanism(path, phase_name=phase)
            gas = ct.Solution(str(path), phase)
            unit.update(species=len(network.species), reactions=len(network.reactions))
            functional("species-order", list(network.species_ids) == gas.species_names)
            # Compare equations structurally, independently of source spelling.
            functional("reaction-count", len(network.reactions) == gas.n_reactions)
            for i, reaction in enumerate(network.reactions):
                reference = gas.reaction(i)
                stoich = {k: reference.products.get(k, 0) - reference.reactants.get(k, 0)
                          for k in reference.products.keys() | reference.reactants.keys()}
                functional(f"reaction-order-{i}", reaction.stoichiometry == {
                    k: v for k, v in stoich.items() if v
                } and reaction.reversible == reference.reversible)
            functional("conservation", network.check_conservation())
            restored = ReactionNetworkSpec.from_dict(json.loads(json.dumps(network.to_dict())))
            functional("json-replay", restored.to_dict() == network.to_dict())
            for t, pressure in product((600, 1000, 1500), (.2, 3)):
                gas.TPX = t, pressure * ct.one_atm, [1] * gas.n_species
                rates = restored.reaction_rates(dict(zip(gas.species_names, gas.concentrations,
                                                        strict=True)),
                                                volume_L=1, temperature_K=t)
                for reaction, expected in zip(network.reactions, gas.net_rates_of_progress,
                                               strict=True):
                    numerical(f"net/{t}/{pressure}/{reaction.reaction_id}",
                              rates[reaction.reaction_id], expected, 2e-8, 1e-18)
                for species in gas.species():
                    state = restored.imported_gas_thermochemistry[species.name].evaluate(t)
                    h, s = species.thermo.h(t) / 1000, species.thermo.s(t) / 1000
                    for field, expected in (("cp_J_mol_K", species.thermo.cp(t) / 1000),
                                            ("enthalpy_J_mol", h), ("entropy_J_mol_K", s),
                                            ("gibbs_J_mol", h - t*s)):
                        numerical(f"{field}/{t}/{pressure}/{species.name}",
                                  getattr(state, field), expected, 2e-10, 2e-7)
            unit["source_resolution"] = network.metadata["source_resolution"]
        except Exception as error:
            unit["error"] = repr(error)
        unit["passed"] = "error" not in unit and all(c["passed"] for c in checks)
        results.append(unit)
        elapsed = time.monotonic() - start
        print(f"units={number}/9 passed={sum(u['passed'] for u in results)} "
              f"elapsed_s={elapsed:.2f} units_per_s={number/elapsed:.2f} "
              f"eta_s={(9-number)*elapsed/number:.2f}", flush=True)
    output = {"mode": "development", "reference": f"Cantera {ct.__version__}",
              "planned_units": 9, "executed_units": len(results),
              "passed_units": sum(u["passed"] for u in results),
              "executed_checks": sum(len(u["checks"]) for u in results),
              "passed_checks": sum(c["passed"] for u in results for c in u["checks"]),
              "elapsed_s": time.monotonic() - start, "units": results}
    write(directory / "RESULTS.json", output)
    summary = {k: v for k, v in output.items() if k != "units"}
    summary["raw_results"] = str(directory / "RESULTS.json")
    summary["units"] = [{k: v for k, v in unit.items() if k not in {"checks", "source_resolution"}}
                        | {"executed_checks": len(unit["checks"]),
                           "passed_checks": sum(c["passed"] for c in unit["checks"]),
                           "failures": [c for c in unit["checks"] if not c["passed"]]}
                        for unit in results]
    write(directory / "SUMMARY.json", summary)
    return output["passed_units"] == 9


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
