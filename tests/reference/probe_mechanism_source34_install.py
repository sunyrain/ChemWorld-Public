"""Self-contained installed source import and real thermal-world continuation."""

import json
import sys
from pathlib import Path

from chemworld.physchem import import_mechanism
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.gas_mechanism_world import gas_mechanism_world


def run(directory):
    previous_references = {name for name in ("cantera", "teqp") if name in sys.modules}
    directory = Path(directory)
    directory.mkdir()
    bank = {
        "units": {"energy": "cal", "quantity": "mol"},
        "molecules": [{"name": name, "composition": {"H": 2}, "thermo": {
            "model": "constant-cp", "cp0": 30, "h0": -100, "s0": 20,
            "T-min": 200, "T-max": 2000,
        }} for name in ("A", "B")],
        "kinetics": [{"units": {"time": "min"}},
                     {"equation": "A => B", "rate-constant": {"A": 120, "b": 0, "Ea": 0}}],
    }
    main = {"units": {"energy": "J", "quantity": "kmol"}, "phases": [{
        "name": "gas", "thermo": "ideal-gas", "kinetics": "gas",
        "species": [{"bank.json/molecules": "all"}], "reactions": ["bank.json/kinetics"],
    }]}
    (directory / "bank.json").write_text(json.dumps(bank), encoding="utf-8")
    (directory / "main.json").write_text(json.dumps(main), encoding="utf-8")
    network = import_mechanism(directory / "main.json")
    assert network.reactions[0].rate_law.parameters["A"] == 2
    assert abs(network.imported_gas_thermochemistry["A"].evaluate(700).cp_J_mol_K - 125.52) < 1e-12
    runtime = ChemWorldRuntime(composition=gas_mechanism_world(network, {"A": 1e-4}))
    restored = None
    try:
        runtime.reset(seed=71)
        first = runtime.execute({"operation": "heat", "target_temperature_K": 750,
                                 "duration_s": 1, "stirring_speed_rpm": 600})
        assert first.info["transaction_status"] == "committed", first.info
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for engine in (runtime, restored):
            result = engine.execute({"operation": "wait", "duration_s": 1,
                                     "stirring_speed_rpm": 600})
            assert result.info["transaction_status"] == "committed", result.info
            assert abs(result.info["outcome_diagnostics"]["energy_balance_residual_J"]) < 1e-5
            assert engine._state.ledger.time_s == 2
            assert abs(sum(engine._state.species_amounts.values()) - 1e-4) < 1e-12
        assert runtime._state.to_dict() == restored._state.to_dict()
        assert {name for name in ("cantera", "teqp") if name in sys.modules} == previous_references
        return {"source_documents": 2, "world_advances": 3, "exact_continuation": True}
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


if __name__ == "__main__":
    assert all(name not in sys.modules for name in ("cantera", "teqp"))
    print(json.dumps(run(Path.cwd() / "source34-inputs")))
