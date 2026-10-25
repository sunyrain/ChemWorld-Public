"""Full elementary reactants can be regenerated, with unchanged world ownership."""

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.physchem import AdaptiveHybridModel, AdaptiveHybridSpec
from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.stochastic_adaptive_hybrid import MOLECULAR_QUANTUM
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.stochastic_adaptive_world import stochastic_adaptive_world

VOLUME_L = 1e-15
AVOGADRO = 6.02214076e23


def network(equation, orders, coefficient=0.01, *, formulas=None):
    return ReactionNetworkSpec(
        "regenerated-elementary",
        tuple(SpeciesSpec(k, (formulas or {}).get(k, "CH2O")) for k in ("A", "B")),
        (
            ReactionSpec.from_equation(
                reaction_id="reaction",
                equation=equation,
                forward_orders=orders,
                rate_law=RateLawSpec(
                    "reaction",
                    "mass_action",
                    {"k": coefficient * (AVOGADRO * VOLUME_L) ** (sum(orders.values()) - 1)},
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    "equation,orders,changes",
    [
        ("A + B => 2 A", {"A": 1, "B": 1}, [1, -1]),
        ("2 A => A + B", {"A": 2}, [-1, 1]),
        ("2 A + B => 3 A", {"A": 2, "B": 1}, [1, -1]),
        ("A + 2 B => 2 A + B", {"A": 1, "B": 2}, [1, -1]),
    ],
)
def test_regenerated_full_reactants_match_independent_drift_and_jump(equation, orders, changes):
    model = AdaptiveHybridModel(network(equation, orders), AdaptiveHybridSpec(VOLUME_L))
    values = {"A": 20, "B": 30}
    flux = 0.01 * math.prod(values[k] ** n for k, n in orders.items())
    jump = 0.01 * math.prod(math.prod(values[k] - j for j in range(n)) for k, n in orders.items())
    np.testing.assert_allclose(
        model.deterministic_drift([20, 30], ()), flux * np.array(changes), atol=1e-10, rtol=1e-12
    )
    assert model.propensities([20, 30], ("reaction",))[0] == pytest.approx(
        jump, abs=1e-10, rel=1e-12
    )
    consumed = next(i for i, n in enumerate(changes) if n < 0)
    missing = [20.0, 30.0]
    missing[consumed] = 0.5
    assert model.propensities(missing, ("reaction",)) == (0.0,)
    assert model.channels[0].participants == {"A", "B"}


@pytest.mark.parametrize(
    "equation,orders,formulas",
    [
        ("2 A => B", {"A": 1}, {"B": "C2H4O2"}),
        ("3 A + B => 4 A", {"A": 3, "B": 1}, None),
        ("A + B => 2 A", {"A": 0.5, "B": 1}, None),
        ("A + B => 2 A", {"A": 2}, None),
    ],
)
def test_invalid_full_elementary_reactants_remain_rejected(equation, orders, formulas):
    with pytest.raises(ValueError, match="elementary"):
        AdaptiveHybridModel(
            network(equation, orders, formulas=formulas), AdaptiveHybridSpec(VOLUME_L)
        )


def composition(method):
    chemistry = network("A + B => 2 A", {"A": 1, "B": 1}, 0.002)
    spec = AdaptiveHybridSpec(VOLUME_L, method=method, lower_limit=39, upper_limit=40, step_s=0.125)
    return stochastic_adaptive_world(
        chemistry, spec, {"A": 39.25 * MOLECULAR_QUANTUM, "B": 260.75 * MOLECULAR_QUANTUM}, seed=17
    )


def execute(session, entry, action):
    return session.execute(action).info if entry == "native" else session.step(action)[4]


@pytest.mark.parametrize("method", ["RK4", "LSODA"])
@pytest.mark.parametrize("noise", ["sequential", "keyed"])
@pytest.mark.parametrize("entry", ["native", "gym"])
def test_regenerated_world_full_json_and_fresh_process(method, noise, entry, tmp_path):
    cls = ChemWorldRuntime if entry == "native" else ChemWorldEnv
    session = cls(composition=composition(method), observation_noise_mode=noise)
    session.reset(seed=7)
    try:
        for action in (
            {"operation": "wait", "duration_s": 0.375},
            {"operation": "add_component", "component": 0, "amount_mol": 0.5 * MOLECULAR_QUANTUM},
            {"operation": "measure", "instrument": "sde_amount_meter"},
        ):
            assert execute(session, entry, action)["transaction_status"] == "committed"
        payload = json.loads(json.dumps(session.snapshot().to_dict()))
        suffix = {"operation": "wait", "duration_s": 0.625}
        restored = ChemWorldRuntime.from_snapshot(payload)
        try:
            result = restored.execute(suffix)
            assert result.info["transaction_status"] == "committed"
            expected = json.loads(
                json.dumps({"state": restored._state.to_dict(), "info": result.info})
            )
        finally:
            restored.close()
        info = execute(session, entry, suffix)
        assert info["transaction_status"] == "committed"
        assert info["outcome_diagnostics"]["stochastic_work"]["drift_evaluations"] > 0
        assert json.loads(json.dumps(session._state.to_dict())) == expected["state"]
        memory = session._state.process.stochastic_continuation
        assert memory["events_committed"] > 0
        assert memory["low_species"] == []
        assert (
            abs(math.fsum(session._state.species_amounts.values()) / MOLECULAR_QUANTUM - 300.5)
            < 1e-4
        )
        checkpoint, output = tmp_path / "checkpoint.json", tmp_path / "result.json"
        checkpoint.write_text(json.dumps(payload), encoding="utf-8")
        code = (
            "import json,sys;from pathlib import Path;"
            "from chemworld.runtime import ChemWorldRuntime;"
            "e=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')));"
            "r=e.execute({'operation':'wait','duration_s':.625});"
            "assert r.info['transaction_status']=='committed';"
            "Path(sys.argv[2]).write_text(json.dumps({'state':e._state.to_dict(),'info':r.info}),encoding='utf-8');e.close()"
        )
        child = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-c", code, str(checkpoint), str(output)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        assert json.loads(output.read_text(encoding="utf-8")) == expected
    finally:
        session.close()
