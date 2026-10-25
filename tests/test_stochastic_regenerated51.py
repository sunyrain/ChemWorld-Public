"""Full elementary reactants in the original integer strategies and world."""

import copy
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.stochastic_kinetics import (
    AVOGADRO,
    TAU_ALGORITHM,
    EventBudgetError,
    EventChannel,
    advance_stochastic,
    initial_continuation,
)
from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy, _critical
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.stochastic_world import stochastic_world

DIRECT = "gillespie-direct-v1"
NEXT = "gibson-bruck-next-reaction-v1"
ALGORITHMS = (DIRECT, NEXT, TAU_ALGORITHM)
VOLUME = 1e-15
CASES = (
    ("A + B => 2 A", {"A": 1, "B": 1}),
    ("2 A => A + B", {"A": 2}),
    ("2 A + B => 3 A", {"A": 2, "B": 1}),
    ("A + 2 B => 2 A + B", {"A": 1, "B": 2}),
    ("A + B => A + C", {"A": 1, "B": 1}),
)


def network(
    equation=CASES[0][0],
    orders=None,
    *,
    algorithm=DIRECT,
    coefficient=0.002,
    policy=None,
    max_events=1000000,
):
    orders = CASES[0][1] if orders is None else orders
    declaration = {"volume_L": VOLUME, "algorithm": algorithm, "max_events_per_advance": max_events}
    if algorithm == TAU_ALGORITHM:
        declaration["tau_policy"] = (policy or TauLeapPolicy(max_tau_s=0.01)).to_dict()
    return ReactionNetworkSpec(
        "integer-regenerated51",
        tuple(SpeciesSpec(k, "CH2O") for k in ("A", "B", "C")),
        (
            ReactionSpec.from_equation(
                reaction_id="reaction",
                equation=equation,
                forward_orders=orders,
                rate_law=RateLawSpec(
                    "reaction",
                    "mass_action",
                    {"k": coefficient * (AVOGADRO * VOLUME) ** (sum(orders.values()) - 1)},
                ),
            ),
        ),
        stochastic=declaration,
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("equation,orders", CASES)
def test_full_reactants_propensity_and_zero_availability(algorithm, equation, orders):
    model = network(equation, orders, algorithm=algorithm)
    channel = model.stochastic_channels[0]
    counts = {"A": 20, "B": 30, "C": 0}
    independent = 0.002 * math.prod(
        math.prod(counts[k] - j for j in range(n)) for k, n in orders.items()
    )
    assert dict(channel.reactants) == orders
    assert channel.propensity(counts) == pytest.approx(independent, rel=1e-13)
    assert (
        ReactionNetworkSpec.from_dict(json.loads(json.dumps(model.to_dict()))).to_dict()
        == model.to_dict()
    )
    for key, order in orders.items():
        assert channel.propensity({**counts, key: order - 1}) == 0


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("orders", ({"A": 0.5, "B": 1}, {"A": 2}, {"A": 3, "B": 1}))
def test_non_elementary_or_missing_consumed_reactants_rejected(algorithm, orders):
    with pytest.raises(ValueError):
        network(orders=orders, algorithm=algorithm)


@pytest.mark.parametrize("n,expected", ((0, 0), (1, 0), (2, 1), (3, 2), (10, 9)))
def test_partial_regeneration_has_correct_isolated_firing_bound(n, expected):
    channel = EventChannel("partial", (("A", -1), ("B", 1)), (("A", 2),), 1)
    assert channel.firing_limit({"A": n, "B": 0}) == expected


def test_tau_classifies_partial_regeneration_by_full_reactants():
    channel = EventChannel("partial", (("A", -1), ("B", 1)), (("A", 2),), 1)
    assert _critical((channel,), {"A": 10, "B": 0}, [90], TauLeapPolicy()) == [True]
    assert _critical((channel,), {"A": 2, "B": 0}, [2], TauLeapPolicy(critical_threshold=1)) == [
        True
    ]


@pytest.mark.parametrize("seed", (1, 7, 17))
def test_tau_partial_depletion_cannot_consume_last_regenerated_molecule(seed):
    model = network(
        CASES[1][0],
        CASES[1][1],
        algorithm=TAU_ALGORITHM,
        coefficient=1,
        policy=TauLeapPolicy(epsilon=1, max_tau_s=10, critical_threshold=1, ssa_threshold=0.001),
    )
    result = advance_stochastic(
        model, {"A": 2, "B": 0, "C": 0}, initial_continuation(seed, algorithm=TAU_ALGORITHM), 100
    )
    assert result.counts == {"A": 1, "B": 1, "C": 0}
    assert len(result.events) == 1 and not result.batches


def test_tampered_tau_batch_cannot_spend_a_regenerated_reactant():
    model = network(
        CASES[1][0],
        CASES[1][1],
        algorithm=TAU_ALGORITHM,
        coefficient=1,
        policy=TauLeapPolicy(epsilon=1, max_tau_s=1, critical_threshold=1, ssa_threshold=0.001),
    )
    result = advance_stochastic(
        model, {"A": 100, "B": 0, "C": 0}, initial_continuation(17, algorithm=TAU_ALGORITHM), 1e-8
    )
    memory = copy.deepcopy(result.continuation)
    assert memory["pending"]["mode"] == "leap"
    memory["pending"]["firings"] = [
        100
    ]  # Nonnegative net endpoint, but the last firing needs two A.
    before = copy.deepcopy(memory)
    with pytest.raises(ValueError, match="reactant"):
        advance_stochastic(model, result.counts, memory, 0)
    assert memory == before


def catalytic_switch(algorithm):
    declaration = {"volume_L": VOLUME, "algorithm": algorithm}
    return ReactionNetworkSpec(
        "integer-catalyst-switch",
        tuple(SpeciesSpec(k, "H2") for k in ("A", "B", "C", "D")),
        (
            ReactionSpec.from_equation(
                reaction_id="catalysed",
                equation="A + C => B + C",
                forward_orders={"A": 1, "C": 1},
                rate_law=RateLawSpec("catalysed", "mass_action", {"k": 3 * AVOGADRO * VOLUME}),
            ),
            ReactionSpec.from_equation(
                reaction_id="off",
                equation="C => D",
                rate_law=RateLawSpec("off", "mass_action", {"k": 20}),
            ),
            ReactionSpec.from_equation(
                reaction_id="on",
                equation="D => C",
                rate_law=RateLawSpec("on", "mass_action", {"k": 20}),
            ),
        ),
        stochastic=declaration,
    )


@pytest.mark.parametrize("algorithm", (DIRECT, NEXT))
def test_catalyst_dependencies_and_actual_reactivation(algorithm):
    from chemworld.physchem.stochastic_next_reaction import channel_dependencies

    model = catalytic_switch(algorithm)
    assert channel_dependencies(model.stochastic_channels)[1] == (0, 1, 2)
    counts = {"A": 20, "B": 0, "C": 0, "D": 1}
    result = advance_stochastic(model, counts, initial_continuation(17, algorithm=algorithm), 3)
    seen = set()
    for _, identity in result.events:
        channel = next(c for c in model.stochastic_channels if c.reaction_id == identity)
        assert all(counts[k] >= r for k, r in channel.reactants)
        for k, change in channel.changes:
            counts[k] += change
        seen.add(identity)
    assert counts == result.counts and seen == {"catalysed", "off", "on"}
    assert counts["A"] + counts["B"] == 20 and counts["C"] + counts["D"] == 1


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_new_chemistry_budget_failure_keeps_inputs_and_world(algorithm):
    model = network(algorithm=algorithm, max_events=1)
    counts = {"A": 39, "B": 261, "C": 0}
    memory = initial_continuation(17, algorithm=algorithm)
    old = copy.deepcopy((counts, memory))
    with pytest.raises(EventBudgetError):
        advance_stochastic(model, counts, memory, 10)
    assert (counts, memory) == old
    runtime = ChemWorldRuntime(composition=stochastic_world(model, counts, seed=17))
    runtime.reset(seed=7)
    try:
        before = runtime._state.to_dict()
        result = runtime.execute({"operation": "wait", "duration_s": 10})
        assert result.info["operation_outcome"] == "numerical_failure"
        after = runtime._state.to_dict()
        assert after["species_amounts"] == before["species_amounts"]
        assert (
            after["process"]["stochastic_continuation"]
            == before["process"]["stochastic_continuation"]
        )
        assert after["ledger"]["time_s"] == 0
    finally:
        runtime.close()


def composition(algorithm):
    payload = stochastic_world(
        network(algorithm=algorithm), {"A": 39, "B": 261, "C": 0}, seed=17
    ).to_dict()
    payload["physics"]["initial_states"]["reactor"]["process"]["stochastic_continuation"][
        "sampling_rng_state"
    ] = np.random.PCG64(np.random.SeedSequence([17, 3710])).state
    payload["components"].extend(
        [
            {"kind": "material_input", "parameters": {}},
            {
                "kind": "observation",
                "parameters": {"instruments": ["molecule_counter", "molecule_capture"]},
            },
        ]
    )
    payload["task"].update(
        operations=["add_component", "wait", "measure", "terminate"],
        instruments=["molecule_counter", "molecule_capture"],
    )
    return WorldCompositionSpec.from_dict(payload)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("noise", ("sequential", "keyed"))
@pytest.mark.parametrize("entry", ("native", "gym"))
def test_new_chemistry_original_world_interventions_json_and_new_process(
    algorithm, noise, entry, tmp_path
):
    cls = ChemWorldRuntime if entry == "native" else ChemWorldEnv
    session = cls(composition=composition(algorithm), observation_noise_mode=noise)
    session.reset(seed=7)
    try:
        for action in (
            {"operation": "wait", "duration_s": 1},
            {"operation": "add_component", "component": 1, "amount_mol": 10 / AVOGADRO},
            {"operation": "measure", "instrument": "molecule_counter"},
            {"operation": "measure", "instrument": "molecule_capture"},
        ):
            info = session.execute(action).info if entry == "native" else session.step(action)[4]
            assert info["transaction_status"] == "committed"
        checkpoint = json.loads(json.dumps(session.snapshot().to_dict()))
        action = {"operation": "wait", "duration_s": 2}
        restored = ChemWorldRuntime.from_snapshot(checkpoint)
        try:
            info = restored.execute(action).info
            expected = json.loads(json.dumps({"state": restored._state.to_dict(), "info": info}))
        finally:
            restored.close()
        info = session.execute(action).info if entry == "native" else session.step(action)[4]
        assert info["transaction_status"] == "committed"
        assert json.loads(json.dumps({"state": session._state.to_dict(), "info": info})) == expected
        source, target = tmp_path / "checkpoint.json", tmp_path / "result.json"
        source.write_text(json.dumps(checkpoint), encoding="utf-8")
        code = (
            "import json,sys;from pathlib import Path;"
            "from chemworld.runtime import ChemWorldRuntime;"
            "r=ChemWorldRuntime.from_snapshot("
            "json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')));"
            "v=r.execute({'operation':'wait','duration_s':2});"
            "Path(sys.argv[2]).write_text("
            "json.dumps({'state':r._state.to_dict(),'info':v.info}),encoding='utf-8');r.close()"
        )
        child = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-c", code, str(source), str(target)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        assert json.loads(target.read_text(encoding="utf-8")) == expected
    finally:
        session.close()
