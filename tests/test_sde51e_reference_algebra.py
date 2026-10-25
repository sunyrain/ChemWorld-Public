"""Independent finite-step weak polynomial versus the full reaction stoichiometry."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).with_name("reference")))
from sde51e_scalar_formula import scalar_endpoint
from sde51e_statistics import cases, native_model

from chemworld.physchem import ReactionSDE, RI5Draw, ri5_step

ANALYTIC_CASES = [c for c in cases() if c["kind"].startswith("ou") or c["kind"] == "gbm"]


@pytest.mark.parametrize("case", ANALYTIC_CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("hat", [-math.sqrt(3), 0.0, math.sqrt(3)])
@pytest.mark.parametrize("tilde", [-1, 1])
def test_declared_reaction_direction_in_independent_scalar_ri5(case, hat, tilde):
    network, spec = native_model(case)
    model = ReactionSDE(network, spec)
    initial = np.array([case["initial"][k] for k in network.species_ids])
    endpoint = ri5_step(
        model.drift, model.diffusion, 0, initial, case["step_s"], RI5Draw((hat,), (tilde,))
    )
    expected_a = scalar_endpoint(case, hat)
    expected = {"A": expected_a, "B": sum(case["initial"].values()) - expected_a}
    np.testing.assert_allclose(
        endpoint.values, [expected[k] for k in network.species_ids], atol=1e-10, rtol=1e-10
    )
