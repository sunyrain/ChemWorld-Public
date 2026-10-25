"""Finite reachable-state CME from raw integer chemistry, with two numerical evaluations."""

import math
from collections import deque

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.stats import poisson


def law(unit):
    keys = tuple(unit["initial"])
    origin = tuple(unit["initial"][k] for k in keys)
    states, indices, queue = [origin], {origin: 0}, deque([origin])
    edges = []
    while queue:
        state = queue.popleft()
        values, row = dict(zip(keys, state, strict=True)), []
        for reaction in unit["reactions"]:
            rate = reaction["coefficient"]
            for key, order in reaction["orders"].items():
                n = values[key]
                rate *= math.prod(range(n - order + 1, n + 1)) if n >= order else 0
            if rate <= 0:
                continue
            target = tuple(values[k] + reaction["changes"].get(k, 0) for k in keys)
            if min(target) < 0:
                raise ValueError("reference reaction lacks whole reactants")
            if target not in indices:
                if len(states) >= 10000:
                    raise ValueError("reference reachable-state budget exceeded")
                indices[target] = len(states)
                states.append(target)
                queue.append(target)
            row.append((indices[target], rate))
        edges.append(row)
    q = np.zeros((len(states), len(states)))
    for index, row in enumerate(edges):
        for target, rate in row:
            q[index, target] += rate
        q[index, index] -= math.fsum(rate for _, rate in row)
    dense = expm(q * unit["duration_s"])[0]
    origin_vector = np.zeros(len(states))
    origin_vector[0] = 1
    action = expm_multiply(csr_matrix(q.T * unit["duration_s"]), origin_vector)
    maximum_rate = float(max(-q.diagonal()))
    probability = np.zeros(len(states))
    probability[0] = 1
    terms, tail = 0, 0.0
    if maximum_rate:
        transition = np.eye(len(states)) + q / maximum_rate
        vector = probability.copy()
        mu = maximum_rate * unit["duration_s"]
        weight = math.exp(-mu)
        if weight == 0:
            raise ArithmeticError("reference uniformization origin weight underflow")
        probability *= weight
        while True:
            tail = float(poisson.sf(terms, mu))
            if tail <= 1e-14:
                break
            terms += 1
            if terms >= 10000:
                raise ValueError("reference Poisson series budget exceeded")
            vector = vector @ transition
            weight *= mu / terms
            probability += weight * vector
    off_diagonal = q.copy()
    np.fill_diagonal(off_diagonal, 0)
    diagnostics = {
        "state_count": len(states),
        "maximum_generator_row_residual": float(max(abs(q.sum(axis=1)))),
        "minimum_offdiagonal": float(np.min(off_diagonal)),
        "minimum_probability": float(min(probability)),
        "probability_mass_residual": abs(float(math.fsum(probability)) - 1),
        "uniformization_terms": terms,
        "uniformization_tail": tail,
        "maximum_expm_uniformization_difference": float(max(abs(dense - probability))),
        "maximum_exponential_action_uniformization_difference": float(
            max(abs(action - probability))
        ),
        "dense_expm_crosscheck_passed": bool(max(abs(dense - probability)) <= 1e-11),
        "maximum_exit_rate": maximum_rate,
    }
    passed = (
        diagnostics["maximum_generator_row_residual"] <= 1e-11
        and diagnostics["minimum_offdiagonal"] >= 0
        and diagnostics["minimum_probability"] >= 0
        and diagnostics["probability_mass_residual"] <= 1e-11
        and diagnostics["uniformization_tail"] <= 1e-14
        and diagnostics["maximum_exponential_action_uniformization_difference"] <= 1e-11
    )
    return {
        "reference_evaluator": "uniformization-and-exponential-action-v2",
        "keys": keys,
        "support": states,
        "probability": probability.tolist(),
        "expm_probability": dense.tolist(),
        "exponential_action_probability": action.tolist(),
        "generator": q.tolist(),
        "time_basis": "declared_analytic_horizon",
        "duration_s": unit["duration_s"],
        "diagnostics": diagnostics,
        "passed": bool(passed),
    }
