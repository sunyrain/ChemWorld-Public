"""Finite-library graph growth using read-only edge fluxes on original core ODEs."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from math import fsum, sqrt
from typing import Any

from chemworld.physchem.mechanism_growth_binding import (
    enumerate_bound_candidates,
    make_growth_network,
    validate_growth_library_identity,
)
from chemworld.physchem.mechanism_growth_specs import (
    CoreEdgeGrowthPolicy,
    GrowthCondition,
    GrowthResourceLimits,
    MechanismGrowthCheckpoint,
    MechanismGrowthLibrary,
    MechanismGrowthResult,
)
from chemworld.physchem.reaction_family import ReactionFamily
from chemworld.physchem.reaction_network import BatchSolverOptions, ReactionNetworkSpec


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _condition_fluxes(
    core: ReactionNetworkSpec,
    edge: ReactionNetworkSpec,
    condition: GrowthCondition,
    solver: BatchSolverOptions,
) -> dict[str, Any]:
    # Only the core enters the existing ODE; edge is an algebraic observer.
    trajectory = core.integrate_batch(
        condition.initial_amounts_mol,
        volume_L=condition.volume_L,
        temperature_K=condition.temperature_K,
        duration_s=condition.sample_times_s[-1],
        evaluation_times_s=condition.sample_times_s,
        solver_options=solver,
    )
    edge_species = sorted(set(edge.species_ids) - set(core.species_ids))
    samples = []
    for index, time_s in enumerate(trajectory.times_s):
        amounts = {
            s: trajectory.amounts_mol[i][index] for i, s in enumerate(trajectory.species_ids)
        }
        core_rates = core.reaction_rates(
            amounts, volume_L=condition.volume_L, temperature_K=condition.temperature_K
        )
        derivatives = core.amount_derivatives_from_rates(core_rates, volume_L=1.0)
        characteristic = sqrt(fsum(v * v for v in derivatives.values()))
        observed = {s: amounts.get(s, 0.0) for s in edge.species_ids}
        rates = edge.reaction_rates(
            observed, volume_L=condition.volume_L, temperature_K=condition.temperature_K
        )
        contributions = {
            s: {
                r.reaction_id: r.stoichiometry.get(s, 0) * rates[r.reaction_id]
                for r in edge.reactions
                if r.stoichiometry.get(s, 0) > 0
            }
            for s in edge_species
        }
        samples.append(
            {
                "time_s": time_s,
                "core_amounts_mol": amounts,
                "core_net_production_mol_L_s": derivatives,
                "characteristic_rate_mol_L_s": characteristic,
                "candidate_rates_mol_L_s": rates,
                "edge_production_mol_L_s": {s: fsum(v.values()) for s, v in contributions.items()},
                "edge_reaction_contributions_mol_L_s": contributions,
            }
        )
    return {
        "condition_id": condition.condition_id,
        "trajectory": trajectory.to_dict(),
        "samples": samples,
    }


def _decisions(pending: dict[str, Any], policy: CoreEdgeGrowthPolicy) -> dict[str, Any]:
    edge = ReactionNetworkSpec.from_dict(pending["edge_network"])
    core_ids = set(pending["core_ids"])
    witnesses: dict[str, dict[str, Any]] = {}
    keep_scores: dict[str, float] = {}
    for result in pending["conditions"]:
        for sample in result["samples"]:
            characteristic = sample["characteristic_rate_mol_L_s"]
            threshold = max(
                policy.promotion_absolute_threshold_mol_L_s,
                policy.promotion_relative_threshold * characteristic,
            )
            keep = max(
                policy.keep_absolute_threshold_mol_L_s,
                policy.keep_relative_threshold * characteristic,
            )
            for species, flux in sample["edge_production_mol_L_s"].items():
                score = flux / threshold
                keep_scores[species] = max(keep_scores.get(species, 0.0), flux / keep)
                if species not in witnesses or score > witnesses[species]["threshold_ratio"]:
                    witnesses[species] = {
                        "condition_id": result["condition_id"],
                        "time_s": sample["time_s"],
                        "flux_mol_L_s": flux,
                        "characteristic_rate_mol_L_s": characteristic,
                        "threshold_mol_L_s": threshold,
                        "threshold_ratio": score,
                        "normalized_flux": flux / characteristic if characteristic else None,
                        "zero_characteristic_rule": "explicit_absolute_threshold",
                        "contributions_mol_L_s": sample["edge_reaction_contributions_mol_L_s"][
                            species
                        ],
                    }
    triggered = {s for s, w in witnesses.items() if w["threshold_ratio"] > 1.0}
    participating = {
        key
        for s in triggered
        for key, value in witnesses[s]["contributions_mol_L_s"].items()
        if value > 0
    }
    promoted = set(triggered)
    for reaction in edge.reactions:
        if reaction.reaction_id in participating:
            promoted.update(
                s for s, n in reaction.stoichiometry.items() if n > 0 and s not in core_ids
            )
    pruned = []
    for reaction in edge.reactions:
        new_products = {s for s, n in reaction.stoichiometry.items() if n > 0 and s not in core_ids}
        if new_products and all(keep_scores.get(s, 0) <= 1 for s in new_products):
            pruned.append(reaction.reaction_id)
    return {
        "witnesses": witnesses,
        "triggered_species": sorted(triggered),
        "promoted_species": sorted(promoted),
        "promotion_reactions": sorted(participating),
        "pruned_edge_reactions": sorted(pruned),
        "keep_scores": keep_scores,
    }


def grow_mechanism(
    library: MechanismGrowthLibrary,
    families: Sequence[ReactionFamily],
    seed_species_ids: Sequence[str],
    conditions: Sequence[GrowthCondition],
    *,
    policy: CoreEdgeGrowthPolicy | None = None,
    resources: GrowthResourceLimits | None = None,
    solver_options: BatchSolverOptions | None = None,
    checkpoint: MechanismGrowthCheckpoint | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> MechanismGrowthResult:
    """Generate, observe and promote; resume only unchanged design with larger budgets.

    Resource limits apply at complete generation/condition/round boundaries.
    Integrator nfev is accounted, not advertised as an interruptible hard limit.
    No core reaction or species is pruned, and no edge inventory is integrated.
    """
    policy = policy or CoreEdgeGrowthPolicy()
    resources = resources or GrowthResourceLimits()
    solver = solver_options or BatchSolverOptions(
        method="DOP853", rtol=1e-10, atol_mol=1e-13, max_step_s=0.025
    )
    # Execute the same fixed declarations that are recorded, even if a progress
    # consumer later mutates its original dictionaries. No second truth is read.
    library = MechanismGrowthLibrary.from_dict(_copy(library.to_dict()))
    validate_growth_library_identity(library, policy)
    families = tuple(
        sorted(
            (ReactionFamily.from_dict(_copy(f.to_dict())) for f in families),
            key=lambda f: f.family_id,
        )
    )
    conditions = tuple(
        sorted(
            (GrowthCondition(**_copy(c.to_dict())) for c in conditions),
            key=lambda c: c.condition_id,
        )
    )
    seeds = sorted(seed_species_ids)
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) - library.species_by_id.keys():
        raise ValueError("initial core requires unique declared library species")
    if not families or len({f.family_id for f in families}) != len(families):
        raise ValueError("growth requires unique explicit reaction families")
    if not conditions or len({c.condition_id for c in conditions}) != len(conditions):
        raise ValueError("growth requires unique nonempty conditions")
    if any(set(c.initial_amounts_mol) - set(seeds) for c in conditions):
        raise ValueError("every initial inventory must belong to the initial core")
    for condition in conditions:
        for binding in library.species:
            # Validate the declared fixed temperature domain before any solve.
            from chemworld.physchem.thermochemistry import species_thermo_from_cantera_yaml

            species_thermo_from_cantera_yaml(
                species_id=binding.species.species_id, **binding.thermo_declaration
            ).evaluate(condition.temperature_K)
    design = _copy(
        {
            "library": library.to_dict(),
            "families": [f.to_dict() for f in families],
            "seeds": seeds,
            "conditions": [c.to_dict() for c in conditions],
            "policy": policy.to_dict(),
            "solver": solver.to_dict(),
        }
    )
    initial = make_growth_network(seeds, (), library, network_id="grown-core")
    if checkpoint is None:
        state: dict[str, Any] = {
            "core_ids": seeds,
            "completed_rounds": 0,
            "condition_solves": 0,
            "nfev": 0,
            "history": [],
            "pending": None,
            "network": initial.to_dict(),
            "stops": [],
            "dormant_edge_keys": [],
            "last_generation_attempt": None,
            "resource_history": [{"limits": resources.to_dict(), "after_condition_solves": 0}],
            "status": "running",
        }
    else:
        saved = checkpoint.to_dict()
        if saved["design"] != design:
            raise ValueError(
                "resuming growth cannot change library, families, conditions or scientific policy"
            )
        state = saved["state"]
        required = {
            "core_ids",
            "completed_rounds",
            "condition_solves",
            "nfev",
            "history",
            "pending",
            "network",
            "stops",
            "dormant_edge_keys",
            "last_generation_attempt",
            "resource_history",
            "status",
        }
        if set(state) != required:
            raise ValueError("invalid growth checkpoint state fields")
        if (
            any(
                type(state[k]) is not int or state[k] < 0
                for k in ("completed_rounds", "condition_solves", "nfev")
            )
            or state["completed_rounds"] != len(state["history"])
            or set(seeds) - set(state["core_ids"])
            or set(state["core_ids"]) - library.species_by_id.keys()
            or len(state["core_ids"]) != len(set(state["core_ids"]))
        ):
            raise ValueError("inconsistent growth checkpoint counters or core identities")
        previous = state["resource_history"][-1]["limits"]
        if any(resources.to_dict()[k] < v for k, v in previous.items()):
            raise ValueError("resumption may only increase resource limits")
        if resources.to_dict() != previous:
            state["resource_history"].append(
                {"limits": resources.to_dict(), "after_condition_solves": state["condition_solves"]}
            )
        if state["status"] not in {"resource_limit", "threshold_satisfied"}:
            raise ValueError("only resource stops or completed results can be resumed unchanged")
        if state["pending"] is not None:
            pending_ids = [r["condition_id"] for r in state["pending"]["conditions"]]
            if state["pending"]["core_ids"] != state["core_ids"] or pending_ids != [
                c.condition_id for c in conditions[: len(pending_ids)]
            ]:
                raise ValueError("inconsistent pending growth condition order")

    def finish(status: str, reasons: list[dict[str, Any]]) -> MechanismGrowthResult:
        state["status"] = status
        if status != "threshold_satisfied":
            state["stops"].append(
                {
                    "status": status,
                    "reasons": reasons,
                    "completed_rounds": state["completed_rounds"],
                    "condition_solves": state["condition_solves"],
                }
            )
        network = ReactionNetworkSpec.from_dict(state["network"])
        network = replace(
            network,
            metadata={
                **network.metadata,
                "growth_complete": status == "threshold_satisfied",
                "growth_status": status,
            },
        )
        return MechanismGrowthResult(
            status, network, tuple(_copy(reasons)), MechanismGrowthCheckpoint(design, _copy(state))
        )

    if state["status"] == "threshold_satisfied":
        return finish("threshold_satisfied", [])
    while True:
        if len(state["core_ids"]) > resources.maximum_core_species:
            return finish("resource_limit", [{"kind": "maximum_core_species"}])
        if state["pending"] is None:
            if state["completed_rounds"] >= resources.maximum_rounds:
                return finish("resource_limit", [{"kind": "maximum_rounds"}])
            generated, discovered, missing = enumerate_bound_candidates(
                state["core_ids"],
                library,
                families,
                policy,
            )
            state["last_generation_attempt"] = generated.to_dict()
            if not generated.complete:
                return finish(
                    "generation_truncated",
                    [{"kind": reason} for reason in generated.truncated_reasons],
                )
            if missing:
                return finish("missing_data", missing)
            core_ids = set(state["core_ids"])
            core_reactions = [
                r
                for r in discovered.values()
                if not (r.stoichiometry.keys() | r.forward_orders.keys() | r.reverse_orders.keys())
                - core_ids
            ]
            previous_keys = set(ReactionNetworkSpec.from_dict(state["network"]).reaction_ids)
            if previous_keys - {r.reaction_id for r in core_reactions}:
                return finish(
                    "missing_data",
                    [
                        {
                            "kind": "previous_core_reaction_missing",
                            "keys": sorted(previous_keys - {r.reaction_id for r in core_reactions}),
                        }
                    ],
                )
            edge_reactions = [r for r in discovered.values() if r not in core_reactions]
            core_network = make_growth_network(
                state["core_ids"], core_reactions, library, network_id="grown-core"
            )
            state["network"] = core_network.to_dict()
            edge_ids = core_ids | {s for r in edge_reactions for s in r.stoichiometry}
            edge_network = make_growth_network(
                sorted(edge_ids), edge_reactions, library, network_id="read-only-edge-observer"
            )
            state["pending"] = {
                "round": state["completed_rounds"] + 1,
                "core_ids": state["core_ids"][:],
                "generation": generated.to_dict(),
                "core_network": core_network.to_dict(),
                "edge_network": edge_network.to_dict(),
                "conditions": [],
            }
        pending = state["pending"]
        if len(pending["edge_network"]["reactions"]) > resources.maximum_edge_candidates:
            return finish(
                "resource_limit",
                [
                    {
                        "kind": "maximum_edge_candidates",
                        "actual": len(pending["edge_network"]["reactions"]),
                    }
                ],
            )
        core_network = ReactionNetworkSpec.from_dict(pending["core_network"])
        edge_network = ReactionNetworkSpec.from_dict(pending["edge_network"])
        for condition in conditions[len(pending["conditions"]) :]:
            if state["condition_solves"] >= resources.maximum_condition_solves:
                return finish("resource_limit", [{"kind": "maximum_condition_solves"}])
            state["condition_solves"] += 1
            try:
                observed = _condition_fluxes(core_network, edge_network, condition, solver)
            except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
                return finish(
                    "integration_failed",
                    [
                        {
                            "kind": type(error).__name__,
                            "condition_id": condition.condition_id,
                            "message": str(error),
                        }
                    ],
                )
            state["nfev"] += observed["trajectory"]["solver_diagnostic"]["nfev"]
            pending["conditions"].append(observed)
            if progress is not None:
                progress(
                    {
                        "stage": "core_trajectory_and_edge_observer",
                        "round": pending["round"],
                        "condition_completed": len(pending["conditions"]),
                        "condition_total": len(conditions),
                        "core_species": len(state["core_ids"]),
                        "condition_solves": state["condition_solves"],
                        "nfev": state["nfev"],
                    }
                )
        decisions = _decisions(pending, policy)
        enlarged = sorted(set(state["core_ids"]) | set(decisions["promoted_species"]))
        if len(enlarged) > resources.maximum_core_species:
            pending["proposed_decisions"] = decisions
            return finish(
                "resource_limit",
                [{"kind": "maximum_core_species", "required_atomic_core_size": len(enlarged)}],
            )
        active_edges = set(edge_network.reaction_ids) - set(decisions["pruned_edge_reactions"])
        decisions["active_edge_reactions"] = sorted(active_edges)
        decisions["reactivated_edge_reactions"] = sorted(
            set(state["dormant_edge_keys"]) & active_edges
        )
        state["dormant_edge_keys"] = sorted(
            (set(state["dormant_edge_keys"]) | set(decisions["pruned_edge_reactions"]))
            - active_edges
            - set(core_network.reaction_ids)
        )
        pending.pop("proposed_decisions", None)
        state["history"].append({**pending, "decisions": decisions})
        state["completed_rounds"] += 1
        state["pending"] = None
        if not decisions["promoted_species"]:
            return finish("threshold_satisfied", [])
        # Promotion only changes the mechanism. The next conditions start from
        # their original inventories; no shadow-edge amounts ever become stock.
        state["core_ids"] = enlarged
        now_internal = [
            r
            for r in (*core_network.reactions, *edge_network.reactions)
            if not (r.stoichiometry.keys() | r.forward_orders.keys() | r.reverse_orders.keys())
            - set(enlarged)
        ]
        state["network"] = make_growth_network(
            enlarged, now_internal, library, network_id="grown-core"
        ).to_dict()
