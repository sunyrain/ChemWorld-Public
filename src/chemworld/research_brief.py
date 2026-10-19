"""Opt-in public commissions and local prior records for Work II research tasks.

This interface changes participant information, never chemistry or action permissions.
The evaluator constructs a scoped record; no world-state reader generates advice here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from chemworld.foundation import equipment_settings
from chemworld.world.scoring import PARTITION_S0_EXTRACTION_EFFICIENCY_V3

VERSION = "chemworld-research-brief-0.1"
TASKS = {
    "RX": "reaction-mechanism-explanation",
    "EQ": "equilibrium-characterization",
    "BC": "low-budget-characterization",
    "PA": "partition-discovery",
    "FL": "flow-reaction-optimization",
}
MISSIONS = {
    "RX": "Investigate target formation, competing conversion, and thermal/time history. "
    "Explain what your experiments support and predict new interventions.",
    "EQ": "Characterize the bounded aqueous acid-dissociation and precipitation response. "
    "Report identifiable effective relationships, uncertainty, and equivalent explanations.",
    "BC": "Use the limited shared experimental budget to diagnose reaction behavior, "
    "predict new conditions, and state what remains unresolved.",
    "PA": "Establish a predictive account of target material in both liquid phases "
    "across media and phase ratios, including sampling and separation losses.",
    "FL": "Identify a configuration-dependent feasible operating window from outlet "
    "reaction measurements and temperature/pressure readouts; recommend a supported point.",
}
MEASUREMENT_NOTES = {
    "RX": "Temperature is the actual fluid sensor reading; heat target is a boundary control. "
    "Reaction yield/conversion and byproduct signals require paid assays.",
    "BC": "All batches, instruments, samples and materials share one budget. "
    "Stopping early alone is not evidence of an informative diagnosis.",
    "EQ": "pH_normalized = pH/14. Acid dissociation is a fraction, precipitation_signal "
    "is a bounded normalized proxy. This is a weak-acid/precipitation slice, not general "
    "aqueous chemistry. equilibrium_confidence is an environment diagnostic, not your confidence.",
    "PA": "product_in_organic and product_in_aqueous are fractions of the initial target "
    "inventory, not concentrations or fractions of the current selected phase. Measure both "
    "before phase removal. Sampling consumes inventory; discarded phases are not free to recover.",
    "FL": "Configured volume V=Q*tau and tube length change with the flow/residence settings. "
    "The temperature target is a wall boundary, not the fluid temperature. run_flow requires "
    "fresh configuration. Nominal processed volume is not delivered product mass.",
}


def normalize_research_brief(
    payload: Mapping[str, Any] | None, *, task_id: str | None, scoring_contract_id: str
) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "card",
        "prior_record",
    }:
        raise ValueError("research_brief requires schema_version, card, prior_record only")
    card = payload["card"]
    if payload["schema_version"] != VERSION or card not in TASKS or TASKS[card] != task_id:
        raise ValueError("research_brief version/card does not match the runtime task")
    if card == "PA" and scoring_contract_id != PARTITION_S0_EXTRACTION_EFFICIENCY_V3:
        raise ValueError("PA research requires the fixed-initial-target observation contract")
    prior = payload["prior_record"]
    if prior is not None:
        if not isinstance(prior, Mapping) or set(prior) != {
            "scope_actions",
            "metric",
            "estimate",
            "half_width",
        }:
            raise ValueError(
                "prior_record must contain one scoped numerical claim, without arm labels"
            )
        if not isinstance(prior["scope_actions"], list) or not prior["scope_actions"]:
            raise ValueError("prior scope_actions must be a nonempty public action list")
        if not all(
            isinstance(a, dict) and isinstance(a.get("operation"), str)
            for a in prior["scope_actions"]
        ):
            raise ValueError("prior scope must use public action dictionaries")
        if prior["metric"] not in {
            "yield",
            "conversion",
            "byproduct_signal",
            "pH_normalized",
            "acid_dissociation_fraction",
            "precipitation_signal",
            "product_in_organic",
            "product_in_aqueous",
            "flow_conversion",
        }:
            raise ValueError("prior metric is outside the declared measured channels")
        for key in ("estimate", "half_width"):
            value = prior[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError("prior estimates and widths must be finite numbers")
        if not 0 <= prior["estimate"] <= 1 or not 0 < prior["half_width"] <= 1:
            raise ValueError("prior estimate/width outside normalized measurement range")
    return deepcopy(dict(payload))


def public_research_brief(config: dict[str, Any]) -> dict[str, Any]:
    card = config["card"]
    return {
        **deepcopy(config),
        "commission": MISSIONS[card],
        "measurement_contract": MEASUREMENT_NOTES[card],
        "prior_status": "A local archival estimate, if present, can be checked and revised. "
        "It is not a complete mechanism and does not constrain your model representation.",
        "deliverable": "Use free text, equations, diagrams or optional code. Distinguish "
        "supported relationships, competing explanations, uncertainty and testable predictions.",
        "score_status": "The native scalar score is a diagnostic, not the research objective.",
        "sensor_contract": "Ideal temperature, pressure, volume and clock telemetry is public. "
        "Chemical composition is available only through purchased instruments.",
    }


def research_operational_state(env: Any, info: dict[str, Any]) -> dict[str, Any]:
    """An explicit allowlist: no species, rate parameters or solver truth."""
    state = env._state
    phases = {} if state.phases is None else state.phases.phases
    closed = bool(info.get("experiment_ended"))
    fresh = (
        info.get("operation_type") == "measure" and info.get("transaction_status") == "committed"
    )
    flow = equipment_settings(state.equipment, "flow_reactor")
    public_flow = {
        k: float(flow[k])
        for k in (
            "flow_rate_mL_min",
            "residence_time_s",
            "reactor_volume_L",
            "inlet_temperature_K",
            "boundary_temperature_K",
            "outlet_temperature_K",
            "outlet_pressure_Pa",
            "pressure_drop_Pa",
        )
        if k in flow
    }
    geometry = flow.get("geometry", {})
    public_geometry = {
        k: float(geometry[k]) for k in ("length_m", "inner_diameter_m") if k in geometry
    }
    return {
        "schema_version": "chemworld-research-operational-state-0.1",
        "temperature_K": float(state.temperature_K),
        "pressure_Pa": float(state.pressure_Pa),
        "active_volume_L": float(state.volume_L),
        "process_time_s": float(state.ledger.time_s),
        "state_scope": "next_batch_initial_state"
        if closed and info.get("next_experiment_ready")
        else "current_batch",
        "measurement_scope": "just_closed_batch"
        if closed and fresh
        else "current_batch"
        if fresh
        else "no_new_measurement",
        "phases": {
            k: {
                "volume_L": float(v.volume_L),
                "selected": bool(v.selected),
                "settled": bool(v.settled),
            }
            for k, v in phases.items()
        },
        "flow": public_flow,
        "geometry": public_geometry,
    }
