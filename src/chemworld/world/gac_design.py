"""Read-only fresh-bed design using the original finite 49C process feed."""

from __future__ import annotations

from typing import Any

from chemworld.foundation import WorldState
from chemworld.physchem.aqueous_adsorption import aqueous_adsorption_profile
from chemworld.physchem.gac_design import (
    GACDesignSpec,
    GACFeedConditions,
    evaluate_gac_design,
)
from chemworld.runtime.aqueous_adsorption import (
    aqueous_adsorption_record,
    unpack_aqueous_adsorption,
)


def analyze_gac_design_world(
    state: WorldState,
    spec: GACDesignSpec,
    *,
    anchor_vessel_id: str | None = None,
) -> dict[str, Any]:
    """Read current feed/pump; predict a separate fresh bed at held conditions.

    Existing mixed-contactor media and its loading do not become this design's
    GAC bed. The reported duration only measures the feed before its pickup
    boundary; it does not promise the original collector or thermal capacity.
    """
    states = {**state.inactive_vessels, state.vessel_id: state.replace(inactive_vessels={})}
    anchors = [
        tank
        for name, tank in states.items()
        if (anchor_vessel_id is None or name == anchor_vessel_id)
        and aqueous_adsorption_record(tank) is not None
    ]
    if len(anchors) != 1:
        raise ValueError("GAC query requires one unambiguous original aqueous-adsorption anchor")
    anchor = anchors[0].replace(
        inactive_vessels={
            key: value for key, value in states.items() if key != anchors[0].vessel_id
        },
    )
    record, original, inventory = unpack_aqueous_adsorption(anchor)
    control = record.settings.get("flow_control", {})
    if not control.get("configured") or control.get("flow_path", "process") != "process":
        raise ValueError("GAC query requires configured original process feed, not rinse")
    if spec.target_species not in original.solute_ids:
        raise ValueError("GAC target must be an original neutral aqueous solute")
    point = aqueous_adsorption_profile(
        original,
        inventory,
        control.get("requested_flow_m3_s", 0.0),
        "process",
    )
    flow = point["actual_flow_m3_s"]
    if flow <= 0:
        raise ValueError("GAC query requires a positive actual process pump flow")
    amounts = inventory.amounts_mol[0]
    volume = original.volume_m3(amounts)
    concentration = (
        amounts[spec.target_species] * original.masses_kg_mol[spec.target_species] / volume
    )
    density = sum(amounts[k] * original.masses_kg_mol[k] for k in original.component_ids) / volume
    viscosity = original.viscosity.evaluate(
        original.temperature_K,
        original.pressure_Pa,
        sum(amounts[k] for k in original.solute_ids) / volume,
    )
    feed = GACFeedConditions(
        flow,
        concentration,
        original.temperature_K,
        original.pressure_Pa,
        density,
        viscosity,
        f"original {original.database.database_id}; current vessel {anchor.vessel_id}",
    )
    report = evaluate_gac_design(spec, feed)
    available = max(0.0, volume - original.pickup_volume_L * 1e-3)
    required = flow * report["metrics"]["operational_time_s"]
    report["world_query"] = {
        "anchor_vessel_id": anchor.vessel_id,
        "selected_vessel_id": state.vessel_id,
        "state_time_s": state.ledger.time_s,
        "feed_amounts_mol": dict(amounts),
        "feed_volume_m3": volume,
        "pickup_volume_m3": original.pickup_volume_L * 1e-3,
        "available_feed_volume_m3": available,
        "feed_duration_before_pickup_s": available / flow,
        "replacement_feed_volume_m3": required,
        "source_covers_replacement": available >= required,
        "replacement_feed_coverage_fraction": available / required,
        "requested_flow_m3_s": control["requested_flow_m3_s"],
        "actual_flow_m3_s": flow,
        "original_pump_operating_point": {
            key: value
            for key, value in point.items()
            if key not in {"adsorption_rates_mol_s", "bath_heat_into_system_W"}
        },
        "inert_solutes": [key for key in original.solute_ids if key != spec.target_species],
        "volume_basis": "original water n times constant partial volume; zero solute volume",
        "density_basis": "all original species masses / same actual aqueous volume",
        "viscosity_source": original.viscosity.source_id,
        "world_advanced": False,
        "existing_media_used_as_fresh_bed": False,
        "boundary": (
            "fresh design at held feed; source-only coverage, not a runtime duration guarantee"
        ),
    }
    return report
