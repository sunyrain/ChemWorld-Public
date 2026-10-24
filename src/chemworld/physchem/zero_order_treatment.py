"""Empirical multistage liquid splits on the original finite n/H integrator."""

from __future__ import annotations

from collections.abc import Mapping
from math import fsum, isclose
from typing import Any

import numpy as np

from chemworld.physchem.phase_inventory import PhaseInventoryRates, integrate_phase_inventory
from chemworld.physchem.transport import pipe_pressure_drop, pump_operating_point
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.physchem.zero_order_treatment_specs import ZeroOrderTreatmentSpec, finite_number


def validate_treatment_amounts(spec: ZeroOrderTreatmentSpec, amounts: Mapping[str, float]) -> None:
    if set(amounts) != set(spec.properties.component_ids):
        raise ValueError("treatment liquid must declare exactly the original component identities")
    for value in amounts.values():
        finite_number(value, "liquid inventory")
    if sum(amounts.values()) and amounts[spec.water_id] <= 0:
        raise ValueError("this treatment template requires water-bearing homogeneous liquids")


def treatment_inventory(
    spec: ZeroOrderTreatmentSpec,
    feed: Mapping[str, float],
    *,
    collectors: tuple[Mapping[str, float], ...] | None = None,
) -> TransportInventory:
    if collectors is None:
        collectors = tuple(
            dict.fromkeys(spec.properties.component_ids, 0.0) for _ in range(len(spec.stages) + 1)
        )
    if len(collectors) != len(spec.stages) + 1:
        raise ValueError("one product and every stage concentrate need finite inventories")
    rows = tuple(dict(n) for n in (feed, *collectors))
    for n in rows:
        validate_treatment_amounts(spec, n)
    return TransportInventory(
        rows, tuple(spec.enthalpy(i, n, spec.temperature_K) for i, n in enumerate(rows))
    )


def treatment_profile(
    spec: ZeroOrderTreatmentSpec,
    feed: Mapping[str, float],
    requested_flow_m3_s: float,
) -> dict[str, Any]:
    validate_treatment_amounts(spec, feed)
    finite_number(requested_flow_m3_s, "requested liquid flow")
    _, volumes = spec.molar_terms()
    feed_volume = fsum(feed[k] * volumes[k] for k in volumes)
    if feed_volume == 0:
        if requested_flow_m3_s:
            raise ValueError("cannot draw from an empty liquid source")
        point = {
            "actual_flow_m3_s": 0.0,
            "pressure_rise_Pa": 0.0,
            "shaft_power_W": 0.0,
            "hydraulic_power_W": 0.0,
            "external_loss_W": 0.0,
            "flow_limited": False,
        }
    else:
        fluid = spec.properties.evaluate(dict(feed), spec.temperature_K)[0]
        point = pump_operating_point(
            spec.pump,
            requested_flow_m3_s,
            lambda flow: (
                pipe_pressure_drop(
                    spec.pipe,
                    fluid,
                    volumetric_flow_m3_s=flow,
                ).pressure_drop_total_Pa
            ),
        )
    inlet = {
        k: feed[k] * point["actual_flow_m3_s"] / feed_volume if feed_volume else 0.0
        for k in volumes
    }
    current, stages = dict(inlet), []
    for stage in spec.stages:
        volume_rate = fsum(current[k] * volumes[k] for k in volumes)
        if volume_rate > stage.maximum_inlet_flow_m3_s * (1 + 1e-12):
            raise ValueError(f"stage {stage.stage_id} outside declared inlet-flow domain")
        if volume_rate and any(
            current[k] / volume_rate > limit * (1 + 1e-12)
            for k, limit in stage.maximum_inlet_concentration_mol_m3.items()
        ):
            raise ValueError(f"stage {stage.stage_id} outside declared concentration domain")
        product = {
            k: value
            * (stage.water_recovery if k == spec.water_id else 1 - stage.removal_to_concentrate[k])
            for k, value in current.items()
        }
        reject = {k: current[k] - product[k] for k in volumes}
        stages.append(
            {
                "stage_id": stage.stage_id,
                "inlet_mol_s": current,
                "product_mol_s": product,
                "reject_mol_s": reject,
                "inlet_volume_m3_s": volume_rate,
                "additional_electrical_power_W": stage.specific_electricity_kWh_m3
                * volume_rate
                * 3.6e6,
            }
        )
        current = product
    return {
        **point,
        "inlet_mol_s": inlet,
        "product_mol_s": current,
        "stages": stages,
        "additional_electrical_power_W": sum(
            stage["additional_electrical_power_W"] for stage in stages
        ),
        "scope": "empirical component split; no removal mechanism or sludge model",
    }


def advance_zero_order_treatment(
    spec: ZeroOrderTreatmentSpec,
    state: TransportInventory,
    requested_flow_m3_s: float,
    duration_s: float,
    *,
    progress: Any = None,
) -> tuple[Any, dict[str, Any]]:
    if len(state.amounts_mol) != len(spec.stages) + 2:
        raise ValueError("finite treatment inventory must contain feed, product and each reject")
    if len(state.enthalpies_J) != len(state.amounts_mol):
        raise ValueError("every treatment inventory has exactly one enthalpy")
    for i, n in enumerate(state.amounts_mol):
        validate_treatment_amounts(spec, n)
        if not isclose(
            state.enthalpies_J[i],
            spec.enthalpy(i, n, spec.temperature_K),
            rel_tol=1e-9,
            abs_tol=1e-7,
        ):
            raise ValueError("treatment H does not match the declared original-property isotherm")
    h, v = spec.molar_terms()
    profile = treatment_profile(spec, state.amounts_mol[0], requested_flow_m3_s)
    rows = (
        {k: -n for k, n in profile["inlet_mol_s"].items()},
        profile["product_mol_s"],
        *(stage["reject_mol_s"] for stage in profile["stages"]),
    )
    matrix = np.asarray([[n[k] for k in spec.properties.component_ids] for n in rows])
    hdot = np.asarray([fsum(n[k] * h[k] for k in h) for n in rows])
    shaft, hydraulic = profile["shaft_power_W"], profile["hydraulic_power_W"]
    extra = profile["additional_electrical_power_W"]
    bath = float(sum(hdot)) - hydraulic - extra

    def volume(amounts: Mapping[str, float]) -> float:
        return fsum(amounts[k] * v[k] for k in v) * 1000

    capacities = (spec.product_capacity_L, *(s.reject_capacity_L for s in spec.stages))
    if volume(state.amounts_mol[0]) < spec.pickup_volume_L - 1e-10 or any(
        volume(n) > capacity + 1e-10
        for n, capacity in zip(state.amounts_mol[1:], capacities, strict=True)
    ):
        raise ValueError("treatment starts outside its finite pickup/capacity domain")

    def capacity_stop(index: int, capacity: float) -> Any:
        def stop(candidate: Any, temperatures: Any) -> float:
            return capacity - volume(candidate.amounts_mol[index])

        return stop

    stops = []
    if requested_flow_m3_s:
        stops = [
            ("feed_pickup_exhausted", lambda s, _: volume(s.amounts_mol[0]) - spec.pickup_volume_L)
        ]
        stops.extend(
            (
                "product_full" if i == 1 else f"reject:{spec.stages[i - 2].stage_id}:full",
                capacity_stop(i, capacity),
            )
            for i, capacity in enumerate(capacities, 1)
        )
        if profile["actual_flow_m3_s"] == 0:
            stops.insert(0, ("no_forward_pump_drive", lambda s, _: 0.0))
    result = integrate_phase_inventory(
        spec,
        state,
        duration_s=duration_s,
        rates=lambda candidate, temperatures: PhaseInventoryRates(
            matrix,
            hdot,
            shaft + extra + max(bath, 0.0),
            shaft - hydraulic + max(-bath, 0.0),
            (),
        ),
        closure_indices=(),
        stop_functions=stops,
        enthalpy_bounds=lambda i, n: (-np.inf, np.inf),
        initial_boundary="stop",
        progress=progress,
        solver_id="finite_zero_order_treatment_nH_v1",
    )
    dt, final = result.actual_duration_s, result.final_state
    for i, n in enumerate(final.amounts_mol):
        validate_treatment_amounts(spec, n)
        if not isclose(
            final.enthalpies_J[i],
            spec.enthalpy(i, n, spec.temperature_K),
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise RuntimeError("treatment failed original-property enthalpy closure")
    receipt = {
        "schema": "native-treatment-throughput-v1",
        "elapsed_s": dt,
        "feed_volume_m3": profile["actual_flow_m3_s"] * dt,
        "product_volume_m3": fsum(profile["product_mol_s"][k] * v[k] for k in v) * dt,
        "reject_volumes_m3": {
            s.stage_id: fsum(row["reject_mol_s"][k] * v[k] for k in v) * dt
            for s, row in zip(spec.stages, profile["stages"], strict=True)
        },
        "stage_inlet_volumes_m3": {
            row["stage_id"]: row["inlet_volume_m3_s"] * dt for row in profile["stages"]
        },
        "pump_work_J": shaft * dt,
        "hydraulic_work_J": hydraulic * dt,
        "additional_electrical_work_J": extra * dt,
        "motor_loss_J": (shaft - hydraulic) * dt,
        "bath_heat_into_liquid_J": bath * dt,
    }
    return result, {
        "profile": profile,
        "receipt": receipt,
        "liquid_enthalpy_change_J": sum(final.enthalpies_J) - sum(state.enthalpies_J),
        "scope": "fixed-T empirical split; shaft+additional electricity-motor+bath=H",
    }
