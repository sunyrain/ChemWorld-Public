"""Complete ASM1 in finite original carrier tanks and quality StateSlot views.

The shared CoupledSystem advances carrier mol and explicitly labelled kg/mol-eq
coordinates together. Its temporary vectors never become a second inventory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isclose, isfinite
from typing import Any

from chemworld.physchem.activated_sludge_kinetics import (
    ASM1_COMPONENTS,
    ASM1_REACTIONS,
    COMPONENT_UNITS,
    PARTICULATE_COMPONENTS,
    asm1_quality,
    asm1_rates,
    asm1_stoichiometry,
)
from chemworld.physchem.activated_sludge_specs import UTILITY_FIELDS, ActivatedSludgeSpec
from chemworld.physchem.coupled_system import (
    CoupledCoordinate,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    HeatFlux,
    InternalTransfer,
    ProcessContribution,
)
from chemworld.physchem.solver_backend import ODESolverPolicy
from chemworld.physchem.transport import (
    PumpOperatingPoint,
    pipe_pressure_drop,
    pump_operating_point,
)

WORK_FIELDS = ("aeration_electricity_J", "throughput_m3", "hydraulic_work_J")


@dataclass(frozen=True)
class ActivatedSludgeInventory:
    water_mol: tuple[float, ...]
    components: tuple[dict[str, float], ...]
    extents: tuple[dict[str, float], ...]
    utilities: dict[str, float]
    work: dict[str, float]
    time_s: float = 0.0


def activated_sludge_inventory(
    spec: ActivatedSludgeSpec,
    volumes_m3: Sequence[float],
    concentrations: Sequence[Mapping[str, float]],
    *,
    oxygen_supply_kg: float | None = None,
) -> ActivatedSludgeInventory:
    """Author extensive quality quantities and the one real carrier inventory."""
    if len(volumes_m3) != len(spec.member_vessel_ids) or len(concentrations) != len(volumes_m3):
        raise ValueError("ASM1 initialization requires every original vessel")
    result = ActivatedSludgeInventory(
        tuple(v / spec.water_molar_volume_m3 for v in volumes_m3),
        tuple(
            {k: float(row[k]) * volume for k in ASM1_COMPONENTS}
            for volume, row in zip(volumes_m3, concentrations, strict=True)
        ),
        tuple(dict.fromkeys(ASM1_REACTIONS, 0.0) for _ in spec.reactor_vessel_ids),
        {
            "oxygen_supply_kg": spec.oxygen_supply_capacity_kg
            if oxygen_supply_kg is None
            else oxygen_supply_kg,
            "oxygen_offgas_kg": 0.0,
            "nitrogen_offgas_kg": 0.0,
        },
        dict.fromkeys(WORK_FIELDS, 0.0),
    )
    validate_inventory(spec, result)
    return result


def validate_inventory(spec: ActivatedSludgeSpec, inventory: ActivatedSludgeInventory) -> None:
    size, reactors = len(spec.member_vessel_ids), len(spec.reactor_vessel_ids)
    if (
        len(inventory.water_mol) != size
        or len(inventory.components) != size
        or len(inventory.extents) != reactors
        or set(inventory.utilities) != set(UTILITY_FIELDS)
        or set(inventory.work) != set(WORK_FIELDS)
        or any(set(row) != set(ASM1_COMPONENTS) for row in inventory.components)
        or any(set(row) != set(ASM1_REACTIONS) for row in inventory.extents)
    ):
        raise ValueError("ASM1 inventory and original declared members/fields differ")
    values = (
        *inventory.water_mol,
        inventory.time_s,
        *inventory.utilities.values(),
        *inventory.work.values(),
        *(v for row in inventory.components for v in row.values()),
        *(v for row in inventory.extents for v in row.values()),
    )
    if any(isinstance(v, bool) or not isfinite(v) or v < 0 for v in values):
        raise ValueError("ASM1 actual inventory and process receipts must be finite nonnegative")
    volumes = [n * spec.water_molar_volume_m3 for n in inventory.water_mol]
    if (
        volumes[0] < spec.feed_pickup_m3 * (1 - 1e-9)
        or any(v > c * (1 + 1e-9) for v, c in zip(volumes, spec.capacities_m3, strict=True))
        or any(
            not isclose(volumes[i + 1], v, rel_tol=1e-9, abs_tol=1e-15)
            for i, v in enumerate(spec.reactor_volumes_m3)
        )
        or any(
            v == 0 and any(row.values())
            for v, row in zip(volumes, inventory.components, strict=True)
        )
    ):
        raise ValueError("ASM1 actual carrier/quality stock leaves finite tank geometry")
    for key, maximum in zip(
        UTILITY_FIELDS,
        (
            spec.oxygen_supply_capacity_kg,
            spec.oxygen_offgas_capacity_kg,
            spec.nitrogen_offgas_capacity_kg,
        ),
        strict=True,
    ):
        if inventory.utilities[key] > maximum * (1 + 1e-9):
            raise ValueError("ASM1 utility inventory exceeds its finite capacity")
    if inventory.work["aeration_electricity_J"] > spec.maximum_aeration_energy_J * (1 + 1e-9):
        raise ValueError("ASM1 aeration electricity exceeds its declared finite budget")


def flow_profile(spec: ActivatedSludgeSpec, requested_flow_m3_s: float) -> dict[str, Any]:
    if (
        isinstance(requested_flow_m3_s, bool)
        or not isfinite(requested_flow_m3_s)
        or requested_flow_m3_s < 0
    ):
        raise ValueError("ASM1 requested feed flow must be finite and nonnegative")
    ri, rs, ef = (
        spec.internal_recycle_fraction,
        spec.sludge_recycle_fraction,
        spec.clarifier_soluble_effluent_fraction,
    )
    train_ratio = 1 / (1 - ri - (1 - ri) * (1 - ef) * rs)
    ratios = {
        "feed": 1.0,
        "internal_recycle": train_ratio * ri,
        "sludge_recycle": train_ratio * (1 - ri) * (1 - ef) * rs,
    }
    fluid = spec.properties.evaluate({"H2O": 1.0}, spec.temperature_K)[0]

    def pump(name: str, q: float) -> PumpOperatingPoint:
        device, pipe = (
            (spec.pump, spec.pipe) if name == "feed" else (spec.recycle_pump, spec.recycle_pipe)
        )
        return pump_operating_point(
            device,
            q,
            lambda value: (
                pipe_pressure_drop(pipe, fluid, volumetric_flow_m3_s=value).pressure_drop_total_Pa
            ),
        )

    limits = [
        pump(name, requested_flow_m3_s * ratio)["actual_flow_m3_s"] / ratio
        for name, ratio in ratios.items()
        if ratio > 0
    ]
    actual = min(limits, default=0.0)
    points = {name: pump(name, actual * ratio) for name, ratio in ratios.items()}
    train = actual * train_ratio
    return {
        "requested_flow_m3_s": requested_flow_m3_s,
        "actual_flow_m3_s": actual,
        "train_flow_m3_s": train,
        "internal_recycle_m3_s": train * ri,
        "clarifier_flow_m3_s": train * (1 - ri),
        "sludge_recycle_m3_s": train * (1 - ri) * (1 - ef) * rs,
        "effluent_flow_m3_s": train * (1 - ri) * ef,
        "waste_flow_m3_s": train * (1 - ri) * (1 - ef) * (1 - rs),
        "shaft_power_W": sum(p["shaft_power_W"] for p in points.values()),
        "hydraulic_power_W": sum(p["hydraulic_power_W"] for p in points.values()),
        "motor_loss_W": sum(p["external_loss_W"] for p in points.values()),
        "pumps": points,
    }


def _component_key(i: int, name: str) -> str:
    return f"quality:{i}:{name}"


def _extent_key(i: int, reaction: str) -> str:
    return f"extent:{i}:{reaction}"


@dataclass(frozen=True)
class _CarrierThermodynamics:
    spec: ActivatedSludgeSpec

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return tuple(f"water:{i}" for i in range(len(self.spec.member_vessel_ids)))

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        return self.spec.properties.heat_capacity(
            {"H2O": sum(state.amounts_mol.values())}, state.temperature_K
        )

    def internal_energy_j(self, state: CoupledStateView) -> float:
        amount = sum(state.amounts_mol.values())
        return self.spec.properties.enthalpy({"H2O": amount}, state.temperature_K) - (
            self.spec.pressure_Pa * amount * self.spec.water_molar_volume_m3
        )

    def chemical_energy_rate_w(
        self,
        state: CoupledStateView,
        derivatives: Mapping[str, float],
        _contributions: Sequence[ProcessContribution],
    ) -> float:
        u = self.spec.properties.enthalpy({"H2O": 1.0}, state.temperature_K) - (
            self.spec.pressure_Pa * self.spec.water_molar_volume_m3
        )
        return u * sum(derivatives.values())


@dataclass(frozen=True)
class ActivatedSludgeAdvance:
    final: ActivatedSludgeInventory
    actual_duration_s: float
    stopping_event: str | None
    diagnostic: dict[str, Any]
    trajectory: tuple[ActivatedSludgeInventory, ...]


def advance_activated_sludge(
    spec: ActivatedSludgeSpec,
    initial: ActivatedSludgeInventory,
    requested_flow_m3_s: float,
    duration_s: float,
    *,
    evaluation_times_s: Sequence[float] | None = None,
) -> ActivatedSludgeAdvance:
    validate_inventory(spec, initial)
    profile = flow_profile(spec, requested_flow_m3_s)
    nreactors, size = len(spec.reactor_vessel_ids), len(spec.member_vessel_ids)
    matrix = asm1_stoichiometry(spec.parameters)
    n2_coefficient = (1 - spec.parameters.Y_H) / (2.86 * spec.parameters.Y_H)
    feed_volume = initial.water_mol[0] * spec.water_molar_volume_m3
    feed_c = {k: initial.components[0][k] / feed_volume for k in ASM1_COMPONENTS}
    thermodynamics = _CarrierThermodynamics(spec)
    water_keys = thermodynamics.inventory_keys
    coordinate_values = {
        **{
            _component_key(i, k): row[k]
            for i, row in enumerate(initial.components)
            for k in ASM1_COMPONENTS
        },
        **{
            _extent_key(i, r): row[r]
            for i, row in enumerate(initial.extents)
            for r in ASM1_REACTIONS
        },
        **initial.utilities,
        **initial.work,
    }
    # Three actual pumps carry feed and the two original source recycle branches.
    water_edges = [
        (0, 1, profile["actual_flow_m3_s"]),
        *((i, i + 1, profile["train_flow_m3_s"]) for i in range(1, nreactors)),
        (nreactors, 1, profile["internal_recycle_m3_s"] + profile["sludge_recycle_m3_s"]),
        (nreactors, size - 2, profile["effluent_flow_m3_s"]),
        (nreactors, size - 1, profile["waste_flow_m3_s"]),
    ]
    water_transfers = tuple(
        InternalTransfer(
            f"asm1-water:{source}:{target}",
            water_keys[source],
            water_keys[target],
            q / spec.water_molar_volume_m3,
        )
        for source, target, q in water_edges
        if source != target and q > 0
    )

    def source_rates(state: CoupledStateView) -> tuple[dict[str, float], dict[str, Any]]:
        dc = dict.fromkeys(coordinate_values, 0.0)
        # Signed continuous trial transport is algebraically conservative. Only
        # the rate-domain extension sees max(0, C); accepted coordinates remain
        # strictly nonnegative and are never repaired or rounded.
        concentrations = [
            feed_c,
            *(
                {k: state.coordinates[_component_key(i + 1, k)] / v for k in ASM1_COMPONENTS}
                for i, v in enumerate(spec.reactor_volumes_m3)
            ),
        ]

        def transfer(
            source: int, target: int, q: float, factors: Mapping[str, float] | None = None
        ) -> None:
            if source == target:
                return
            for k in ASM1_COMPONENTS:
                flux = q * concentrations[source][k] * (1.0 if factors is None else factors[k])
                dc[_component_key(source, k)] -= flux
                dc[_component_key(target, k)] += flux

        transfer(0, 1, profile["actual_flow_m3_s"])
        for i in range(1, nreactors):
            transfer(i, i + 1, profile["train_flow_m3_s"])
        transfer(nreactors, 1, profile["internal_recycle_m3_s"])
        splits = {
            k: spec.clarifier_particulate_effluent_fraction
            if k in PARTICULATE_COMPONENTS
            else spec.clarifier_soluble_effluent_fraction
            for k in ASM1_COMPONENTS
        }
        transfer(nreactors, size - 2, profile["clarifier_flow_m3_s"], splits)
        transfer(
            nreactors,
            size - 1,
            profile["clarifier_flow_m3_s"],
            {k: (1 - f) * (1 - spec.sludge_recycle_fraction) for k, f in splits.items()},
        )
        transfer(
            nreactors,
            1,
            profile["clarifier_flow_m3_s"],
            {k: (1 - f) * spec.sludge_recycle_fraction for k, f in splits.items()},
        )
        aeration = []
        all_rates = []
        for i, (volume, kla) in enumerate(
            zip(spec.reactor_volumes_m3, spec.kla_per_hour, strict=True)
        ):
            c = {k: max(v, 0.0) for k, v in concentrations[i + 1].items()}
            rates = {r: rate * volume for r, rate in asm1_rates(spec.parameters, c).items()}
            all_rates.append(rates)
            for r, rate in rates.items():
                dc[_extent_key(i, r)] = rate
                for k, coefficient in matrix[r].items():
                    dc[_component_key(i + 1, k)] += coefficient * rate
            dc["nitrogen_offgas_kg"] += n2_coefficient * rates["R2"]
            oxygen = kla / 3600 * volume * (spec.oxygen_equilibrium_kg_m3 - c["S_O"])
            dc[_component_key(i + 1, "S_O")] += oxygen
            dc["oxygen_supply_kg"] -= max(oxygen, 0.0)
            dc["oxygen_offgas_kg"] += max(-oxygen, 0.0)
            if spec.electricity_mode == "calculated":
                power = spec.oxygen_equilibrium_kg_m3 * kla * volume / 1.8 * 1000
            elif spec.electricity_mode == "fixed":
                power = spec.fixed_electricity_kWh_m3 * profile["train_flow_m3_s"] * 3.6e6
            else:
                power = 0.0
            aeration.append({"oxygen_transfer_kg_s": oxygen, "electricity_W": power})
        dc["aeration_electricity_J"] = sum(row["electricity_W"] for row in aeration)
        dc["throughput_m3"] = profile["actual_flow_m3_s"]
        dc["hydraulic_work_J"] = profile["hydraulic_power_W"]
        return dc, {"reaction_extent_rates": all_rates, "aeration": aeration}

    def process(state: CoupledStateView) -> ProcessContribution:
        dc, _ = source_rates(state)
        return ProcessContribution(internal_transfers=water_transfers, coordinate_rates=dc)

    start = CoupledState(
        dict(zip(water_keys, initial.water_mol, strict=True)),
        spec.temperature_K,
        sum(initial.water_mol) * spec.water_molar_volume_m3 * 1000,
        initial.time_s,
        coordinates=coordinate_values,
    )
    dc0, rate0 = source_rates(start)
    events: list[CoupledEvent] = []
    if profile["actual_flow_m3_s"] > 0:
        events.extend(
            (
                CoupledEvent(
                    "asm1_feed_pickup",
                    lambda s: (
                        s.amounts_mol[water_keys[0]] * spec.water_molar_volume_m3
                        - spec.feed_pickup_m3
                    ),
                ),
                CoupledEvent(
                    "asm1_effluent_full",
                    lambda s: (
                        spec.product_capacity_m3
                        - s.amounts_mol[water_keys[-2]] * spec.water_molar_volume_m3
                    ),
                ),
                CoupledEvent(
                    "asm1_sludge_full",
                    lambda s: (
                        spec.sludge_capacity_m3
                        - s.amounts_mol[water_keys[-1]] * spec.water_molar_volume_m3
                    ),
                ),
            )
        )
    if any(spec.kla_per_hour):
        events.extend(
            (
                CoupledEvent(
                    "asm1_oxygen_supply_exhausted",
                    lambda s: (
                        1e-30
                        if s.coordinates["oxygen_supply_kg"] == 0
                        and source_rates(s)[0]["oxygen_supply_kg"] >= 0
                        else s.coordinates["oxygen_supply_kg"]
                    ),
                ),
                CoupledEvent(
                    "asm1_oxygen_offgas_full",
                    lambda s: spec.oxygen_offgas_capacity_kg - s.coordinates["oxygen_offgas_kg"],
                ),
            )
        )
    events.extend(
        (
            CoupledEvent(
                "asm1_nitrogen_offgas_full",
                lambda s: spec.nitrogen_offgas_capacity_kg - s.coordinates["nitrogen_offgas_kg"],
            ),
            CoupledEvent(
                "asm1_electricity_budget",
                lambda s: spec.maximum_aeration_energy_J - s.coordinates["aeration_electricity_J"],
            ),
            CoupledEvent(
                "asm1_aeration_power_limit",
                lambda _s: (
                    spec.aeration_power_limit_W - dc0["aeration_electricity_J"]
                    if spec.aeration_power_limit_W != dc0["aeration_electricity_J"]
                    else 1e-30
                ),
            ),
            CoupledEvent(
                "asm1_bath_cooling_limit",
                lambda _s: (
                    spec.maximum_bath_cooling_W - profile["hydraulic_power_W"]
                    if spec.maximum_bath_cooling_W != profile["hydraulic_power_W"]
                    else 1e-30
                ),
            ),
        )
    )
    for i in range(nreactors):
        for component in ("S_NH", "S_ALK"):
            key = _component_key(i + 1, component)

            def available(s: CoupledStateView, key: str = key) -> float:
                value = s.coordinates[key]
                # A zero inert/producing pool is valid. Only the consuming side
                # is a finite reaction-domain boundary; no extra Monod law.
                if value == 0 and source_rates(s)[0][key] >= 0:
                    return 1e-30
                return value

            events.append(CoupledEvent(f"asm1_{i + 1}_{component}_exhausted", available))
    declaration = spec.balance_declaration()
    laws = {"carrier_water": dict.fromkeys(water_keys, 1.0)}
    for quantity in declaration["quantities"]:
        weights = {
            _component_key(i, quantity): 1.0 for i in range(size) if quantity in ASM1_COMPONENTS
        }
        if quantity == "S_O":
            weights.update(oxygen_supply_kg=1.0, oxygen_offgas_kg=1.0)
        if quantity == "N2_N":
            weights["nitrogen_offgas_kg"] = 1.0
        for i in range(nreactors):
            for reaction, row in declaration["reactions"].items():
                coefficient = row["stoichiometry"].get(quantity, 0.0)
                if coefficient:
                    weights[_extent_key(i, reaction)] = -coefficient
        laws[f"asm1_source_budget:{quantity}"] = weights
    coordinates = []
    for key, value in coordinate_values.items():
        unit = (
            COMPONENT_UNITS[key.rsplit(":", 1)[1]]
            if key.startswith("quality:")
            else "m^3"
            if key == "throughput_m3"
            else "J"
            if key.endswith("_J")
            else "kg"
        )
        scale = max(abs(value), 0.001 if unit == "mol_eq" else 1.0 if unit == "J" else 1e-6)
        coordinates.append(CoupledCoordinate(key, unit, scale, minimum=0.0))
    system = CoupledSystem(
        thermodynamics,
        (CoupledProcess("complete-asm1", process),),
        heat_flux=lambda _s: HeatFlux(
            work_W=profile["shaft_power_W"], loss_W=profile["motor_loss_W"]
        ),
        events=tuple(replace(event, feasible_side=True) for event in events),
        conservation_laws=laws,
        inventory_scales_mol={
            k: max(n, 1e-3) for k, n in zip(water_keys, initial.water_mol, strict=True)
        },
        isothermal=True,
        policy=ODESolverPolicy(
            "complete-asm1-finite-cstr-v1",
            method="BDF",
            rtol=spec.rtol,
            atol=spec.atol,
            max_step_s=spec.maximum_step_s,
        ),
        coordinates=tuple(coordinates),
    )
    result = system.advance(start, duration_s, evaluation_times_s=evaluation_times_s)

    def unpack(state: CoupledState) -> ActivatedSludgeInventory:
        c = state.coordinates
        inventory = ActivatedSludgeInventory(
            tuple(state.amounts_mol[k] for k in water_keys),
            tuple({k: c[_component_key(i, k)] for k in ASM1_COMPONENTS} for i in range(size)),
            tuple({r: c[_extent_key(i, r)] for r in ASM1_REACTIONS} for i in range(nreactors)),
            {k: c[k] for k in UTILITY_FIELDS},
            {k: c[k] for k in WORK_FIELDS},
            state.time_s,
        )
        validate_inventory(spec, inventory)
        return inventory

    final = unpack(result.final_state)
    work = {k: final.work[k] - initial.work[k] for k in WORK_FIELDS}
    diagnostic = {
        "flow": profile,
        "initial_rates": rate0,
        "final_rates": source_rates(result.final_state)[1],
        "solver": result.solver_diagnostic,
        "source_budget_residuals": dict(result.conservation_residuals),
        "pump_shaft_work_J": result.final_state.work_J,
        "pump_motor_loss_J": result.final_state.heat_loss_J,
        "carrier_bath_heat_J": result.final_state.boundary_heat_J,
        "carrier_energy_residual_J": result.energy_balance_residual_J,
        **work,
        "scope": "complete ASM1 empirical quality plus finite original carrier water",
        "energy_scope": "water sensible/pump only; aeration electricity is empirical utility; "
        "biochemical formation heat and gas-side thermodynamics unspecified",
    }
    return ActivatedSludgeAdvance(
        final,
        result.actual_duration_s,
        result.stopping_event,
        diagnostic,
        tuple(unpack(s) for s in result.trajectory),
    )


def inventory_quality(
    spec: ActivatedSludgeSpec, inventory: ActivatedSludgeInventory
) -> dict[str, dict[str, float]]:
    validate_inventory(spec, inventory)
    return {
        name: asm1_quality(
            spec.parameters, {k: value / volume if volume else 0.0 for k, value in row.items()}
        )
        for name, water, row in zip(
            spec.member_vessel_ids, inventory.water_mol, inventory.components, strict=True
        )
        for volume in (water * spec.water_molar_volume_m3,)
    }
