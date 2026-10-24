"""Finite slurry pumping, prescribed jar conditioning and empirical clarification.

The conditioner/separator has zero runtime holdup. Q*tau values are engineering
design volumes only. Dispersed and floc are two phases of the same real solid;
conversion conserves its molar H/V. No dose-response or metal chemistry is implied.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isclose, pi
from typing import Any

from chemworld.physchem.coupled_system import (
    DEFAULT_COUPLED_ODE_POLICY,
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
from chemworld.physchem.suspension_treatment_specs import (
    SuspensionTankInventory,
    SuspensionTreatmentSpec,
    finite,
)
from chemworld.physchem.transport import FluidState, pipe_pressure_drop, pump_operating_point
from chemworld.physchem.transport_devices import TransportInventory


def phase_amounts(spec: SuspensionTreatmentSpec, tank: SuspensionTankInventory) -> dict[str, float]:
    if set(tank.liquid_mol) != set(spec.material.liquid.component_ids):
        raise ValueError("suspension liquid identities differ from declared material")
    return {
        **{"liquid:" + k: v for k, v in tank.liquid_mol.items()},
        "dispersed": tank.dispersed_mol,
        "floc": tank.floc_mol,
    }


def suspension_volume(spec: SuspensionTreatmentSpec, n: Mapping[str, float]) -> float:
    return sum(n[k] * spec.molar_volumes[k] for k in spec.phase_keys)


def suspension_enthalpy(
    spec: SuspensionTreatmentSpec, n: Mapping[str, float], temperature: float | None = None
) -> float:
    t = spec.temperature_K if temperature is None else temperature
    liquid = {k: n["liquid:" + k] for k in spec.material.liquid.component_ids}
    return spec.material.liquid.enthalpy(liquid, t) + (
        n["dispersed"] + n["floc"]
    ) * spec.material.particle.enthalpy_J_mol(t)


def suspension_treatment_inventory(
    spec: SuspensionTreatmentSpec,
    feed: SuspensionTankInventory,
    product: SuspensionTankInventory | None = None,
    sludge: SuspensionTankInventory | None = None,
) -> TransportInventory:
    empty = SuspensionTankInventory(dict.fromkeys(spec.material.liquid.component_ids, 0.0))
    rows = tuple(phase_amounts(spec, tank) for tank in (feed, product or empty, sludge or empty))
    validate_amounts(spec, rows)
    return TransportInventory(rows, tuple(suspension_enthalpy(spec, n) for n in rows))


def validate_amounts(spec: SuspensionTreatmentSpec, rows: Sequence[Mapping[str, float]]) -> None:
    if len(rows) != 3:
        raise ValueError("suspension requires three original finite tanks")
    for i, n in enumerate(rows):
        if set(n) != set(spec.phase_keys):
            raise ValueError("suspension inventory contains undeclared phase/species identities")
        for value in n.values():
            finite(value, "suspension amount")
        volume = suspension_volume(spec, n)
        if volume * 1000 > spec.capacities_L[i] * (1 + 1e-9):
            raise ValueError("suspension exceeds original finite capacity")
        if i == 0 and volume * 1000 < spec.pickup_volume_L * (1 - 1e-9):
            raise ValueError("suspension feed is below pickup")
        solid_v = (n["dispersed"] + n["floc"]) * spec.material.particle.molar_volume_m3_mol
        # The declared pump viscosity domain applies to pumped feed, not the settled receiver.
        if i == 0 and solid_v / volume > spec.material.maximum_solid_volume_fraction:
            raise ValueError("feed solid fraction exceeds declared suspension viscosity domain")
    feed_v = suspension_volume(spec, rows[0])
    measured = rows[0]["dispersed"] * spec.material.particle.molecular_weight_kg_mol / feed_v
    if not isclose(measured, spec.jar_test.initial_tss_kg_m3, rel_tol=2e-8, abs_tol=1e-10):
        raise ValueError(
            "initial measured TSS disagrees with original dispersed particle inventory"
        )


def suspension_treatment_profile(
    spec: SuspensionTreatmentSpec, rows: Sequence[Mapping[str, float]], requested_flow_m3_s: float
) -> dict[str, Any]:
    finite(requested_flow_m3_s, "requested suspension flow")
    feed_v = suspension_volume(spec, rows[0])
    mass = sum(rows[0][k] * spec.molecular_weights[k] for k in spec.phase_keys)
    fluid = FluidState(mass / feed_v, spec.material.viscosity_Pa_s)
    pump = pump_operating_point(
        spec.pump,
        requested_flow_m3_s,
        lambda q: (
            pipe_pressure_drop(spec.pipe, fluid, volumetric_flow_m3_s=q).pressure_drop_total_Pa
        ),
    )
    q = pump["actual_flow_m3_s"]
    inlet = {k: rows[0][k] * q / feed_v for k in spec.phase_keys}
    # All groups share identical molar H/V: the exact material balance gives Qout=Qin.
    outlet_q = sum(inlet[k] * spec.molar_volumes[k] for k in spec.phase_keys)
    retained = (
        spec.jar_test.final_tss_kg_m3 * outlet_q / spec.material.particle.molecular_weight_kg_mol
    )
    converted = inlet["dispersed"] - retained
    if spec.jar_test.final_ntu == spec.jar_test.initial_ntu:
        # Identical measured endpoints explicitly mean no regrouping. Defining
        # this endpoint exactly also permits BDF Jacobian perturbations off the
        # validated constant-feed-composition manifold, without negative transfer.
        converted, retained = 0.0, inlet["dispersed"]
    if converted < -1e-12 * max(inlet["dispersed"], 1e-20):
        raise ValueError("jar endpoint requires negative particle conversion")
    if converted < 0:  # floating-point equality at the declared no-conversion endpoint only
        converted, retained = 0.0, inlet["dispersed"]
    conditioned = {**inlet, "dispersed": retained, "floc": inlet["floc"] + converted}
    split = spec.clarifier
    capture = {
        k: (
            split.dispersed_capture
            if k == "dispersed"
            else split.floc_capture
            if k == "floc"
            else split.sludge_liquid_fraction
        )
        for k in spec.phase_keys
    }
    sludge = {k: v * capture[k] for k, v in conditioned.items()}
    product = {k: v * (1 - capture[k]) for k, v in conditioned.items()}
    transfers = []
    for dest, fraction in ((1, {k: 1 - v for k, v in capture.items()}), (2, capture)):
        for k in spec.phase_keys:
            rate = retained if k == "dispersed" else inlet[k]
            transfers.append((0, k, dest, k, rate * fraction[k]))
        transfers.append((0, "dispersed", dest, "floc", converted * fraction["floc"]))
    dn = ({k: -v for k, v in inlet.items()}, product, sludge)
    dh = [sum(spec.molar_enthalpies[k] * row[k] for k in row) for row in dn]
    m = spec.mixing
    rapid_each = q * m.rapid_retention_s
    rapid = m.rapid_velocity_gradient_s_inv**2 * fluid.viscosity_Pa_s * rapid_each * m.rapid_basins
    paddle_speed = pi * m.paddle_length_m * m.paddle_rotation_Hz
    floc = (
        (
            0.5
            * m.paddle_drag_coefficient
            * m.paddle_length_m
            * m.paddle_width_m
            * m.paddle_wheels
            * m.paddles_per_wheel
            * fluid.density_kg_m3
            * (m.paddle_velocity_fraction * paddle_speed) ** 3
        )
        if q > 0
        else 0.0
    )
    mixing = rapid + floc
    return {
        **pump,
        "inlet_mol_s": inlet,
        "conditioned_mol_s": conditioned,
        "product_mol_s": product,
        "sludge_mol_s": sludge,
        "transfers": transfers,
        "conditioned_flow_m3_s": outlet_q,
        "converted_particle_mol_s": converted,
        "density_kg_m3": fluid.density_kg_m3,
        "viscosity_Pa_s": fluid.viscosity_Pa_s,
        "rapid_basin_volume_each_m3": rapid_each,
        "rapid_basin_volume_total_m3": rapid_each * m.rapid_basins,
        "floc_basin_volume_m3": q * m.floc_retention_s,
        "paddle_speed_m_s": paddle_speed,
        "rapid_mixing_power_W": rapid,
        "floc_mixing_power_W": floc,
        "mixing_power_W": mixing,
        "enthalpy_rates_W": dh,
        "bath_heat_into_material_W": sum(dh) - pump["hydraulic_power_W"] - mixing,
        "volumes_L": [1000 * suspension_volume(spec, n) for n in rows],
    }


@dataclass(frozen=True)
class _SuspensionThermodynamics:
    spec: SuspensionTreatmentSpec

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return tuple(f"{i}:{k}" for i in range(3) for k in self.spec.phase_keys)

    def rows(self, state: CoupledStateView) -> tuple[dict[str, float], ...]:
        return tuple(
            {k: state.amounts_mol[f"{i}:{k}"] for k in self.spec.phase_keys} for i in range(3)
        )

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        return sum(
            self.spec.material.liquid.heat_capacity(
                {k: n["liquid:" + k] for k in self.spec.material.liquid.component_ids},
                state.temperature_K,
            )
            + (n["dispersed"] + n["floc"]) * self.spec.material.particle.heat_capacity_J_mol_K
            for n in self.rows(state)
        )

    def internal_energy_j(self, state: CoupledStateView) -> float:
        return sum(
            suspension_enthalpy(self.spec, n, state.temperature_K)
            - self.spec.pressure_Pa * suspension_volume(self.spec, n)
            for n in self.rows(state)
        )

    def chemical_energy_rate_w(
        self,
        state: CoupledStateView,
        derivatives: Mapping[str, float],
        contributions: Sequence[ProcessContribution],
    ) -> float:
        return sum(
            derivatives[f"{i}:{k}"]
            * (self.spec.molar_enthalpies[k] - self.spec.pressure_Pa * self.spec.molar_volumes[k])
            for i in range(3)
            for k in self.spec.phase_keys
        )


@dataclass(frozen=True)
class SuspensionTreatmentAdvance:
    initial_state: TransportInventory
    final_state: TransportInventory
    actual_duration_s: float
    stopping_event: str | None
    solver_diagnostic: Mapping[str, Any]
    energy_balance_residual_J: float
    conservation_residuals: Mapping[str, float]
    trajectory: tuple[TransportInventory, ...]


def advance_suspension_treatment(
    spec: SuspensionTreatmentSpec,
    initial: TransportInventory,
    requested_flow_m3_s: float,
    duration_s: float,
    *,
    evaluation_times_s: Sequence[float] | None = None,
) -> tuple[SuspensionTreatmentAdvance, dict[str, Any]]:
    validate_amounts(spec, initial.amounts_mol)
    if len(initial.enthalpies_J) != 3 or any(
        not isclose(h, suspension_enthalpy(spec, n), rel_tol=1e-9, abs_tol=1e-7)
        for n, h in zip(initial.amounts_mol, initial.enthalpies_J, strict=True)
    ):
        raise ValueError("suspension original H disagrees with actual multiphase material")
    thermo = _SuspensionThermodynamics(spec)

    def point(s: CoupledStateView) -> dict[str, Any]:
        rows = thermo.rows(s)
        if suspension_volume(spec, rows[0]) * 1000 < spec.pickup_volume_L:
            # Root finding may evaluate beyond pickup, even beyond zero feed.
            # This isolated feed is withdrawn proportionally, so its composition
            # is invariant. Extend only the constitutive rates using that valid
            # composition; signed event volumes and accepted inventories remain
            # untouched, and the original pickup root terminates the interval.
            rows = (initial.amounts_mol[0], *rows[1:])
        return suspension_treatment_profile(spec, rows, requested_flow_m3_s)

    def bath_point(s: CoupledStateView) -> dict[str, Any]:
        return point(replace(s, amounts_mol={k: max(v, 0) for k, v in s.amounts_mol.items()}))

    def process(s: CoupledStateView) -> ProcessContribution:
        p = point(s)
        return ProcessContribution(
            internal_transfers=tuple(
                InternalTransfer(f"{i}:{a}->{j}:{b}", f"{i}:{a}", f"{j}:{b}", rate)
                for i, a, j, b, rate in p["transfers"]
            ),
            coordinate_rates={
                **{f"H{i}": h for i, h in enumerate(p["enthalpy_rates_W"])},
                "throughput_m3": p["actual_flow_m3_s"],
                "hydraulic_J": p["hydraulic_power_W"],
                "pump_J": p["shaft_power_W"],
                "mixing_J": p["mixing_power_W"],
                "bath_heating_J": max(p["bath_heat_into_material_W"], 0),
                "bath_cooling_J": max(-p["bath_heat_into_material_W"], 0),
            },
        )

    coordinate_names = (
        "H0",
        "H1",
        "H2",
        "throughput_m3",
        "hydraulic_J",
        "pump_J",
        "mixing_J",
        "bath_heating_J",
        "bath_cooling_J",
    )
    start = CoupledState(
        {f"{i}:{k}": n[k] for i, n in enumerate(initial.amounts_mol) for k in spec.phase_keys},
        spec.temperature_K,
        sum(1000 * suspension_volume(spec, n) for n in initial.amounts_mol),
        initial.time_s,
        coordinates=dict.fromkeys(coordinate_names, 0.0),
    )
    flowing = point(start)
    events = []
    if requested_flow_m3_s > 0:
        events = [
            CoupledEvent("no_forward_flow", lambda s: point(s)["actual_flow_m3_s"]),
            CoupledEvent(
                "feed_pickup_exhausted",
                lambda s: 1000 * suspension_volume(spec, thermo.rows(s)[0]) - spec.pickup_volume_L,
            ),
        ]

        def receiver_event(index: int) -> Callable[[CoupledStateView], float]:
            def capacity(state: CoupledStateView) -> float:
                return spec.capacities_L[index] - 1000 * suspension_volume(
                    spec, thermo.rows(state)[index]
                )

            return capacity

        events.extend(
            CoupledEvent(f"{label}_full", receiver_event(i))
            for i, label in ((1, "clarified"), (2, "sludge"))
        )
    events.extend(
        (
            CoupledEvent(
                "bath_heating_limit",
                lambda s: spec.bath.maximum_power_W - bath_point(s)["bath_heat_into_material_W"],
            ),
            CoupledEvent(
                "bath_cooling_limit",
                lambda s: (
                    spec.bath.maximum_cooling_power_W + bath_point(s)["bath_heat_into_material_W"]
                ),
            ),
        )
    )
    compositions = {
        "liquid:" + p.component.identifier: dict(p.component.composition)
        for p in spec.material.liquid.packages
    }
    from chemworld.physchem.specs import ComponentSpec

    particle = dict(
        ComponentSpec(spec.material.particle.species_id, spec.material.particle.formula).composition
    )
    compositions.update({"dispersed": particle, "floc": particle})
    laws = {
        e: {f"{i}:{k}": compositions[k].get(e, 0) for i in range(3) for k in spec.phase_keys}
        for e in sorted({e for row in compositions.values() for e in row})
    }
    system = CoupledSystem(
        thermo,
        (CoupledProcess("suspension-treatment", process),),
        heat_flux=lambda s: HeatFlux(
            work_W=point(s)["shaft_power_W"] + point(s)["mixing_power_W"],
            loss_W=point(s)["external_loss_W"],
        ),
        events=tuple(events),
        conservation_laws=laws,
        isothermal=True,
        inventory_scales_mol={
            f"{i}:{k}": max(sum(n[k] for n in initial.amounts_mol), 1e-3)
            for i in range(3)
            for k in spec.phase_keys
        },
        policy=replace(DEFAULT_COUPLED_ODE_POLICY, rtol=1e-10),
        coordinates=tuple(
            CoupledCoordinate(
                k, "m3" if k == "throughput_m3" else "J", 1e-6 if k == "throughput_m3" else 1.0
            )
            for k in coordinate_names
        ),
    )
    result = system.advance(start, duration_s, evaluation_times_s=evaluation_times_s)

    def original(candidate: CoupledState) -> TransportInventory:
        n = thermo.rows(candidate)
        h = tuple(initial.enthalpies_J[i] + candidate.coordinates[f"H{i}"] for i in range(3))
        for row, value in zip(n, h, strict=True):
            if not isclose(value, suspension_enthalpy(spec, row), rel_tol=1e-9, abs_tol=3e-6):
                raise RuntimeError("integrated original total material H disagrees with properties")
        return TransportInventory(
            n,
            h,
            candidate.time_s,
            initial.heater_energy_J + candidate.coordinates["bath_heating_J"],
            initial.condenser_energy_J + candidate.coordinates["bath_cooling_J"],
        )

    final = original(result.final_state)
    validate_amounts(spec, final.amounts_mol)
    if result.energy_balance_residual_J is None:
        raise RuntimeError("suspension requires the original U balance")
    c = result.final_state.coordinates
    dh = sum(final.enthalpies_J) - sum(initial.enthalpies_J)
    diagnostic = {
        "elapsed_s": result.actual_duration_s,
        "requested_flow_m3_s": requested_flow_m3_s,
        "initial_operating_point": flowing,
        "final_operating_point": suspension_treatment_profile(
            spec, final.amounts_mol, requested_flow_m3_s
        ),
        "throughput_m3": c["throughput_m3"],
        "pump_work_J": c["pump_J"],
        "mixing_work_J": c["mixing_J"],
        "hydraulic_work_J": c["hydraulic_J"],
        "motor_loss_J": result.final_state.heat_loss_J,
        "bath_heat_into_material_J": result.final_state.boundary_heat_J,
        "bath_heating_J": c["bath_heating_J"],
        "bath_cooling_J": c["bath_cooling_J"],
        "enthalpy_change_J": dh,
        "boundary_work_J": 0.0,
        "enthalpy_residual_J": dh
        - result.final_state.boundary_heat_J
        - c["hydraulic_J"]
        - c["mixing_J"],
        "conditioning_model": "prescribed jar endpoint; no dose-response or additive chemistry",
        "holdup_model": "zero runtime process holdup; Q*tau is design volume only",
    }
    return SuspensionTreatmentAdvance(
        initial,
        final,
        result.actual_duration_s,
        result.stopping_event,
        result.solver_diagnostic,
        float(result.energy_balance_residual_J),
        result.conservation_residuals,
        tuple(original(s) for s in result.trajectory),
    ), diagnostic
