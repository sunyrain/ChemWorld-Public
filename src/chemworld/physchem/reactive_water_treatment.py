"""Finite pumped CSTR on original reaction, liquid and CoupledSystem models.

The inlet has a finite pump and pipe. The outlet is explicitly ideal overflow at
the same pressure/elevation, with no resistance/holdup or kinetic energy. Pure
volumes are constant; additive volume is conserved by every reaction. This
explicit incompressibility makes d(H-PV)/dT=Cp; no generic Cp-to-Cv rule is used.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from math import isclose, isfinite
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
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.transport import (
    PipeSpec,
    PumpSpec,
    pipe_pressure_drop,
    pump_operating_point,
)
from chemworld.physchem.transport_devices import LiquidTransportProperties, TransportInventory

SCHEMA = "finite-reactive-water-treatment-v1"


def _key(i: int, species: str) -> str:
    return f"{i}:{species}"


@dataclass(frozen=True)
class ReactiveWaterTreatmentSpec:
    properties: LiquidTransportProperties
    network: ReactionNetworkSpec
    bath: ThermalDeviceSpec
    pump: PumpSpec
    pipe: PipeSpec
    capacities_L: tuple[float, float, float] = (0.1, 0.005, 0.1)
    pickup_volume_L: float = 0.0001
    reactor_vessel_id: str = "treatment_reactor"
    product_vessel_id: str = "product"

    def __post_init__(self) -> None:
        # Own nested declarations, not caller-mutable network/property mappings.
        object.__setattr__(
            self, "properties", LiquidTransportProperties.from_dict(self.properties.to_dict())
        )
        object.__setattr__(self, "network", ReactionNetworkSpec.from_dict(self.network.to_dict()))
        object.__setattr__(self, "capacities_L", tuple(self.capacities_L))
        if (
            len(self.capacities_L) != 3
            or any(
                isinstance(v, bool) or not isfinite(v) or v <= 0
                for v in (*self.capacities_L, self.pickup_volume_L)
            )
            or self.pickup_volume_L >= self.capacities_L[0]
        ):
            raise ValueError("reactive treatment needs finite positive capacities and pickup")
        if (
            not self.reactor_vessel_id
            or not self.product_vessel_id
            or self.reactor_vessel_id == self.product_vessel_id
        ):
            raise ValueError("reactor and product require distinct original vessel identities")
        prop = self.properties
        if (
            self.bath.mode != "isothermal"
            or not self.bath.enabled
            or not prop.temperature_bounds_K[0]
            <= self.temperature_K
            <= prop.temperature_bounds_K[1]
            or self.pipe.elevation_change_m != 0
        ):
            raise ValueError("reactive treatment requires declared common T/P and zero elevation")
        if (
            prop.activity_model is not None
            or prop.incompressible_model is not None
            or prop.volume_mixing_method not in (None, "Amgat")
            or set(prop.component_ids) != set(self.network.species_ids)
            or self.network.stochastic is not None
        ):
            raise ValueError("reactive treatment requires additive neutral liquid properties")
        species = {s.species_id: s for s in self.network.species}
        for package in prop.packages:
            component, correlations = package.component, package.by_property("liquid_density")
            s = species[component.identifier]
            if (
                s.charge
                or s.phase != "liquid"
                or component.charge
                or component.default_phase != "liquid"
                or component.formula != s.formula
            ):
                raise ValueError("reactive treatment requires matching neutral liquid identities")
            if (
                len(correlations) != 1
                or correlations[0].equation_id != "linear_liquid_density"
                or correlations[0].coefficients.get("alpha") != 0
            ):
                raise ValueError(
                    "reactive treatment requires explicit constant pure liquid volumes"
                )
        for reaction in self.network.reactions:
            law = reaction.rate_law
            if (
                law.equation_id
                not in {"mass_action", "arrhenius", "modified_arrhenius", "reversible_arrhenius"}
                or law.uses_activities
                or law.parameters.get("mechanism_schema")
                or law.parameters.get("reverse_basis")
                or reaction.equilibrium_model_id
                or reaction.delta_h_J_per_mol != 0
            ):
                raise ValueError(
                    "use declared liquid kinetics and original H; no extra reaction heat"
                )
            delta_v = sum(nu * self.molar_volumes_m3[k] for k, nu in reaction.stoichiometry.items())
            scale = sum(
                abs(nu) * self.molar_volumes_m3[k] for k, nu in reaction.stoichiometry.items()
            )
            if abs(delta_v) > 1e-12 * scale:
                raise ValueError("every treatment reaction must conserve additive liquid volume")

    @property
    def temperature_K(self) -> float:  # noqa: N802
        return self.bath.setpoint_K

    @cached_property
    def molar_volumes_m3(self) -> dict[str, float]:
        return {
            k: self.properties.evaluate(
                {j: float(j == k) for j in self.properties.component_ids}, self.temperature_K
            )[2]
            for k in self.properties.component_ids
        }

    @cached_property
    def molar_enthalpies_J(self) -> dict[str, float]:  # noqa: N802
        return {
            k: self.properties.enthalpy(
                {j: float(j == k) for j in self.properties.component_ids}, self.temperature_K
            )
            for k in self.properties.component_ids
        }

    def volume_m3(self, n: Mapping[str, float]) -> float:
        return sum(n[k] * v for k, v in self.molar_volumes_m3.items())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "properties": self.properties.to_dict(),
            "network": self.network.to_dict(),
            "bath": self.bath.to_dict(),
            "pump": asdict(self.pump),
            "pipe": asdict(self.pipe),
            "capacities_L": list(self.capacities_L),
            "pickup_volume_L": self.pickup_volume_L,
            "reactor_vessel_id": self.reactor_vessel_id,
            "product_vessel_id": self.product_vessel_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReactiveWaterTreatmentSpec":
        values = dict(payload)
        if values.pop("schema", None) != SCHEMA:
            raise ValueError("unsupported reactive treatment schema")
        return cls(
            **{
                **values,
                "properties": LiquidTransportProperties.from_dict(values["properties"]),
                "network": ReactionNetworkSpec.from_dict(values["network"]),
                "bath": ThermalDeviceSpec.from_dict(values["bath"]),
                "pump": PumpSpec(**values["pump"]),
                "pipe": PipeSpec(**values["pipe"]),
            }
        )


def reactive_water_treatment_inventory(
    spec: ReactiveWaterTreatmentSpec,
    feed: Mapping[str, float],
    reactor: Mapping[str, float],
    product: Mapping[str, float] | None = None,
) -> TransportInventory:
    amounts = (
        dict(feed),
        dict(reactor),
        dict(product) if product is not None else dict.fromkeys(spec.properties.component_ids, 0.0),
    )
    for i, n in enumerate(amounts):
        if set(n) != set(spec.properties.component_ids) or any(
            isinstance(v, bool) or not isfinite(v) or v < 0 for v in n.values()
        ):
            raise ValueError("reactive treatment requires declared nonnegative liquid amounts")
        volume = 1000 * spec.volume_m3(n)
        if (
            volume > spec.capacities_L[i] * (1 + 1e-10)
            or (i < 2 and volume <= 0)
            or (i == 1 and not isclose(volume, spec.capacities_L[1], rel_tol=1e-9, abs_tol=1e-12))
            or (i == 0 and volume < spec.pickup_volume_L * (1 - 1e-9))
        ):
            raise ValueError("reactive treatment leaves original finite liquid capacity/pickup")
    return TransportInventory(
        amounts, tuple(spec.properties.enthalpy(n, spec.temperature_K) for n in amounts)
    )


def reactive_water_treatment_profile(
    spec: ReactiveWaterTreatmentSpec,
    amounts: Sequence[Mapping[str, float]],
    requested_flow_m3_s: float,
) -> dict[str, Any]:
    """Current feed properties and actual pump operating point, evaluated each RHS."""
    if (
        isinstance(requested_flow_m3_s, bool)
        or not isfinite(requested_flow_m3_s)
        or requested_flow_m3_s < 0
        or len(amounts) != 3
    ):
        raise ValueError("reactive treatment needs three tanks and nonnegative requested flow")
    fluid, _, feed_volume = spec.properties.evaluate(amounts[0], spec.temperature_K)
    point = pump_operating_point(
        spec.pump,
        requested_flow_m3_s,
        lambda q: (
            pipe_pressure_drop(spec.pipe, fluid, volumetric_flow_m3_s=q).pressure_drop_total_Pa
        ),
    )
    q = point["actual_flow_m3_s"]
    reactor_volume = spec.volume_m3(amounts[1])
    inlet = {k: amounts[0][k] * q / feed_volume for k in spec.properties.component_ids}
    outlet = {k: amounts[1][k] * q / reactor_volume for k in spec.properties.component_ids}
    rates = spec.network.reaction_rates(
        amounts[1], volume_L=reactor_volume * 1000, temperature_K=spec.temperature_K
    )
    reaction = spec.network.amount_derivatives_from_rates(rates, volume_L=reactor_volume * 1000)
    dn = (
        {k: -inlet[k] for k in inlet},
        {k: inlet[k] - outlet[k] + reaction[k] for k in inlet},
        outlet,
    )
    dh = [sum(spec.molar_enthalpies_J[k] * row[k] for k in row) for row in dn]
    bath = sum(dh) - point["hydraulic_power_W"]
    return {
        **point,
        "inlet_mol_s": inlet,
        "outlet_mol_s": outlet,
        "reaction_mol_s": reaction,
        "reaction_progress_mol_s": {k: v * reactor_volume * 1000 for k, v in rates.items()},
        "enthalpy_rates_W": dh,
        "bath_heat_into_liquids_W": bath,
        "volumes_L": [1000 * spec.volume_m3(n) for n in amounts],
    }


@dataclass(frozen=True)
class _LiquidThermodynamics:
    spec: ReactiveWaterTreatmentSpec

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return tuple(_key(i, k) for i in range(3) for k in self.spec.properties.component_ids)

    def rows(self, state: CoupledStateView) -> tuple[dict[str, float], ...]:
        return tuple(
            {k: state.amounts_mol[_key(i, k)] for k in self.spec.properties.component_ids}
            for i in range(3)
        )

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        # Constant pure volumes: dU/dT=dH/dT. This is not a generic liquid Cv.
        return sum(
            self.spec.properties.heat_capacity(n, state.temperature_K) for n in self.rows(state)
        )

    def internal_energy_j(self, state: CoupledStateView) -> float:
        return sum(
            self.spec.properties.enthalpy(n, state.temperature_K)
            - self.spec.properties.pressure_Pa * self.spec.volume_m3(n)
            for n in self.rows(state)
        )

    def chemical_energy_rate_w(
        self,
        state: CoupledStateView,
        derivatives: Mapping[str, float],
        contributions: Sequence[ProcessContribution],
    ) -> float:
        return sum(
            derivatives[_key(i, k)]
            * (
                self.spec.molar_enthalpies_J[k]
                - self.spec.properties.pressure_Pa * self.spec.molar_volumes_m3[k]
            )
            for i in range(3)
            for k in self.spec.properties.component_ids
        )


@dataclass(frozen=True)
class ReactiveWaterTreatmentAdvance:
    initial_state: TransportInventory
    final_state: TransportInventory
    actual_duration_s: float
    stopping_event: str | None
    solver_diagnostic: Mapping[str, Any]
    energy_balance_residual_J: float
    conservation_residuals: Mapping[str, float]
    trajectory: tuple[TransportInventory, ...]


def advance_reactive_water_treatment(
    spec: ReactiveWaterTreatmentSpec,
    initial: TransportInventory,
    requested_flow_m3_s: float,
    duration_s: float,
    *,
    evaluation_times_s: Sequence[float] | None = None,
) -> tuple[ReactiveWaterTreatmentAdvance, dict[str, Any]]:
    validated = reactive_water_treatment_inventory(spec, *initial.amounts_mol)
    if len(initial.enthalpies_J) != 3 or any(
        not isclose(a, b, rel_tol=1e-9, abs_tol=1e-7)
        for a, b in zip(initial.enthalpies_J, validated.enthalpies_J, strict=True)
    ):
        raise ValueError("original treatment H must match its declared same-temperature property")
    thermodynamics = _LiquidThermodynamics(spec)

    def point(state: CoupledStateView) -> dict[str, Any]:
        return reactive_water_treatment_profile(
            spec, thermodynamics.rows(state), requested_flow_m3_s
        )

    def bath_point(state: CoupledStateView) -> dict[str, Any]:
        # CoupledSystem exposes signed trial n to root functions. Constitutive
        # bath rates use the same zero-boundary extension as its RHS; volume
        # events below retain the original signed values. Accepted amounts and
        # H still undergo the original solver/property checks, never clipping.
        return point(
            replace(state, amounts_mol={k: max(v, 0.0) for k, v in state.amounts_mol.items()})
        )

    def process(state: CoupledStateView) -> ProcessContribution:
        p = point(state)
        return ProcessContribution(
            amount_rates_mol_s={_key(1, k): v for k, v in p["reaction_mol_s"].items()},
            reaction_progress_mol_s=p["reaction_progress_mol_s"],
            internal_transfers=tuple(
                InternalTransfer(f"{edge}:{k}", _key(edge, k), _key(edge + 1, k), p[field][k])
                for edge, field in ((0, "inlet_mol_s"), (1, "outlet_mol_s"))
                for k in spec.properties.component_ids
            ),
            coordinate_rates={
                **{f"H{i}": v for i, v in enumerate(p["enthalpy_rates_W"])},
                "throughput_m3": p["actual_flow_m3_s"],
                "hydraulic_J": p["hydraulic_power_W"],
                "bath_heating_J": max(p["bath_heat_into_liquids_W"], 0),
                "bath_cooling_J": max(-p["bath_heat_into_liquids_W"], 0),
            },
        )

    # Pickup/full inhibit flowing actions only; zero-flow batch reaction remains possible.
    flowing = point(
        CoupledState(
            {
                k: initial.amounts_mol[i][s]
                for i in range(3)
                for s in spec.properties.component_ids
                for k in [_key(i, s)]
            },
            spec.temperature_K,
            sum(1000 * spec.volume_m3(n) for n in initial.amounts_mol),
        )
    )
    events: list[CoupledEvent] = []
    if flowing["actual_flow_m3_s"] > 0:
        events.extend(
            (
                CoupledEvent(
                    "feed_pickup_exhausted",
                    lambda s: (
                        1000 * spec.volume_m3(thermodynamics.rows(s)[0]) - spec.pickup_volume_L
                    ),
                ),
                CoupledEvent(
                    "product_full",
                    lambda s: (
                        spec.capacities_L[2] - 1000 * spec.volume_m3(thermodynamics.rows(s)[2])
                    ),
                ),
            )
        )
    events.extend(
        (
            CoupledEvent(
                "bath_heating_limit",
                lambda s: spec.bath.maximum_power_W - bath_point(s)["bath_heat_into_liquids_W"],
            ),
            CoupledEvent(
                "bath_cooling_limit",
                lambda s: (
                    spec.bath.maximum_cooling_power_W + bath_point(s)["bath_heat_into_liquids_W"]
                ),
            ),
        )
    )
    coordinate_names = (
        "H0",
        "H1",
        "H2",
        "throughput_m3",
        "hydraulic_J",
        "bath_heating_J",
        "bath_cooling_J",
    )
    coordinates = tuple(
        CoupledCoordinate(
            k, "m3" if k == "throughput_m3" else "J", 1e-6 if k == "throughput_m3" else 1.0
        )
        for k in coordinate_names
    )
    element_names = sorted({e for s in spec.network.species for e in s.composition})
    laws = {
        e: {
            _key(i, s.species_id): s.composition.get(e, 0)
            for i in range(3)
            for s in spec.network.species
        }
        for e in element_names
    }
    system = CoupledSystem(
        thermodynamics,
        (CoupledProcess("reactive-treatment", process),),
        heat_flux=lambda s: HeatFlux(
            work_W=point(s)["shaft_power_W"], loss_W=point(s)["external_loss_W"]
        ),
        events=tuple(events),
        conservation_laws=laws,
        inventory_scales_mol={
            _key(i, k): max(sum(n[k] for n in initial.amounts_mol), 1e-3)
            for i in range(3)
            for k in spec.properties.component_ids
        },
        isothermal=True,
        policy=replace(DEFAULT_COUPLED_ODE_POLICY, rtol=1e-10),
        coordinates=coordinates,
    )
    start = CoupledState(
        {
            _key(i, k): n[k]
            for i, n in enumerate(initial.amounts_mol)
            for k in spec.properties.component_ids
        },
        spec.temperature_K,
        sum(1000 * spec.volume_m3(n) for n in initial.amounts_mol),
        initial.time_s,
        coordinates=dict.fromkeys(coordinate_names, 0.0),
    )
    result = system.advance(start, duration_s, evaluation_times_s=evaluation_times_s)

    def original(candidate: CoupledState) -> TransportInventory:
        n = thermodynamics.rows(candidate)
        h = tuple(initial.enthalpies_J[i] + candidate.coordinates[f"H{i}"] for i in range(3))
        for i, row in enumerate(n):
            if not isclose(
                h[i], spec.properties.enthalpy(row, spec.temperature_K), rel_tol=1e-9, abs_tol=3e-6
            ):
                raise RuntimeError("integrated original treatment H disagrees with same properties")
        return TransportInventory(
            n,
            h,
            candidate.time_s,
            initial.heater_energy_J + candidate.coordinates["bath_heating_J"],
            initial.condenser_energy_J + candidate.coordinates["bath_cooling_J"],
        )

    final = original(result.final_state)
    if result.energy_balance_residual_J is None:
        raise RuntimeError("physical treatment requires its original U energy balance")
    c = result.final_state.coordinates
    dh = sum(final.enthalpies_J) - sum(initial.enthalpies_J)
    diagnostic = {
        "elapsed_s": result.actual_duration_s,
        "requested_flow_m3_s": requested_flow_m3_s,
        "initial_operating_point": flowing,
        "final_operating_point": reactive_water_treatment_profile(
            spec, final.amounts_mol, requested_flow_m3_s
        ),
        "throughput_m3": c["throughput_m3"],
        "pump_work_J": result.final_state.work_J,
        "hydraulic_work_J": c["hydraulic_J"],
        "motor_loss_J": result.final_state.heat_loss_J,
        "bath_heat_into_liquids_J": result.final_state.boundary_heat_J,
        "bath_heating_J": c["bath_heating_J"],
        "bath_cooling_J": c["bath_cooling_J"],
        "enthalpy_change_J": dh,
        "boundary_work_J": 0.0,
        "enthalpy_residual_J": dh - result.final_state.boundary_heat_J - c["hydraulic_J"],
        "reaction_heat_model": "original additive H only; no extra reaction heat",
        "outlet_model": "same-P/elevation ideal zero-resistance overflow; zero holdup/KE",
    }
    return ReactiveWaterTreatmentAdvance(
        initial,
        final,
        result.actual_duration_s,
        result.stopping_event,
        result.solver_diagnostic,
        float(result.energy_balance_residual_J),
        result.conservation_residuals,
        tuple(original(s) for s in result.trajectory),
    ), diagnostic
