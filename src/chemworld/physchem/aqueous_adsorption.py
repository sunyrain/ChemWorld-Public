"""Five finite neutral aqueous tanks and one original persistent sorbent lot.

Ideal dilute water has explicit constant solvent volume and zero solute partial
volumes. Adsorbed volume is neglected. The mixed contactor and outlet have zero
hydraulic resistance; the inlet pipe and bounded pump determine the actual flow.
This is a mixed Langmuir/LDF contactor, not a GAC column or CPHSDM approximation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from math import isclose, isfinite
from typing import Any

from chemworld.foundation.state_ownership import StateSlot
from chemworld.physchem.aqueous_database import AqueousDatabase
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.physchem.aqueous_transport import AqueousTransportLaw
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
from chemworld.physchem.elements import molecular_weight
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.sorbent_kinetics import (
    SorbentKineticsSpec,
    accepted_sorbent_slot,
    make_sorbent_slot,
    sorbent_inventory,
    sorbent_key,
    sorbent_process,
    validate_sorbent_slot,
)
from chemworld.physchem.transport import (
    FluidState,
    PipeSpec,
    PumpSpec,
    pipe_pressure_drop,
    pump_operating_point,
)

SCHEMA = "finite-aqueous-adsorption-v1"


def _key(index: int, species: str) -> str:
    return f"liquid:{index}:{species}"


@dataclass(frozen=True)
class AqueousAdsorptionSpec:
    database: AqueousDatabase
    sorbent: SorbentKineticsSpec
    viscosity: AqueousTransportLaw
    bath: ThermalDeviceSpec
    pump: PumpSpec
    pipe: PipeSpec
    media_mass_kg: float
    capacities_L: tuple[float, ...] = (0.1, 0.005, 0.1, 0.1, 0.1)
    pickup_volume_L: float = 0.001
    pressure_Pa: float = 1e5
    maximum_solute_concentration_mol_L: float = 0.25
    contact_vessel_id: str = "adsorber"
    product_vessel_id: str = "product"
    rinse_vessel_id: str = "rinse"
    waste_vessel_id: str = "waste"
    media_equipment_id: str = "adsorption_media"

    def __post_init__(self) -> None:
        object.__setattr__(self, "database", AqueousDatabase.from_dict(self.database.to_dict()))
        object.__setattr__(self, "sorbent", SorbentKineticsSpec.from_dict(self.sorbent.to_dict()))
        object.__setattr__(self, "capacities_L", tuple(self.capacities_L))
        numbers = (
            *self.capacities_L,
            self.pickup_volume_L,
            self.pressure_Pa,
            self.maximum_solute_concentration_mol_L,
        )
        names = (
            self.contact_vessel_id,
            self.product_vessel_id,
            self.rinse_vessel_id,
            self.waste_vessel_id,
        )
        if (
            len(self.capacities_L) != 5
            or any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in numbers)
            or not isfinite(self.media_mass_kg)
            or self.media_mass_kg < 0
            or isinstance(self.media_mass_kg, bool)
            or len(set(names)) != 4
            or not all(names)
            or not self.media_equipment_id
            or self.pickup_volume_L >= min(self.capacities_L[0], self.capacities_L[3])
        ):
            raise ValueError(
                "aqueous adsorption requires five finite distinct tanks and a finite media lot"
            )
        if (
            self.database.reactions
            or self.database.activity_model.family != "ideal"
            or any(s.charge or s.phase not in {"solvent", "aqueous"} for s in self.database.species)
            or any(
                s.standard_state.pressure_model != "constant-volume" for s in self.database.species
            )
            or any(self.standards[k].molar_volume_m3_mol != 0 for k in self.solute_ids)
            or self.water_volume_m3_mol <= 0
            or not set(self.sorbent.component_ids) <= set(self.solute_ids)
        ):
            raise ValueError(
                "adsorption needs ideal neutral water, zero solute volume and no bulk reactions"
            )
        if (
            self.bath.mode != "isothermal"
            or not self.bath.enabled
            or self.pipe.elevation_change_m != 0
            or self.viscosity.unit != "Pa*s"
            or self.viscosity.source_id not in {s.source_id for s in self.database.sources}
            or self.maximum_solute_concentration_mol_L * 1000
            > self.viscosity.maximum_concentration_mol_m3
        ):
            raise ValueError("adsorption requires sourced viscosity, common T/P and zero elevation")
        self.sorbent.affinities(self.temperature_K)
        self.viscosity.evaluate(self.temperature_K, self.pressure_Pa, 0.0)

    @property
    def temperature_K(self) -> float:  # noqa: N802
        return self.bath.setpoint_K

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(s.species_id for s in self.database.species)

    @property
    def solute_ids(self) -> tuple[str, ...]:
        return tuple(s.species_id for s in self.database.species if s.phase == "aqueous")

    @property
    def water_id(self) -> str:
        return self.database.solvent_species.species_id

    @property
    def water_volume_m3_mol(self) -> float:
        value = self.database.solvent_species.standard_state.molar_volume_m3_mol
        if value is None:
            raise ValueError("actual solvent volume must be declared")
        return value

    @cached_property
    def standards(self) -> dict[str, Any]:
        return self.database.standard_states(self.temperature_K, self.pressure_Pa)

    @cached_property
    def masses_kg_mol(self) -> dict[str, float]:
        return {
            s.species_id: molecular_weight(dict(s.composition)) / 1000
            for s in self.database.species
        }

    def volume_m3(self, n: Mapping[str, float]) -> float:
        return n[self.water_id] * self.water_volume_m3_mol

    def enthalpy(self, n: Mapping[str, float]) -> float:
        if not any(n.values()):
            return 0.0
        return aqueous_phase_properties(
            self.database, n, temperature_K=self.temperature_K, pressure_Pa=self.pressure_Pa
        ).enthalpy_J

    def initial_slot(self, loading_mol_per_kg: Mapping[str, float] | None = None) -> StateSlot:
        return make_sorbent_slot(
            self.sorbent,
            owner_id=self.media_equipment_id,
            location_id=self.contact_vessel_id,
            mass_kg=self.media_mass_kg,
            initial_loading_mol_per_kg=loading_mol_per_kg,
        )

    def validate_slot(self, slot: StateSlot) -> None:
        validate_sorbent_slot(slot, self.sorbent)
        wanted = self.initial_slot()
        if (
            slot.spec != wanted.spec
            or slot.location_id != self.contact_vessel_id
            or slot.values["mass_kg"] != self.media_mass_kg
            or any(slot.values[k] != wanted.values[k] for k in ("sites_mol", "heat_capacity_J_K"))
        ):
            raise ValueError(
                "aqueous adsorption media must retain its original owner/location/declaration"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name
                not in {"database", "sorbent", "viscosity", "bath", "pump", "pipe", "capacities_L"}
            },
            "database": self.database.to_dict(),
            "sorbent": self.sorbent.to_dict(),
            "viscosity": self.viscosity.to_dict(),
            "bath": self.bath.to_dict(),
            "pump": asdict(self.pump),
            "pipe": asdict(self.pipe),
            "capacities_L": list(self.capacities_L),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AqueousAdsorptionSpec":
        data = dict(payload)
        if data.pop("schema", None) != SCHEMA:
            raise ValueError("unsupported aqueous adsorption schema")
        return cls(
            **{
                **data,
                "database": AqueousDatabase.from_dict(data["database"]),
                "sorbent": SorbentKineticsSpec.from_dict(data["sorbent"]),
                "viscosity": AqueousTransportLaw(**data["viscosity"]),
                "bath": ThermalDeviceSpec.from_dict(data["bath"]),
                "pump": PumpSpec(**data["pump"]),
                "pipe": PipeSpec(**data["pipe"]),
            }
        )


@dataclass(frozen=True)
class AqueousAdsorptionInventory:
    amounts_mol: tuple[dict[str, float], ...]
    enthalpies_J: tuple[float, ...]
    sorbent_slot: StateSlot
    time_s: float = 0.0


def aqueous_adsorption_inventory(
    spec: AqueousAdsorptionSpec,
    amounts: Sequence[Mapping[str, float]],
    *,
    sorbent_slot: StateSlot | None = None,
    initial_loading_mol_per_kg: Mapping[str, float] | None = None,
    time_s: float = 0.0,
) -> AqueousAdsorptionInventory:
    if len(amounts) != 5 or not isfinite(time_s) or time_s < 0:
        raise ValueError("aqueous adsorption needs five original inventories and finite time")
    if sorbent_slot is not None and initial_loading_mol_per_kg is not None:
        raise ValueError("cannot reinitialize an existing sorbent lot")
    rows = tuple(dict(n) for n in amounts)
    for i, n in enumerate(rows):
        if set(n) != set(spec.component_ids) or any(
            isinstance(v, bool) or not isfinite(v) or v < 0 for v in n.values()
        ):
            raise ValueError("aqueous adsorption needs declared nonnegative species")
        volume = spec.volume_m3(n) * 1000
        if (
            volume > spec.capacities_L[i] * (1 + 1e-9)
            or (i in (0, 3) and volume < spec.pickup_volume_L * (1 - 1e-9))
            or (i == 1 and not isclose(volume, spec.capacities_L[1], rel_tol=1e-9, abs_tol=1e-12))
            or (volume == 0 and any(n.values()))
            or (
                volume > 0
                and sum(n[k] for k in spec.solute_ids) / volume
                > spec.maximum_solute_concentration_mol_L * (1 + 1e-9)
            )
        ):
            raise ValueError("aqueous adsorption exceeds finite volume/holdup/concentration domain")
    slot = spec.initial_slot(initial_loading_mol_per_kg) if sorbent_slot is None else sorbent_slot
    spec.validate_slot(slot)
    return AqueousAdsorptionInventory(rows, tuple(spec.enthalpy(n) for n in rows), slot, time_s)


@dataclass(frozen=True)
class _Thermodynamics:
    spec: AqueousAdsorptionSpec
    slot: StateSlot

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return (
            *(_key(i, k) for i in range(5) for k in self.spec.component_ids),
            *(sorbent_key(self.slot, k) for k in self.spec.sorbent.component_ids),
        )

    def species_energy_j_mol(self, key: str, temperature_K: float) -> float:
        standards = (
            self.spec.standards
            if temperature_K == self.spec.temperature_K
            else self.spec.database.standard_states(temperature_K, self.spec.pressure_Pa)
        )
        for k in self.spec.sorbent.component_ids:
            if key == sorbent_key(self.slot, k):
                return standards[k].enthalpy_J_mol + self.spec.sorbent.adsorption_energy_J_mol[k]
        for i in range(5):
            for k in self.spec.component_ids:
                if key == _key(i, k):
                    volume = self.spec.water_volume_m3_mol if k == self.spec.water_id else 0.0
                    return standards[k].enthalpy_J_mol - self.spec.pressure_Pa * volume
        raise ValueError("unknown aqueous adsorption caloric inventory")

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        standards = (
            self.spec.standards
            if state.temperature_K == self.spec.temperature_K
            else self.spec.database.standard_states(state.temperature_K, self.spec.pressure_Pa)
        )
        return (
            sum(
                state.amounts_mol[_key(i, k)] * standards[k].heat_capacity_J_mol_K
                for i in range(5)
                for k in self.spec.component_ids
            )
            + sum(
                state.amounts_mol[sorbent_key(self.slot, k)] * standards[k].heat_capacity_J_mol_K
                for k in self.spec.sorbent.component_ids
            )
            + self.slot.values["heat_capacity_J_K"]
        )

    def internal_energy_j(self, state: CoupledStateView) -> float:
        return sum(
            n * self.species_energy_j_mol(k, state.temperature_K)
            for k, n in state.amounts_mol.items()
        ) + self.slot.values["heat_capacity_J_K"] * (
            state.temperature_K - self.spec.sorbent.reference_temperature_K
        )

    def chemical_energy_rate_w(
        self,
        state: CoupledStateView,
        derivatives: Mapping[str, float],
        contributions: Sequence[ProcessContribution],
    ) -> float:
        return sum(
            v * self.species_energy_j_mol(k, state.temperature_K) for k, v in derivatives.items()
        )

    def rows(self, state: CoupledStateView) -> tuple[dict[str, float], ...]:
        return tuple(
            {k: state.amounts_mol[_key(i, k)] for k in self.spec.component_ids} for i in range(5)
        )


def _operating_point(
    spec: AqueousAdsorptionSpec, rows: Sequence[Mapping[str, float]], requested: float, path: str
) -> dict[str, Any]:
    if (
        path not in {"process", "rinse"}
        or isinstance(requested, bool)
        or not isfinite(requested)
        or requested < 0
    ):
        raise ValueError("adsorption needs process/rinse and a nonnegative finite flow request")
    source, target = (0, 2) if path == "process" else (3, 4)
    n = rows[source]
    volume = spec.volume_m3(n)
    concentration = sum(n[k] for k in spec.solute_ids) / volume
    viscosity = spec.viscosity.evaluate(spec.temperature_K, spec.pressure_Pa, concentration)
    mass = sum(n[k] * spec.masses_kg_mol[k] for k in spec.component_ids)
    capacity = sum(n[k] * spec.standards[k].heat_capacity_J_mol_K for k in spec.component_ids)
    fluid = FluidState(mass / volume, viscosity, capacity / mass)
    point = pump_operating_point(
        spec.pump,
        requested,
        lambda q: (
            pipe_pressure_drop(spec.pipe, fluid, volumetric_flow_m3_s=q).pressure_drop_total_Pa
        ),
    )
    q = point["actual_flow_m3_s"]
    return {
        **point,
        "flow_path": path,
        "source_index": source,
        "target_index": target,
        "inlet_mol_s": {k: n[k] * q / volume for k in spec.component_ids},
        "outlet_mol_s": {k: rows[1][k] * q / spec.volume_m3(rows[1]) for k in spec.component_ids},
    }


def _initial_view(
    spec: AqueousAdsorptionSpec,
    inventory: AqueousAdsorptionInventory,
    coordinates: Mapping[str, float] | None = None,
) -> CoupledState:
    return CoupledState(
        {
            **{
                _key(i, k): n[k]
                for i, n in enumerate(inventory.amounts_mol)
                for k in spec.component_ids
            },
            **sorbent_inventory(inventory.sorbent_slot, spec.sorbent),
        },
        spec.temperature_K,
        spec.capacities_L[1],
        inventory.time_s,
        coordinates={} if coordinates is None else coordinates,
    )


def aqueous_adsorption_profile(
    spec: AqueousAdsorptionSpec,
    inventory: AqueousAdsorptionInventory,
    requested_flow_m3_s: float,
    flow_path: str = "process",
) -> dict[str, Any]:
    spec.validate_slot(inventory.sorbent_slot)
    thermo = _Thermodynamics(spec, inventory.sorbent_slot)
    contact = sorbent_process(
        inventory.sorbent_slot,
        spec.sorbent,
        fluid_keys={k: _key(1, k) for k in spec.sorbent.component_ids},
        thermodynamics=thermo,
    )
    state = _initial_view(spec, inventory)
    contribution = contact.evaluate(state)
    rates = dict.fromkeys(spec.sorbent.component_ids, 0.0)
    for transfer in contribution.internal_transfers:
        for k in rates:
            if transfer.destination_key == sorbent_key(inventory.sorbent_slot, k):
                rates[k] += transfer.rate_mol_s
            elif transfer.source_key == sorbent_key(inventory.sorbent_slot, k):
                rates[k] -= transfer.rate_mol_s
    point = _operating_point(spec, inventory.amounts_mol, requested_flow_m3_s, flow_path)
    return {
        **point,
        "adsorption_rates_mol_s": rates,
        "bath_heat_into_system_W": sum(
            rates[k] * spec.sorbent.adsorption_energy_J_mol[k] for k in rates
        )
        - point["hydraulic_power_W"],
        "volumes_L": [spec.volume_m3(n) * 1000 for n in inventory.amounts_mol],
    }


@dataclass(frozen=True)
class AqueousAdsorptionAdvance:
    initial_state: AqueousAdsorptionInventory
    final_state: AqueousAdsorptionInventory
    actual_duration_s: float
    stopping_event: str | None
    solver_diagnostic: Mapping[str, Any]
    energy_balance_residual_J: float
    conservation_residuals: Mapping[str, float]


def advance_aqueous_adsorption(
    spec: AqueousAdsorptionSpec,
    initial: AqueousAdsorptionInventory,
    requested_flow_m3_s: float,
    duration_s: float,
    *,
    flow_path: str = "process",
) -> tuple[AqueousAdsorptionAdvance, dict[str, Any]]:
    validated = aqueous_adsorption_inventory(
        spec, initial.amounts_mol, sorbent_slot=initial.sorbent_slot, time_s=initial.time_s
    )
    if len(initial.enthalpies_J) != 5 or any(
        not isclose(a, b, rel_tol=1e-9, abs_tol=1e-7)
        for a, b in zip(initial.enthalpies_J, validated.enthalpies_J, strict=True)
    ):
        raise ValueError("original aqueous liquid H differs from declared same-source calorics")
    thermo = _Thermodynamics(spec, initial.sorbent_slot)
    original_contact = sorbent_process(
        initial.sorbent_slot,
        spec.sorbent,
        fluid_keys={k: _key(1, k) for k in spec.sorbent.component_ids},
        thermodynamics=thermo,
    )

    def evaluate(state: CoupledStateView) -> tuple[ProcessContribution, dict[str, Any]]:
        rows = thermo.rows(state)
        point = _operating_point(spec, rows, requested_flow_m3_s, flow_path)
        source, target = point["source_index"], point["target_index"]
        adsorption = original_contact.evaluate(state)
        transfers = (
            *adsorption.internal_transfers,
            *(
                InternalTransfer(f"inlet:{k}", _key(source, k), _key(1, k), point["inlet_mol_s"][k])
                for k in spec.component_ids
            ),
            *(
                InternalTransfer(
                    f"outlet:{k}", _key(1, k), _key(target, k), point["outlet_mol_s"][k]
                )
                for k in spec.component_ids
            ),
        )
        dn = dict.fromkeys(thermo.inventory_keys, 0.0)
        for transfer in transfers:
            dn[transfer.source_key] -= transfer.rate_mol_s
            dn[transfer.destination_key] += transfer.rate_mol_s
        dh = [
            sum(dn[_key(i, k)] * spec.standards[k].enthalpy_J_mol for k in spec.component_ids)
            for i in range(5)
        ]
        bath = thermo.chemical_energy_rate_w(state, dn, ()) - point["hydraulic_power_W"]
        contribution = ProcessContribution(
            internal_transfers=transfers,
            coordinate_rates={
                **{f"H{i}": v for i, v in enumerate(dh)},
                "throughput_m3": point["actual_flow_m3_s"],
                "hydraulic_J": point["hydraulic_power_W"],
                "bath_heating_J": max(bath, 0.0),
                "bath_cooling_J": max(-bath, 0.0),
            },
        )
        return contribution, {**point, "bath_heat_into_system_W": bath}

    def bath_rate(state: CoupledStateView) -> float:
        # Same transient zero-boundary extension as the original coupled RHS;
        # signed water/capacity event coordinates are not changed.
        view = replace(state, amounts_mol={k: max(v, 0.0) for k, v in state.amounts_mol.items()})
        return float(evaluate(view)[1]["bath_heat_into_system_W"])

    names = (
        *(f"H{i}" for i in range(5)),
        "throughput_m3",
        "hydraulic_J",
        "bath_heating_J",
        "bath_cooling_J",
    )
    start = _initial_view(spec, initial, dict.fromkeys(names, 0.0))
    first = evaluate(start)[1]
    source, target = first["source_index"], first["target_index"]
    events: list[CoupledEvent] = []
    if first["actual_flow_m3_s"] > 0:
        events.extend(
            (
                CoupledEvent(
                    f"{flow_path}_source_pickup",
                    lambda s: 1000 * spec.volume_m3(thermo.rows(s)[source]) - spec.pickup_volume_L,
                ),
                CoupledEvent(
                    f"{flow_path}_collector_full",
                    lambda s: (
                        spec.capacities_L[target] - 1000 * spec.volume_m3(thermo.rows(s)[target])
                    ),
                ),
            )
        )
    events.extend(
        (
            CoupledEvent("bath_heating_limit", lambda s: spec.bath.maximum_power_W - bath_rate(s)),
            CoupledEvent(
                "bath_cooling_limit", lambda s: spec.bath.maximum_cooling_power_W + bath_rate(s)
            ),
        )
    )
    # Total neutral concentration may rise during desorption, unlike fixed source composition.
    events.append(
        CoupledEvent(
            "contact_concentration_limit",
            lambda s: (
                spec.maximum_solute_concentration_mol_L * spec.capacities_L[1]
                - sum(s.amounts_mol[_key(1, k)] for k in spec.solute_ids)
            ),
        )
    )
    laws = {
        k: {
            **{_key(i, k): 1.0 for i in range(5)},
            **(
                {sorbent_key(initial.sorbent_slot, k): 1.0}
                if k in spec.sorbent.component_ids
                else {}
            ),
        }
        for k in spec.component_ids
    }
    system = CoupledSystem(
        thermo,
        (
            CoupledProcess(
                "aqueous-adsorption-flow",
                lambda s: evaluate(s)[0],
                validate_state=original_contact.validate_state,
            ),
        ),
        heat_flux=lambda s: HeatFlux(
            work_W=_operating_point(spec, thermo.rows(s), requested_flow_m3_s, flow_path)[
                "shaft_power_W"
            ],
            loss_W=_operating_point(spec, thermo.rows(s), requested_flow_m3_s, flow_path)[
                "external_loss_W"
            ],
        ),
        events=tuple(events),
        conservation_laws=laws,
        inventory_scales_mol={k: max(v, 1e-3) for k, v in start.amounts_mol.items()},
        coordinates=tuple(
            CoupledCoordinate(
                k, "m3" if k == "throughput_m3" else "J", 1e-6 if k == "throughput_m3" else 1.0
            )
            for k in names
        ),
        isothermal=True,
        policy=replace(DEFAULT_COUPLED_ODE_POLICY, rtol=1e-10),
    )
    result = system.advance(start, duration_s)
    endpoint = result.final_state
    amounts = thermo.rows(endpoint)
    enthalpies = tuple(initial.enthalpies_J[i] + endpoint.coordinates[f"H{i}"] for i in range(5))
    for n, h in zip(amounts, enthalpies, strict=True):
        if not isclose(h, spec.enthalpy(n), rel_tol=1e-9, abs_tol=3e-6):
            raise RuntimeError("integrated original aqueous H disagrees with same-source endpoint")
    slot = accepted_sorbent_slot(initial.sorbent_slot, spec.sorbent, endpoint)
    aqueous_adsorption_inventory(spec, amounts, sorbent_slot=slot, time_s=endpoint.time_s)
    if result.energy_balance_residual_J is None:
        raise RuntimeError("physical aqueous adsorption needs its original U balance")
    final = AqueousAdsorptionInventory(amounts, enthalpies, slot, endpoint.time_s)
    c = endpoint.coordinates
    diagnostic = {
        "flow_path": flow_path,
        "actual_duration_s": result.actual_duration_s,
        "initial_operating_point": first,
        "final_operating_point": aqueous_adsorption_profile(
            spec, final, requested_flow_m3_s, flow_path
        ),
        "throughput_m3": c["throughput_m3"],
        "pump_work_J": endpoint.work_J,
        "hydraulic_work_J": c["hydraulic_J"],
        "motor_loss_J": endpoint.heat_loss_J,
        "bath_heat_into_system_J": endpoint.boundary_heat_J,
        "bath_heating_J": c["bath_heating_J"],
        "bath_cooling_J": c["bath_cooling_J"],
        "internal_energy_change_J": thermo.internal_energy_j(endpoint)
        - thermo.internal_energy_j(start),
        "net_adsorbed_mol": {
            k: slot.values[f"loading:{k}"] - initial.sorbent_slot.values[f"loading:{k}"]
            for k in spec.sorbent.component_ids
        },
        "bath_owner": "single whole-group bath at feed anchor",
        "boundary_work_J": 0.0,
        "hydraulic_scope": "inlet pipe only; mixed contactor and overflow have zero resistance",
    }
    return AqueousAdsorptionAdvance(
        initial,
        final,
        result.actual_duration_s,
        result.stopping_event,
        result.solver_diagnostic,
        float(result.energy_balance_residual_J),
        result.conservation_residuals,
    ), diagnostic
