"""Finite Na/Ca exchange with the original liquid stocks and one charged resin lot.

Gaines--Thomas equivalent fractions and ideal molality activities determine
one reversible exchange rate. Six finite liquids share an isothermal bath;
only the mixed contact holdup is constant. No axial bed or backwash model.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from math import exp, isclose, isfinite, log
from typing import Any

from chemworld.foundation.state_ownership import (
    CapacityConstraint,
    StateFieldSpec,
    StateSlot,
    StateSlotSpec,
)
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
from chemworld.physchem.multiion_membrane_properties import MultiIonMembraneProperties
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.transport import (
    FluidState,
    PipeSpec,
    PumpSpec,
    pipe_pressure_drop,
    pump_operating_point,
)

SCHEMA = "finite-cation-exchange-v1"
RESIN_STATE_TYPE = "fixed-cation-exchange-v1"
FARADAY = 96485.33212331001
ROUTES = {"process": (0, 2), "regenerate": (3, 5), "rinse": (4, 5)}


def _key(index: int, species: str) -> str:
    return f"liquid:{index}:{species}"


@dataclass(frozen=True)
class CationExchangeSpec:
    properties: MultiIonMembraneProperties
    bath: ThermalDeviceSpec
    pump: PumpSpec
    pipe: PipeSpec
    media_mass_kg: float
    charge_capacity_mol_kg: float
    reference_selectivity: float
    exchange_enthalpy_J_mol: float
    rate_coefficient_s_inv: float
    source_id: str
    reference_temperature_K: float = 298.15
    carrier_heat_capacity_J_kg_K: float = 1000.0
    capacities_L: tuple[float, ...] = (0.1, 0.005, 0.1, 0.1, 0.1, 0.1)
    pickup_volume_L: float = 0.001
    pressure_Pa: float = 1e5
    maximum_ion_concentration_mol_L: float = 0.25
    sodium_id: str = "Na"
    calcium_id: str = "Ca"
    chloride_id: str = "Cl"
    contact_vessel_id: str = "exchange_contact"
    product_vessel_id: str = "product"
    regenerant_vessel_id: str = "regenerant"
    rinse_vessel_id: str = "rinse"
    waste_vessel_id: str = "waste"
    media_equipment_id: str = "exchange_resin"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "properties", MultiIonMembraneProperties.from_dict(self.properties.to_dict())
        )
        object.__setattr__(self, "capacities_L", tuple(self.capacities_L))
        positive = (
            *self.capacities_L,
            self.pickup_volume_L,
            self.pressure_Pa,
            self.maximum_ion_concentration_mol_L,
            self.media_mass_kg,
            self.charge_capacity_mol_kg,
            self.reference_selectivity,
            self.reference_temperature_K,
        )
        if (
            len(self.capacities_L) != 6
            or any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in positive)
            or any(
                isinstance(v, bool) or not isfinite(v) or v < 0
                for v in (self.rate_coefficient_s_inv, self.carrier_heat_capacity_J_kg_K)
            )
            or not isfinite(self.exchange_enthalpy_J_mol)
            or self.pickup_volume_L >= min(self.capacities_L[i] for i in (0, 3, 4))
            or len(set(self.member_ids)) != 5
            or not all(self.member_ids)
            or not self.media_equipment_id
        ):
            raise ValueError(
                "cation exchange needs six finite distinct tanks and a charged resin lot"
            )
        expected = {
            self.sodium_id: ({"Na": 1.0}, 1),
            self.calcium_id: ({"Ca": 1.0}, 2),
            self.chloride_id: ({"Cl": 1.0}, -1),
        }
        db = self.database
        if (
            db.activity_model.family != "ideal"
            or {
                s.species_id: (dict(s.composition), s.charge)
                for s in db.species
                if s.phase == "aqueous"
            }
            != expected
            or dict(db.solvent_species.composition) != {"H": 2.0, "O": 1.0}
            or any(self.standards[k].molar_volume_m3_mol != 0 for k in self.ion_ids)
            or self.source_id not in {s.source_id for s in db.sources}
            or self.bath.mode != "isothermal"
            or not self.bath.enabled
            or self.pipe.elevation_change_m != 0
            or self.maximum_ion_concentration_mol_L * 1000
            > self.properties.viscosity.maximum_concentration_mol_m3
            or not self.properties.temperature_bounds_K[0]
            <= self.reference_temperature_K
            <= self.properties.temperature_bounds_K[1]
        ):
            raise ValueError(
                "cation exchange requires sourced ideal Na/Ca/Cl water, "
                "zero ion volume and common T/P"
            )
        self.selectivity(self.temperature_K)
        self.properties.viscosity.evaluate(self.temperature_K, self.pressure_Pa, 0.0)

    @property
    def member_ids(self) -> tuple[str, ...]:
        return (
            self.contact_vessel_id,
            self.product_vessel_id,
            self.regenerant_vessel_id,
            self.rinse_vessel_id,
            self.waste_vessel_id,
        )

    @property
    def database(self) -> Any:
        return self.properties.database

    @property
    def temperature_K(self) -> float:  # noqa: N802
        return self.bath.setpoint_K

    @property
    def component_ids(self) -> tuple[str, ...]:
        return self.properties.component_ids

    @property
    def ion_ids(self) -> tuple[str, ...]:
        return self.properties.ion_ids

    @property
    def water_id(self) -> str:
        return self.properties.water_id

    @property
    def charge_equivalents_mol(self) -> float:
        return self.media_mass_kg * self.charge_capacity_mol_kg

    @property
    def exchange_entropy_J_mol_K(self) -> float:  # noqa: N802
        return self.exchange_enthalpy_J_mol / self.reference_temperature_K + R * log(
            self.reference_selectivity
        )

    @cached_property
    def standards(self) -> dict[str, Any]:
        return self.database.standard_states(self.temperature_K, self.pressure_Pa)

    @cached_property
    def masses_kg_mol(self) -> dict[str, float]:
        return {
            s.species_id: molecular_weight(dict(s.composition)) / 1000
            for s in self.database.species
        }

    def selectivity(self, temperature_K: float) -> float:
        if (
            not self.properties.temperature_bounds_K[0]
            <= temperature_K
            <= self.properties.temperature_bounds_K[1]
        ):
            raise ValueError("cation selectivity leaves its declared temperature domain")
        value = self.reference_selectivity * exp(
            -self.exchange_enthalpy_J_mol
            / R
            * (1 / temperature_K - 1 / self.reference_temperature_K)
        )
        if not isfinite(value) or value <= 0:
            raise ValueError("cation selectivity must remain finite and positive")
        return value

    def volume_m3(self, n: Mapping[str, float]) -> float:
        return n[self.water_id] * self.properties.water_volume_m3_mol

    def enthalpy(self, n: Mapping[str, float]) -> float:
        return self.properties.enthalpy(n, self.temperature_K, self.pressure_Pa)

    def initial_slot(self, calcium_equivalent_fraction: float = 0.0) -> StateSlot:
        f = calcium_equivalent_fraction
        if isinstance(f, bool) or not isfinite(f) or not 0 <= f <= 1:
            raise ValueError("initial calcium equivalent fraction must be in [0,1]")
        q = self.charge_equivalents_mol
        fields = (
            StateFieldSpec("mass_kg", "kg", conserved_quantity=f"{self.media_equipment_id}:mass"),
            StateFieldSpec(
                "fixed_charge_eq_mol",
                "mol",
                conserved_quantity=f"{self.media_equipment_id}:charge_equivalents",
            ),
            StateFieldSpec("heat_capacity_J_K", "J/K"),
            StateFieldSpec("bound_sodium_mol", "mol", species_id=self.sodium_id),
            StateFieldSpec("bound_calcium_mol", "mol", species_id=self.calcium_id),
        )
        declaration = StateSlotSpec(
            f"{self.media_equipment_id}:resin",
            "equipment",
            self.media_equipment_id,
            RESIN_STATE_TYPE,
            fields,
            mechanism_id="gaines-thomas-Na-Ca-v1",
            batch_reset="preserve",
            heat_capacity_field="heat_capacity_J_K",
            capacities=(
                CapacityConstraint(
                    "fixed_charge_eq_mol", (("bound_sodium_mol", 1.0), ("bound_calcium_mol", 2.0))
                ),
            ),
        )
        return StateSlot(
            declaration,
            self.contact_vessel_id,
            {
                "mass_kg": self.media_mass_kg,
                "fixed_charge_eq_mol": q,
                "heat_capacity_J_K": self.media_mass_kg * self.carrier_heat_capacity_J_kg_K,
                "bound_sodium_mol": q * (1 - f),
                "bound_calcium_mol": q * f / 2,
            },
        )

    def validate_slot(self, slot: StateSlot) -> None:
        wanted = self.initial_slot()
        if (
            slot.spec != wanted.spec
            or slot.location_id != self.contact_vessel_id
            or any(
                slot.values[k] != wanted.values[k]
                for k in ("mass_kg", "fixed_charge_eq_mol", "heat_capacity_J_K")
            )
        ):
            raise ValueError("original resin owner/location/charge/carrier declaration differs")
        for values in (slot.values, slot.initial_values):
            if values is None or not isclose(
                values["bound_sodium_mol"] + 2 * values["bound_calcium_mol"],
                values["fixed_charge_eq_mol"],
                rel_tol=1e-10,
                abs_tol=1e-14,
            ):
                raise ValueError("resin counterions must exactly neutralize its fixed charge")

    def bound_key(self, species: str) -> str:
        return f"{self.media_equipment_id}:resin:{species}"

    def to_dict(self) -> dict[str, Any]:
        special = {"properties", "bath", "pump", "pipe", "capacities_L"}
        return {
            "schema": SCHEMA,
            **{k: getattr(self, k) for k in self.__dataclass_fields__ if k not in special},
            "properties": self.properties.to_dict(),
            "bath": self.bath.to_dict(),
            "pump": asdict(self.pump),
            "pipe": asdict(self.pipe),
            "capacities_L": list(self.capacities_L),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CationExchangeSpec":
        data = dict(payload)
        if data.pop("schema", None) != SCHEMA:
            raise ValueError("unsupported cation exchange schema")
        return cls(
            **{
                **data,
                "properties": MultiIonMembraneProperties.from_dict(data["properties"]),
                "bath": ThermalDeviceSpec.from_dict(data["bath"]),
                "pump": PumpSpec(**data["pump"]),
                "pipe": PipeSpec(**data["pipe"]),
            }
        )


@dataclass(frozen=True)
class CationExchangeInventory:
    amounts_mol: tuple[dict[str, float], ...]
    enthalpies_J: tuple[float, ...]
    resin_slot: StateSlot
    time_s: float = 0.0


def cation_exchange_inventory(
    spec: CationExchangeSpec,
    amounts: Sequence[Mapping[str, float]],
    *,
    resin_slot: StateSlot | None = None,
    initial_calcium_equivalent_fraction: float | None = None,
    time_s: float = 0.0,
) -> CationExchangeInventory:
    if len(amounts) != 6 or not isfinite(time_s) or time_s < 0:
        raise ValueError("cation exchange requires six original finite inventories and time")
    if resin_slot is not None and initial_calcium_equivalent_fraction is not None:
        raise ValueError("cannot reset an existing resin lot")
    rows = tuple(dict(n) for n in amounts)
    for i, n in enumerate(rows):
        spec.properties.validate_amounts(n, empty=i in (2, 5))
        if any(isinstance(v, bool) for v in n.values()):
            raise ValueError("ion amounts cannot be boolean")
        v = spec.volume_m3(n) * 1000
        if (
            v > spec.capacities_L[i] * (1 + 1e-9)
            or (i in (0, 3, 4) and v < spec.pickup_volume_L * (1 - 1e-9))
            or (i == 1 and not isclose(v, spec.capacities_L[1], rel_tol=1e-9, abs_tol=1e-12))
            or (
                v > 0
                and sum(n[k] for k in spec.ion_ids) / v
                > spec.maximum_ion_concentration_mol_L * (1 + 1e-9)
            )
        ):
            raise ValueError("cation liquid exceeds finite volume/holdup/concentration domain")
    slot = (
        spec.initial_slot(
            0.0
            if initial_calcium_equivalent_fraction is None
            else initial_calcium_equivalent_fraction
        )
        if resin_slot is None
        else resin_slot
    )
    spec.validate_slot(slot)
    return CationExchangeInventory(rows, tuple(spec.enthalpy(n) for n in rows), slot, time_s)


@dataclass(frozen=True)
class _Thermodynamics:
    spec: CationExchangeSpec
    slot: StateSlot

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return (
            *(_key(i, k) for i in range(6) for k in self.spec.component_ids),
            self.spec.bound_key(self.spec.sodium_id),
            self.spec.bound_key(self.spec.calcium_id),
        )

    def species_energy_j_mol(self, key: str, temperature_K: float) -> float:
        s = self.spec
        standards = (
            s.standards
            if temperature_K == s.temperature_K
            else s.database.standard_states(temperature_K, s.pressure_Pa)
        )
        for k, offset in ((s.sodium_id, 0.0), (s.calcium_id, s.exchange_enthalpy_J_mol)):
            if key == s.bound_key(k):
                return standards[k].enthalpy_J_mol + offset
        for i in range(6):
            for k in s.component_ids:
                if key == _key(i, k):
                    return standards[k].enthalpy_J_mol - (
                        s.pressure_Pa * s.properties.water_volume_m3_mol if k == s.water_id else 0.0
                    )
        raise ValueError("unknown original resin/liquid caloric inventory")

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        s = self.spec
        standards = (
            s.standards
            if state.temperature_K == s.temperature_K
            else s.database.standard_states(state.temperature_K, s.pressure_Pa)
        )
        return (
            sum(
                state.amounts_mol[_key(i, k)] * standards[k].heat_capacity_J_mol_K
                for i in range(6)
                for k in s.component_ids
            )
            + sum(
                state.amounts_mol[s.bound_key(k)] * standards[k].heat_capacity_J_mol_K
                for k in (s.sodium_id, s.calcium_id)
            )
            + self.slot.values["heat_capacity_J_K"]
        )

    def internal_energy_j(self, state: CoupledStateView) -> float:
        return sum(
            n * self.species_energy_j_mol(k, state.temperature_K)
            for k, n in state.amounts_mol.items()
        ) + self.slot.values["heat_capacity_J_K"] * (
            state.temperature_K - self.spec.reference_temperature_K
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
            {k: state.amounts_mol[_key(i, k)] for k in self.spec.component_ids} for i in range(6)
        )


def _initial_view(
    spec: CationExchangeSpec,
    initial: CationExchangeInventory,
    coordinates: Mapping[str, float] | None = None,
) -> CoupledState:
    return CoupledState(
        {
            **{
                _key(i, k): n[k]
                for i, n in enumerate(initial.amounts_mol)
                for k in spec.component_ids
            },
            spec.bound_key(spec.sodium_id): initial.resin_slot.values["bound_sodium_mol"],
            spec.bound_key(spec.calcium_id): initial.resin_slot.values["bound_calcium_mol"],
        },
        spec.temperature_K,
        spec.capacities_L[1],
        initial.time_s,
        coordinates={} if coordinates is None else coordinates,
    )


def _exchange(spec: CationExchangeSpec, state: CoupledStateView) -> dict[str, Any]:
    q = spec.charge_equivalents_mol
    na, ca = (state.amounts_mol[spec.bound_key(k)] for k in (spec.sodium_id, spec.calcium_id))
    kg = state.amounts_mol[_key(1, spec.water_id)] * spec.properties.water_mass_kg_mol
    ana, aca = (state.amounts_mol[_key(1, k)] / kg for k in (spec.sodium_id, spec.calcium_id))
    tna, tca = na / q, 2 * ca / q
    selectivity = spec.selectivity(state.temperature_K)
    forward, reverse = aca * tna * tna, ana * ana * tca / selectivity
    rate = q * spec.rate_coefficient_s_inv * (forward - reverse)
    affinity = (
        R * state.temperature_K * log(reverse / forward) if forward > 0 and reverse > 0 else None
    )
    # On the constant-Q carrier, d[n_i(ln(theta_i)-1)]/dn_i=ln(theta_i).
    mix_g = (
        R
        * state.temperature_K
        * sum(n * (log(theta) - 1) for n, theta in ((na, tna), (ca, tca)) if n > 0)
    )
    return {
        "sodium_equivalent_fraction": tna,
        "calcium_equivalent_fraction": tca,
        "sodium_activity": ana,
        "calcium_activity": aca,
        "selectivity": selectivity,
        "forward_activity_product": forward,
        "reverse_activity_product": reverse,
        "exchange_rate_mol_s": rate,
        "exchange_gibbs_J_mol": affinity,
        "dissipation_W": None if affinity is None else -rate * affinity,
        "resin_mixing_gibbs_J": mix_g,
        "fixed_resin_charge_C": -FARADAY * q,
        "bound_counterion_charge_C": FARADAY * (na + 2 * ca),
    }


def _operating_point(
    spec: CationExchangeSpec, rows: Sequence[Mapping[str, float]], requested: float, path: str
) -> dict[str, Any]:
    if (
        path not in ROUTES
        or isinstance(requested, bool)
        or not isfinite(requested)
        or requested < 0
    ):
        raise ValueError("cation exchange requires process/regenerate/rinse and nonnegative flow")
    source, target = ROUTES[path]
    n = rows[source]
    v = spec.volume_m3(n)
    mu = spec.properties.viscosity.evaluate(
        spec.temperature_K, spec.pressure_Pa, sum(n[k] for k in spec.ion_ids) / v
    )
    mass = sum(n[k] * spec.masses_kg_mol[k] for k in spec.component_ids)
    cp = sum(n[k] * spec.standards[k].heat_capacity_J_mol_K for k in spec.component_ids)
    fluid = FluidState(mass / v, mu, cp / mass)
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
        "inlet_mol_s": {k: n[k] * q / v for k in spec.component_ids},
        "outlet_mol_s": {k: rows[1][k] * q / spec.volume_m3(rows[1]) for k in spec.component_ids},
    }


def cation_exchange_profile(
    spec: CationExchangeSpec,
    inventory: CationExchangeInventory,
    requested_flow_m3_s: float,
    flow_path: str = "process",
) -> dict[str, Any]:
    spec.validate_slot(inventory.resin_slot)
    point = _operating_point(spec, inventory.amounts_mol, requested_flow_m3_s, flow_path)
    exchange = _exchange(spec, _initial_view(spec, inventory))
    return {
        **point,
        **exchange,
        "bath_heat_into_system_W": exchange["exchange_rate_mol_s"] * spec.exchange_enthalpy_J_mol
        - point["hydraulic_power_W"],
        "volumes_L": [spec.volume_m3(n) * 1000 for n in inventory.amounts_mol],
    }


@dataclass(frozen=True)
class CationExchangeAdvance:
    initial_state: CationExchangeInventory
    final_state: CationExchangeInventory
    actual_duration_s: float
    stopping_event: str | None
    solver_diagnostic: Mapping[str, Any]
    energy_balance_residual_J: float
    conservation_residuals: Mapping[str, float]


def advance_cation_exchange(
    spec: CationExchangeSpec,
    initial: CationExchangeInventory,
    requested_flow_m3_s: float,
    duration_s: float,
    *,
    flow_path: str = "process",
) -> tuple[CationExchangeAdvance, dict[str, Any]]:
    checked = cation_exchange_inventory(
        spec, initial.amounts_mol, resin_slot=initial.resin_slot, time_s=initial.time_s
    )
    if len(initial.enthalpies_J) != 6 or any(
        not isclose(a, b, rel_tol=1e-9, abs_tol=1e-7)
        for a, b in zip(initial.enthalpies_J, checked.enthalpies_J, strict=True)
    ):
        raise ValueError("original aqueous liquid H differs from declared same-source calorics")
    thermo = _Thermodynamics(spec, initial.resin_slot)
    names = (
        *(f"H{i}" for i in range(6)),
        "throughput_m3",
        "hydraulic_J",
        "bath_heating_J",
        "bath_cooling_J",
    )
    start = _initial_view(spec, initial, dict.fromkeys(names, 0.0))

    def evaluate(state: CoupledStateView) -> tuple[ProcessContribution, dict[str, Any]]:
        rows = thermo.rows(state)
        point = _operating_point(spec, rows, requested_flow_m3_s, flow_path)
        r = _exchange(spec, state)["exchange_rate_mol_s"]
        na_liq, ca_liq = _key(1, spec.sodium_id), _key(1, spec.calcium_id)
        na_bound, ca_bound = spec.bound_key(spec.sodium_id), spec.bound_key(spec.calcium_id)
        exchange = (
            InternalTransfer(
                "exchange:Ca",
                ca_liq if r >= 0 else ca_bound,
                ca_bound if r >= 0 else ca_liq,
                abs(r),
            ),
            InternalTransfer(
                "exchange:Na",
                na_bound if r >= 0 else na_liq,
                na_liq if r >= 0 else na_bound,
                2 * abs(r),
            ),
        )
        transfers = (
            *exchange,
            *(
                InternalTransfer(
                    f"inlet:{k}",
                    _key(point["source_index"], k),
                    _key(1, k),
                    point["inlet_mol_s"][k],
                )
                for k in spec.component_ids
            ),
            *(
                InternalTransfer(
                    f"outlet:{k}",
                    _key(1, k),
                    _key(point["target_index"], k),
                    point["outlet_mol_s"][k],
                )
                for k in spec.component_ids
            ),
        )
        dn = dict.fromkeys(thermo.inventory_keys, 0.0)
        for transfer in transfers:
            dn[transfer.source_key] -= transfer.rate_mol_s
            dn[transfer.destination_key] += transfer.rate_mol_s
        bath = thermo.chemical_energy_rate_w(state, dn, ()) - point["hydraulic_power_W"]
        return ProcessContribution(
            internal_transfers=transfers,
            coordinate_rates={
                **{
                    f"H{i}": sum(
                        dn[_key(i, k)] * spec.standards[k].enthalpy_J_mol
                        for k in spec.component_ids
                    )
                    for i in range(6)
                },
                "throughput_m3": point["actual_flow_m3_s"],
                "hydraulic_J": point["hydraulic_power_W"],
                "bath_heating_J": max(bath, 0.0),
                "bath_cooling_J": max(-bath, 0.0),
            },
        ), {**point, "bath_heat_into_system_W": bath}

    def bath_rate(state: CoupledStateView) -> float:
        view = replace(state, amounts_mol={k: max(v, 0.0) for k, v in state.amounts_mol.items()})
        return float(evaluate(view)[1]["bath_heat_into_system_W"])

    first = evaluate(start)[1]
    source, target = first["source_index"], first["target_index"]
    events: list[CoupledEvent] = []
    if first["actual_flow_m3_s"] > 0:
        events.extend(
            (
                CoupledEvent(
                    f"{flow_path}_source_pickup",
                    lambda s: spec.volume_m3(thermo.rows(s)[source]) * 1000 - spec.pickup_volume_L,
                ),
                CoupledEvent(
                    f"{flow_path}_collector_full",
                    lambda s: (
                        spec.capacities_L[target] - spec.volume_m3(thermo.rows(s)[target]) * 1000
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
            CoupledEvent(
                "contact_concentration_limit",
                lambda s: (
                    spec.maximum_ion_concentration_mol_L * spec.capacities_L[1]
                    - sum(s.amounts_mol[_key(1, k)] for k in spec.ion_ids)
                ),
            ),
        )
    )
    laws = {
        k: {
            **{_key(i, k): 1.0 for i in range(6)},
            **({spec.bound_key(k): 1.0} if k in (spec.sodium_id, spec.calcium_id) else {}),
        }
        for k in spec.component_ids
    }
    laws["resin_charge_equivalents"] = {
        spec.bound_key(spec.sodium_id): 1.0,
        spec.bound_key(spec.calcium_id): 2.0,
    }
    # Each flowing vessel transports any trial charge to another vessel; it is
    # not individually closed off the electroneutral manifold used by numerical
    # Jacobian probes. Total liquid charge is closed even for those trial states.
    laws["liquid_charge"] = {
        _key(i, k): float(spec.properties.charges[k]) for i in range(6) for k in spec.ion_ids
    }
    system = CoupledSystem(
        thermo,
        (CoupledProcess("aqueous-cation-exchange-flow", lambda s: evaluate(s)[0]),),
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
    rows = thermo.rows(endpoint)
    enthalpies = tuple(initial.enthalpies_J[i] + endpoint.coordinates[f"H{i}"] for i in range(6))
    if any(
        not isclose(h, spec.enthalpy(n), rel_tol=1e-9, abs_tol=3e-6)
        for n, h in zip(rows, enthalpies, strict=True)
    ):
        raise RuntimeError("integrated original aqueous H differs from same-source endpoint")
    slot = replace(
        initial.resin_slot,
        values={
            **initial.resin_slot.values,
            "bound_sodium_mol": endpoint.amounts_mol[spec.bound_key(spec.sodium_id)],
            "bound_calcium_mol": endpoint.amounts_mol[spec.bound_key(spec.calcium_id)],
        },
    )
    cation_exchange_inventory(spec, rows, resin_slot=slot, time_s=endpoint.time_s)
    if result.energy_balance_residual_J is None:
        raise RuntimeError("cation exchange needs its original U balance")
    final = CationExchangeInventory(rows, enthalpies, slot, endpoint.time_s)
    c = endpoint.coordinates
    diagnostic = {
        "flow_path": flow_path,
        "actual_duration_s": result.actual_duration_s,
        "initial_operating_point": first,
        "final_operating_point": cation_exchange_profile(
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
        "net_calcium_bound_mol": slot.values["bound_calcium_mol"]
        - initial.resin_slot.values["bound_calcium_mol"],
        "bath_owner": "single whole-group bath at feed anchor",
        "boundary_work_J": 0.0,
        "hydraulic_scope": "inlet pipe only; mixed contactor and overflow have zero resistance",
    }
    return CationExchangeAdvance(
        initial,
        final,
        result.actual_duration_s,
        result.stopping_event,
        result.solver_diagnostic,
        float(result.energy_balance_residual_J),
        result.conservation_residuals,
    ), diagnostic
