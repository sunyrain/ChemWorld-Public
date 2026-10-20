"""One local inventory/temperature integrator; processes never own a clock.

Views and results are transient solver representations of the caller's inventory,
not a second persistent world ledger. Runtime adapters retain ownership of commit,
resource settlement and observation. All rates use mol, s, J, K and L explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import pairwise
from math import isfinite
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from scipy.sparse import lil_matrix

from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode


def _finite_mapping(values: Mapping[str, float]) -> Mapping[str, float]:
    copied = {str(key): float(value) for key, value in values.items()}
    if any(not key or not isfinite(value) for key, value in copied.items()):
        raise ValueError("inventory/rate mappings require named finite quantities")
    return MappingProxyType(copied)


@dataclass(frozen=True)
class CoupledStateView:
    """Read-only rates view; event roots may inspect signed solver trial amounts.

    Process evaluations use a recorded boundary extension, while accepted
    CoupledState inventories are validated independently and stay nonnegative.
    """

    amounts_mol: Mapping[str, float]
    temperature_K: float
    volume_L: float
    time_s: float = 0.0
    coordinates: Mapping[str, float] = field(default_factory=dict, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "amounts_mol", _finite_mapping(self.amounts_mol))
        object.__setattr__(self, "coordinates", _finite_mapping(self.coordinates))
        if (
            any(
                not isfinite(value) or value <= 0.0
                for value in (
                    self.temperature_K,
                    self.volume_L,
                )
            )
            or not isfinite(self.time_s)
            or self.time_s < 0.0
        ):
            raise ValueError("coupled state requires finite temperature, volume and time")


@dataclass(frozen=True)
class CoupledState(CoupledStateView):
    boundary_heat_J: float = 0.0
    chemical_energy_change_J: float = 0.0
    heat_loss_J: float = 0.0
    work_J: float = 0.0
    exchanged_mol: Mapping[str, float] = field(default_factory=dict)
    exchange_energy_J: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(value < 0.0 for value in self.amounts_mol.values()):
            raise ValueError("state inventory cannot be negative")
        for name in ("exchanged_mol", "exchange_energy_J"):
            object.__setattr__(self, name, _finite_mapping(getattr(self, name)))
        if any(value < 0.0 for value in self.exchanged_mol.values()):
            raise ValueError("cumulative port quantities must be nonnegative")
        if any(
            not isfinite(getattr(self, name))
            for name in (
                "boundary_heat_J",
                "chemical_energy_change_J",
                "heat_loss_J",
                "work_J",
            )
        ):
            raise ValueError("cumulative energy quantities must be finite")


@dataclass(frozen=True)
class InternalTransfer:
    exchange_id: str
    source_key: str
    destination_key: str
    rate_mol_s: float

    def __post_init__(self) -> None:
        if not all((self.exchange_id, self.source_key, self.destination_key)):
            raise ValueError("internal transfer requires identity and two endpoints")
        if self.source_key == self.destination_key:
            raise ValueError("internal transfer requires different endpoints")
        if not isfinite(self.rate_mol_s) or self.rate_mol_s < 0.0:
            raise ValueError("internal transfer rate must be finite and nonnegative")


@dataclass(frozen=True)
class MaterialPort:
    port_id: str
    inventory_key: str
    source: str
    destination: str
    direction: str
    unit: str = "mol"

    def __post_init__(self) -> None:
        if not all((self.port_id, self.inventory_key, self.source, self.destination)):
            raise ValueError("material port needs identity, inventory and endpoints")
        if self.direction not in {"in", "out"} or self.source == self.destination:
            raise ValueError("invalid material port direction/endpoints")
        if self.unit != "mol":
            raise ValueError("coupled material ports use mol")

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == "in" else -1.0


@dataclass(frozen=True)
class EnergyPort:
    """An explicit heat/work or non-molar carrier enthalpy boundary, in J/s.

    Carrier volumes stay in their declared L inventory; they are never packed
    into ``amounts_mol``. Enthalpies relative to a reference may be signed.
    """

    port_id: str
    source: str
    destination: str
    direction: str

    def __post_init__(self) -> None:
        if (
            not all((self.port_id, self.source, self.destination))
            or self.source == self.destination
            or self.direction not in {"in", "out"}
        ):
            raise ValueError("energy port requires identity, endpoints and direction")

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == "in" else -1.0


@dataclass(frozen=True)
class ProcessContribution:
    amount_rates_mol_s: Mapping[str, float] = field(default_factory=dict)
    reaction_progress_mol_s: Mapping[str, float] = field(default_factory=dict)
    internal_transfers: tuple[InternalTransfer, ...] = ()
    port_rates_mol_s: Mapping[str, float] = field(default_factory=dict)
    port_enthalpies_J_mol: Mapping[str, float] = field(default_factory=dict)
    # Compatibility for the published liquid enthalpy model only. Native U
    # models reject this term, so a reaction cannot release its energy twice.
    legacy_reaction_energy_W: float | None = None
    energy_port_rates_W: Mapping[str, float] = field(default_factory=dict)
    # Non-material coordinates use their declared unit per second. Charge,
    # voltage and displacement must never masquerade as molar inventory.
    coordinate_rates: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "amount_rates_mol_s",
            "reaction_progress_mol_s",
            "port_rates_mol_s",
            "port_enthalpies_J_mol",
            "energy_port_rates_W",
            "coordinate_rates",
        ):
            object.__setattr__(self, name, _finite_mapping(getattr(self, name)))
        object.__setattr__(self, "internal_transfers", tuple(self.internal_transfers))
        if any(value < 0.0 for value in self.port_rates_mol_s.values()):
            raise ValueError("material port rates must be nonnegative")
        if set(self.port_rates_mol_s) != set(self.port_enthalpies_J_mol):
            raise ValueError(
                "each executed material port requires an explicit inlet/outlet enthalpy"
            )
        if self.legacy_reaction_energy_W is not None and not isfinite(
            self.legacy_reaction_energy_W
        ):
            raise ValueError("reaction energy rate must be finite")


@dataclass(frozen=True)
class CoupledEvent:
    event_id: str
    value: Callable[[CoupledStateView], float]
    direction: int = -1

    def __post_init__(self) -> None:
        if not self.event_id or self.direction not in {-1, 0, 1}:
            raise ValueError("stopping event needs an identity and crossing direction")


@dataclass(frozen=True)
class CoupledProcess:
    process_id: str
    evaluate: Callable[[CoupledStateView], ProcessContribution]
    events: tuple[CoupledEvent, ...] = ()
    validate_state: Callable[[CoupledStateView], None] | None = None
    # Optional inventory-row dependencies. Omission means unknown/dense, never
    # independence. Temperature/energy/port coupling is assembled centrally.
    inventory_dependencies: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if not self.process_id:
            raise ValueError("process requires a stable identity")
        object.__setattr__(self, "events", tuple(self.events))
        if self.inventory_dependencies is not None:
            object.__setattr__(self, "inventory_dependencies", MappingProxyType({
                key: tuple(columns) for key, columns in self.inventory_dependencies.items()
            }))


@dataclass(frozen=True)
class HeatFlux:
    heater_W: float = 0.0
    loss_W: float = 0.0
    work_W: float = 0.0

    def __post_init__(self) -> None:
        if any(not isfinite(value) for value in (self.heater_W, self.loss_W, self.work_W)):
            raise ValueError("thermal powers must be finite")


class CoupledThermodynamics(Protocol):
    @property
    def inventory_keys(self) -> tuple[str, ...]: ...

    def heat_capacity_j_k(self, state: CoupledStateView) -> float: ...

    def internal_energy_j(self, state: CoupledStateView) -> float | None: ...

    def chemical_energy_rate_w(
        self,
        state: CoupledStateView,
        derivatives: Mapping[str, float],
        contributions: Sequence[ProcessContribution],
    ) -> float: ...


@dataclass(frozen=True)
class SpeciesInternalEnergy:
    u_ref_J_mol: float
    cv_J_mol_K: float

    def __post_init__(self) -> None:
        if not isfinite(self.u_ref_J_mol) or not isfinite(self.cv_J_mol_K):
            raise ValueError("species internal energy parameters must be finite")
        if self.cv_J_mol_K < 0.0:
            raise ValueError("species heat capacity cannot be negative")


@dataclass(frozen=True)
class ConstantCvThermodynamics:
    species: Mapping[str, SpeciesInternalEnergy]
    support_heat_capacity_J_K: float = 0.0
    reference_temperature_K: float = 350.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", MappingProxyType(dict(self.species)))
        if not isfinite(self.support_heat_capacity_J_K) or self.support_heat_capacity_J_K < 0.0:
            raise ValueError("support heat capacity must be finite and nonnegative")
        if not isfinite(self.reference_temperature_K) or self.reference_temperature_K <= 0:
            raise ValueError("reference temperature must be finite and positive")
        if any(not key for key in self.species):
            raise ValueError("thermal inventory keys cannot be empty")

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return tuple(self.species)

    @cached_property
    def _property_model(self):
        from chemworld.physchem.property_packages import (
            PropertyPackageThermodynamics,
            SpeciesCaloricModel,
        )
        from chemworld.physchem.property_reports import CaloricReference
        from chemworld.physchem.specs import PropertyCorrelation

        return PropertyPackageThermodynamics({
            key: SpeciesCaloricModel(
                PropertyCorrelation(
                    correlation_id=f"native-constant-cv:{key}",
                    property_id="isochoric_heat_capacity", equation_id="cp_polynomial",
                    coefficients={"a": value.cv_J_mol_K}, input_units={"temperature": "K"},
                    output_unit="J/(mol*K)", source_note="Declared native constant-Cv model",
                ),
                CaloricReference(temperature_K=self.reference_temperature_K,
                                 energy_basis="internal_energy", energy_J_mol=value.u_ref_J_mol,
                                 source_note="Declared native U reference; no phase conversion"),
            ) for key, value in self.species.items()
        }, self.support_heat_capacity_J_K, self.reference_temperature_K)

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        return self._property_model.heat_capacity_j_k(state)

    def species_energy_j_mol(self, key: str, temperature_K: float) -> float:
        return self._property_model.species_energy_j_mol(key, temperature_K)

    def internal_energy_j(self, state: CoupledStateView) -> float:
        return self._property_model.internal_energy_j(state)

    def chemical_energy_rate_w(
        self,
        state: CoupledStateView,
        derivatives: Mapping[str, float],
        contributions: Sequence[ProcessContribution],
    ) -> float:
        return self._property_model.chemical_energy_rate_w(state, derivatives, contributions)


@dataclass(frozen=True)
class CoupledRates:
    amount_rates_mol_s: Mapping[str, float]
    temperature_rate_K_s: float
    heater_W: float
    chemical_energy_W: float
    loss_W: float
    work_W: float
    port_rates_mol_s: Mapping[str, float]
    port_energy_W: Mapping[str, float]
    coordinate_rates: Mapping[str, float] = field(default_factory=dict)

    def legacy_batch_vector(self, keys: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [
                *(self.amount_rates_mol_s[key] for key in keys),
                self.temperature_rate_K_s,
                self.heater_W,
                self.chemical_energy_W,
                self.loss_W,
            ]
        )


def assemble_coupled_rates(
    state: CoupledStateView,
    *,
    processes: Sequence[CoupledProcess],
    thermodynamics: CoupledThermodynamics,
    heat_flux: Callable[[CoupledStateView], HeatFlux],
    ports: Sequence[MaterialPort] = (),
    isothermal: bool = False,
    energy_ports: Sequence[EnergyPort] = (),
    prescribed_temperature_rate_K_s: float | None = None,
) -> CoupledRates:
    """Read one immutable candidate, sum once, and derive its single dT/dt."""
    keys = thermodynamics.inventory_keys
    if set(state.amounts_mol) != set(keys):
        raise ValueError("candidate inventory must match the compiled thermal layout")
    derivatives = dict.fromkeys(keys, 0.0)
    coordinate_rates = dict.fromkeys(state.coordinates, 0.0)
    port_map = {port.port_id: port for port in ports}
    energy_map = {port.port_id: port for port in energy_ports}
    all_ports = (*ports, *energy_ports)
    if len({port.port_id for port in all_ports}) != len(all_ports):
        raise ValueError("boundary port identities must be unique")
    port_rates = dict.fromkeys(port_map, 0.0)
    port_energy = dict.fromkeys((port.port_id for port in all_ports), 0.0)
    contributions = []
    identities: set[str] = set()
    for process in processes:
        contribution = process.evaluate(state)
        contributions.append(contribution)
        for key, rate in contribution.coordinate_rates.items():
            if key not in coordinate_rates:
                raise ValueError(
                    f"process {process.process_id} references unknown coordinate {key}")
            coordinate_rates[key] += rate
        for key, rate in contribution.amount_rates_mol_s.items():
            if key not in derivatives:
                raise ValueError(f"process {process.process_id} references unknown inventory {key}")
            derivatives[key] += rate
        for reaction in contribution.reaction_progress_mol_s:
            identity = f"reaction:{reaction}"
            if identity in identities:
                raise ValueError(f"duplicate reaction contribution: {reaction}")
            identities.add(identity)
        for transfer in contribution.internal_transfers:
            identity = f"transfer:{transfer.exchange_id}"
            if identity in identities:
                raise ValueError(f"duplicate internal exchange: {transfer.exchange_id}")
            identities.add(identity)
            if (
                transfer.source_key not in derivatives
                or transfer.destination_key not in derivatives
            ):
                raise ValueError(
                    "internal transfer endpoints must belong to the compiled inventory"
                )
            derivatives[transfer.source_key] -= transfer.rate_mol_s
            derivatives[transfer.destination_key] += transfer.rate_mol_s
        for port_id, rate in contribution.port_rates_mol_s.items():
            if port_id not in port_map:
                raise ValueError(f"undeclared material port: {port_id}")
            if f"port:{port_id}" in identities:
                raise ValueError(f"duplicate material port contribution: {port_id}")
            identities.add(f"port:{port_id}")
            port = port_map[port_id]
            derivatives[port.inventory_key] += port.sign * rate
            port_rates[port_id] = rate
            port_energy[port_id] = rate * contribution.port_enthalpies_J_mol[port_id]
        for port_id, power in contribution.energy_port_rates_W.items():
            if port_id not in energy_map:
                raise ValueError(f"undeclared energy port: {port_id}")
            if f"port:{port_id}" in identities:
                raise ValueError(f"duplicate energy port contribution: {port_id}")
            identities.add(f"port:{port_id}")
            port_energy[port_id] = power
    chemical = thermodynamics.chemical_energy_rate_w(state, derivatives, contributions)
    capacity = thermodynamics.heat_capacity_j_k(state)
    if not isfinite(capacity) or capacity <= 0.0 or not isfinite(chemical):
        raise ValueError("candidate requires positive finite heat capacity and finite energy rates")
    thermal = heat_flux(state)
    flow_energy = sum(port.sign * port_energy[port.port_id] for port in all_ports)
    heater = thermal.heater_W
    if prescribed_temperature_rate_K_s is not None:
        if isothermal or not isfinite(prescribed_temperature_rate_K_s):
            raise ValueError("prescribed temperature rate must be finite and unambiguous")
        heater = (
            chemical
            + capacity * prescribed_temperature_rate_K_s
            + thermal.loss_W
            - thermal.work_W
            - flow_energy
        )
    elif isothermal:
        # The thermostat supplies/removes the actual balancing heat. Reaction
        # energy remains internal and is never added as another external input.
        heater = chemical + thermal.loss_W - thermal.work_W - flow_energy
    temperature_rate = (
        0.0
        if isothermal
        else prescribed_temperature_rate_K_s
        if prescribed_temperature_rate_K_s is not None
        else (heater - thermal.loss_W + thermal.work_W + flow_energy - chemical) / capacity
    )
    return CoupledRates(
        derivatives,
        temperature_rate,
        heater,
        chemical,
        thermal.loss_W,
        thermal.work_W,
        port_rates,
        port_energy,
        coordinate_rates,
    )


DEFAULT_COUPLED_ODE_POLICY = ODESolverPolicy(
    solver_id="chemworld_coupled_bdf_v1",
    method="BDF",
    rtol=1e-8,
    atol=1e-12,
)


@dataclass(frozen=True)
class CoupledAdvanceResult:
    initial_state: CoupledState
    final_state: CoupledState
    trajectory: tuple[CoupledState, ...]
    outcome: str
    stopping_event: str | None
    solver_diagnostic: Mapping[str, Any]
    energy_balance_residual_J: float | None
    conservation_residuals: Mapping[str, float]
    roundoff_corrections_mol: Mapping[str, float]
    ports: tuple[MaterialPort, ...] = ()
    energy_ports: tuple[EnergyPort, ...] = ()

    @property
    def actual_duration_s(self) -> float:
        return self.final_state.time_s - self.initial_state.time_s

    @property
    def actual_exchanges(self) -> tuple[dict[str, Any], ...]:
        """Actual nonzero port executions, never receipts inferred from a deficit."""
        result = []
        for port in self.ports:
            amount = self.final_state.exchanged_mol.get(port.port_id, 0.0) - (
                self.initial_state.exchanged_mol.get(port.port_id, 0.0)
            )
            if amount > 0.0:
                result.append(
                    {
                        "exchange_id": port.port_id,
                        "inventory_key": port.inventory_key,
                        "source": port.source,
                        "destination": port.destination,
                        "direction": port.direction,
                        "unit": port.unit,
                        "amount": amount,
                        "energy_J": self.final_state.exchange_energy_J.get(port.port_id, 0.0)
                        - self.initial_state.exchange_energy_J.get(port.port_id, 0.0),
                    }
                )
        for port in self.energy_ports:
            energy = self.final_state.exchange_energy_J.get(port.port_id, 0.0) - (
                self.initial_state.exchange_energy_J.get(port.port_id, 0.0)
            )
            if energy:
                result.append(
                    {
                        "exchange_id": port.port_id,
                        "source": port.source,
                        "destination": port.destination,
                        "direction": port.direction,
                        "unit": "J",
                        "energy_J": energy,
                    }
                )
        return tuple(result)


@dataclass(frozen=True)
class CoupledCoordinate:
    """Named non-material coordinate; persistent ownership stays with the caller."""

    name: str
    unit: str
    scale: float
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self):
        if not self.name or not self.unit or not isfinite(self.scale) or self.scale <= 0:
            raise ValueError("coordinate requires a name, native unit and positive finite scale")
        if any(v is not None and not isfinite(v) for v in (self.minimum, self.maximum)):
            raise ValueError("coordinate bounds must be finite when declared")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("coordinate bounds are reversed")

    def validate(self, value):
        if (not isfinite(value)
                or (self.minimum is not None and value < self.minimum)
                or (self.maximum is not None and value > self.maximum)):
            raise ValueError(f"coordinate {self.name} is outside its declared domain")


@dataclass(frozen=True)
class CoupledSystem:
    thermodynamics: CoupledThermodynamics
    processes: tuple[CoupledProcess, ...] = ()
    heat_flux: Callable[[CoupledStateView], HeatFlux] = lambda _state: HeatFlux()
    ports: tuple[MaterialPort, ...] = ()
    events: tuple[CoupledEvent, ...] = ()
    conservation_laws: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    inventory_scales_mol: Mapping[str, float] = field(default_factory=dict)
    temperature_scale_K: float = 350.0
    energy_scale_J: float = 1.0
    isothermal: bool = False
    policy: ODESolverPolicy = DEFAULT_COUPLED_ODE_POLICY
    energy_ports: tuple[EnergyPort, ...] = ()
    coordinates: tuple[CoupledCoordinate, ...] = ()

    def __post_init__(self) -> None:
        for name in ("processes", "ports", "events", "energy_ports", "coordinates"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        keys = self.thermodynamics.inventory_keys
        coordinate_keys = tuple(item.name for item in self.coordinates)
        if len({*keys, *coordinate_keys}) != len(keys) + len(coordinate_keys):
            raise ValueError("material and coordinate identities must be unique")
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate thermal inventory keys")
        if len({p.process_id for p in self.processes}) != len(self.processes):
            raise ValueError("duplicate process owner")
        if len({p.port_id for p in (*self.ports, *self.energy_ports)}) != (
            len(self.ports) + len(self.energy_ports)
        ):
            raise ValueError("duplicate boundary port identity")
        if any(port.inventory_key not in keys for port in self.ports):
            raise ValueError("material port refers to unknown inventory")
        for process in self.processes:
            dependencies = process.inventory_dependencies
            if dependencies is not None and (
                set(dependencies) - set(keys)
                or any(set(columns) - set(keys) for columns in dependencies.values())
            ):
                raise ValueError("process sparsity refers to unknown inventory")
        all_events = self.all_events
        if len({event.event_id for event in all_events}) != len(all_events):
            raise ValueError("duplicate stopping event identity")
        scales = _finite_mapping(self.inventory_scales_mol)
        if set(scales) - set(keys) or any(value <= 0.0 for value in scales.values()):
            raise ValueError("inventory scales must refer to known keys and be positive")
        object.__setattr__(self, "inventory_scales_mol", scales)
        laws = {name: _finite_mapping(weights) for name, weights in self.conservation_laws.items()}
        if any(
            not name or not weights or not any(weights.values()) for name, weights in laws.items()
        ):
            raise ValueError("declared conservation laws need meaningful nonzero weights")
        if any(set(weights) - {*keys, *coordinate_keys} for weights in laws.values()):
            raise ValueError("conservation law refers to unknown inventory")
        object.__setattr__(self, "conservation_laws", MappingProxyType(laws))
        if any(
            not isfinite(value) or value <= 0
            for value in (
                self.temperature_scale_K,
                self.energy_scale_J,
            )
        ):
            raise ValueError("temperature and energy scales must be positive and finite")

    @property
    def all_events(self) -> tuple[CoupledEvent, ...]:
        return self.events + tuple(event for process in self.processes for event in process.events)

    def jacobian_sparsity(self):
        """Assemble inventory stencils plus conservative thermal/receipt dependencies.

        Cumulative receipts do not feed back into rates. Unknown process
        dependencies stay dense, so adding a spatial process cannot hide a
        reaction or material-port dependence from numerical differentiation.
        """
        if not any(p.inventory_dependencies is not None for p in self.processes):
            return None
        keys = self.thermodynamics.inventory_keys
        count = len(keys)
        base_size = count + 5 + 2 * len(self.ports) + len(self.energy_ports)
        size = base_size + len(self.coordinates)
        pattern = lil_matrix((size, size), dtype=bool)
        positions = {key: i for i, key in enumerate(keys)}
        for process in self.processes:
            if process.inventory_dependencies is None:
                pattern[:count, :count] = True
            else:
                for row, columns in process.inventory_dependencies.items():
                    for column in columns:
                        pattern[positions[row], positions[column]] = True
        pattern[:count, count] = True
        if not self.isothermal:
            pattern[count, :count + 1] = True
        pattern[count + 1:, :count + 1] = True
        # Signed coordinates may influence any physical rate. Receipt columns
        # remain independent: only declared evolving coordinates feed back.
        pattern[:, base_size:] = True
        return pattern.tocsr()

    def advance(
        self,
        state: CoupledState,
        duration_s: float,
        *,
        evaluation_times_s: Sequence[float] | None = None,
    ) -> CoupledAdvanceResult:
        if not isfinite(duration_s) or duration_s < 0:
            raise ValueError("duration_s must be finite and nonnegative")
        keys = self.thermodynamics.inventory_keys
        coordinate_keys = tuple(item.name for item in self.coordinates)
        if set(state.amounts_mol) != set(keys):
            raise ValueError("initial state does not match the compiled inventory")
        if set(state.coordinates) != set(coordinate_keys):
            raise ValueError("initial state does not match the declared coordinates")
        for item in self.coordinates:
            item.validate(state.coordinates[item.name])
        validate_thermal_state = getattr(self.thermodynamics, "validate_state", None)
        if validate_thermal_state is not None:
            validate_thermal_state(state)
        for process in self.processes:
            if process.validate_state is not None:
                process.validate_state(state)
        port_ids = {port.port_id for port in self.ports}
        energy_ids = port_ids | {port.port_id for port in self.energy_ports}
        energy_boundaries = (*self.ports, *self.energy_ports)
        if set(state.exchanged_mol) - port_ids or set(state.exchange_energy_J) - energy_ids:
            raise ValueError("initial exchange ledger contains undeclared material ports")
        count = len(keys)
        coordinate_offset = count + 5 + len(self.ports) + len(energy_boundaries)
        scales = np.asarray(
            [
                *(
                    self.inventory_scales_mol.get(key, max(state.amounts_mol[key], 1e-12))
                    for key in keys
                ),
                self.temperature_scale_K,
                *(self.energy_scale_J for _ in range(4)),
                *(
                    self.inventory_scales_mol.get(
                        port.inventory_key,
                        max(
                            state.amounts_mol[port.inventory_key],
                            1e-12,
                        ),
                    )
                    for port in self.ports
                ),
                *(self.energy_scale_J for _ in energy_boundaries),
                *(item.scale for item in self.coordinates),
            ]
        )
        initial = np.asarray(
            [
                *(state.amounts_mol[key] for key in keys),
                state.temperature_K,
                *([0.0] * (4 + len(self.ports) + len(energy_boundaries))),
                *(state.coordinates[key] for key in coordinate_keys),
            ]
        )
        corrections: dict[str, float] = {}
        trial_extensions: dict[str, float] = {}
        property_trial_warnings: set[str] = set()

        def amounts(vector: np.ndarray, *, record: bool = False) -> dict[str, float]:
            result = {}
            for index, key in enumerate(keys):
                value = float(vector[index])
                tolerance = max(1e-24, 1e-10 * scales[index])
                if not isfinite(value) or (record and value < -tolerance):
                    raise RuntimeError(f"coupled inventory outside nonnegative domain: {key}")
                result[key] = max(value, 0.0)
                if record and value < 0:
                    corrections[key] = max(corrections.get(key, 0.0), -value)
                elif not record and value < 0:
                    # Extend rates to the boundary only for solver trials; root
                    # functions see the signed trial inventory. Accepted states
                    # still undergo the strict domain and conservation checks.
                    trial_extensions[key] = max(trial_extensions.get(key, 0.0), -value)
            return result

        def view(time: float, vector: np.ndarray) -> CoupledStateView:
            return CoupledStateView(
                amounts(vector), float(vector[count]), state.volume_L, state.time_s + time,
                coordinates=dict(zip(coordinate_keys, vector[coordinate_offset:], strict=True)),
            )

        def rhs(time: float, scaled: np.ndarray) -> np.ndarray:
            candidate = view(time, scaled * scales)
            validity = getattr(self.thermodynamics, "validity_warnings", None)
            if validity is not None:
                property_trial_warnings.update(validity(candidate))
            rates = assemble_coupled_rates(
                candidate,
                processes=self.processes,
                thermodynamics=self.thermodynamics,
                heat_flux=self.heat_flux,
                ports=self.ports,
                isothermal=self.isothermal,
                energy_ports=self.energy_ports,
            )
            for law, weights in self.conservation_laws.items():
                changes = [
                    weights.get(key, 0.0) * rate
                    for key, rate in {**rates.amount_rates_mol_s, **rates.coordinate_rates}.items()
                ]
                external = [
                    port.sign
                    * weights.get(port.inventory_key, 0.0)
                    * rates.port_rates_mol_s[port.port_id]
                    for port in self.ports
                ]
                residual = sum(changes) - sum(external)
                scale = sum(abs(value) for value in (*changes, *external))
                if abs(residual) > 1e-14 + 1e-8 * scale:
                    raise RuntimeError(f"coupled conservation law rate failed: {law}={residual}")
            return (
                np.asarray(
                    [
                        *(rates.amount_rates_mol_s[key] for key in keys),
                        rates.temperature_rate_K_s,
                        rates.heater_W,
                        rates.chemical_energy_W,
                        rates.loss_W,
                        rates.work_W,
                        *(rates.port_rates_mol_s[port.port_id] for port in self.ports),
                        *(rates.port_energy_W[port.port_id] for port in energy_boundaries),
                        *(rates.coordinate_rates[key] for key in coordinate_keys),
                    ]
                )
                / scales
            )

        def snapshot(time: float, vector: np.ndarray) -> CoupledState:
            offset = count + 5
            return CoupledState(
                amounts(vector, record=True),
                float(vector[count]),
                state.volume_L,
                state.time_s + time,
                state.boundary_heat_J + float(vector[count + 1]),
                state.chemical_energy_change_J + float(vector[count + 2]),
                state.heat_loss_J + float(vector[count + 3]),
                state.work_J + float(vector[count + 4]),
                {
                    port.port_id: state.exchanged_mol.get(port.port_id, 0.0)
                    + float(vector[offset + i])
                    for i, port in enumerate(self.ports)
                },
                {
                    port.port_id: state.exchange_energy_J.get(port.port_id, 0.0)
                    + float(vector[offset + len(self.ports) + i])
                    for i, port in enumerate(energy_boundaries)
                },
                coordinates=dict(zip(coordinate_keys, vector[coordinate_offset:], strict=True)),
            )

        grid = None if evaluation_times_s is None else tuple(float(t) for t in evaluation_times_s)
        if grid is not None and (
            any(not isfinite(t) or not 0 <= t <= duration_s for t in grid)
            or any(b < a for a, b in pairwise(grid))
        ):
            raise ValueError("evaluation times must be ordered within the requested interval")
        stopping_event = None
        if duration_s > 0.0:
            for event in self.all_events:
                value = float(event.value(state))
                if not isfinite(value):
                    raise ValueError("initial stopping event must be finite")
                if (
                    (event.direction < 0 and value <= 0.0)
                    or (event.direction > 0 and value >= 0.0)
                    or (event.direction == 0 and value == 0.0)
                ):
                    stopping_event = event.event_id
                    break
        if duration_s == 0.0 or stopping_event is not None:
            trajectory = (state,)
            diagnostic: dict[str, Any] = {"solver_calls": 0}
        else:
            callbacks = []
            for event in self.all_events:

                def callback(time: float, scaled: np.ndarray, spec: CoupledEvent = event) -> float:
                    raw = scaled * scales
                    trial = CoupledStateView(
                        dict(zip(keys, raw[:count], strict=True)),
                        float(raw[count]),
                        state.volume_L,
                        state.time_s + time,
                        coordinates=dict(zip(
                            coordinate_keys, raw[coordinate_offset:], strict=True)),
                    )
                    value = float(spec.value(trial))
                    if not isfinite(value):
                        raise RuntimeError(f"nonfinite stopping event: {spec.event_id}")
                    return value

                callback.terminal = True  # type: ignore[attr-defined]
                callback.direction = event.direction  # type: ignore[attr-defined]
                callbacks.append(callback)
            policy = replace(self.policy, event_names=tuple(e.event_id for e in self.all_events))
            sparsity = self.jacobian_sparsity() if policy.method in {"BDF", "Radau"} else None
            sparse_options = {} if sparsity is None else {"jac_sparsity": sparsity}
            report = solve_ode(
                rhs,
                initial / scales,
                time_span_s=(0.0, duration_s),
                evaluation_times_s=(tuple(sorted({0.0, duration_s, *grid})) if grid else None),
                policy=policy,
                events=callbacks or None,
                **sparse_options,
            )
            report.raise_for_failure("coupled physical advance")
            points = [(float(t), report.y[:, i] * scales) for i, t in enumerate(report.t)]
            if report.diagnostic.status == 1:
                terminal = report.terminal_point
                if terminal is None:
                    raise RuntimeError("solver stopped without an event state")
                terminal_time, terminal_vector, event_index = terminal
                stopping_event = self.all_events[event_index].event_id
                if points and abs(points[-1][0] - terminal_time) < 1e-12:
                    points[-1] = (terminal_time, terminal_vector * scales)
                else:
                    points.append((terminal_time, terminal_vector * scales))
            trajectory = tuple(snapshot(t, values) for t, values in points)
            diagnostic = {
                **report.diagnostic.to_dict(),
                "solver_calls": 1,
                "inventory_scales_mol": dict(zip(keys, scales[:count], strict=True)),
                "temperature_scale_K": self.temperature_scale_K,
                "energy_scale_J": self.energy_scale_J,
                "rhs_boundary_extensions_mol": trial_extensions,
                "accepted_nonnegative_relative_tolerance": 1e-10,
            }
            if property_trial_warnings:
                diagnostic["property_trial_warnings"] = sorted(property_trial_warnings)
        final = trajectory[-1]
        for point in trajectory:
            for item in self.coordinates:
                item.validate(point.coordinates[item.name])
            if validate_thermal_state is not None:
                validate_thermal_state(point)
            for process in self.processes:
                if process.validate_state is not None:
                    process.validate_state(point)
        u_before = self.thermodynamics.internal_energy_j(state)
        u_after = self.thermodynamics.internal_energy_j(final)
        energy_residual = None
        if u_before is not None and u_after is not None:
            delta_heat = final.boundary_heat_J - state.boundary_heat_J
            delta_loss = final.heat_loss_J - state.heat_loss_J
            delta_work = final.work_J - state.work_J
            flow_energy = sum(
                port.sign
                * (
                    final.exchange_energy_J.get(port.port_id, 0.0)
                    - state.exchange_energy_J.get(port.port_id, 0.0)
                )
                for port in energy_boundaries
            )
            energy_residual = (
                u_after - u_before - delta_heat + delta_loss - delta_work - flow_energy
            )
            scale = max(
                1.0,
                abs(u_after - u_before),
                abs(delta_heat) + abs(delta_loss) + abs(delta_work) + abs(flow_energy),
            )
            if abs(energy_residual) > 1e-8 + 1e-6 * scale:
                raise RuntimeError(f"coupled internal energy failed to close: {energy_residual} J")
        residuals = {}
        before_values = {**state.amounts_mol, **state.coordinates}
        after_values = {**final.amounts_mol, **final.coordinates}
        for law, weights in self.conservation_laws.items():
            residual = sum(
                weights.get(key, 0.0) * (after_values[key] - before_values[key])
                for key in before_values
            ) - sum(
                port.sign
                * weights.get(port.inventory_key, 0.0)
                * (
                    final.exchanged_mol.get(port.port_id, 0.0)
                    - state.exchanged_mol.get(port.port_id, 0.0)
                )
                for port in self.ports
            )
            scale = sum(abs(weight * before_values[key]) for key, weight in weights.items())
            if abs(residual) > 1e-12 + 1e-8 * scale:
                raise RuntimeError(f"coupled conservation law failed: {law}={residual}")
            residuals[law] = residual
        return CoupledAdvanceResult(
            state,
            final,
            trajectory,
            "stopped" if stopping_event else "success",
            stopping_event,
            diagnostic,
            energy_residual,
            residuals,
            corrections,
            self.ports,
            self.energy_ports,
        )
