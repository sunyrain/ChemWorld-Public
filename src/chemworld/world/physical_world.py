"""Physical inputs for WorldCompositionSpec, independent of task policy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import EquipmentLedger, VesselLedger
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.runtime.constitution_factory import make_chemworld_constitution
from chemworld.runtime.mechanisms import compile_native_network
from chemworld.world.domain_modules import DomainModuleRegistry
from chemworld.world.parameters import load_chemworld_parameters
from chemworld.world.scenario import WORLD_LAW_ID, ScenarioInstance, ScenarioSpec

if TYPE_CHECKING:
    from chemworld.world.composition import WorldCompositionSpec


@dataclass(frozen=True, init=False)
class CompositionPhysics:
    """Common physical inputs with one instance-scoped model collection.

    Legacy constructor fields are normalized at this boundary. JSON always uses
    versioned model declarations; decoding never imports types from input text.
    """

    network: ReactionNetworkSpec
    initial_states: Mapping[str, WorldState]
    parameter_seed: int = 0
    feed_sources: dict[str, str] = field(default_factory=dict)
    lab_clock: LabClockSpec | None = None
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    advance_bindings: dict[str, str] = field(default_factory=dict)
    modules: DomainModuleRegistry = field(
        default_factory=DomainModuleRegistry.default, repr=False, compare=False
    )

    def __init__(
        self,
        network: ReactionNetworkSpec,
        initial_states: Mapping[str, WorldState],
        parameter_seed: int = 0,
        *,
        feed_sources: Mapping[str, str] | None = None,
        lab_clock: LabClockSpec | None = None,
        models: Mapping[str, Mapping[str, Any]] | None = None,
        advance_bindings: Mapping[str, str] | None = None,
        modules: DomainModuleRegistry | None = None,
        **legacy_models: Mapping[str, Any],
    ) -> None:
        modules = modules if modules is not None else DomainModuleRegistry.default()
        declarations = modules.model_specs()
        aliases = {
            s.legacy_field: s.model_id for s in declarations.values() if s.legacy_field is not None
        }
        if set(legacy_models) - aliases.keys():
            raise ValueError(
                f"unsupported physical model fields: {sorted(set(legacy_models) - aliases.keys())}"
            )
        collection = dict(models or {})
        # dataclasses.replace passes the current canonical collection as well as
        # an explicit legacy override. This remains an input-only conversion.
        collection.update({aliases[key]: value for key, value in legacy_models.items()})
        if set(collection) - declarations.keys():
            raise ValueError(
                f"unknown physical model types: {sorted(set(collection) - declarations.keys())}"
            )
        if (
            isinstance(parameter_seed, bool)
            or not isinstance(parameter_seed, int)
            or parameter_seed < 0
        ):
            raise ValueError("physics.parameter_seed must be a nonnegative integer")
        normalized = {}
        for model_id, bindings in collection.items():
            if not isinstance(bindings, Mapping) or any(
                not isinstance(key, str) or not key for key in bindings
            ):
                raise ValueError("physical model bindings must use nonempty string identities")
            declaration = declarations[model_id]
            normalized[model_id] = {
                key: declaration.decode(deepcopy(declaration.encode(model)))
                for key, model in bindings.items()
            }
        for name, value in {
            "network": network,
            "initial_states": initial_states,
            "parameter_seed": parameter_seed,
            "feed_sources": dict(feed_sources or {}),
            "lab_clock": lab_clock,
            "models": normalized,
            "advance_bindings": dict(advance_bindings or {}),
            "modules": modules,
        }.items():
            object.__setattr__(self, name, value)
        if advance_bindings is None:
            self._normalize_legacy_advance_inputs()

    def _normalize_legacy_advance_inputs(self) -> None:
        """Old inputs gain explicit solver choices once, before serialization."""
        strategies = self.declared_advance_strategies(self.modules)
        defaults = {
            key: strategy
            for key, strategy in strategies.items()
            if self.modules.advance_specs()[key].legacy_input_default
        }
        bindings = {}
        for vessel, state in self.initial_states.items():
            if any(s.supports(state) for key, s in strategies.items() if key not in defaults):
                continue
            candidates = [key for key, s in defaults.items() if s.supports_legacy_input(state)]
            if len(candidates) > 1:
                raise ValueError(f"ambiguous legacy advance input for {vessel!r}")
            if candidates:
                bindings[vessel] = candidates[0]
        object.__setattr__(self, "advance_bindings", bindings)

    def declared_advance_strategies(self, modules: DomainModuleRegistry) -> dict[str, Any]:
        # Capability queries need physical bindings, not a TaskSpec or a live
        # thermal service. World validators receive the full composition below.
        return modules.build_advance_strategies(SimpleNamespace(physics=self, modules=modules))

    def advance_strategy(
        self, state: WorldState, modules: DomainModuleRegistry, *, required: bool = False,
    ) -> tuple[str, Any] | None:
        from chemworld.runtime.advance_paths import select_advance_strategy

        return select_advance_strategy(
            self.declared_advance_strategies(modules),
            state,
            bindings=self.advance_bindings,
            required=required,
        )

    def supports_measurement_clock(self, state: WorldState, modules: DomainModuleRegistry) -> bool:
        selected = self.advance_strategy(state, modules)
        return selected is not None and modules.advance_specs()[selected[0]].timed_measurements

    def models_of(self, model_id: str) -> Mapping[str, Any]:
        return self.models.get(model_id, {})

    def __getattr__(self, name: str) -> Any:
        # Historical Python readers receive the same collection, never a second
        # stored configuration. Production consumers use models_of explicitly.
        modules = object.__getattribute__(self, "modules")
        for declaration in modules.model_specs().values():
            if name == declaration.legacy_field:
                return self.models_of(declaration.model_id)
        raise AttributeError(name)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, modules: DomainModuleRegistry | None = None,
    ) -> CompositionPhysics:
        modules = modules if modules is not None else DomainModuleRegistry.default()
        declarations = modules.model_specs()
        aliases = {s.legacy_field: s for s in declarations.values() if s.legacy_field is not None}
        if set(payload) - {
            "network",
            "initial_states",
            "parameter_seed",
            "feed_sources",
            "lab_clock",
            "models",
            "advance_bindings",
            *aliases,
        }:
            raise ValueError("unsupported composition physics fields")
        states = payload.get("initial_states")
        if not isinstance(states, Mapping) or not states:
            raise ValueError("physics.initial_states must specify every vessel")
        required = {
            "species_amounts",
            "volume_L",
            "temperature_K",
            "pressure_Pa",
            "phase",
            "vessel_id",
            "units",
            "equipment",
        }
        for name, state in states.items():
            if not isinstance(state, Mapping) or not required <= state.keys():
                raise ValueError(
                    f"physics initial state {name!r} must explicitly declare {sorted(required)}"
                )
        records = payload.get("models", {})
        if not isinstance(records, Mapping):
            raise ValueError("physics.models must be a versioned type mapping")
        records = dict(records)
        for alias, declaration in aliases.items():
            if alias in payload:
                if declaration.model_id in records:
                    raise ValueError("duplicate canonical and legacy physical model configuration")
                records[declaration.model_id] = {
                    "version": declaration.version,
                    "bindings": payload[alias],
                }
        models = {}
        for model_id, record in records.items():
            if model_id not in declarations:
                raise ValueError(f"unknown physical model type: {model_id!r}")
            declaration = declarations[model_id]
            if not isinstance(record, Mapping) or set(record) != {"version", "bindings"}:
                raise ValueError("physical model record requires version and bindings")
            if record["version"] != declaration.version:
                raise ValueError(f"unsupported physical model version for {model_id!r}")
            bindings = record["bindings"]
            if not isinstance(bindings, Mapping) or any(
                not isinstance(key, str) or not key for key in bindings
            ):
                raise ValueError("physical model bindings must use nonempty string identities")
            models[model_id] = {
                key: declaration.decode(deepcopy(value)) for key, value in bindings.items()
            }
        return cls(
            ReactionNetworkSpec.from_dict(payload["network"]),
            {str(name): WorldState.from_dict(dict(state)) for name, state in states.items()},
            payload.get("parameter_seed", 0),
            feed_sources=payload.get("feed_sources", {}),
            lab_clock=LabClockSpec(**payload["lab_clock"]) if "lab_clock" in payload else None,
            models=models,
            advance_bindings=payload.get("advance_bindings"),
            modules=modules,
        )

    def to_dict(self) -> dict[str, Any]:
        declarations = self.modules.model_specs()
        return {
            **({"lab_clock": self.lab_clock.to_dict()} if self.lab_clock is not None else {}),
            "network": {
                "network_id": self.network.network_id,
                "species": [s.to_dict() for s in self.network.species],
                "reactions": [r.to_dict() for r in self.network.reactions],
                "units": deepcopy(self.network.units),
                "metadata": deepcopy(self.network.metadata),
                **(
                    {"stochastic": deepcopy(self.network.stochastic)}
                    if self.network.stochastic is not None
                    else {}
                ),
            },
            "initial_states": {
                name: state.to_dict() for name, state in self.initial_states.items()
            },
            "parameter_seed": self.parameter_seed,
            **({"feed_sources": dict(self.feed_sources)} if self.feed_sources else {}),
            "models": {
                model_id: {
                    "version": declarations[model_id].version,
                    "bindings": {
                        key: deepcopy(declarations[model_id].encode(model))
                        for key, model in bindings.items()
                    },
                }
                for model_id, bindings in self.models.items()
            },
            "advance_bindings": dict(self.advance_bindings),
        }

    def scenario_spec(self, composition_id: str, split: str) -> ScenarioSpec:
        return ScenarioSpec(
            composition_id,
            WORLD_LAW_ID,
            "composed",
            split,
            "authored",
            self.parameter_seed,
            0,
            f"{composition_id}:initial",
            "explicit",
            (),
            (),
        )

    @property
    def finite_supply_vessels(self) -> set[str]:
        return set(self.feed_sources.values()) | {
            m.condensed_background.source_vessel_id
            for m in self.models_of("thermal").values()
            if m.condensed_background is not None
            and m.condensed_background.source_vessel_id is not None
        }

    def declared_temperature_bounds_k(
        self, state: WorldState, modules: DomainModuleRegistry,
    ) -> tuple[float, float] | None:
        """Native constitutive domain, independent of the legacy liquid schema."""
        selected = self.advance_strategy(state, modules)
        bounds = None if selected is None else getattr(selected[1], "temperature_bounds_k", None)
        return None if bounds is None else bounds(state)

    def available_operations(
        self, operations: Iterable[str], modules: DomainModuleRegistry, vessel: str,
    ) -> tuple[str, ...]:
        """Intersect declared thermal actions with the installed device capability."""
        from chemworld.runtime.aqueous_filtration import aqueous_filter_record
        from chemworld.runtime.aqueous_gas_transfer import aqueous_gas_transfer_record
        from chemworld.runtime.aqueous_mineral import aqueous_record
        from chemworld.runtime.aqueous_mineral_kinetics import database_mineral_record
        from chemworld.runtime.aqueous_precipitation import precipitation_record
        from chemworld.runtime.aqueous_reactive_transport import aqueous_transport_members
        from chemworld.runtime.calphad_thermal import CalphadThermalSpec
        from chemworld.runtime.electrodialysis import electrodialysis_record
        from chemworld.runtime.molecular_flash_thermal import MolecularFlashThermalSpec
        from chemworld.runtime.molecular_multiphase_thermal import MolecularMultiphaseThermalSpec

        local_state = self.initial_states[vessel]
        if precipitation_record(local_state) is not None:
            allowed_precipitation = {
                "wait", "route_material", "select_vessel", "terminate",
            }
            if aqueous_filter_record(local_state) is not None:
                allowed_precipitation.add("filter_crystals")
            operations = tuple(op for op in operations if op in allowed_precipitation)
        if aqueous_gas_transfer_record(local_state) is not None:
            operations = tuple(op for op in operations if op in {
                "wait", "select_vessel", "terminate",
            })
        if vessel in aqueous_transport_members(self):
            operations = tuple(op for op in operations if op in {
                "wait", "select_vessel", "terminate",
            })
        if vessel in self.models_of("stochastic_hybrid"):
            from chemworld.physchem.stochastic_adaptive_hybrid import AdaptiveHybridSpec

            adaptive = isinstance(self.models_of("stochastic_hybrid")[vessel], AdaptiveHybridSpec)
            operations = tuple(op for op in operations if op in {
                "wait", "measure", "configure_instrument", "select_vessel", "terminate",
            } or (adaptive and op == "add_component"))
        if vessel in self.models_of("stochastic_sde"):
            operations = tuple(op for op in operations if op in {
                "wait", "add_component", "measure", "configure_instrument", "select_vessel",
                "terminate", 'set_sde_parameters',
            })
        if not getattr(self.models_of('stochastic_sde').get(vessel), 'controls', ()):
            operations = tuple(op for op in operations if op != 'set_sde_parameters')
        if vessel in self.models_of("molecular_process"):
            operations = tuple(op for op in operations if op in {"wait", "terminate"})
        if isinstance(self.models_of("calphad_material").get(vessel), CalphadThermalSpec):
            operations = tuple(op for op in operations if op in {
                "heat", "wait", "measure", "configure_instrument", "pause_control",
                "resume_control", "select_vessel", "terminate",
            })
        if isinstance(self.models_of("flash").get(vessel),
                      (MolecularFlashThermalSpec, MolecularMultiphaseThermalSpec)):
            operations = tuple(op for op in operations if op in {
                "heat", "wait", "measure", "configure_instrument", "pause_control",
                "resume_control", "select_vessel", "terminate",
            })
        if electrodialysis_record(local_state) is not None:
            operations = tuple(op for op in operations if op in {
                "configure_electrical_control", "wait", "select_vessel", "terminate",
            })
        elif not any(
            record.attached_vessel_id == vessel and "electrical_control" in record.settings
            for record in cast(EquipmentLedger, local_state.equipment).equipment.values()
        ):
            operations = tuple(op for op in operations if op != "configure_electrical_control")

        if database_mineral_record(self.initial_states[vessel]) is not None:
            operations = tuple(op for op in operations if op in {
                "wait", "select_vessel", "terminate",
            })

        thermal = self.models_of("thermal").get(vessel)
        if thermal is not None and thermal.uses_eos_pressure:
            operations = tuple(op for op in operations if op in {
                "heat", "wait", "pause_control", "resume_control", "measure",
                "select_vessel", "configure_instrument", "terminate",
                "queue_lab_heat", "queue_lab_measure",
            })
        if self.lab_clock is not None:
            from chemworld.runtime.scheduling import shared_operation_allowed

            return tuple(
                op
                for op in operations
                if shared_operation_allowed(op, self.initial_states[vessel], modules)
                and (op != "set_model_parameters" or getattr(
                    self.models_of("sbml").get(vessel), "controlled_parameters", ()))
                and (
                    op != "queue_lab_heat"
                    or (self.lab_clock.heater_ids and self.declared_temperature_bounds_k(
                        self.initial_states[vessel], modules,
                    ) is not None)
                )
                and (op != "queue_lab_measure" or self.lab_clock.analyzers)
            )

        state = self.initial_states[vessel]
        aqueous = aqueous_record(state)
        native_aqueous = aqueous is not None
        if native_aqueous:
            operations = tuple(
                op
                for op in operations
                if op
                in {
                    "add_component",
                    "wait",
                    "sample",
                    "measure",
                    "select_vessel",
                    "terminate",
                }
                or (op == "route_material" and "mineral_kinetics" in aqueous.settings)
            )
        if vessel in self.models_of("crystal"):
            operations = tuple(
                op
                for op in operations
                if op
                in {
                    "seed_crystals",
                    "cool_crystallize",
                    "filter_crystals",
                    "resuspend_crystals",
                    "heat",
                    "wait",
                    "select_vessel",
                    "measure",
                    "configure_instrument",
                    "create_container",
                    "transfer_material",
                    "pause_control",
                    "resume_control",
                    "terminate",
                }
            )
        carrier_sources = {
            m.condensed_background.source_vessel_id
            for m in self.models_of("thermal").values()
            if m.condensed_background is not None
        }
        if vessel in carrier_sources:
            operations = tuple(op for op in operations if op in {"select_vessel", "terminate"})
        if any(s.spec.state_type == "competitive-sorbent-v1" for s in state.state_slots.slots):
            from chemworld.runtime.thermal_fixed_bed_world import bed_record

            bed_controls = {"configure_thermal_bed"} if bed_record(state) is not None else set()
            operations = tuple(
                op
                for op in operations
                if op
                in bed_controls | {
                    "dry",
                    "reconnect_heater",
                    "heat",
                    "wait",
                    "add_solvent",
                    "route_material",
                    "select_vessel",
                    "measure",
                    "configure_instrument",
                    "pause_control",
                    "resume_control",
                    "terminate",
                }
            )
        if vessel in self.models_of("concentration"):
            operations = tuple(
                op
                for op in operations
                if op
                in {
                    "concentrate",
                    "heat",
                    "wait",
                    "measure",
                    "configure_instrument",
                    "pause_control",
                    "resume_control",
                    "select_vessel",
                    "terminate",
                }
            )
        if vessel in self.models_of("column"):
            operations = tuple(
                op
                for op in operations
                if op
                in {
                    "distill",
                    "collect_fraction",
                    "heat",
                    "wait",
                    "measure",
                    "configure_instrument",
                    "pause_control",
                    "resume_control",
                    "select_vessel",
                    "terminate",
                }
            )
        if vessel in self.models_of("liquid"):
            operations = tuple(
                op
                for op in operations
                if op
                in {
                    "add_component",
                    "mix",
                    "wash",
                    "settle",
                    "wait",
                    "measure",
                    "route_material",
                    "select_vessel",
                    "terminate",
                }
            )
            if self.models_of("liquid")[vessel].wash_feed_id not in self.feed_sources:
                operations = tuple(op for op in operations if op != "wash")
        if vessel in self.models_of("flash"):
            operations = tuple(
                op
                for op in operations
                if op
                in {
                    "add_component",
                    "heat",
                    "wait",
                    "measure",
                    "configure_instrument",
                    "route_material",
                    "select_vessel",
                    "pause_control",
                    "resume_control",
                    "terminate",
                }
            )
        if (
            self.feed_sources
            and state.phase != "gas"
            and vessel not in self.models_of("flash")
            and vessel not in self.models_of("liquid")
            and not native_aqueous
        ):
            operations = tuple(op for op in operations if op != "add_component")
        if state.phase == "gas":
            operations = tuple(
                op
                for op in operations
                if op
                not in {
                    "add_reagent",
                    "add_solvent",
                    "add_catalyst",
                }
                and (
                    op != "add_component"
                    or (
                        self.feed_sources
                        and (
                            vessel in self.models_of("thermal") or vessel in self.models_of("flash")
                        )
                    )
                )
            )
        selected = self.advance_strategy(state, modules)
        supported = set() if selected is None else set(selected[1].supported_operations)
        if selected is not None:
            state_operations = getattr(selected[1], "supported_operations_for_state", None)
            if state_operations is not None:
                supported = set(state_operations(state))
            for record in cast(EquipmentLedger, state.equipment).equipment.values():
                if (
                    record.equipment_type == "thermal_device"
                    and record.attached_vessel_id == vessel
                ):
                    supported.update(
                        ThermalDeviceSpec.from_dict(
                            record.settings["thermal_device"]
                        ).supported_operations
                    )
        specs = modules.operation_specs()
        return tuple(
            op
            for op in operations
            if specs[op].service_id not in {"reaction_thermal", "control_program"}
            or op in supported
        )

    def generate(
        self, spec: ScenarioSpec, seed: int, interventions: tuple[dict[str, Any], ...] = (),
    ) -> ScenarioInstance:
        if interventions:
            raise ValueError(
                "native compositions declare physics directly; scenario interventions do not apply"
            )
        first = next(iter(self.initial_states.values()))
        amounts = {
            name: sum(s.species_amounts.get(name, 0.0) for s in self.initial_states.values())
            for name in self.network.species_ids
        }
        return ScenarioInstance(
            spec,
            load_chemworld_parameters(spec.split, self.parameter_seed),
            deepcopy(first),
            compile_native_network(self.network, initial_amounts=amounts),
        )

    def validate(self, spec: WorldCompositionSpec) -> None:
        from chemworld.runtime.scheduling import validate_lab

        validate_lab(spec)
        vessels = {v["id"]: v for v in spec.vessels}
        if set(self.initial_states) != set(vessels):
            raise ValueError("physics.initial_states must match the declared vessels exactly")
        declarations = spec.modules.model_specs()
        owners: dict[tuple[str, str], str] = {}
        for model_id, declaration in declarations.items():
            if model_id in self.models and declaration != self.modules.model_specs().get(model_id):
                raise ValueError(
                    "physical model declaration differs from its decoded configuration"
                )
            bindings = self.models_of(model_id)
            if declaration.binding == "vessel" and set(bindings) - vessels.keys():
                raise ValueError(f"{model_id} model must belong to a declared vessel")
            if declaration.component_kind is not None and set(bindings) != {
                c.vessel for c in spec.components if c.kind == declaration.component_kind
            }:
                raise ValueError(f"{model_id} models require matching component declarations")
            if declaration.exclusive_group is not None:
                for vessel in bindings:
                    key = (declaration.exclusive_group, vessel)
                    if key in owners:
                        raise ValueError(
                            f"conflicting physical model owners: {owners[key]} and {model_id} "
                            f"at {vessel}"
                        )
                    owners[key] = model_id
        if set(self.models) - declarations.keys():
            raise ValueError("physical configuration requires uninstalled model declarations")
        if len(self.feed_sources) > 16 or len(set(self.feed_sources.values())) != len(
            self.feed_sources
        ):
            raise ValueError("finite feeds require unique source vessels, at most 16")
        for feed_id, vessel in self.feed_sources.items():
            if not isinstance(feed_id, str) or not feed_id or vessel not in self.initial_states:
                raise ValueError("finite feed must name a declared source vessel")
            source = self.initial_states[vessel]
            if (
                (
                    source.phase != "gas"
                    and not (
                        any(
                            d.accepts_liquid_feed and self.models_of(key)
                            for key, d in declarations.items()
                        )
                        or {"aqueous_mineral", "mineral_kinetics"} & set(spec.component_kinds)
                    )
                )
                or source.phase not in {"gas", "liquid"}
                or source.state_slots.slots
                or any(
                    vessel in self.models_of(key)
                    for key, d in declarations.items()
                    if d.binding == "vessel"
                )
                or any(
                    c.vessel == vessel and spec.modules.components()[c.kind].processes
                    for c in spec.components
                )
            ):
                raise ValueError(
                    "finite source requires a declared fixed-temperature material boundary"
                )
            for model_id, declaration in declarations.items():
                if declaration.validate_feed is not None:
                    for model in self.models_of(model_id).values():
                        declaration.validate_feed(source, model)
        if self.network.stochastic is None and bool(self.network.reactions) != (
            "reaction" in spec.component_kinds or bool(self.models_of("stochastic_hybrid"))
            or bool(self.models_of("stochastic_sde"))
        ):
            raise ValueError("bulk reactions require a reaction component and a nonempty network")
        instance = self.generate(self.scenario_spec(spec.composition_id, spec.world_split), 0)
        constitution = make_chemworld_constitution(
            instance.compiled_mechanism, modules=spec.modules
        )
        for name, state in self.initial_states.items():
            if state.vessel_id != name or state.inactive_vessels:
                raise ValueError(
                    "initial states must be flat local states with matching vessel IDs"
                )
            if state.terminated or state.ledger.time_s != 0:
                raise ValueError(
                    "initial states must be unstarted; use snapshots to resume a session"
                )
            local_vessels = cast(VesselLedger, state.vessels)
            if local_vessels.vessels[name].max_volume_L != vessels[name]["capacity_L"]:
                raise ValueError("initial-state vessel capacity differs from the composition")
            bulk_species = {s.species_id for s in self.network.species if s.phase != "surface"}
            if not set(state.species_amounts) <= bulk_species:
                raise ValueError("initial-state bulk species must belong to the explicit network")
            expected_units = {
                "amount": "mol",
                "volume": "L",
                "temperature": "K",
                "pressure": "Pa",
                "time": "s",
            }
            if any(state.units.get(key) != unit for key, unit in expected_units.items()):
                raise ValueError("initial-state quantities must use native mol/L/K/Pa/s units")
            if state.phase == "fluid":
                from chemworld.physchem.helmholtz_equilibrium_thermal import (
                    HelmholtzEquilibriumIsochoricThermodynamics,
                )
                from chemworld.physchem.humid_air_thermal import HumidAirIsochoricThermodynamics
                from chemworld.physchem.water_if97_thermal import IF97IsochoricThermodynamics
                from chemworld.runtime.aqueous_gas_transfer import aqueous_gas_transfer_record

                thermal = self.models_of("thermal").get(name)
                headspace = aqueous_gas_transfer_record(state) is not None and any(
                    c.kind == "aqueous_gas_transfer" and c.vessel == name
                    for c in spec.components
                )
                if not headspace and (thermal is None or not isinstance(
                        thermal.thermodynamics,
                        (HelmholtzEquilibriumIsochoricThermodynamics, IF97IsochoricThermodynamics,
                         HumidAirIsochoricThermodynamics))):
                    raise ValueError(
                        "aggregate fluid phase requires a declared EOS or finite headspace closure")
            elif state.phase not in {"liquid", "gas", "solid", "formal"}:
                raise ValueError("initial state needs an explicit supported phase")
            report = constitution.check_state(state)
            if not report.passed:
                failed = [check.name for check in report.checks if not check.passed]
                raise ValueError(f"invalid physical initial state {name!r}: {failed}")
        for declaration in declarations.values():
            if declaration.validate_world is not None:
                declaration.validate_world(self, spec)
        from chemworld.runtime.advance_paths import select_advance_strategy

        strategies = spec.modules.build_advance_strategies(spec)
        if set(self.advance_bindings) - vessels.keys() or any(
            key not in strategies for key in self.advance_bindings.values()
        ):
            raise ValueError("advance bindings require declared vessels and installed strategies")
        for strategy in strategies.values():
            validate_world = getattr(strategy, "validate_world", None)
            if validate_world is not None:
                validate_world()
        boundaries = self.finite_supply_vessels | set().union(
            *(
                getattr(strategy, "preserved_source_vessels", set())
                for strategy in strategies.values()
            )
        )
        for vessel, state in self.initial_states.items():
            active = vessel not in boundaries and (
                vessel in self.advance_bindings
                or any(
                    d.exclusive_group is not None and vessel in self.models_of(key)
                    for key, d in declarations.items()
                )
                or any(s.spec.time_evolution == "coupled" for s in state.state_slots.slots)
                or any(
                    c.vessel == vessel and spec.modules.components()[c.kind].processes
                    for c in spec.components
                )
            )
            select_advance_strategy(
                strategies,
                state,
                bindings=self.advance_bindings,
                required=active,
            )
