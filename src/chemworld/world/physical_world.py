"""Physical inputs for WorldCompositionSpec, independent of task policy."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.physchem.coupled_concentration import CoupledConcentrationSpec
from chemworld.physchem.crystal_thermal_kinetics import CrystalThermalSpec
from chemworld.physchem.dynamic_distillation import DynamicColumnSpec
from chemworld.physchem.flash_units import FlashSystemSpec
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec, ThermalSystemSpec
from chemworld.physchem.phase_equilibrium_units import StrictLiquidSystemSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.sorbent_kinetics import SorbentKineticsSpec
from chemworld.physchem.surface_specs import surface_spec_from_dict
from chemworld.runtime.constitution_factory import make_chemworld_constitution
from chemworld.runtime.mechanisms import compile_native_network
from chemworld.world.parameters import load_chemworld_parameters
from chemworld.world.scenario import WORLD_LAW_ID, ScenarioInstance, ScenarioSpec


@dataclass(frozen=True)
class CompositionPhysics:
    """Explicit network and per-vessel initial states, using existing contracts.

    This is the physical portion of a composition, not a second world schema.
    Task seeds only drive observations; the physical parameter seed is fixed here.
    """

    network: ReactionNetworkSpec
    initial_states: dict[str, WorldState]
    parameter_seed: int = 0
    thermal_models: dict[str, ThermalSystemSpec] = field(default_factory=dict)
    feed_sources: dict[str, str] = field(default_factory=dict)
    flash_models: dict[str, FlashSystemSpec] = field(default_factory=dict)
    liquid_models: dict[str, StrictLiquidSystemSpec] = field(default_factory=dict)
    column_models: dict[str, DynamicColumnSpec] = field(default_factory=dict)
    concentration_models: dict[str, CoupledConcentrationSpec] = field(default_factory=dict)
    sorbent_models: dict[str, SorbentKineticsSpec] = field(default_factory=dict)
    crystal_models: dict[str, CrystalThermalSpec] = field(default_factory=dict)
    lab_clock: LabClockSpec | None = None
    surface_models: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CompositionPhysics:
        if set(payload) - {
            "network",
            "initial_states",
            "parameter_seed",
            "thermal_models",
            "feed_sources",
            "flash_models",
            "liquid_models",
            "column_models",
            "concentration_models",
            "sorbent_models",
            "crystal_models",
            "lab_clock",
            "surface_models",
        }:
            raise ValueError("unsupported composition physics fields")
        states = payload.get("initial_states")
        if not isinstance(states, Mapping) or not states:
            raise ValueError("physics.initial_states must specify every vessel")
        seed = payload.get("parameter_seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("physics.parameter_seed must be a nonnegative integer")
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
        return cls(
            ReactionNetworkSpec.from_dict(payload["network"]),
            {str(name): WorldState.from_dict(dict(state)) for name, state in states.items()},
            seed,
            {
                name: ThermalSystemSpec.from_dict(model)
                for name, model in payload.get("thermal_models", {}).items()
            },
            dict(payload.get("feed_sources", {})),
            {
                name: FlashSystemSpec.from_dict(model)
                for name, model in payload.get("flash_models", {}).items()
            },
            {
                name: StrictLiquidSystemSpec.from_dict(model)
                for name, model in payload.get("liquid_models", {}).items()
            },
            {
                name: DynamicColumnSpec.from_dict(model)
                for name, model in payload.get("column_models", {}).items()
            },
            {
                name: CoupledConcentrationSpec.from_dict(model)
                for name, model in payload.get("concentration_models", {}).items()
            },
            {
                name: SorbentKineticsSpec.from_dict(model)
                for name, model in payload.get("sorbent_models", {}).items()
            },
            {
                name: CrystalThermalSpec.from_dict(model)
                for name, model in payload.get("crystal_models", {}).items()
            },
            LabClockSpec(**payload["lab_clock"]) if "lab_clock" in payload else None,
            {name: surface_spec_from_dict(model)
             for name, model in payload.get("surface_models", {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
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
            **({"surface_models": {k: v.to_dict() for k, v in self.surface_models.items()}}
               if self.surface_models else {}),
            **(
                {"flash_models": {k: v.to_dict() for k, v in self.flash_models.items()}}
                if self.flash_models
                else {}
            ),
            **(
                {"liquid_models": {k: v.to_dict() for k, v in self.liquid_models.items()}}
                if self.liquid_models
                else {}
            ),
            **(
                {"column_models": {k: v.to_dict() for k, v in self.column_models.items()}}
                if self.column_models
                else {}
            ),
            **(
                {
                    "concentration_models": {
                        k: v.to_dict() for k, v in self.concentration_models.items()
                    }
                }
                if self.concentration_models
                else {}
            ),
            **(
                {"sorbent_models": {k: v.to_dict() for k, v in self.sorbent_models.items()}}
                if self.sorbent_models
                else {}
            ),
            **(
                {"crystal_models": {k: v.to_dict() for k, v in self.crystal_models.items()}}
                if self.crystal_models
                else {}
            ),
            **(
                {
                    "thermal_models": {
                        key: model.to_dict() for key, model in self.thermal_models.items()
                    }
                }
                if self.thermal_models
                else {}
            ),
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
    def finite_supply_vessels(self):
        return set(self.feed_sources.values()) | {
            m.condensed_background.source_vessel_id
            for m in self.thermal_models.values()
            if m.condensed_background is not None
            and m.condensed_background.source_vessel_id is not None
        }

    def declared_temperature_bounds_k(self, state, modules):
        """Native constitutive domain, independent of the legacy liquid schema."""
        if state.vessel_id in self.thermal_models:
            model = self.thermal_models[state.vessel_id]
            return model.minimum_temperature_K, model.maximum_temperature_K
        bounds = []
        for factory in modules.advance_factories().values():
            strategy = factory(None)
            if strategy.supports(state) and hasattr(strategy, "temperature_bounds_k"):
                bounds.append(strategy.temperature_bounds_k(state))
        if len(bounds) > 1:
            raise ValueError("multiple native strategies own the temperature domain")
        return bounds[0] if bounds else None

    def available_operations(self, operations, modules, vessel: str) -> tuple[str, ...]:
        """Intersect declared thermal actions with the installed device capability."""
        from chemworld.runtime.aqueous_mineral import aqueous_record

        if self.lab_clock is not None:
            from chemworld.runtime.scheduling import ALLOWED

            return tuple(
                op for op in operations if op in ALLOWED
                and (op != "queue_lab_heat" or
                     (self.lab_clock.heater_id and vessel in self.thermal_models))
                and (op != "queue_lab_measure" or self.lab_clock.analyzer)
            )

        state = self.initial_states[vessel]
        aqueous = aqueous_record(state)
        native_aqueous = aqueous is not None
        if native_aqueous:
            operations = tuple(op for op in operations if op in {
                "add_component", "wait", "sample", "measure", "select_vessel", "terminate",
            } or (op == "route_material" and "mineral_kinetics" in aqueous.settings))
        if vessel in self.crystal_models:
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
            for m in self.thermal_models.values()
            if m.condensed_background is not None
        }
        if vessel in carrier_sources:
            operations = tuple(op for op in operations if op in {"select_vessel", "terminate"})
        if any(s.spec.state_type == "competitive-sorbent-v1" for s in state.state_slots.slots):
            operations = tuple(
                op
                for op in operations
                if op
                in {
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
        if vessel in self.concentration_models:
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
        if vessel in self.column_models:
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
        if vessel in self.liquid_models:
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
            if self.liquid_models[vessel].wash_feed_id not in self.feed_sources:
                operations = tuple(op for op in operations if op != "wash")
        if vessel in self.flash_models:
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
            and vessel not in self.flash_models
            and vessel not in self.liquid_models
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
                        and (vessel in self.thermal_models or vessel in self.flash_models)
                    )
                )
            )
        if (
            vessel not in self.thermal_models
            and vessel not in self.flash_models
            and vessel not in self.liquid_models
            and vessel not in self.column_models
            and vessel not in self.concentration_models
            and state.phase == "liquid"
            and not any(
                factory(None).supports(state) for factory in modules.advance_factories().values()
            )
        ):
            return operations  # Existing liquid presets retain their installed controller.
        supported = set()
        for factory in modules.advance_factories().values():
            strategy = factory(None)
            if strategy.supports(state):
                supported.update(strategy.supported_operations)
        if (
            bool(supported)
            or vessel in self.thermal_models
            or vessel in self.flash_models
            or vessel in self.liquid_models
            or vessel in self.column_models
            or vessel in self.concentration_models
        ):
            supported.add("wait")
            for record in state.equipment.equipment.values():
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

    def generate(self, spec: ScenarioSpec, seed: int, interventions=()) -> ScenarioInstance:
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

    def validate(self, spec) -> None:
        from chemworld.runtime.scheduling import validate_lab

        validate_lab(spec)
        vessels = {v["id"]: v for v in spec.vessels}
        if set(self.initial_states) != set(vessels):
            raise ValueError("physics.initial_states must match the declared vessels exactly")
        if (
            {c.vessel for c in spec.components if c.kind == "crystal_thermal"}
            != set(self.crystal_models)
            or set(self.crystal_models) - set(self.thermal_models)
            or set(self.crystal_models)
            & (
                self.flash_models.keys()
                | self.liquid_models.keys()
                | self.column_models.keys()
                | self.concentration_models.keys()
            )
        ):
            raise ValueError("crystal thermal models require unique declared thermal compartments")
        for name in self.crystal_models:
            from math import isclose

            from chemworld.physchem.elements import molecular_weight

            crystal = self.crystal_models[name]
            species = next(
                (s for s in self.network.species if s.species_id == crystal.target_component), None
            )
            if species is None:
                raise ValueError("crystal target must belong to the authored molecular network")
            if not isclose(
                crystal.kinetics.target_molecular_weight_kg_mol,
                molecular_weight(species.composition) / 1000,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "crystal molecular weight must match its authored molecular identity"
                )
            if any(
                c.vessel == name and spec.modules.components()[c.kind].processes
                for c in spec.components
            ):
                raise ValueError(
                    "this crystal thermal slice excludes simultaneous reactive processes"
                )
        contacts = [c for c in spec.components if c.kind == "sorbent_contact"]
        references = [c.parameters.get("sorbent_id") for c in contacts]
        if any(not isinstance(key, str) or not key for key in references):
            raise ValueError("sorbent components require an explicit kinetic model identity")
        if set(references) != set(self.sorbent_models) or any(
            key != model.sorbent_id for key, model in self.sorbent_models.items()
        ):
            raise ValueError("sorbent components must select exactly their declared kinetic models")
        if len({(c.vessel, c.parameters.get("sorbent_id")) for c in contacts}) != len(contacts):
            raise ValueError("one sorbent contribution must own each local kinetic model")
        if any(c.vessel not in self.thermal_models for c in contacts):
            raise ValueError("sorbent contact requires the existing coupled thermal strategy")
        if any(
            self.initial_states[c.vessel].phase != "liquid"
            or self.thermal_models[c.vessel].condensed_background is None
            for c in contacts
        ):
            raise ValueError("native sorbent contact requires its declared condensed carrier model")
        if any(
            self.thermal_models[c.vessel].condensed_background.source_vessel_id is None
            for c in contacts
        ):
            raise ValueError("native sorbent wash requires a declared finite carrier source")
        from chemworld.runtime.condensed_material_routing import validate_background_sources

        validate_background_sources(self, spec)
        for factory in spec.modules.advance_factories().values():
            strategy = factory(spec)
            validate_world = getattr(strategy, "validate_world", None)
            if validate_world is not None:
                validate_world()
        if set(self.thermal_models) - set(vessels):
            raise ValueError("thermal model must belong to a declared vessel")
        if set(self.flash_models) - set(vessels) or set(self.flash_models) & set(
            self.thermal_models
        ):
            raise ValueError("flash thermal strategy needs a unique declared vessel")
        declared_flash = {c.vessel for c in spec.components if c.kind == "vapor_liquid_equilibrium"}
        if declared_flash != set(self.flash_models):
            raise ValueError("flash models require matching VLE component declarations")
        declared_liquid = {c.vessel for c in spec.components if c.kind == "liquid_equilibrium"}
        if (
            declared_liquid != set(self.liquid_models)
            or set(self.liquid_models) - set(vessels)
            or set(self.liquid_models) & (set(self.flash_models) | set(self.thermal_models))
        ):
            raise ValueError("strict liquid models need matching, unique component declarations")
        declared_columns = {c.vessel for c in spec.components if c.kind == "dynamic_distillation"}
        if (
            declared_columns != set(self.column_models)
            or set(self.column_models) - set(vessels)
            or set(self.column_models)
            & (set(self.thermal_models) | set(self.flash_models) | set(self.liquid_models))
        ):
            raise ValueError("dynamic columns need matching, unique component declarations")
        declared_concentrators = {
            c.vessel for c in spec.components if c.kind == "vacuum_concentration"
        }
        if (
            declared_concentrators != set(self.concentration_models)
            or set(self.concentration_models) - set(vessels)
            or set(self.concentration_models)
            & (
                set(self.thermal_models)
                | set(self.flash_models)
                | set(self.liquid_models)
                | set(self.column_models)
            )
        ):
            raise ValueError("concentrators need matching, unique component declarations")
        for vessel in self.concentration_models:
            if any(
                spec.modules.components()[c.kind].processes
                for c in spec.components
                if c.vessel == vessel
            ):
                raise ValueError("this concentration slice does not support kinetic processes")
        for vessel in self.column_models:
            if any(
                spec.modules.components()[c.kind].processes
                for c in spec.components
                if c.vessel == vessel
            ):
                raise ValueError("this dynamic column slice does not support kinetic processes")
        for vessel in self.liquid_models:
            if any(
                spec.modules.components()[c.kind].processes
                for c in spec.components
                if c.vessel == vessel
            ):
                raise ValueError("this isothermal liquid slice does not support kinetic processes")
        for vessel in self.flash_models:
            if any(
                spec.modules.components()[c.kind].processes
                for c in spec.components
                if c.vessel == vessel
            ):
                raise ValueError("this VLE slice does not support simultaneous kinetic processes")
        if len(self.feed_sources) > 16 or len(set(self.feed_sources.values())) != len(
            self.feed_sources
        ):
            raise ValueError("finite feeds require unique source vessels, at most 16")
        for feed_id, vessel in self.feed_sources.items():
            if not isinstance(feed_id, str) or not feed_id or vessel not in self.initial_states:
                raise ValueError("finite feed must name a declared source vessel")
            source = self.initial_states[vessel]
            if (
                (source.phase != "gas" and not (self.flash_models or self.liquid_models
                    or {"aqueous_mineral", "mineral_kinetics"} & set(spec.component_kinds)))
                or source.phase not in {"gas", "liquid"}
                or source.state_slots.slots
                or vessel in self.thermal_models
                or vessel in self.flash_models
                or vessel in self.liquid_models
                or vessel in self.column_models
                or vessel in self.concentration_models
                or any(
                    c.vessel == vessel and spec.modules.components()[c.kind].processes
                    for c in spec.components
                )
            ):
                raise ValueError(
                    "finite source requires a declared fixed-temperature material boundary"
                )
            for model in self.thermal_models.values():
                if set(source.species_amounts) <= set(model.thermodynamics.inventory_keys):
                    for key, amount in source.species_amounts.items():
                        if amount:
                            try:
                                model.thermodynamics.species_energy_j_mol(key, source.temperature_K)
                            except RuntimeError as exc:
                                raise ValueError(
                                    "finite source is outside its caloric data domain"
                                ) from exc
            for model in self.flash_models.values():
                from chemworld.runtime.flash_thermal import inventory_enthalpy_j

                if set(source.species_amounts) != set(model.properties.component_ids):
                    raise ValueError("finite flash source must share its component identities")
                inventory_enthalpy_j(source, model)
            for model in self.liquid_models.values():
                from chemworld.runtime.liquid_equilibrium import validate_liquid_state

                validate_liquid_state(source, model, require_equilibrium=False)
        if self.network.stochastic is None and bool(self.network.reactions) != (
            "reaction" in spec.component_kinds
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
            if state.vessels.vessels[name].max_volume_L != vessels[name]["capacity_L"]:
                raise ValueError("initial-state vessel capacity differs from the composition")
            bulk_species = {s.species_id for s in self.network.species if s.phase != "surface"}
            if not set(state.species_amounts) <= bulk_species:
                raise ValueError("initial-state bulk species must belong to the explicit network")
            if name in self.thermal_models:
                required = set(self.thermal_models[name].thermodynamics.species) & bulk_species
                if not required <= state.species_amounts.keys():
                    raise ValueError("initial state omits a declared local thermal inventory")
            # Disconnected apparatus may contain different declared species.
            # Local strategy/process validation below still requires every
            # inventory it actually evolves, plus its caloric data. Padding a
            # gas vessel with fictitious zero-valued battery/liquid inventories
            # would instead demand meaningless cross-phase property tables.
            expected_units = {
                "amount": "mol",
                "volume": "L",
                "temperature": "K",
                "pressure": "Pa",
                "time": "s",
            }
            if any(state.units.get(key) != unit for key, unit in expected_units.items()):
                raise ValueError("initial-state quantities must use native mol/L/K/Pa/s units")
            if state.phase not in {"liquid", "gas", "solid"}:
                raise ValueError("initial state needs an explicit supported phase")
            report = constitution.check_state(state)
            if not report.passed:
                failed = [check.name for check in report.checks if not check.passed]
                raise ValueError(f"invalid physical initial state {name!r}: {failed}")
            if name in self.crystal_models:
                from chemworld.runtime.crystal_thermal import validate_crystal_state

                validate_crystal_state(state, self.crystal_models[name], self.thermal_models[name])
            elif name in self.thermal_models:
                from chemworld.runtime.coupled_thermal import assemble_local_system

                assemble_local_system(state, self, spec.modules, spec.components)
            if name in self.flash_models:
                from chemworld.runtime.flash_thermal import validate_flash_state

                validate_flash_state(state, self.flash_models[name])
            if name in self.liquid_models:
                from chemworld.runtime.liquid_equilibrium import validate_liquid_state

                validate_liquid_state(state, self.liquid_models[name])
            if name in self.column_models:
                from chemworld.runtime.dynamic_column import validate_column_state

                validate_column_state(state, self.column_models[name])
            if name in self.concentration_models:
                from chemworld.runtime.concentration import validate_concentration_state

                validate_concentration_state(state, self.concentration_models[name])
