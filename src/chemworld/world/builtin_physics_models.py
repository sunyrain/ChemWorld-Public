"""Codecs and binding checks owned by the built-in domain declarations."""

from math import isclose

from chemworld.physchem.coupled_concentration import CoupledConcentrationSpec
from chemworld.physchem.dynamic_distillation import DynamicColumnSpec
from chemworld.physchem.flash_units import FlashSystemSpec
from chemworld.physchem.heat_transfer_units import ThermalSystemSpec
from chemworld.physchem.phase_equilibrium_units import StrictLiquidSystemSpec
from chemworld.physchem.sorbent_kinetics import SorbentKineticsSpec
from chemworld.physchem.surface_specs import surface_spec_from_dict
from chemworld.world.domain_modules import PhysicsModelSpec


def _encode(model):
    return model.to_dict()


def _decode_crystal(payload):
    # Installing a declaration does not require loading its numerical model.
    from chemworld.physchem.crystal_thermal_kinetics import CrystalThermalSpec

    return CrystalThermalSpec.from_dict(payload)


def _decode_calphad_material(payload):
    if payload.get("model_type") == "calphad_multiphase_thermal_v1":
        from chemworld.runtime.calphad_thermal import CalphadThermalSpec

        return CalphadThermalSpec.from_dict(payload)
    from chemworld.physchem.calphad_material import CefMaterialSpec

    return CefMaterialSpec.from_dict(payload)


def _decode_sbml(payload):
    from chemworld.physchem.sbml_system import SBMLSystemSpec

    return SBMLSystemSpec.from_dict(payload)


def _decode_stochastic_hybrid(payload):
    from chemworld.physchem.stochastic_hybrid import StochasticHybridSpec

    return StochasticHybridSpec.from_dict(payload)


def _decode_molecular_process(payload):
    from chemworld.runtime.molecular_process_runtime import MolecularProcessReceiverSpec

    return MolecularProcessReceiverSpec.from_dict(payload)


def _decode_flash(payload):
    if payload.get("model_type") == "molecular_flash_thermal_v1":
        from chemworld.runtime.molecular_flash_thermal import MolecularFlashThermalSpec

        return MolecularFlashThermalSpec.from_dict(payload)
    return FlashSystemSpec.from_dict(payload)


def _calphad_material(physics, spec):
    from chemworld.runtime.solid_solutions import CEF_TYPE

    for vessel in physics.models_of("calphad_material"):
        records = [r for r in physics.initial_states[vessel].equipment.equipment.values()
                   if r.equipment_type == CEF_TYPE and r.attached_vessel_id == vessel]
        if len(records) != 1:
            raise ValueError("CEF model binding requires exactly one installed material reference")


def _without_processes(spec, vessels, label):
    if any(
        c.vessel in vessels and spec.modules.components()[c.kind].processes for c in spec.components
    ):
        raise ValueError(f"this {label} slice does not support simultaneous kinetic processes")


def _thermal(physics, spec):
    from chemworld.runtime.condensed_material_routing import validate_background_sources
    from chemworld.runtime.coupled_thermal import assemble_local_system

    validate_background_sources(physics, spec)
    bulk_species = {s.species_id for s in physics.network.species if s.phase != "surface"}
    for name, model in physics.models_of("thermal").items():
        state = physics.initial_states[name]
        required = set(model.thermodynamics.inventory_keys) & bulk_species
        if not required <= state.species_amounts.keys():
            raise ValueError("initial state omits a declared local thermal inventory")
        if model.uses_eos_pressure:
            _without_processes(spec, (name,), "declared EOS thermal")
        if name not in physics.models_of("crystal"):
            assemble_local_system(state, physics, spec.modules, spec.components)


def _thermal_feed(source, model):
    if model.uses_eos_pressure:
        raise ValueError("EOS world feed needs an explicitly declared EOS enthalpy port")
    if set(source.species_amounts) <= set(model.thermodynamics.inventory_keys):
        for key, amount in source.species_amounts.items():
            if amount:
                try:
                    model.thermodynamics.species_energy_j_mol(key, source.temperature_K)
                except RuntimeError as exc:
                    raise ValueError("finite source is outside its caloric data domain") from exc


def _flash(physics, spec):
    from chemworld.runtime.flash_thermal import validate_flash_state

    _without_processes(spec, physics.models_of("flash"), "VLE")
    for name, model in physics.models_of("flash").items():
        validate_flash_state(physics.initial_states[name], model)


def _flash_feed(source, model):
    from chemworld.runtime.flash_thermal import inventory_enthalpy_j
    from chemworld.runtime.molecular_flash_thermal import MolecularFlashThermalSpec

    if isinstance(model, MolecularFlashThermalSpec):
        raise ValueError("molecular two-phase caloric branch does not declare material feeds")
    if set(source.species_amounts) != set(model.properties.component_ids):
        raise ValueError("finite flash source must share its component identities")
    inventory_enthalpy_j(source, model)


def _liquid(physics, spec):
    from chemworld.runtime.liquid_equilibrium import validate_liquid_state

    _without_processes(spec, physics.models_of("liquid"), "isothermal liquid")
    for name, model in physics.models_of("liquid").items():
        validate_liquid_state(physics.initial_states[name], model)


def _liquid_feed(source, model):
    from chemworld.runtime.liquid_equilibrium import validate_liquid_state

    validate_liquid_state(source, model, require_equilibrium=False)


def _column(physics, spec):
    from chemworld.runtime.dynamic_column import validate_column_state

    _without_processes(spec, physics.models_of("column"), "dynamic column")
    for name, model in physics.models_of("column").items():
        validate_column_state(physics.initial_states[name], model)


def _concentration(physics, spec):
    from chemworld.runtime.concentration import validate_concentration_state

    _without_processes(spec, physics.models_of("concentration"), "concentration")
    for name, model in physics.models_of("concentration").items():
        validate_concentration_state(physics.initial_states[name], model)


def _crystal(physics, spec):
    from chemworld.physchem.elements import molecular_weight
    from chemworld.runtime.crystal_thermal import validate_crystal_state

    crystals, thermal = physics.models_of("crystal"), physics.models_of("thermal")
    if set(crystals) - set(thermal):
        raise ValueError("crystal thermal models require unique declared thermal compartments")
    _without_processes(spec, crystals, "crystal thermal")
    for name, crystal in crystals.items():
        species = next(
            (s for s in physics.network.species if s.species_id == crystal.target_component), None
        )
        if species is None:
            raise ValueError("crystal target must belong to the authored molecular network")
        if not isclose(
            crystal.kinetics.target_molecular_weight_kg_mol,
            molecular_weight(species.composition) / 1000,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("crystal molecular weight must match its authored molecular identity")
        validate_crystal_state(physics.initial_states[name], crystal, thermal[name])


def _sorbent(physics, spec):
    contacts = [c for c in spec.components if c.kind == "sorbent_contact"]
    references = [c.parameters.get("sorbent_id") for c in contacts]
    models, thermal = physics.models_of("sorbent"), physics.models_of("thermal")
    if any(not isinstance(key, str) or not key for key in references):
        raise ValueError("sorbent components require an explicit kinetic model identity")
    if set(references) != set(models) or any(
        key != model.sorbent_id for key, model in models.items()
    ):
        raise ValueError("sorbent components must select exactly their declared kinetic models")
    if len({(c.vessel, c.parameters.get("sorbent_id")) for c in contacts}) != len(contacts):
        raise ValueError("one sorbent contribution must own each local kinetic model")
    if any(c.vessel not in thermal for c in contacts):
        raise ValueError("sorbent contact requires the existing coupled thermal strategy")
    if any(
        physics.initial_states[c.vessel].phase != "liquid"
        or thermal[c.vessel].condensed_background is None
        for c in contacts
    ):
        raise ValueError("native sorbent contact requires its declared condensed carrier model")
    # A shared closed-system clock exposes no wash/input operation. Its finite
    # initial carrier is sufficient for adsorption and thermal evolution.
    if physics.lab_clock is None and any(
        thermal[c.vessel].condensed_background.source_vessel_id is None for c in contacts
    ):
        raise ValueError("native sorbent wash requires a declared finite carrier source")


def _surface(physics, spec):
    models = physics.models_of("surface")
    references = [
        c.parameters.get("surface_id")
        for c in spec.components
        if c.kind == "surface_reaction" and c.parameters.get("surface_id") is not None
    ]
    if set(references) - set(models) or any(
        key != model.mechanism_id for key, model in models.items()
    ):
        raise ValueError(
            "surface components must select declared mechanisms with matching identities"
        )


THERMAL = PhysicsModelSpec(
    "thermal",
    "1",
    ThermalSystemSpec.from_dict,
    _encode,
    legacy_field="thermal_models",
    exclusive_group="thermal_evolution",
    validate_world=_thermal,
    validate_feed=_thermal_feed,
)
FLASH = PhysicsModelSpec(
    "flash",
    "1",
    _decode_flash,
    _encode,
    legacy_field="flash_models",
    component_kind="vapor_liquid_equilibrium",
    exclusive_group="thermal_evolution",
    validate_world=_flash,
    validate_feed=_flash_feed,
    accepts_liquid_feed=True,
)
LIQUID = PhysicsModelSpec(
    "liquid",
    "1",
    StrictLiquidSystemSpec.from_dict,
    _encode,
    legacy_field="liquid_models",
    component_kind="liquid_equilibrium",
    exclusive_group="thermal_evolution",
    validate_world=_liquid,
    validate_feed=_liquid_feed,
    accepts_liquid_feed=True,
)
COLUMN = PhysicsModelSpec(
    "column",
    "1",
    DynamicColumnSpec.from_dict,
    _encode,
    legacy_field="column_models",
    component_kind="dynamic_distillation",
    exclusive_group="thermal_evolution",
    validate_world=_column,
)
CONCENTRATION = PhysicsModelSpec(
    "concentration",
    "1",
    CoupledConcentrationSpec.from_dict,
    _encode,
    legacy_field="concentration_models",
    component_kind="vacuum_concentration",
    exclusive_group="thermal_evolution",
    validate_world=_concentration,
)
CRYSTAL = PhysicsModelSpec(
    "crystal",
    "1",
    _decode_crystal,
    _encode,
    legacy_field="crystal_models",
    component_kind="crystal_thermal",
    validate_world=_crystal,
)
SORBENT = PhysicsModelSpec(
    "sorbent",
    "1",
    SorbentKineticsSpec.from_dict,
    _encode,
    legacy_field="sorbent_models",
    binding="named",
    validate_world=_sorbent,
)
SURFACE = PhysicsModelSpec(
    "surface",
    "1",
    surface_spec_from_dict,
    _encode,
    legacy_field="surface_models",
    binding="named",
    validate_world=_surface,
)
CALPHAD_MATERIAL = PhysicsModelSpec(
    "calphad_material", "1", _decode_calphad_material, _encode,
    component_kind="calphad_phase", exclusive_group="thermal_evolution",
    validate_world=_calphad_material,
)
MOLECULAR_PROCESS = PhysicsModelSpec(
    "molecular_process", "1", _decode_molecular_process, _encode,
    component_kind="molecular_process", exclusive_group="thermal_evolution",
)
SBML = PhysicsModelSpec("sbml", "1", _decode_sbml, _encode,
                       component_kind="sbml_model", exclusive_group="thermal_evolution")
STOCHASTIC_HYBRID = PhysicsModelSpec(
    "stochastic_hybrid", "1", _decode_stochastic_hybrid, _encode,
    exclusive_group="thermal_evolution",
)
