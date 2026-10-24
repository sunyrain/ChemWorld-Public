"""Explicit real constituent identities for the fixed solution consumer block."""

from chemworld.data.incompressible_bindings import bind_incompressible_liquid
from chemworld.physchem.phase_evaluation import PhasePropertyModel
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.transport_devices import LiquidTransportProperties

IDENTITIES = {
    "MEG": ("glycol", "C2H6O2", "107-21-1"),
    "MPG": ("glycol", "C3H8O2", "57-55-6"),
    "MCA": ("salt", "CaCl2", "10043-52-4"),
}


def material(name):
    water = ComponentSpec("water", "H2O", cas_number="7732-18-5", aliases=("solvent",))
    if name == "Water":
        model = bind_incompressible_liquid(name, water=water)
    else:
        key, formula, cas = IDENTITIES[name.removesuffix("2")]
        solute = ComponentSpec(
            key, formula, cas_number=cas, default_phase="solid" if key == "salt" else "liquid"
        )
        model = bind_incompressible_liquid(name, water=water, solute=solute)
    packages = tuple(ComponentPropertyPackage(c, ()) for c in model.components)
    phase = PhasePropertyModel(packages, phase="liquid", incompressible_model=model)
    return model, packages, phase


def amounts(model, mass_fraction, mass_kg=1.0):
    return {
        c.identifier: mass_kg
        * (mass_fraction if c.identifier == model.solute_id else 1 - mass_fraction)
        / (c.molecular_weight_g_mol / 1000)
        for c in model.components
    }


def transport_properties(name):
    model, packages, _ = material(name)
    return LiquidTransportProperties(
        packages, temperature_bounds_K=(290.0, 312.0), incompressible_model=model
    )
