"""Explicit liquid identity/composition binding on the original property reports.

The caller declares which material a correlation describes. Catalog membership
alone never supplies that identity or qualifies a solid/slurry as a liquid.
The checked convenience bindings live in data.incompressible_bindings.
"""

from dataclasses import dataclass
from math import fsum, isfinite

from chemworld.physchem.incompressible import IncompressibleFluid
from chemworld.physchem.property_reports import PropertyEvaluation, StatePropertyReport
from chemworld.physchem.specs import ComponentSpec, resolve_component_identifier

_FIELDS = {
    "pressure": ("pressure_Pa", "Pa", False),
    "density": ("mass_density_kg_m3", "kg/m^3", False),
    "cp": ("cp_J_kg_K", "J/(mol*K)", True),
    "source_cp": ("source_cp_J_kg_K", "J/(mol*K)", True),
    "dH_dT_P": ("cp_J_kg_K", "J/(mol*K)", True),
    "enthalpy": ("enthalpy_J_kg", "J/mol", True),
    "internal_energy": ("internal_energy_J_kg", "J/mol", True),
    "entropy": ("entropy_J_kg_K", "J/(mol*K)", True),
    "gibbs_energy": ("gibbs_energy_J_kg", "J/mol", True),
    "thermal_expansion": ("thermal_expansion_K_inv", "1/K", False),
}
_PARTIALS = {
    "partial_enthalpy": ("enthalpy_J_kg", "dh_dx_J_kg", "J/mol"),
    "partial_entropy": ("entropy_J_kg_K", "ds_dx_J_kg_K", "J/(mol*K)"),
    "partial_internal_energy": (None, None, "J/mol"),
    "partial_volume": (None, None, "m^3/mol"),
}
INCOMPRESSIBLE_PROPERTY_QUANTITIES = (
    *_FIELDS,
    *_PARTIALS,
    "molar_volume",
    "dV_dT_P",
    "dV_dP_T",
    "isothermal_compressibility",
    "viscosity",
    "thermal_conductivity",
)


def _identity(component):
    return (
        component.identifier,
        component.formula,
        component.cas_number,
        component.charge,
        component.molecular_weight_g_mol,
        component.inchi,
        component.inchi_key,
    )


@dataclass(frozen=True)
class IncompressibleMaterialModel:
    fluid: IncompressibleFluid
    components: tuple[ComponentSpec, ...]
    solute_id: str | None
    binding_source: str
    preparation_molar_volumes_m3_mol: tuple[float, ...] | None = None
    preparation_reference_temperature_K: float | None = None
    preparation_source: str = ""

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if (
            not isinstance(self.fluid, IncompressibleFluid)
            or not components
            or any(not isinstance(c, ComponentSpec) for c in components)
            or len({c.identifier for c in components}) != len(components)
            or not isinstance(self.binding_source, str)
            or not self.binding_source.strip()
        ):
            raise ValueError(
                "incompressible material requires explicit components and source binding"
            )
        pure = self.fluid.fraction_basis == "pure"
        if (
            len(components) != (1 if pure else 2)
            or any(c.charge for c in components)
            or (pure and self.solute_id is not None)
            or (not pure and self.solute_id not in {c.identifier for c in components})
        ):
            raise ValueError("incompressible binding requires neutral pure or binary constituents")
        object.__setattr__(self, "components", components)
        volumes, temperature = (
            self.preparation_molar_volumes_m3_mol,
            self.preparation_reference_temperature_K,
        )
        if self.fluid.fraction_basis == "volume":
            if (
                volumes is None
                or len(volumes) != len(components)
                or any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in volumes)
                or isinstance(temperature, bool)
                or temperature is None
                or not isfinite(temperature)
                or temperature <= 0
                or not isinstance(self.preparation_source, str)
                or not self.preparation_source.strip()
            ):
                raise ValueError("volume concentration needs explicit preparation volumes/T/source")
            object.__setattr__(self, "preparation_molar_volumes_m3_mol", tuple(volumes))
        elif volumes is not None or temperature is not None or self.preparation_source:
            raise ValueError("preparation volumes apply only to a volume-fraction source fit")

    @property
    def component_ids(self):
        return tuple(c.identifier for c in self.components)

    def validate_components(self, components):
        supplied = {_identity(c) for c in components}
        if len(supplied) != len(components) or supplied != {_identity(c) for c in self.components}:
            raise ValueError(
                "incompressible model/component identities or molecular weights differ"
            )

    def inventory_coordinates(self, amounts):
        """Return the caller inventory's mass and declared x(n), never a new stock."""
        values = dict.fromkeys(self.component_ids, 0.0)
        for alias, value in amounts.items():
            if isinstance(value, bool) or not isfinite(value) or value < 0:
                raise ValueError("incompressible amounts must be finite and nonnegative")
            key = resolve_component_identifier(self.components, alias)
            values[key] += value
        masses = {c.identifier: c.molecular_weight_g_mol / 1000 for c in self.components}
        total = fsum(values.values())
        mass = fsum(values[k] * masses[k] for k in values)
        if not 0 < total < float("inf") or not 0 < mass < float("inf"):
            raise ValueError("incompressible property state requires positive finite material")
        basis = self.fluid.fraction_basis
        if basis == "pure":
            return values, total, mass, 1.0, dict.fromkeys(values, 0.0)
        weights = (
            masses
            if basis == "mass"
            else dict.fromkeys(values, 1.0)
            if basis == "mole"
            else (dict(zip(self.component_ids, self.preparation_molar_volumes_m3_mol, strict=True)))
        )
        denominator = fsum(values[k] * weights[k] for k in values)
        if not 0 < denominator < float("inf"):
            raise ValueError("nonfinite concentration basis inventory")
        x = values[self.solute_id] * weights[self.solute_id] / denominator
        self.fluid.concentration(x, basis=basis)
        slopes = {k: weights[k] * (float(k == self.solute_id) - x) / denominator for k in values}
        return values, total, mass, x, slopes

    def available_quantities(self):
        result = dict.fromkeys(INCOMPRESSIBLE_PROPERTY_QUANTITIES, True)
        fits = dict(self.fluid.correlations)
        result["viscosity"] = fits["viscosity"] is not None
        result["thermal_conductivity"] = fits["conductivity"] is not None
        return result

    def to_dict(self):
        return {
            "model": "incompressible-material-v1",
            "fluid": self.fluid.to_dict(),
            "components": [c.to_dict() for c in self.components],
            "solute_id": self.solute_id,
            "binding_source": self.binding_source,
            "preparation_molar_volumes_m3_mol": (
                None
                if self.preparation_molar_volumes_m3_mol is None
                else list(self.preparation_molar_volumes_m3_mol)
            ),
            "preparation_reference_temperature_K": self.preparation_reference_temperature_K,
            "preparation_source": self.preparation_source,
        }

    @classmethod
    def from_dict(cls, payload):
        data = dict(payload)
        if data.pop("model") != "incompressible-material-v1":
            raise ValueError("unknown incompressible material schema")
        data["fluid"] = IncompressibleFluid.from_dict(data["fluid"])
        data["components"] = tuple(ComponentSpec.from_dict(c) for c in data["components"])
        return cls(**data)


def evaluate_incompressible_property_state(
    packages, model, component_amounts_mol, *, temperature_K, pressure_Pa, quantities
):
    if not isinstance(model, IncompressibleMaterialModel):
        raise TypeError("incompressible property evaluation requires a material binding")
    model.validate_components(tuple(p.component for p in packages))
    requested = set(quantities)
    available = model.available_quantities()
    if (
        not requested
        or len(requested) != len(quantities)
        or any(not available.get(key, False) for key in requested)
    ):
        raise ValueError("unsupported or duplicate incompressible property quantity")
    amounts, total, mass, x, slopes = model.inventory_coordinates(component_amounts_mol)
    T, p = temperature_K, pressure_Pa
    state = model.fluid.state(T, p, x)
    mass_per_mole = mass / total
    volume = state.specific_volume_m3_kg
    volume_x = -state.drho_dx_kg_m3 / state.mass_density_kg_m3**2
    values = {
        key: (getattr(state, field) * (mass_per_mole if molar else 1.0), unit)
        for key, (field, unit, molar) in _FIELDS.items()
    }
    values.update(
        {
            "molar_volume": (mass_per_mole * volume, "m^3/mol"),
            "dV_dT_P": (mass_per_mole * volume * state.thermal_expansion_K_inv, "m^3/(mol*K)"),
            "dV_dP_T": (0.0, "m^3/(mol*Pa)"),
            "isothermal_compressibility": (0.0, "1/Pa"),
        }
    )
    for key, field, unit in (
        ("viscosity", "viscosity", "Pa*s"),
        ("thermal_conductivity", "conductivity", "W/(m*K)"),
    ):
        if key in requested:
            values[key] = (model.fluid.property_value(field, T, p, x), unit)

    def evaluation(key, value, unit):
        return PropertyEvaluation(
            key,
            model.fluid.fluid_id,
            "incompressible-native-v1",
            value,
            unit,
            {"temperature": T, "pressure": p, "concentration": x},
        )

    partials = {}
    for key in requested & _PARTIALS.keys():
        field, derivative, unit = _PARTIALS[key]
        if key == "partial_volume":
            value, slope = volume, volume_x
        elif key == "partial_internal_energy":
            value, slope = state.internal_energy_J_kg, state.dh_dx_J_kg - p * volume_x
        else:
            value, slope = getattr(state, field), getattr(state, derivative)
        partials[key] = {
            c.identifier: evaluation(
                key,
                c.molecular_weight_g_mol / 1000 * value + mass * slope * slopes[c.identifier],
                unit,
            )
            for c in model.components
        }
    return StatePropertyReport(
        "incompressible:" + model.fluid.fluid_id,
        "liquid",
        T,
        p,
        amounts,
        {key: evaluation(key, *values[key]) for key in quantities if key in values},
        component_properties=partials,
        model_details={
            "parameter_source": model.fluid.source,
            "material_binding_source": model.binding_source,
            "fraction_basis": state.fraction_basis,
            "fraction": x,
            "mass_kg": mass,
            "molar_mass_kg_mol": mass_per_mole,
            "reference": model.fluid.reference.to_dict(),
            "reference_temperature_outside_fit": state.reference_temperature_outside_fit,
            "freezing_boundary_available": state.freezing_boundary_available,
            "saturation_boundary_available_at_temperature": (
                state.saturation_boundary_available_at_temperature
            ),
            "caloric_scope": state.caloric_scope,
            "stability_scope": state.stability_scope,
            "preparation_molar_volumes_m3_mol": model.preparation_molar_volumes_m3_mol,
            "preparation_reference_temperature_K": model.preparation_reference_temperature_K,
            "preparation_source": model.preparation_source,
            "partial_scope": "derivatives_of_declared_reference; no_equilibrium_mu",
        },
    )
