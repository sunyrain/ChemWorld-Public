"""IF97 on the existing component and phase property reports, without state."""

from dataclasses import dataclass

from chemworld.physchem.property_reports import PropertyEvaluation, StatePropertyReport
from chemworld.physchem.water_if97 import (
    WATER_CRITICAL_TEMPERATURE_K,
    WATER_GAS_CONSTANT_J_KG_K,
)
from chemworld.physchem.water_if97_solvers import water_if97_tp

# Factor is one for mass/intensive fields, molar mass for mass-specific fields.
_STATE_FIELDS = {
    "pressure": ("pressure_Pa", "Pa", False),
    "density": ("mass_density_kg_m3", "kg/m^3", False),
    "enthalpy": ("enthalpy_J_kg", "J/mol", True),
    "internal_energy": ("internal_energy_J_kg", "J/mol", True),
    "entropy": ("entropy_J_kg_K", "J/(mol*K)", True),
    "cp": ("cp_J_kg_K", "J/(mol*K)", True),
    "cv": ("cv_J_kg_K", "J/(mol*K)", True),
    "dH_dT_P": ("cp_J_kg_K", "J/(mol*K)", True),
    "dU_dT_V": ("cv_J_kg_K", "J/(mol*K)", True),
    "speed_of_sound": ("speed_of_sound_m_s", "m/s", False),
    "gibbs_energy": ("gibbs_energy_J_kg", "J/mol", True),
    "isothermal_compressibility": ("isothermal_compressibility_Pa_inv", "1/Pa", False),
    "thermal_expansion": ("thermal_expansion_K_inv", "1/K", False),
}
IF97_PROPERTY_QUANTITIES = (
    *_STATE_FIELDS, "molar_volume", "dV_dT_P", "dV_dP_T", "chemical_potential",
    "compressibility_factor", "helmholtz_energy",
)
_TRANSPORT = {"viscosity", "thermal_conductivity", "surface_tension"}


@dataclass(frozen=True)
class IF97WaterModel:
    """Fixed IAPWS release and datum; no mutable parameters or material stock."""

    def validate_component(self, component):
        if (component.cas_number != "7732-18-5" or component.charge != 0
                or component.composition != {"H": 2, "O": 1}):
            raise ValueError("IF97 requires an explicitly identified neutral H2O, CAS 7732-18-5")

    def to_dict(self):
        return {"model": "iapws-if97-2012-water-v1"}

    @classmethod
    def from_dict(cls, payload):
        if dict(payload) != {"model": "iapws-if97-2012-water-v1"}:
            raise ValueError("unknown or modified IF97 water declaration")
        return cls()


def evaluate_if97_property_state(package, model, *, temperature_K, pressure_Pa, phase, quantities):
    if not isinstance(model, IF97WaterModel) or phase not in {"gas", "liquid"}:
        raise ValueError("IF97 properties require the water model and a fluid phase")
    model.validate_component(package.component)
    requested = set(quantities)
    if (not requested or len(requested) != len(quantities)
            or requested - (set(IF97_PROPERTY_QUANTITIES) | _TRANSPORT)):
        raise ValueError("unsupported or duplicate IF97 property quantity")
    above_critical_temperature = temperature_K >= WATER_CRITICAL_TEMPERATURE_K
    if above_critical_temperature and phase != "gas":
        raise ValueError("IF97 above Tc uses the gas report label for a single fluid")
    selected = None if above_critical_temperature else ("vapor" if phase == "gas" else "liquid")
    state = water_if97_tp(temperature_K, pressure_Pa, phase=selected)
    mass = package.component.molecular_weight_g_mol / 1000
    volume = state.specific_volume_m3_kg * mass
    values = {key: (getattr(state, field) * (mass if molar else 1), unit)
              for key, (field, unit, molar) in _STATE_FIELDS.items()}
    values.update({
        "molar_volume": (volume, "m^3/mol"),
        "dV_dT_P": (volume * state.thermal_expansion_K_inv, "m^3/(mol*K)"),
        "dV_dP_T": (-volume * state.isothermal_compressibility_Pa_inv, "m^3/(mol*Pa)"),
        "chemical_potential": (state.gibbs_energy_J_kg * mass, "J/mol"),
        "helmholtz_energy": ((state.internal_energy_J_kg
                              - temperature_K * state.entropy_J_kg_K) * mass, "J/mol"),
        "compressibility_factor": (state.pressure_Pa * state.specific_volume_m3_kg
                                    / (WATER_GAS_CONSTANT_J_KG_K * temperature_K), "dimensionless"),
    })

    def evaluation(key):
        value, unit = values[key]
        return PropertyEvaluation(key, "iapws-if97-2012", f"IF97_region_{state.region}",
                                  value, unit, {"temperature": state.temperature_K,
                                                "pressure": state.pressure_Pa})

    properties = {key: evaluation(key) for key in quantities
                  if key != "chemical_potential" and key not in _TRANSPORT}
    if requested & _TRANSPORT:
        from chemworld.physchem.property_packages import evaluate_property_state

        report = evaluate_property_state(
            (package,), {package.component.identifier: 1.0}, temperature_K=temperature_K,
            pressure_Pa=state.pressure_Pa, phase=phase,
            quantities=tuple(key for key in quantities if key in _TRANSPORT),
            validity_policy="raise")
        properties.update(report.properties)
        properties = {key: properties[key] for key in quantities if key in properties}
    partials = ({"chemical_potential": {package.component.identifier:
                                       evaluation("chemical_potential")}}
                if "chemical_potential" in requested else {})
    return StatePropertyReport(
        "iapws-if97-2012", phase, state.temperature_K, state.pressure_Pa,
        {package.component.identifier: 1.0}, properties, component_properties=partials,
        model_details={
            "source": "IAPWS R7-97(2012)", "region": state.region,
            "caloric_reference": "IF97 original water reference; no added formation datum",
            "molar_conversion_kg_mol": mass,
            "specific_gas_constant_J_kg_K": WATER_GAS_CONSTANT_J_KG_K,
            "stability_scope": state.stability_scope,
            "phase_scope": "single_fluid_above_Tc" if above_critical_temperature else phase,
            "transport_scope": "only separately declared original package correlations",
        })
