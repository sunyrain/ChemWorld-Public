"""Bind native Helmholtz states to the existing component property reports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from chemworld.physchem.helmholtz_solvers import HelmholtzPhaseModel
from chemworld.physchem.property_reports import (
    PhaseLabel,
    PropertyEvaluation,
    StatePropertyReport,
)

if TYPE_CHECKING:
    from chemworld.physchem.property_packages import ComponentPropertyPackage


_STATE_FIELDS = {
    "pressure": ("pressure_Pa", "Pa"),
    "density": ("mass_density_kg_m3", "kg/m^3"),
    "enthalpy": ("enthalpy_J_mol", "J/mol"),
    "internal_energy": ("internal_energy_J_mol", "J/mol"),
    "entropy": ("entropy_J_mol_K", "J/(mol*K)"),
    "cp": ("cp_J_mol_K", "J/(mol*K)"),
    "cv": ("cv_J_mol_K", "J/(mol*K)"),
    "dH_dT_P": ("cp_J_mol_K", "J/(mol*K)"),
    "dU_dT_V": ("cv_J_mol_K", "J/(mol*K)"),
    "speed_of_sound": ("speed_of_sound_m_s", "m/s"),
    "compressibility_factor": ("compressibility_factor", "dimensionless"),
    "gibbs_energy": ("gibbs_energy_J_mol", "J/mol"),
    "helmholtz_energy": ("helmholtz_energy_J_mol", "J/mol"),
    "isothermal_compressibility": ("isothermal_compressibility_Pa_inv", "1/Pa"),
    "thermal_expansion": ("thermal_expansion_K_inv", "1/K"),
}


def evaluate_helmholtz_property_state(
    package: ComponentPropertyPackage, model: HelmholtzPhaseModel, *, temperature_K: float,
    pressure_Pa: float, phase: str, quantities: Sequence[str],
) -> StatePropertyReport:
    """Use the same EOS for pressure, energy and derivatives on a one-mole basis.

    The explicit phase selects a homogeneous branch. The solver's metastability
    and stability-scope diagnostics remain visible; no global phase claim is
    inferred from the caller's label. Absolute caloric offsets are the original
    EOS convention, with no additional pure-Cp integration or datum added.
    """
    if not isinstance(model, HelmholtzPhaseModel) or phase not in {"gas", "liquid"}:
        raise ValueError("Helmholtz properties require a phase model and a fluid phase")
    requested = set(quantities)
    extra = {"molar_volume", "dV_dT_P", "dV_dP_T", "chemical_potential"}
    transport = {"viscosity", "thermal_conductivity", "surface_tension"}
    if not requested or requested - (_STATE_FIELDS.keys() | extra | transport):
        raise ValueError("unsupported Helmholtz property; transport needs a declared relation")
    fluid, component = model.fluid, package.component
    if fluid.casrn:
        if fluid.casrn != component.cas_number:
            raise ValueError("Helmholtz source CAS must match the existing component identity")
    elif fluid.fluid_id not in {component.identifier, *component.aliases}:
        raise ValueError("a source without CAS must explicitly match the component name or alias")
    solved = model.solve_tp(temperature_K=temperature_K, pressure_Pa=pressure_Pa, phase_hint=phase)
    state = solved.state
    if not solved.success or state is None:
        raise ValueError(f"Helmholtz property state did not converge: {solved.failure_reason}")
    rho = state.molar_density_mol_m3
    values = {key: (float(getattr(state, attribute)), unit)
              for key, (attribute, unit) in _STATE_FIELDS.items()}
    values.update({
        "molar_volume": (1 / rho, "m^3/mol"),
        "dV_dT_P": (state.dP_dT_rho_Pa_K / (rho**2 * state.dP_drho_T_Pa_m3_mol),
                     "m^3/(mol*K)"),
        "dV_dP_T": (-1 / (rho**2 * state.dP_drho_T_Pa_m3_mol), "m^3/(mol*Pa)"),
        "chemical_potential": (state.gibbs_energy_J_mol, "J/mol"),
    })
    model_id = f"helmholtz:{fluid.fluid_id}:{fluid.eos_index}"

    def evaluation(key: str) -> PropertyEvaluation:
        value, unit = values[key]
        return PropertyEvaluation(key, model_id, "pure_helmholtz_single_energy", value, unit,
                                  {"temperature": temperature_K, "pressure": state.pressure_Pa},
                                  state.warnings)

    properties = {key: evaluation(key) for key in quantities
                  if key != "chemical_potential" and key not in transport}
    if requested & transport:
        from chemworld.physchem.property_packages import evaluate_property_state

        # Use the original explicit transport selection and validity checks.
        # No caloric or volume value from that path replaces the EOS state.
        transport_report = evaluate_property_state(
            (package,), {component.identifier: 1.0}, temperature_K=temperature_K,
            pressure_Pa=state.pressure_Pa, phase=phase,
            quantities=tuple(key for key in quantities if key in transport),
            validity_policy="raise",
        )
        properties.update(transport_report.properties)
        properties = {key: properties[key] for key in quantities if key in properties}
    partials = ({"chemical_potential": {component.identifier: evaluation("chemical_potential")}}
                if "chemical_potential" in requested else {})
    return StatePropertyReport(
        model_id, cast(PhaseLabel, phase), temperature_K, state.pressure_Pa,
        {component.identifier: 1.0}, properties,
        component_properties=partials, model_details={
            "source": fluid.source, "citation": fluid.citation, "eos_index": fluid.eos_index,
            "source_fluid_id": fluid.fluid_id, "source_cas": fluid.casrn,
            "source_molar_mass_kg_mol": fluid.molar_mass_kg_mol,
            "source_gas_constant_J_mol_K": fluid.gas_constant_J_mol_K,
            "caloric_reference": "original EOS ideal lead and enthalpy/entropy offsets",
            "pseudo_pure": fluid.pseudo_pure, "state_solver": dict(solved.diagnostics),
        },
    )
