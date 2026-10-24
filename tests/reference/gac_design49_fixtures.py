"""Fixed 49E synthetic coverage; no upstream model is imported by native code."""

from dataclasses import replace

from chemworld.examples.gac_design import gac_design_example
from chemworld.physchem.transport import PumpSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.aqueous_adsorption_world import aqueous_adsorption_world_spec
from reference.aqueous_adsorption49_fixtures import initial_amounts, spec_case

METRICS = (
    "equilibrium_loading_kg_kg",
    "distribution_parameter",
    "biot_number",
    "minimum_stanton_number",
    "throughput_parameter",
    "bulk_density_kg_m3",
    "interstitial_velocity_m_s",
    "bed_length_m",
    "bed_diameter_m",
    "bed_area_m2",
    "bed_volume_m3",
    "carbon_mass_kg",
    "residence_time_s",
    "minimum_ebct_s",
    "minimum_residence_time_s",
    "minimum_operational_time_s",
    "operational_time_s",
    "bed_volumes_treated",
    "carbon_usage_kg_s",
    "average_concentration_ratio",
    "adsorbed_mass_kg",
    "film_transfer_m_s",
    "surface_diffusion_m2_s",
    "average_outlet_target_kg_s",
)


def numerical_cases():
    cases = []
    for film in (False, True):
        for surface in (False, True):
            for i, concentration in enumerate((0.002, 0.02, 0.2)):
                spec, feed = gac_design_example(calculated_film=film, calculated_surface=surface)
                spec = replace(
                    spec,
                    empirical=replace(spec.empirical, freundlich_exponent=(0.45, 0.65, 0.85)[i]),
                    bed=replace(
                        spec.bed,
                        particle_diameter_m=(0.0007, 0.001, 0.0014)[i],
                        void_fraction=(0.35, 0.4, 0.5)[i],
                        empty_bed_contact_time_s=(120.0, 600.0, 1800.0)[i],
                        superficial_velocity_m_s=(0.003, 0.01, 0.02)[i],
                    ),
                    replacement_concentration_ratio=(0.1, 0.5, 0.9)[i],
                )
                feed = replace(
                    feed,
                    concentration_kg_m3=concentration,
                    volumetric_flow_m3_s=(1e-4, 1e-3, 1e-2)[i],
                    temperature_K=(288.15, 298.15, 318.15)[i],
                    pressure_Pa=(1e5, 1.5e5, 2e5)[i],
                    liquid_density_kg_m3=(990.0, 1000.0, 1010.0)[i],
                    dynamic_viscosity_Pa_s=(0.0012, 0.001, 0.0008)[i],
                )
                cases.append((f"film-{film}-surface-{surface}-level-{i}", spec, feed))
    return cases


def commit(runtime, operation, **kwargs):
    result = runtime.execute({"operation": operation, **kwargs})
    if result.info["transaction_status"] != "committed":
        raise RuntimeError(result.info)
    return result


def world_case(shared=False, advanced=False, *, blocked=False):
    # A low power limit makes the requested flow differ from its actual value.
    original = replace(spec_case(), pump=PumpSpec(1e5, 1e-7 if not blocked else 0.0, 0.7))
    runtime = ChemWorldRuntime(
        composition=aqueous_adsorption_world_spec(
            original,
            initial_amounts(original),
            shared_clock=shared,
        )
    )
    runtime.reset(seed=49)
    commit(runtime, "set_flow_rate", flow_rate_mL_min=10.0, flow_path="process")
    if advanced:
        commit(runtime, "wait" if shared else "run_flow", duration_s=10.0)
        commit(runtime, "select_vessel", vessel="product")
    design, _ = gac_design_example(calculated_film=shared, calculated_surface=advanced)
    if advanced:
        design = replace(
            design,
            empirical=replace(design.empirical, freundlich_k=1e-6),
            bed=replace(design.bed, empty_bed_contact_time_s=120.0),
        )
    return runtime, design
