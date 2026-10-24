"""Explicit transport models, original identities, reference frames and mixture limits."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from chemworld.physchem.gas_transport import (
    GasTransportResponse,
    GasTransportSpec,
    gas_transport_from_network,
    gas_transport_response,
)
from chemworld.physchem.gas_transport_fluxes import gas_transport_fluxes

module_spec = importlib.util.spec_from_file_location(
    "gas_transport36_cases", Path(__file__).parent / "reference/gas_transport36_cases.py"
)
cases = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(cases)


def response(x=(0.7, 0.2, 0.1), **kwargs):
    network = cases.network_from_species(cases.synthetic_species())
    spec = gas_transport_from_network(network, thermal_diffusion="off", **kwargs)
    return gas_transport_response(
        spec,
        network,
        temperature_K=700,
        pressure_Pa=1e5,
        mole_fractions=dict(zip(cases.IDS, x, strict=True)),
    )


def flux(r, *, model="stefan-maxwell", frame="mass", grad=None, pressure=0, temperature=0):
    return gas_transport_fluxes(
        r,
        mole_fraction_gradients_per_m=grad or dict(zip(cases.IDS, (0.01, -0.01, 0), strict=True)),
        pressure_log_gradient_per_m=pressure,
        temperature_gradient_K_m=temperature,
        model=model,
        frame=frame,
        thermal_diffusion="off",
    )


def test_response_reuses_original_species_parameters_and_thermochemistry():
    network = cases.network_from_species(cases.synthetic_species())
    spec = gas_transport_from_network(network, thermal_diffusion="off")
    r = response()
    assert r.species_ids == network.species_ids
    assert spec.parameters[0].component_id == network.species[0].species_id
    assert spec.parameters[0].sigma_m == pytest.approx(3.621e-10)
    assert r.partial_enthalpies_J_mol["O2"] == pytest.approx(-1000 + 29.4 * (700 - 298.15))
    for s in r.species_ids:
        for t in r.species_ids:
            assert r.binary_diffusivities_m2_s[s][t] > 0
            assert r.binary_diffusivities_m2_s[s][t] == r.binary_diffusivities_m2_s[t][s]
    assert (
        GasTransportSpec.from_dict(json.loads(json.dumps(spec.to_dict()))).to_dict()
        == spec.to_dict()
    )
    assert (
        GasTransportResponse.from_dict(json.loads(json.dumps(r.to_dict()))).to_dict() == r.to_dict()
    )
    with pytest.raises(TypeError):
        r.binary_diffusivities_m2_s["O2"]["N2"] = 1


@pytest.mark.parametrize("model", ("mixture-averaged", "stefan-maxwell"))
@pytest.mark.parametrize("frame", ("mass", "molar"))
@pytest.mark.parametrize("x", ((0.7, 0.2, 0.1), (0.4, 0.6, 0), (1 - 1e-12, 1e-12, 0)))
def test_whole_vector_preserves_the_declared_frame_without_erasing_trace_species(model, frame, x):
    r = response(x)
    result = flux(r, model=model, frame=frame, pressure=0.2)
    assert result.frame_residual == pytest.approx(0, abs=1e-15)
    if x[2] == 0:
        assert result.molar_fluxes_mol_m2_s["AR"] == pytest.approx(0, abs=1e-15)
    if model == "stefan-maxwell":
        assert all(abs(v) < 1e-12 for v in result.friction_residuals_per_m.values())


@pytest.mark.parametrize("model", ("mixture-averaged", "stefan-maxwell"))
def test_frames_are_the_same_flux_with_a_reference_velocity_change(model):
    r = response()
    mass, molar = (flux(r, model=model, frame=frame, pressure=0.2) for frame in ("mass", "molar"))
    drift = sum(mass.molar_fluxes_mol_m2_s.values())
    for s in r.species_ids:
        assert molar.molar_fluxes_mol_m2_s[s] == pytest.approx(
            mass.molar_fluxes_mol_m2_s[s] - r.mole_fractions[s] * drift, abs=1e-15
        )
    assert sum(mass.mass_fluxes_kg_m2_s.values()) == pytest.approx(0, abs=1e-15)
    assert abs(sum(molar.mass_fluxes_kg_m2_s.values())) > 1e-10


def test_stefan_maxwell_binary_limit_and_pressure_driving_force():
    r = response((0.4, 0.6, 0))
    zero = dict.fromkeys(cases.IDS, 0.0)
    result = flux(r, frame="molar", grad=zero, pressure=0.2)
    d = (r.mole_fractions["N2"] - r.mass_fractions["N2"]) * 0.2
    expected = -r.molar_density_mol_m3 * r.binary_diffusivities_m2_s["N2"]["O2"] * d
    assert result.molar_fluxes_mol_m2_s["N2"] == pytest.approx(expected, abs=1e-15)
    assert result.molar_fluxes_mol_m2_s["O2"] == pytest.approx(-expected, abs=1e-15)
    assert result.driving_forces_per_m["N2"] == pytest.approx(d)


def test_heat_flux_contains_fourier_and_original_species_enthalpy_once():
    r = response()
    result = flux(r, temperature=3)
    expected = sum(
        result.molar_fluxes_mol_m2_s[s] * r.partial_enthalpies_J_mol[s] for s in r.species_ids
    )
    assert result.diffusion_enthalpy_flux_W_m2 == pytest.approx(expected)
    assert result.fourier_heat_flux_W_m2 == pytest.approx(-3 * r.conductivity_W_m_K)
    assert result.total_heat_flux_W_m2 == pytest.approx(expected - 3 * r.conductivity_W_m_K)
    reverse = flux(r, grad={"N2": -0.01, "O2": 0.01, "AR": 0}, temperature=-3)
    assert reverse.total_heat_flux_W_m2 == pytest.approx(-result.total_heat_flux_W_m2)


@pytest.mark.parametrize("model", ("mixture-averaged", "stefan-maxwell"))
def test_pure_vertex_has_zero_diffusion_under_zero_composition_drive(model):
    r = response((1, 0, 0))
    assert r.diffusion_coefficients_m2_s["mass_mole"]["N2"] is None
    assert r.binary_diffusivities_m2_s["N2"]["N2"] > 0
    result = flux(r, model=model, grad=dict.fromkeys(cases.IDS, 0.0), pressure=0.2, temperature=3)
    assert all(v == 0 for v in result.molar_fluxes_mol_m2_s.values())
    assert result.diffusion_enthalpy_flux_W_m2 == 0
    assert result.total_heat_flux_W_m2 < 0


def test_pure_vertex_uses_full_friction_for_directional_boundary_gradient():
    r = response((1, 0, 0))
    with pytest.raises(ValueError, match="directional"):
        flux(r, model="mixture-averaged")
    result = flux(r, model="stefan-maxwell")
    assert result.frame_residual == pytest.approx(0, abs=1e-15)
    assert all(abs(v) < 1e-12 for v in result.friction_residuals_per_m.values())


def test_single_component_is_not_a_singular_diffusion_problem():
    network = cases.network_from_species(cases.synthetic_species()[:1])
    spec = gas_transport_from_network(network, thermal_diffusion="off")
    r = gas_transport_response(
        spec, network, temperature_K=700, pressure_Pa=1e5, mole_fractions={"N2": 1}
    )
    for model in ("stefan-maxwell", "mixture-averaged"):
        result = gas_transport_fluxes(
            r,
            mole_fraction_gradients_per_m={"N2": 0},
            model=model,
            frame="mass",
            thermal_diffusion="off",
            temperature_gradient_K_m=3,
        )
        assert result.molar_fluxes_mol_m2_s == {"N2": 0}
        assert result.total_heat_flux_W_m2 == pytest.approx(-3 * r.conductivity_W_m_K)


def test_original_species_permutation_preserves_named_transport_and_flux():
    entries = cases.synthetic_species()
    network = cases.network_from_species(list(reversed(entries)))
    r = response()
    reordered = gas_transport_response(
        gas_transport_from_network(network, thermal_diffusion="off"),
        network,
        temperature_K=700,
        pressure_Pa=1e5,
        mole_fractions=r.mole_fractions,
    )
    assert reordered.viscosity_Pa_s == pytest.approx(r.viscosity_Pa_s)
    for model in ("stefan-maxwell", "mixture-averaged"):
        assert flux(reordered, model=model).molar_fluxes_mol_m2_s == pytest.approx(
            flux(r, model=model).molar_fluxes_mol_m2_s
        )


@pytest.mark.parametrize(
    "defect", ("charged", "polar", "no-source", "no-thermo", "soret", "pressure")
)
def test_unsupported_scope_is_explicit(defect):
    entries = cases.synthetic_species()
    if defect == "polar":
        entries[0]["transport"]["dipole"] = 1
    if defect == "no-source":
        entries[0].pop("transport")
    network = cases.network_from_species(entries)
    if defect == "charged":
        network = replace(
            network, species=(replace(network.species[0], charge=1), *network.species[1:])
        )
    if defect == "no-thermo":
        network = replace(network, metadata={})
    with pytest.raises(ValueError):
        spec = gas_transport_from_network(
            network, thermal_diffusion="on" if defect == "soret" else "off"
        )
        gas_transport_response(
            spec,
            network,
            temperature_K=700,
            pressure_Pa=1e7 if defect == "pressure" else 1e5,
            mole_fractions=dict(zip(cases.IDS, (0.7, 0.2, 0.1), strict=True)),
        )


@pytest.mark.parametrize("defect", ("missing", "non-tangent", "nan", "soret"))
def test_invalid_gradients_and_hidden_soret_are_rejected(defect):
    r = response()
    gradient = {"N2": 0.01, "O2": -0.01, "AR": 0}
    if defect == "missing":
        gradient.pop("AR")
    elif defect == "non-tangent":
        gradient["AR"] = 0.01
    elif defect == "nan":
        gradient["AR"] = float("nan")
    with pytest.raises(ValueError):
        gas_transport_fluxes(
            r,
            mole_fraction_gradients_per_m=gradient,
            model="stefan-maxwell",
            frame="mass",
            thermal_diffusion="on" if defect == "soret" else "off",
        )
