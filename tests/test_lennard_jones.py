"""LJ source binding, domain handling and actual native transport consumers."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.data.constants import ConflictingConstantDataError, ConstantSourceUnavailableError
from chemworld.data.lennard_jones import LENNARD_JONES_SOURCES, LennardJonesDatabase
from chemworld.foundation.units import Quantity
from chemworld.physchem.finite_volume import ConservativeDiffusion, DiffusionMesh
from chemworld.physchem.lennard_jones import (
    COLLISION_PAIRS,
    LennardJonesSpec,
    lennard_jones_collision_integral,
    lennard_jones_diameter,
    lennard_jones_diffusivity_report,
    lennard_jones_energy,
    lennard_jones_reduced_temperature,
    lennard_jones_viscosity_report,
)
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
)
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.transport_properties import gas_mixture_effective_diffusivity_ledger


@pytest.fixture
def nitrogen():
    return ComponentSpec("nitrogen", "N2", default_phase="gas", cas_number="7727-37-9")


@pytest.fixture
def parameters(nitrogen):
    return LennardJonesSpec(nitrogen.identifier, 100, 3.7e-10, "Explicit synthetic LJ parameters")


@pytest.fixture
def database(tmp_path):
    for source, diameter in zip(LENNARD_JONES_SOURCES, (3.7, 3.8), strict=True):
        path = tmp_path / source.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("CAS\tChemical\tmolecular_diameter\tStockmayer\n"
                        f"7727-37-9\tNitrogen\t{diameter}\t100\n"
                        "7440-37-1\tArgon\t\t120\n", encoding="utf-8")
    return LennardJonesDatabase(tmp_path)


def test_sources_are_lazy_conflicts_explicit_and_parameter_identity_checked(database, nitrogen):
    assert database.loaded_sources == ()
    with pytest.raises(ConflictingConstantDataError):
        database.resolve("7727-37-9", "molecular_diameter")
    p = database.parameters("7727-37-9", nitrogen, energy_method="MAGALHAES_LJ",
                            diameter_method="POLING_LJ")
    assert p.epsilon_k_K == 100 and p.sigma_m == pytest.approx(3.8e-10)
    assert p.sources["epsilon_k"]["source"]["source_id"].endswith("MAGALHAES_LJ")
    assert p.sources["sigma"]["source"]["source_id"].endswith("POLING_LJ")
    with pytest.raises(ValueError, match="identity"):
        database.parameters("7727-37-9", ComponentSpec("oxygen", "O2"),
                            energy_method="MAGALHAES_LJ", diameter_method="POLING_LJ")
    with pytest.raises(ValueError, match="missing"):
        database.parameters("7440-37-1", ComponentSpec("argon", "Ar", cas_number="7440-37-1"),
                            energy_method="MAGALHAES_LJ", diameter_method="POLING_LJ")


@pytest.mark.parametrize("method", ["Neufeld_Janzen_Aziz", "Kim_Monroe"])
def test_original_input_card_package_and_report_agree_after_json(database, nitrogen, method):
    card = database.viscosity_input("7727-37-9", nitrogen, energy_method="MAGALHAES_LJ",
                                    diameter_method="POLING_LJ", collision_method=method,
                                    temperature_range_K=(100, 1000))
    copied = PropertyInputCard.from_dict(json.loads(json.dumps(card.to_dict())))
    assert copied.to_dict() == card.to_dict()
    p = database.parameters("7727-37-9", nitrogen, energy_method="MAGALHAES_LJ",
                            diameter_method="POLING_LJ")
    assert LennardJonesSpec.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    report = lennard_jones_viscosity_report(nitrogen, p, temperature_K=300,
                                           collision_method=method)
    state = copied.evaluate_state(temperature_K=300, pressure_Pa=101325, phase="gas",
                                   quantities=("viscosity",))
    assert state.properties["viscosity"].value == pytest.approx(report.evaluation.value)
    assert (copied.package.evaluate("gas_viscosity", temperature_K=500).value
            > report.evaluation.value)
    slope = correlation_temperature_derivative(copied.package.correlations[0], temperature_K=300)
    assert slope.to("mPa*s/K").value == pytest.approx(1000 * slope.value)
    with pytest.raises(ValueError):
        copied.package.evaluate("gas_viscosity", temperature_K=1200, validity_policy="raise")


@pytest.mark.parametrize("method", ["Neufeld_Janzen_Aziz", "Kim_Monroe"])
def test_diffusion_reaches_original_mixture_ledger_and_conservative_operator(
    nitrogen, parameters, method,
):
    oxygen = ComponentSpec("oxygen", "O2", default_phase="gas")
    other = LennardJonesSpec("oxygen", 120, 3.5e-10, "Explicit synthetic LJ parameters")

    def diffusivity(pressure):
        return lennard_jones_diffusivity_report(
            nitrogen, parameters, oxygen, other, temperature_K=350, pressure_Pa=pressure,
            collision_method=method, combining_rule="Lorentz_Berthelot",
        ).evaluation.value

    low, high = diffusivity(1e5), diffusivity(2e5)
    assert low == pytest.approx(2 * high)
    ledger = gas_mixture_effective_diffusivity_ledger(
        target_component="nitrogen", component_mole_fractions={"nitrogen": .3, "oxygen": .7},
        binary_diffusivities_m2_s={"oxygen": low},
    )
    assert ledger.mixture_value == pytest.approx(low)
    grid = DiffusionMesh.planar(.01, 2)
    slow = ConservativeDiffusion(grid, high).evaluate((.1, .01))
    fast = ConservativeDiffusion(grid, ledger.mixture_value).evaluate((.1, .01))
    assert np.sum(fast.amount_rates_mol_s) == pytest.approx(0)
    assert fast.amount_rates_mol_s == pytest.approx(2 * slow.amount_rates_mol_s)


def test_estimates_bind_to_original_component_and_si_lengths(nitrogen):
    spec = LennardJonesSpec.estimate(nitrogen, energy_method="BSL_critical",
                                   energy_parameters={"Tc": 126.2},
                                   diameter_method="Tee_Gotoh_Steward_2",
                                   diameter_parameters={"Tc": 126.2, "Pc": 3.4e6, "omega": .04})
    assert spec.sigma_m == pytest.approx(Quantity(spec.sigma_m, "m").to("angstrom").value*1e-10)
    curve = spec.viscosity_correlation(nitrogen, collision_method="Kim_Monroe",
                                       temperature_range_K=(150, 600))
    assert evaluate_correlation(curve, temperature_K=300).value > 0
    assert LennardJonesSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


@pytest.mark.parametrize("method,limit", [("Neufeld_Janzen_Aziz", 100), ("Kim_Monroe", 400)])
def test_all_collision_pairs_upper_domain_stability_and_explicit_extrapolation(method, limit):
    assert len(COLLISION_PAIRS) == 16
    for l_order, s_order in COLLISION_PAIRS:
        assert lennard_jones_collision_integral(limit*.99, method=method, l=l_order, s=s_order) > 0
    for T in (.3, limit):
        with pytest.raises(ValueError, match="T_star"):
            lennard_jones_collision_integral(T, method=method)
        assert lennard_jones_collision_integral(T, method=method, extrapolate=True) > 0
    with pytest.raises(ValueError, match="order"):
        lennard_jones_collision_integral(1, method=method, l=True, s=1)
    with pytest.raises(ValueError, match="order"):
        lennard_jones_collision_integral(1, method=method, l=4, s=5)


def test_missing_parameters_unsupported_methods_and_nonphysical_polynomial():
    with pytest.raises(ValueError, match="exactly"):
        lennard_jones_energy("BSL_critical", Tb=300)
    with pytest.raises(ValueError, match="unknown"):
        lennard_jones_energy("automatic", Tc=500)
    with pytest.raises(ValueError, match="outside"):
        lennard_jones_diameter("Silva_Liu_Macedo", Tc=1029.13, Pc=3.83e5)
    with pytest.raises(ValueError, match="exactly"):
        lennard_jones_reduced_temperature(300, epsilon_k=100, epsilon_J=1e-21)
    with pytest.raises(ValueError, match="positive"):
        lennard_jones_reduced_temperature(300, epsilon_J=0)
    assert lennard_jones_diameter("Flynn", Vc=.0003) != pytest.approx(
        lennard_jones_diameter("Flynn_published", Vc=.0003), abs=1e-15)


def test_unknown_source_does_not_download_and_neutral_model_rejects_ions(tmp_path, parameters):
    db = LennardJonesDatabase(tmp_path)
    with pytest.raises(ConstantSourceUnavailableError):
        db.records("MAGALHAES_LJ")
    with pytest.raises(ValueError, match="ionic"):
        lennard_jones_viscosity_report(ComponentSpec("ion", "NH4+"),
                                       replace(parameters, component_id="ion"),
                                       temperature_K=300, collision_method="Kim_Monroe")
