"""Multifluid physical limits, composition contracts and common consumers."""

import json
import math
from dataclasses import replace

import pytest

from chemworld.data.multifluid import load_multifluid_model, read_multifluid_model
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_scaled_derivative,
)
from chemworld.physchem.multifluid import (
    MultifluidBinaryPair,
    MultifluidDepartureTerm,
    MultifluidModel,
)

R = 8.31446261815324


def synthetic_model(count=3, *, ideal=False, beta=1.07):
    fluids = tuple(
        HelmholtzFluid(
            name,
            str(i + 1),
            0.02 + 0.01 * i,
            R,
            200.0 + 30 * i,
            9000.0 + 1000 * i,
            100.0,
            1000.0,
            1e8,
            (
                HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 2.0 + i, "a2": 3.0 + i}),
                HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5 + 0.5 * i}),
            ),
            ()
            if ideal
            else (
                HelmholtzTerm(
                    "ResidualHelmholtzPower",
                    {
                        "n": [-0.2 - 0.05 * i, 0.03],
                        "d": [1.0, 2.0],
                        "t": [1.0, 0.5],
                        "l": [0.0, 1.0],
                    },
                ),
            ),
            "explicit synthetic source",
        )
        for i, name in enumerate(("A", "B", "C")[:count])
    )
    pairs = tuple(
        MultifluidBinaryPair(
            (a.fluid_id, b.fluid_id),
            beta,
            0.98,
            1 / beta,
            1.02,
            0.0 if ideal else 0.7,
            ()
            if ideal
            else (
                MultifluidDepartureTerm("power", -0.015, 2, 1.2),
                MultifluidDepartureTerm(
                    "gerg", 0.02, 1, 0.7, eta=0.3, epsilon=0.5, beta=0.2, gamma=0.8
                ),
            ),
            "explicit synthetic pair",
        )
        for i, a in enumerate(fluids)
        for b in fluids[i + 1 :]
    )
    return MultifluidModel(fluids, pairs, R, "source_equal", "synthetic test declaration")


def source_payload():
    model = synthetic_model(2)
    pure = []
    for f in model.fluids:
        pure.append(
            {
                "INFO": {"NAME": f.fluid_id, "CAS": f.casrn},
                "EOS": [
                    {
                        "STATES": {
                            "reducing": {
                                "T": f.reducing_temperature_K,
                                "rhomolar": f.reducing_molar_density_mol_m3,
                            },
                            "sat_min_liquid": {"T": f.minimum_temperature_K},
                        },
                        "molar_mass": f.molar_mass_kg_mol,
                        "gas_constant": R,
                        "T_max": 1000.0,
                        "p_max": 1e8,
                        "pseudo_pure": False,
                        "alpha0": [t.to_dict() for t in f.ideal_terms],
                        "alphar": [t.to_dict() for t in f.residual_terms],
                    }
                ],
            }
        )
    pair = model.binary_pairs[0]
    return {
        "fluids": pure,
        "binary_pairs": [
            {
                "Name1": "A",
                "Name2": "B",
                "CAS1": "1",
                "CAS2": "2",
                "betaT": pair.beta_T,
                "gammaT": pair.gamma_T,
                "betaV": pair.beta_v,
                "gammaV": pair.gamma_v,
                "F": 0.7,
                "function": "AB",
                "BibTeX": "synthetic",
            }
        ],
        "departure_functions": [
            {
                "Name": "AB",
                "type": "GERG-2008",
                "Npower": 1,
                "n": [-0.015, 0.02],
                "d": [2.0, 1.0],
                "t": [1.2, 0.7],
                "eta": [0.0, 0.3],
                "epsilon": [0.0, 0.5],
                "beta": [0.0, 0.2],
                "gamma": [0.0, 0.8],
            }
        ],
        "gas_constant_J_mol_K": R,
        "gas_constant_convention": "source_equal",
    }


def test_source_import_codec_and_common_consumers(tmp_path):
    source = source_payload()
    path = tmp_path / "mixture.json"
    path.write_text(json.dumps(source))
    model = read_multifluid_model(path, source="specified source")
    rebuilt = MultifluidModel.from_dict(json.loads(json.dumps(model.to_dict())))
    baseline = synthetic_model(2)
    for candidate in (model, rebuilt):
        assert candidate.state(300.0, 800.0, (0.4, 0.6)) == baseline.state(300.0, 800.0, (0.4, 0.6))
        assert candidate.caloric_model().residual_model.source == "specified source"
        state = candidate.state(300.0, 800.0, (0.4, 0.6))
        root = molecular_density_from_pressure(
            candidate,
            300.0,
            state.pressure_Pa,
            (0.4, 0.6),
            initial_density_mol_m3=500.0,
            density_bounds_mol_m3=(20.0, 2000.0),
        )
        assert root.molar_density_mol_m3 == pytest.approx(800.0, rel=2e-10)
    assert rebuilt.to_dict() == model.to_dict()


@pytest.mark.parametrize("count,index", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
def test_pure_limit_matches_original_pressure_calorics_and_insertion(count, index):
    model = synthetic_model(count)
    xs = tuple(float(i == index) for i in range(count))
    pure = model.fluids[index].state(temperature_K=300.0, molar_density_mol_m3=700.0)
    mixture = model.caloric_state(300.0, 700.0, xs)
    for field in (
        "helmholtz_energy_J_mol",
        "internal_energy_J_mol",
        "enthalpy_J_mol",
        "entropy_J_mol_K",
        "cv_J_mol_K",
        "cp_J_mol_K",
        "gibbs_energy_J_mol",
    ):
        assert getattr(mixture, field) == pytest.approx(getattr(pure, field), rel=1e-12)
    assert mixture.pressure_Pa == pytest.approx(pure.pressure_Pa, rel=1e-12)
    assert all(math.isfinite(m) for m in model.residual_chemical_potentials(300.0, 700.0, xs))


@pytest.mark.parametrize("xs", [(0.25, 0.75), (1.0, 0.0), (0.0, 1.0)])
def test_ideal_mixture_has_exact_energy_and_mixing_entropy(xs):
    model = synthetic_model(2, ideal=True)
    result = model.caloric_state(300.0, 700.0, xs)
    pures = [f.state(temperature_K=300.0, molar_density_mol_m3=700.0) for f in model.fluids]
    for field in ("internal_energy_J_mol", "enthalpy_J_mol", "cv_J_mol_K", "cp_J_mol_K"):
        assert getattr(result, field) == pytest.approx(
            sum(x * getattr(f, field) for x, f in zip(xs, pures, strict=True)), rel=2e-12
        )
    assert result.entropy_J_mol_K == pytest.approx(
        sum(x * f.entropy_J_mol_K for x, f in zip(xs, pures, strict=True))
        - R * sum(x * math.log(x) for x in xs if x),
        rel=2e-12,
    )
    assert result.pressure_Pa == pytest.approx(700.0 * R * 300.0)


def test_pair_orientation_and_component_permutation_do_not_change_science():
    model = synthetic_model()
    xs = (0.2, 0.3, 0.5)
    reversed_pairs = tuple(
        replace(p, component_ids=p.component_ids[::-1], beta_T=1 / p.beta_T, beta_v=1 / p.beta_v)
        for p in model.binary_pairs
    )
    changed = replace(model, fluids=model.fluids[::-1], binary_pairs=reversed_pairs)
    assert changed.state(300.0, 700.0, xs[::-1]).pressure_Pa == pytest.approx(
        model.state(300.0, 700.0, xs).pressure_Pa, rel=1e-12
    )
    assert changed.residual_chemical_potentials(300.0, 700.0, xs[::-1]) == pytest.approx(
        model.residual_chemical_potentials(300.0, 700.0, xs)[::-1], rel=1e-12
    )


def test_two_absent_components_reject_undefined_second_order_but_keep_first():
    model = synthetic_model()
    assert math.isfinite(model.composition_derivative(300.0, 700.0, (1.0, 0.0, 0.0), ("B",)))
    with pytest.raises(ValueError, match="two absent"):
        model.residual_chemical_potential_jacobian(300.0, 700.0, (1.0, 0.0, 0.0))
    analytic = synthetic_model(beta=1.0)
    assert all(
        math.isfinite(v)
        for row in analytic.residual_chemical_potential_jacobian(300.0, 700.0, (1.0, 0.0, 0.0))
        for v in row
    )
    assert math.isfinite(
        molecular_scaled_derivative(
            model, 300.0, 700.0, (1.0, 0.0, 0.0), inverse_temperature_order=3
        )
    )


@pytest.mark.parametrize(
    "xs",
    [
        (0.0, 0.0, 0.0),
        (0.2, 0.3),
        (-0.1, 0.6, 0.5),
        (0.2, 0.3, 0.6),
        (True, 0.0, 0.0),
        (float("nan"), 0.3, 0.7),
    ],
)
def test_invalid_composition_rejected(xs):
    with pytest.raises(ValueError):
        synthetic_model().state(300.0, 700.0, xs)


@pytest.mark.parametrize(
    "t,rho",
    [
        (0.0, 700.0),
        (99.0, 700.0),
        (1001.0, 700.0),
        (300.0, 0.0),
        (300.0, -1.0),
        (float("nan"), 700.0),
        (300.0, True),
    ],
)
def test_invalid_state_domain_rejected(t, rho):
    with pytest.raises(ValueError):
        synthetic_model().state(t, rho, (0.2, 0.3, 0.5))


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "unknown", "bad_r", "implicit_r", "pseudo", "wrong_type"]
)
def test_invalid_model_declarations_rejected(change):
    model = synthetic_model()
    kwargs = {}
    if change == "missing":
        kwargs["binary_pairs"] = model.binary_pairs[:-1]
    if change == "duplicate":
        kwargs["binary_pairs"] = (*model.binary_pairs, model.binary_pairs[0])
    if change == "unknown":
        kwargs["binary_pairs"] = (
            replace(model.binary_pairs[0], component_ids=("A", "X")),
            *model.binary_pairs[1:],
        )
    if change == "bad_r":
        kwargs["gas_constant_J_mol_K"] = R + 1
    if change == "implicit_r":
        kwargs["gas_constant_convention"] = "weighted"
    if change == "pseudo":
        kwargs["fluids"] = (replace(model.fluids[0], pseudo_pure=True), *model.fluids[1:])
    if change == "wrong_type":
        kwargs["fluids"] = ("not a fluid",)
    with pytest.raises(ValueError):
        replace(model, **kwargs)


def test_explicit_r_override_retains_source_constants_and_can_be_audited():
    model = synthetic_model(2)
    changed = replace(
        model, gas_constant_J_mol_K=9.0, gas_constant_convention="common_constant_override"
    )
    assert [f["gas_constant_J_mol_K"] for f in changed.to_dict()["fluids"]] == [R, R]
    assert changed.state(300.0, 700.0, (0.4, 0.6)).pressure_Pa / model.state(
        300.0, 700.0, (0.4, 0.6)
    ).pressure_Pa == pytest.approx(9 / R)


@pytest.mark.parametrize(
    "change",
    [
        "missing_pair",
        "unknown_pair",
        "wrong_cas",
        "missing_dep",
        "unknown_dep",
        "bad_shape",
        "bad_Npower",
        "unknown_root",
        "ambiguous_eos",
        "no_source",
    ],
)
def test_source_diagnostics_do_not_estimate_or_drop_semantics(change):
    payload = source_payload()
    source = "specified"
    if change == "missing_pair":
        payload["binary_pairs"] = []
    if change == "unknown_pair":
        payload["binary_pairs"][0]["coefficient_typo"] = 3.0
    if change == "wrong_cas":
        payload["binary_pairs"][0]["CAS1"] = "not-A"
    if change == "missing_dep":
        payload["departure_functions"] = []
    if change == "unknown_dep":
        payload["departure_functions"][0]["type"] = "unknown-science"
    if change == "bad_shape":
        payload["departure_functions"][0]["eta"] = [0.3]
    if change == "bad_Npower":
        payload["departure_functions"][0]["Npower"] = -1
    if change == "unknown_root":
        payload["hidden_estimation"] = True
    if change == "ambiguous_eos":
        payload["fluids"][0]["EOS"] *= 2
    if change == "no_source":
        source = ""
    with pytest.raises(ValueError):
        load_multifluid_model(payload, source=source)


@pytest.mark.parametrize("where", ["root", "pair", "departure", "pure_term"])
def test_codec_rejects_unknown_fields(where):
    payload = synthetic_model(2).to_dict()
    selected = {
        "root": payload,
        "pair": payload["binary_pairs"][0],
        "departure": payload["binary_pairs"][0]["departure_terms"][0],
        "pure_term": payload["fluids"][0]["residual_terms"][0],
    }[where]
    selected["unknown_parameter"] = 1
    with pytest.raises(ValueError):
        MultifluidModel.from_dict(payload)


def test_common_caloric_codec_preserves_source_r_and_explicit_derived_reference():
    from chemworld.physchem.molecular_mixture import MolecularCaloricModel

    original = synthetic_model(2)
    model = replace(
        original, gas_constant_J_mol_K=9.0, gas_constant_convention="common_constant_override"
    )
    caloric = model.caloric_model()
    assert all(f.gas_constant_J_mol_K == R for f in model.fluids)
    assert all(
        f.gas_constant_J_mol_K == 9.0 and not f.residual_terms for f in caloric.ideal_references
    )
    assert all(
        "original R=" in f.source and "common R=9.0" in f.source for f in caloric.ideal_references
    )
    rebuilt = MolecularCaloricModel.from_dict(json.loads(json.dumps(caloric.to_dict())))
    assert rebuilt.to_dict() == caloric.to_dict()
    assert rebuilt.state(300.0, 700.0, (0.3, 0.7)) == caloric.state(300.0, 700.0, (0.3, 0.7))


@pytest.mark.parametrize("amounts", [(1.0, 2.0, 3.0), (1.0, 0.0, 3.0)])
def test_common_inventory_partial_energies_reuse_multifluid_derivatives(amounts):
    model = synthetic_model(3).caloric_model()
    volume, temp = 0.01, 300.0
    values = dict(zip(model.component_ids, amounts, strict=True))
    partials = model.partial_inventory_energies(temp, volume, values)

    def total_energy(ns):
        total = sum(ns)
        return (
            total
            * model.state(temp, total / volume, tuple(n / total for n in ns)).internal_energy_J_mol
        )

    for i, actual in enumerate(partials):
        plus, minus = list(amounts), list(amounts)
        step = 1e-5
        plus[i] += step
        if amounts[i] > 0:
            minus[i] -= step
            expected = (total_energy(plus) - total_energy(minus)) / (2 * step)
        else:
            # Second-order forward difference preserves the physical nonnegative domain.
            plus2 = list(amounts)
            plus2[i] += 2 * step
            expected = (
                -3 * total_energy(amounts) + 4 * total_energy(plus) - total_energy(plus2)
            ) / (2 * step)
        assert actual == pytest.approx(expected, rel=2e-8, abs=2e-6)


def test_exponential_source_requires_explicit_damping_coefficients():
    payload = source_payload()
    payload["departure_functions"][0] = {
        "Name": "synthetic",
        "type": "Exponential",
        "n": [0.1],
        "d": [1],
        "t": [1.0],
    }
    payload["binary_pairs"][0]["function"] = "synthetic"
    with pytest.raises(ValueError, match="departure source fields"):
        load_multifluid_model(payload, source="explicit test")


def test_public_property_package_consumes_multifluid_through_json():
    from chemworld.physchem.molecular_properties import MolecularEOSSpec
    from chemworld.physchem.phase_evaluation import PhasePropertyModel
    from chemworld.physchem.property_packages import (
        ComponentPropertyPackage,
        evaluate_property_state,
    )
    from chemworld.physchem.property_reports import CaloricReference
    from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation

    model = synthetic_model(2)
    spec = MolecularEOSSpec("explicit-multifluid", model, "vapor", (1.0, 3000.0), 500.0)
    packages = tuple(
        ComponentPropertyPackage(
            ComponentSpec(name, "Ar", metadata={"synthetic": True}),
            (
                PropertyCorrelation(
                    correlation_id=f"test:{name}:cp",
                    property_id="ideal_gas_heat_capacity",
                    equation_id="cp_polynomial",
                    coefficients={"a": 30.0 + 5 * i},
                    input_units={"temperature": "K"},
                    output_unit="J/(mol*K)",
                    validity_ranges={"temperature": (100.0, 1000.0)},
                    source_note="Synthetic explicit constant Cp for consumer integration",
                ),
            ),
        )
        for i, name in enumerate(model.component_ids)
    )
    references = {
        name: CaloricReference(phase="gas", entropy_J_mol_K=0.0, source_note="explicit zero datum")
        for name in model.component_ids
    }
    phase = PhasePropertyModel(packages, phase="gas", references=references, eos=spec)
    restored = PhasePropertyModel.from_dict(json.loads(json.dumps(phase.to_dict())))
    assert restored.to_dict() == phase.to_dict()
    composition = {"A": 0.3, "B": 0.7}
    raw = model.state(300.0, 700.0, tuple(composition.values()))
    quantities = ("enthalpy", "entropy", "molar_volume", "cp", "cv", "fugacity_coefficient")
    result = restored.evaluate(
        composition, temperature_K=300.0, pressure_Pa=raw.pressure_Pa, quantities=quantities
    )
    ideal = evaluate_property_state(
        packages,
        composition,
        temperature_K=300.0,
        pressure_Pa=raw.pressure_Pa,
        phase="gas",
        references=references,
        quantities=quantities,
    )
    assert result.value("molar_volume") == pytest.approx(1 / 700.0, rel=2e-10)
    assert result.value("enthalpy") - ideal.properties["enthalpy"].value == pytest.approx(
        raw.residual_enthalpy_J_mol
    )
    assert result.value("entropy") - ideal.properties["entropy"].value == pytest.approx(
        raw.residual_entropy_J_mol_K + R * math.log(raw.compressibility_factor)
    )


@pytest.mark.parametrize("entry", ["native", "common", "caloric", "root", "property"])
def test_source_pressure_domain_applies_to_all_common_consumers(entry):
    from chemworld.physchem.molecular_eos import molecular_eos_state
    from chemworld.physchem.molecular_properties import MolecularEOSSpec

    original = synthetic_model(2)
    model = replace(
        original, fluids=tuple(replace(f, maximum_pressure_Pa=1e5) for f in original.fluids)
    )
    with pytest.raises(ValueError, match="pressure outside common pure-source validity"):
        if entry == "native":
            model.state(300.0, 700.0, (0.3, 0.7))
        elif entry == "common":
            molecular_eos_state(model, 300.0, 700.0, (0.3, 0.7))
        elif entry == "caloric":
            model.caloric_model().state(300.0, 700.0, (0.3, 0.7))
        elif entry == "root":
            molecular_density_from_pressure(
                model,
                300.0,
                1.7e6,
                (0.3, 0.7),
                initial_density_mol_m3=700.0,
                density_bounds_mol_m3=(1.0, 3000.0),
            )
        else:
            MolecularEOSSpec("explicit", model, "vapor", (1.0, 3000.0), 700.0).evaluate(
                {"A": 0.3, "B": 0.7}, temperature_K=300.0, pressure_Pa=1.7e6, phase="vapor"
            )


def test_ideal_reference_contribution_uses_actual_mixture_pressure_domain():
    from chemworld.physchem.molecular_mixture import MolecularCaloricModel

    original = synthetic_model(2)
    t, rho, xs = 300.0, 700.0, (0.3, 0.7)
    # Attraction makes physical pressure lower than the ideal contribution.
    limit = 0.995 * R * t * rho
    model = replace(
        original, fluids=tuple(replace(f, maximum_pressure_Pa=limit) for f in original.fluids)
    )
    raw = model.state(t, rho, xs)
    assert 0 < raw.pressure_Pa < limit < rho * R * t
    caloric = model.caloric_model()
    state = caloric.state(t, rho, xs)
    assert state.pressure_Pa == raw.pressure_Pa
    assert state.enthalpy_J_mol == original.caloric_state(t, rho, xs).enthalpy_J_mol
    assert [f.maximum_pressure_Pa for f in model.fluids] == [limit, limit]
    assert all("original p_max=" in f.source for f in caloric.ideal_references)
    assert all("mathematical ideal contribution only" in f.source for f in caloric.ideal_references)
    restored = MolecularCaloricModel.from_dict(json.loads(json.dumps(caloric.to_dict())))
    assert restored.to_dict() == caloric.to_dict()
    assert restored.state(t, rho, xs) == state
    assert all(
        math.isfinite(e)
        for e in restored.partial_inventory_energies(t, 1 / rho, {"A": xs[0], "B": xs[1]})
    )
    with pytest.raises(ValueError, match="pressure outside common pure-source validity"):
        restored.state(t, 2 * rho, xs)
    with pytest.raises(ValueError, match="temperature"):
        restored.state(99.0, rho, xs)
