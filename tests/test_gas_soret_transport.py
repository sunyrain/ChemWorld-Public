"""Authored numbers only: no source phase, coefficient extraction, network or flow."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from math import fsum, log, sqrt

import pytest

from chemworld.physchem import gas_soret_transport as soret_module
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.gas_soret_transport import (
    BOLTZMANN_J_K,
    DEBYE_C_M,
    VACUUM_PERMITTIVITY_F_M,
    GasMixtureSoretSpec,
    gas_mixture_soret_coefficients,
    midpoint_temperature_log_gradient,
    soret_a_coefficients,
    soret_cstar,
    soret_pair_well_depths_J,
)
from chemworld.physchem.gas_transport import GasTransportResponse
from chemworld.physchem.specs import ComponentSpec

IDS = ("N2", "O2", "AR")


def parent_declaration(ids=IDS):
    source = DataSourceProvenance(
        "authored-mu-lambda-D-Cstar",
        0,
        "Authored artificial mathematical parameters",
        "fixture-v1",
        "CC0-1.0",
        "a" * 64,
    )
    formulas = {"N2": "N2", "O2": "O2", "AR": "Ar"}
    species, masses, thermo = {}, {}, {}
    for s in ids:
        i = IDS.index(s)
        component = ComponentSpec(s, formulas[s])
        raw_transport = {
            "model": "gas",
            "geometry": "linear" if s != "AR" else "atom",
            "diameter": 3.0 + i * 0.2,
            "well-depth": 100 + i * 30,
            "dipole": 0.5 if s == "O2" else 0.0,
            "polarizability": 0.8 + i * 0.1,
            "note": "Artificial values; not material predictions",
        }
        species[s] = {
            "species_id": s,
            "formula": formulas[s],
            "composition": component.composition,
            "phase": "gas",
            "charge": 0,
            "catalyst": False,
            "observable_aliases": [],
            "metadata": {"source_declaration": {"transport": raw_transport}},
        }
        masses[s] = component.molecular_weight_g_mol / 1000
        thermo[s] = {
            "thermo": {
                "model": "constant-cp",
                "T0": 298.15,
                "h0": "0 J/mol",
                "s0": "30 J/mol/K",
                "cp0": "30 J/mol/K",
                "T-min": 200,
                "T-max": 3000,
            },
            "composition": component.composition,
            "reference_pressure_Pa": 101325,
            "units": {},
        }
    return {
        "model": "gas-transport-polynomial-v1",
        "formula_id": "cantera-3.2-default-gas-transport-polynomials-v1",
        "coefficient_basis": "ascending powers of ln(T/K); binary coefficients normalized at 1 Pa",
        "units": {
            "temperature": "K",
            "pressure": "Pa",
            "viscosity": "Pa*s",
            "conductivity": "W/(m*K)",
            "binary_diffusivity": "m^2/s",
        },
        "species_ids": list(ids),
        "species_declarations": species,
        "thermochemistry_declarations": thermo,
        "molecular_weights_kg_mol": masses,
        "viscosity_coefficients": {s: [0.002 + IDS.index(s) * 0.0005, 0, 0, 0, 0] for s in ids},
        "conductivity_coefficients": {s: [0.001, 0, 0, 0, 0] for s in ids},
        "binary_diffusivity_coefficients": {
            s: {t: [0.0001 * (IDS.index(s) + IDS.index(t) + 1), 0, 0, 0, 0] for t in ids}
            for s in ids
        },
        "source": source.to_dict(),
        "source_data_kind": "authored-synthetic",
        "fit_temperature_range_K": [300, 1200],
        "maximum_pressure_Pa": 200000,
        "thermal_diffusion": "off",
        "mode": "DEFAULT",
    }


def spec_arguments(parent=None):
    parent = parent_declaration() if parent is None else parent
    ids = parent["species_ids"]
    si = {}
    for s in ids:
        raw = parent["species_declarations"][s]["metadata"]["source_declaration"]["transport"]
        si[s] = {
            "diameter_m": 1e-10 * raw["diameter"],
            "well_depth_J": BOLTZMANN_J_K * raw["well-depth"],
            "dipole_C_m": DEBYE_C_M * raw.get("dipole", 0),
            "polarizability_m3": 1e-30 * raw.get("polarizability", 0),
        }
    return {
        "parent_transport_declaration": parent,
        "species_transport_SI": si,
        "cstar_coefficients": {
            s: {t: [1.1 + 0.02 * (IDS.index(s) + IDS.index(t)), *([0] * 8)] for t in ids}
            for s in ids
        },
        "fit_temperature_range_K": (300, 1200),
        "source_data_kind": "authored-synthetic",
        "source": DataSourceProvenance(**parent["source"]),
    }


def authored_response(parent, *, temperature=400.0, pressure=100000.0, fractions=None, ids=None):
    ids = tuple(parent["species_ids"]) if ids is None else tuple(ids)
    x = (
        {s: {"N2": 0.5, "O2": 0.3, "AR": 0.2}[s] for s in ids}
        if fractions is None
        else {s: fractions[s] for s in ids}
    )
    masses = parent["molecular_weights_kg_mol"]
    mean_mass = fsum(x[s] * masses[s] for s in ids)
    y = {s: x[s] * masses[s] / mean_mass for s in ids}
    z = log(temperature)
    mu = {
        s: sqrt(temperature)
        * fsum(c * z**i for i, c in enumerate(parent["viscosity_coefficients"][s])) ** 2
        for s in ids
    }
    binary = {
        s: {
            t: temperature
            * sqrt(temperature)
            * fsum(c * z**i for i, c in enumerate(parent["binary_diffusivity_coefficients"][s][t]))
            / pressure
            for t in ids
        }
        for s in ids
    }
    diffusion = {k: {} for k in ("molar_mole", "mass_mole", "mass_mass")}
    for s in ids:
        other_x, other_y = fsum(x[t] for t in ids if t != s), fsum(y[t] for t in ids if t != s)
        denominator = fsum(x[t] / binary[s][t] for t in ids if t != s)
        if other_x == 0:
            for row in diffusion.values():
                row[s] = None
        else:
            diffusion["molar_mole"][s] = other_x / denominator
            diffusion["mass_mole"][s] = other_y / denominator
            diffusion["mass_mass"][s] = 1 / (
                denominator + x[s] / other_y * fsum(y[t] / binary[s][t] for t in ids if t != s)
            )
    return GasTransportResponse(
        ids,
        temperature,
        pressure,
        x,
        y,
        masses,
        mu,
        dict.fromkeys(ids, 0.1),
        binary,
        diffusion,
        dict.fromkeys(ids, 0.0),
        1.0,
        mean_mass,
        1e-4,
        0.1,
        {
            "transport": parent["model"],
            "formula": parent["formula_id"],
            "mode": "DEFAULT",
            "thermal_diffusion": "off",
            "source_data_kind": parent["source_data_kind"],
            "source_id": parent["source"]["source_id"],
            "source_version": parent["source"]["version"],
            "source_checksum": parent["source"]["checksum"],
        },
    )


def evaluate(spec=None, **kwargs):
    spec = GasMixtureSoretSpec(**spec_arguments()) if spec is None else spec
    parent = spec.to_dict()["parent_transport_declaration"]
    return gas_mixture_soret_coefficients(
        spec,
        authored_response(parent, **kwargs),
        parent_transport_declaration=parent,
    )


def test_pair_si_polar_and_nonpolar_rules_with_independent_hand_expression():
    si = {
        "n": {
            "diameter_m": 2e-10,
            "well_depth_J": 2e-21,
            "dipole_C_m": 0,
            "polarizability_m3": 1e-30,
        },
        "p": {
            "diameter_m": 3e-10,
            "well_depth_J": 8e-21,
            "dipole_C_m": 2e-30,
            "polarizability_m3": 7e-30,
        },
    }
    pairs = soret_pair_well_depths_J(si)
    xi = (
        1
        + 0.25
        * (1e-30 / (2e-10) ** 3)
        * ((2e-30) ** 2 / (4 * 3.141592653589793 * VACUUM_PERMITTIVITY_F_M * (3e-10) ** 3 * 8e-21))
        * 2
    )
    assert pairs["n"]["p"] == pytest.approx(4e-21 * xi**2, rel=3e-15)
    assert pairs["n"]["p"] == pairs["p"]["n"]
    assert pairs["n"]["n"] == pytest.approx(2e-21)
    assert pairs["p"]["p"] == pytest.approx(8e-21)
    both_polar = deepcopy(si)
    both_polar["n"]["dipole_C_m"] = 1e-30
    assert soret_pair_well_depths_J(both_polar)["n"]["p"] == pytest.approx(4e-21)
    both_neutral = deepcopy(si)
    both_neutral["p"]["dipole_C_m"] = 0
    assert soret_pair_well_depths_J(both_neutral)["n"]["p"] == pytest.approx(4e-21)


def test_cstar_nine_ascending_coefficients_and_reduced_temperature_against_decimal():
    coefficients = (1.2, -0.03, 0.004, -0.0001, 2e-5, -1e-7, 2e-8, -3e-10, 4e-11)
    with localcontext() as ctx:
        ctx.prec = 60
        z = (Decimal("1.380649e-23") * Decimal(600) / Decimal("2e-21")).ln()
        expected = float(sum(Decimal(str(c)) * z**i for i, c in enumerate(coefficients)))
    assert soret_cstar(coefficients, 600, 2e-21) == pytest.approx(expected, rel=3e-15)
    assert soret_cstar((1.2, *([0] * 8)), 600, 2e-21) == 1.2


def test_stable_a_uses_actual_cpp_1point065_and_retains_exact_zero_limit():
    fractions = {"a": 0.25, "b": 0.75, "absent": 0.0}
    masses, mu = dict.fromkeys(fractions, 0.02), dict.fromkeys(fractions, 1e-5)
    a = soret_a_coefficients(fractions, masses, mu)
    assert a["a"] == pytest.approx(0.001875 * 0.25 / (0.25 + 1.065 * 0.75))
    assert a["b"] == pytest.approx(0.001875 * 0.75 / (0.75 + 1.065 * 0.25))
    assert a["absent"] == 0
    assert soret_a_coefficients({"one": 1}, {"one": 0.02}, {"one": 1e-5})["one"] == pytest.approx(
        0.001875
    )
    with pytest.raises(TypeError):
        a["a"] = 1


@pytest.mark.parametrize(
    "field,value",
    (
        ("fraction", True),
        ("fraction", -0.1),
        ("fraction", 0.1),
        ("MW", 0),
        ("MW", float("nan")),
        ("mu", -1),
        ("mu", True),
        ("mu", float("inf")),
    ),
)
def test_a_rejects_bad_fractions_mw_and_viscosities(field, value):
    x, mass, mu = {"a": 0.4, "b": 0.6}, {"a": 0.02, "b": 0.03}, {"a": 1e-5, "b": 2e-5}
    {"fraction": x, "MW": mass, "mu": mu}[field]["a"] = value
    with pytest.raises(ValueError):
        soret_a_coefficients(x, mass, mu)


def test_all_species_formula_against_independent_decimal_pair_sum():
    spec = GasMixtureSoretSpec(**spec_arguments())
    response = authored_response(spec.to_dict()["parent_transport_declaration"])
    result = evaluate(spec)
    with localcontext() as ctx:
        ctx.prec = 60

        def d(value):
            return Decimal(str(value))

        x, y = (
            {s: d(row[s]) for s in IDS}
            for row in (response.mole_fractions, response.mass_fractions)
        )
        mu = {s: d(response.species_viscosities_Pa_s[s]) for s in IDS}
        masses = {s: d(response.molecular_weights_kg_mol[s]) for s in IDS}
        mean_mass = sum(x[s] * masses[s] for s in IDS)
        a = {}
        for s in IDS:
            resistance = Decimal(0)
            for t in IDS:
                if t != s:
                    phi = (
                        1 + (mu[s] / mu[t]).sqrt() * (masses[t] / masses[s]).sqrt().sqrt()
                    ) ** 2 / (8 * (1 + masses[s] / masses[t])).sqrt()
                    resistance += x[t] * phi
            a[s] = Decimal("3.75") * mu[s] / masses[s] / (1 + Decimal("1.065") * resistance / x[s])
        unnormalized = {}
        for s in IDS:
            pair_sum = sum(
                (
                    (Decimal("1.2") * d(spec.cstar_coefficients[s][t][0]) - 1)
                    / d(response.binary_diffusivities_m2_s[s][t])
                    / (masses[s] + masses[t])
                    * (y[s] * a[t] - y[t] * a[s])
                )
                for t in IDS
                if t != s
            )
            unnormalized[s] = (
                d(response.diffusion_coefficients_m2_s["mass_mole"][s])
                * masses[s]
                * mean_mass
                * pair_sum
            )
        total = sum(unnormalized.values())
        expected = {s: float(unnormalized[s] - y[s] * total) for s in IDS}
    dt = result.thermal_diffusion_coefficients_kg_m_s
    assert dt == pytest.approx(expected, rel=2e-13, abs=1e-23)
    assert min(dt.values()) < 0 < max(dt.values())
    assert abs(fsum(dt.values())) < 1e-20
    assert "not inferred" in result.receipt["source_qualification"]


def test_pressure_cancels_and_permutation_retains_id_keyed_values():
    baseline = evaluate()
    assert evaluate(pressure=200000).thermal_diffusion_coefficients_kg_m_s == pytest.approx(
        baseline.thermal_diffusion_coefficients_kg_m_s, rel=3e-14
    )
    reordered = evaluate(ids=tuple(reversed(IDS)))
    assert tuple(reordered.thermal_diffusion_coefficients_kg_m_s) == tuple(reversed(IDS))
    assert reordered.thermal_diffusion_coefficients_kg_m_s == pytest.approx(
        baseline.thermal_diffusion_coefficients_kg_m_s, rel=1e-13
    )


@pytest.mark.parametrize(
    "fractions",
    (
        {"N2": 1, "O2": 0, "AR": 0},
        {"N2": 0.4, "O2": 0.6, "AR": 0},
        {"N2": 1 - 1e-25, "O2": 1e-25, "AR": 0},
    ),
)
def test_pure_absent_and_below_source_tiny_remain_real_zeros_without_filling(fractions):
    result = evaluate(fractions=fractions)
    dt = result.thermal_diffusion_coefficients_kg_m_s
    assert dt["AR"] == 0
    if fractions["N2"] == 1 and fractions["O2"] == 0:
        assert dict(dt) == dict.fromkeys(IDS, 0)
        assert result.receipt["pure_species"] == "N2"
    elif fractions["O2"] == 1e-25:
        assert dt["O2"] != 0
    assert abs(fsum(dt.values())) < 1e-20


def test_one_species_returns_zero_without_multiplying_none_composition_d():
    parent = parent_declaration(("N2",))
    spec = GasMixtureSoretSpec(**spec_arguments(parent))
    response = authored_response(parent, fractions={"N2": 1})
    assert response.diffusion_coefficients_m2_s["mass_mole"]["N2"] is None
    result = gas_mixture_soret_coefficients(spec, response, parent_transport_declaration=parent)
    assert result.thermal_diffusion_coefficients_kg_m_s == {"N2": 0}


def test_deep_freeze_plain_json_codec_and_complete_parent_binding():
    args = spec_arguments()
    spec = GasMixtureSoretSpec(**args)
    serialized = spec.to_dict()
    args["cstar_coefficients"]["N2"]["O2"][0] = 999
    args["species_transport_SI"]["N2"]["well_depth_J"] = 999
    args["parent_transport_declaration"]["thermochemistry_declarations"]["N2"]["thermo"]["cp0"] = (
        "999 J/mol/K"
    )
    assert spec.to_dict() == serialized
    restored = GasMixtureSoretSpec.from_dict(json.loads(json.dumps(serialized)))
    assert restored.to_dict() == serialized
    with pytest.raises(FrozenInstanceError):
        spec.formula_id = "changed"
    with pytest.raises(TypeError):
        spec.cstar_coefficients["N2"]["O2"][0] = 1
    with pytest.raises(TypeError):
        spec.parent_transport_declaration["species_declarations"]["N2"]["metadata"]["changed"] = 1
    with pytest.raises(ValueError, match="binding mismatch"):
        spec.validate_parent(args["parent_transport_declaration"])
    result = evaluate(spec)
    json.dumps(result.to_dict(), allow_nan=False)
    with pytest.raises(TypeError):
        result.thermal_diffusion_coefficients_kg_m_s["N2"] = 1


def test_raw_units_do_not_rescale_the_special_source_transport_format():
    parent = parent_declaration()
    parent["species_declarations"]["N2"]["metadata"]["source_declaration"]["transport"]["units"] = {
        "length": "km",
        "temperature": "degC",
    }
    spec = GasMixtureSoretSpec(**spec_arguments(parent))
    assert spec.species_transport_SI["N2"]["diameter_m"] == 1e-10 * 3.0
    assert spec.species_transport_SI["N2"]["well_depth_J"] == BOLTZMANN_J_K * 100


def test_only_public_source_zero_defaults_are_used_and_sigma_epsilon_cannot_be_filled():
    parent = parent_declaration()
    transport = parent["species_declarations"]["N2"]["metadata"]["source_declaration"]["transport"]
    transport.pop("dipole")
    transport.pop("polarizability")
    spec = GasMixtureSoretSpec(**spec_arguments(parent))
    assert spec.species_transport_SI["N2"]["dipole_C_m"] == 0
    assert spec.species_transport_SI["N2"]["polarizability_m3"] == 0
    args = spec_arguments(parent)
    transport.pop("well-depth")
    with pytest.raises(ValueError, match="cannot be filled"):
        GasMixtureSoretSpec(**args)


@pytest.mark.parametrize(
    "field,value",
    (
        ("fit_temperature_range_K", [True, 1200]),
        ("fit_temperature_range_K", "300,1200"),
        ("maximum_pressure_Pa", True),
        ("maximum_pressure_Pa", float("inf")),
    ),
)
def test_parent_numeric_contract_is_checked_before_binding(field, value):
    args = spec_arguments()
    args["parent_transport_declaration"][field] = value
    with pytest.raises(ValueError):
        GasMixtureSoretSpec(**args)


@pytest.mark.parametrize(
    "defect", ("missing-model", "unknown-model", "number-bool", "unknown-NASA7")
)
def test_parent_thermochemistry_remains_a_real_original_declaration(defect):
    args = spec_arguments()
    thermo = args["parent_transport_declaration"]["thermochemistry_declarations"]["N2"]["thermo"]
    if defect == "missing-model":
        thermo.pop("model")
    elif defect == "unknown-model":
        thermo["model"] = "unknown"
    elif defect == "number-bool":
        thermo["T0"] = True
    else:
        thermo.clear()
        thermo.update(
            model="NASA7",
            **{"temperature-ranges": [200, 3000], "data": [[3.5, 0, 0, 0, 0, 0, 0]], "unknown": 1},
        )
    with pytest.raises(ValueError):
        GasMixtureSoretSpec(**args)


def test_boolean_cannot_match_a_unit_atom_count_in_the_parent_thermo_binding():
    args = spec_arguments()
    args["parent_transport_declaration"]["thermochemistry_declarations"]["AR"]["composition"] = {
        "Ar": True
    }
    with pytest.raises(ValueError, match="thermo atom count"):
        GasMixtureSoretSpec(**args)


@pytest.mark.parametrize("field", ("source_id", "version", "license", "checksum"))
def test_complete_source_provenance_cannot_omit_its_identity_or_rights(field):
    args = spec_arguments()
    raw = args["source"].to_dict()
    raw[field] = ""
    args["source"] = raw
    with pytest.raises(ValueError):
        GasMixtureSoretSpec(**args)


@pytest.mark.parametrize(
    "field,value",
    (
        ("diameter_m", 0),
        ("well_depth_J", float("inf")),
        ("dipole_C_m", -1),
        ("polarizability_m3", -1),
        ("diameter_m", True),
    ),
)
def test_pair_parameter_domain_rejects_nonfinite_nonphysical_values(field, value):
    record = {"diameter_m": 3e-10, "well_depth_J": 1e-21, "dipole_C_m": 0, "polarizability_m3": 0}
    record[field] = value
    with pytest.raises(ValueError):
        soret_pair_well_depths_J({"a": record})


@pytest.mark.parametrize(
    "field,value",
    (
        ("construction", "unknown"),
        ("construction", "actualT"),
        ("fit_variable", "absolute-temperature"),
        ("formula_id", "unknown"),
        ("source_data_kind", "measured"),
        ("fit_temperature_range_K", (300, 1199)),
        ("fit_temperature_range_K", (True, 1200)),
    ),
)
def test_unknown_construction_and_modes_are_not_inferred_from_coefficients(field, value):
    with pytest.raises(ValueError):
        GasMixtureSoretSpec(**(spec_arguments() | {field: value}))


@pytest.mark.parametrize(
    "field",
    (
        "species_declarations",
        "thermochemistry_declarations",
        "viscosity_coefficients",
        "binary_diffusivity_coefficients",
        "source",
    ),
)
def test_validate_parent_checks_actual_full_content_not_only_provider_label(field):
    spec = GasMixtureSoretSpec(**spec_arguments())
    parent = spec.to_dict()["parent_transport_declaration"]
    if field == "source":
        parent[field]["citation"] += " changed"
    elif field == "species_declarations":
        parent[field]["N2"]["metadata"]["changed"] = 1
    elif field == "thermochemistry_declarations":
        parent[field]["N2"]["thermo"]["h0"] = "1 J/mol"
    elif field == "viscosity_coefficients":
        parent[field]["N2"][0] *= 1.1
    else:
        parent[field]["N2"]["O2"][0] *= 1.1
    with pytest.raises(ValueError, match="binding mismatch"):
        spec.validate_parent(parent)


@pytest.mark.parametrize(
    "defect",
    (
        "missing",
        "unknown",
        "CK",
        "on",
        "charge",
        "MW",
        "parent-binary-row",
        "source",
        "SI",
        "SI-unknown",
        "SI-bool",
        "Cstar-missing",
        "Cstar-unknown",
        "Cstar-asymmetric",
        "Cstar-degree",
        "Cstar-nan",
    ),
)
def test_spec_rejects_malformed_or_mismatched_parent_si_and_collision_data(defect):
    args = spec_arguments()
    parent = args["parent_transport_declaration"]
    if defect == "missing":
        parent.pop("thermochemistry_declarations")
    elif defect == "unknown":
        parent["unknown"] = 1
    elif defect == "CK":
        parent["mode"] = "CK"
    elif defect == "on":
        parent["thermal_diffusion"] = "on"
    elif defect == "charge":
        parent["species_declarations"]["N2"]["charge"] = 1
    elif defect == "MW":
        parent["molecular_weights_kg_mol"]["N2"] *= 1.1
    elif defect == "parent-binary-row":
        parent["binary_diffusivity_coefficients"]["AR"] = []
    elif defect == "source":
        args["source"] = replace(args["source"], checksum="b" * 64)
    elif defect.startswith("SI"):
        key, value = {
            "SI": ("well_depth_J", 999),
            "SI-unknown": ("unknown", 1),
            "SI-bool": ("diameter_m", True),
        }[defect]
        args["species_transport_SI"]["N2"][key] = value
    elif defect == "Cstar-missing":
        args["cstar_coefficients"]["N2"].pop("N2")
    elif defect == "Cstar-unknown":
        args["cstar_coefficients"]["N2"]["unknown"] = [1] * 9
    elif defect == "Cstar-asymmetric":
        args["cstar_coefficients"]["N2"]["O2"][0] += 1
    elif defect == "Cstar-degree":
        args["cstar_coefficients"]["N2"]["N2"] = [1] * 8
    else:
        args["cstar_coefficients"]["N2"]["N2"][0] = float("nan")
    with pytest.raises(ValueError):
        GasMixtureSoretSpec(**args)


@pytest.mark.parametrize(
    "defect", ("T", "P", "id", "MW", "source", "mode", "mu", "D", "Dm", "x-bool", "y", "nan")
)
def test_base_response_is_not_accepted_by_labels_alone(defect):
    spec = GasMixtureSoretSpec(**spec_arguments())
    parent = spec.to_dict()["parent_transport_declaration"]
    response = authored_response(parent)
    changes = {}
    if defect == "T":
        changes["temperature_K"] = 299
    elif defect == "P":
        changes["pressure_Pa"] = 200001
    elif defect == "id":
        changes["species_ids"] = ("N2", "O2")
    elif defect == "MW":
        changes["molecular_weights_kg_mol"] = dict(response.molecular_weights_kg_mol) | {"N2": 0.05}
    elif defect in {"source", "mode"}:
        changes["models"] = dict(response.models) | {
            "source_id" if defect == "source" else "mode": "unknown"
        }
    elif defect == "mu":
        changes["species_viscosities_Pa_s"] = dict(response.species_viscosities_Pa_s) | {"N2": 0.1}
    elif defect == "D":
        values = {s: dict(v) for s, v in response.binary_diffusivities_m2_s.items()}
        values["N2"]["O2"] = values["O2"]["N2"] = 0.1
        changes["binary_diffusivities_m2_s"] = values
    elif defect == "Dm":
        values = {s: dict(v) for s, v in response.diffusion_coefficients_m2_s.items()}
        values["mass_mole"]["N2"] = 0.1
        changes["diffusion_coefficients_m2_s"] = values
    elif defect == "x-bool":
        changes["mole_fractions"] = dict(response.mole_fractions) | {"N2": True}
    elif defect == "y":
        changes["mass_fractions"] = dict.fromkeys(IDS, 1 / 3)
    else:
        changes["temperature_K"] = float("nan")
    with pytest.raises(ValueError):
        gas_mixture_soret_coefficients(
            spec, replace(response, **changes), parent_transport_declaration=parent
        )


def test_bound_kernel_retains_complete_public_result_without_full_revalidation(monkeypatch):
    spec = GasMixtureSoretSpec(**spec_arguments())
    parent = spec.to_dict()["parent_transport_declaration"]
    response = authored_response(parent)
    expected = gas_mixture_soret_coefficients(
        spec, response, parent_transport_declaration=parent
    ).to_dict()

    def unexpected_full_validation(*args, **kwargs):
        pytest.fail("query-owned bound kernel repeated complete parent/response validation")

    monkeypatch.setattr(soret_module, "_canonical", unexpected_full_validation)
    monkeypatch.setattr(soret_module, "_validate_response", unexpected_full_validation)
    actual = soret_module._bound_mixture_soret_coefficients(spec, response)
    assert actual.to_dict() == expected


@pytest.mark.parametrize(
    "state",
    (
        {},
        {"temperature": 300.0},
        {"temperature": 1200.0},
        {"pressure": 200000.0},
        {"ids": tuple(reversed(IDS))},
        {"fractions": {"N2": 0.6, "O2": 0.4, "AR": 0.0}},
        {"fractions": {"N2": 0.6, "O2": 0.4, "AR": 1e-30}},
        {"fractions": {"N2": 1.0, "O2": 0.0, "AR": 0.0}},
        {"fractions": {"N2": 0.0, "O2": 0.0, "AR": 1.0}},
    ),
)
def test_bound_and_public_complete_results_are_identical_for_authored_states(state):
    parent = parent_declaration()
    for s in IDS:
        parent["viscosity_coefficients"][s][1:] = [1e-5, 1e-6, 1e-7, 1e-8]
        for target in IDS:
            parent["binary_diffusivity_coefficients"][s][target][1:] = [1e-6, 1e-7, 1e-8, 1e-9]
    args = spec_arguments(parent)
    for row in args["cstar_coefficients"].values():
        for coefficients in row.values():
            coefficients[1:] = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    spec = GasMixtureSoretSpec(**args)
    response = authored_response(parent, **state)
    public = gas_mixture_soret_coefficients(spec, response, parent_transport_declaration=parent)
    bound = soret_module._bound_mixture_soret_coefficients(spec, response)
    assert bound.to_dict() == public.to_dict()


def test_single_species_bound_and_public_results_retain_directional_none_and_zero():
    parent = parent_declaration(("AR",))
    spec = GasMixtureSoretSpec(**spec_arguments(parent))
    response = authored_response(parent, fractions={"AR": 1.0})
    public = gas_mixture_soret_coefficients(spec, response, parent_transport_declaration=parent)
    bound = soret_module._bound_mixture_soret_coefficients(spec, response)
    assert bound.to_dict() == public.to_dict()
    assert bound.thermal_diffusion_coefficients_kg_m_s == {"AR": 0.0}
    assert response.diffusion_coefficients_m2_s["mass_mole"]["AR"] is None


@pytest.mark.parametrize(
    "field,value",
    (
        ("temperature_K", 299.0),
        ("temperature_K", 1201.0),
        ("temperature_K", float("nan")),
        ("temperature_K", True),
        ("pressure_Pa", 200001.0),
        ("pressure_Pa", 0.0),
        ("pressure_Pa", float("inf")),
        ("species_ids", ("N2", "O2")),
        ("species_ids", ("N2", "O2", "N2")),
        ("mole_fractions", {"N2": -0.1, "O2": 0.8, "AR": 0.3}),
        ("mole_fractions", {"N2": True, "O2": 0.0, "AR": 0.0}),
        ("mole_fractions", {"N2": float("nan"), "O2": 0.0, "AR": 0.0}),
        ("mole_fractions", {"N2": 0.2, "O2": 0.2, "AR": 0.2}),
        ("mole_fractions", {"N2": 0.5, "O2": 0.5}),
        ("mass_fractions", {"N2": -0.1, "O2": 0.8, "AR": 0.3}),
        ("mass_fractions", {"N2": True, "O2": 0.0, "AR": 0.0}),
        ("mass_fractions", dict.fromkeys(IDS, 1 / 3)),
    ),
)
def test_bound_kernel_rejects_illegal_live_domain_ids_and_fractions(field, value):
    spec = GasMixtureSoretSpec(**spec_arguments())
    response = authored_response(spec.to_dict()["parent_transport_declaration"])
    with pytest.raises(ValueError):
        soret_module._bound_mixture_soret_coefficients(spec, replace(response, **{field: value}))


@pytest.mark.parametrize("defect", ("MW", "source", "mu", "D", "Dm", "pure-Dm"))
def test_bound_kernel_retains_finite_value_mw_source_and_pure_none_guards(defect):
    spec = GasMixtureSoretSpec(**spec_arguments())
    parent = spec.to_dict()["parent_transport_declaration"]
    response = authored_response(
        parent, **({"fractions": {"N2": 1.0, "O2": 0.0, "AR": 0.0}} if defect == "pure-Dm" else {})
    )
    if defect == "MW":
        changes = {
            "molecular_weights_kg_mol": dict(response.molecular_weights_kg_mol) | {"N2": 0.05}
        }
    elif defect == "source":
        changes = {"models": dict(response.models) | {"source_checksum": "b" * 64}}
    elif defect == "mu":
        changes = {"species_viscosities_Pa_s": dict(response.species_viscosities_Pa_s) | {"N2": 0}}
    elif defect == "D":
        values = {s: dict(v) for s, v in response.binary_diffusivities_m2_s.items()}
        values["N2"]["O2"] = values["O2"]["N2"] = float("inf")
        changes = {"binary_diffusivities_m2_s": values}
    else:
        values = {s: dict(v) for s, v in response.diffusion_coefficients_m2_s.items()}
        values["mass_mole"]["N2"] = 1.0 if defect == "pure-Dm" else None
        changes = {"diffusion_coefficients_m2_s": values}
    with pytest.raises(ValueError):
        soret_module._bound_mixture_soret_coefficients(spec, replace(response, **changes))


def test_public_entry_still_authenticates_complete_parent_before_bound_kernel():
    spec = GasMixtureSoretSpec(**spec_arguments())
    parent = spec.to_dict()["parent_transport_declaration"]
    response = authored_response(parent)
    parent["conductivity_coefficients"]["N2"][0] *= 1.1
    with pytest.raises(ValueError, match="binding mismatch"):
        gas_mixture_soret_coefficients(spec, response, parent_transport_declaration=parent)


@pytest.mark.parametrize("defect", ("model", "constants", "units", "unknown", "missing"))
def test_codec_is_strict(defect):
    payload = GasMixtureSoretSpec(**spec_arguments()).to_dict()
    if defect == "model":
        payload["model"] = "unknown"
    elif defect == "constants":
        payload["constants_id"] = "unknown"
    elif defect == "units":
        payload["units"]["well_depth_J"] = "K"
    elif defect == "unknown":
        payload["ignored"] = 1
    else:
        payload.pop("construction")
    with pytest.raises(ValueError):
        GasMixtureSoretSpec.from_dict(payload)


def test_midpoint_gradient_sign_scale_and_zero_without_heat_or_clock_state():
    assert midpoint_temperature_log_gradient(300, 900, 0.01) == 100
    assert midpoint_temperature_log_gradient(900, 300, 0.01) == -100
    assert midpoint_temperature_log_gradient(600, 1800, 0.01) == 100
    assert midpoint_temperature_log_gradient(300, 300, 0.01) == 0
    assert midpoint_temperature_log_gradient(300, 900, 0.01) != pytest.approx(
        (log(900) - log(300)) / 0.01
    )


@pytest.mark.parametrize(
    "values",
    (
        (True, 900, 1),
        (0, 900, 1),
        (300, float("nan"), 1),
        (300, 900, 0),
        (300, 900, False),
        (1e308, 1e308, 1e308),
    ),
)
def test_gradient_rejects_nonphysical_or_nonfinite_intermediates(values):
    with pytest.raises(ValueError):
        midpoint_temperature_log_gradient(*values)


@pytest.mark.parametrize(
    "coefficients,T,epsilon",
    (
        ([1] * 8, 400, 1e-21),
        ([True] * 9, 400, 1e-21),
        ([float("inf")] * 9, 400, 1e-21),
        ([0] * 9, 400, 1e-21),
        ([-1] + [0] * 8, 400, 1e-21),
        ([1] * 9, True, 1e-21),
        ([1] * 9, 400, 0),
    ),
)
def test_cstar_invalid_data_cannot_be_reinterpreted_as_a_different_fit(coefficients, T, epsilon):
    with pytest.raises(ValueError):
        soret_cstar(coefficients, T, epsilon)
