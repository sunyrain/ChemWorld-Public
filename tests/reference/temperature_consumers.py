"""F5 fixed cross-function consumers and full public/source inventory."""
# ruff: noqa: C408, E501 -- full public scientific names in the inventory mapping

import csv
import importlib
import json
from pathlib import Path

import chemicals
from chemicals import combustion, environment, permittivity, refractivity, safety, temperature
from run_assessment_block import COUNTS as F3_COUNTS
from run_electro_optical_block import COUNTS as F2_COUNTS
from run_solubility_block import SOURCES as F1_SOURCES
from run_solubility_block import compare, flash_unit

from chemworld.data.assessments import AssessmentDatabase
from chemworld.data.caloric import CaloricDatabase
from chemworld.data.constants import ConstantDatabase
from chemworld.data.electro_optical import ElectroOpticalDatabase
from chemworld.data.fuels import FUEL_SOURCES, FuelDatabase
from chemworld.data.scientific_inputs import InputValue
from chemworld.foundation.observations import Observation, ObservationChannel, ObservationTime
from chemworld.foundation.units import Quantity
from chemworld.physchem.combustion import combustion_reference_heat, combustion_stoichiometry
from chemworld.physchem.electro_optics import (
    water_permittivity_iapws,
    water_refractive_index_iapws,
)
from chemworld.physchem.elements import molecular_weight, parse_formula
from chemworld.physchem.safety_environment import (
    emission_impact,
    ppmv_to_mass_concentration,
    suzuki_flammability_limit,
)
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.temperature_scales import temperature_input, temperature_observation


def thermal_unit(scale, t, pressure):
    raw = ObservationChannel(
        "thermometer",
        "K",
        t,
        uncertainty=0.02,
        sample_id="calibration",
        sampled_at=ObservationTime("world", 4),
        available_at=ObservationTime("world", 7),
    )
    before = raw.to_dict()
    obs = temperature_observation(raw, scale, "ITS-90")
    obs = Observation.from_native_record(json.loads(json.dumps(obs.native_record())))
    channel = obs.channel("converted_temperature")
    actual_t = temperature_input(
        InputValue(channel.value, "K", channel.uncertainty, "Calibrated observation"),
        "ITS-90",
        "ITS-90",
    )
    expected_t = temperature.T_converter(t, scale, "ITS-90")
    comparisons = [compare(actual_t.value, expected_t, atol=2e-8, rtol=0)]
    phase = flash_unit(actual_t.value, pressure, 0.5)
    eps = water_permittivity_iapws(actual_t.value, 997, constants="chemicals_1_5_2")
    ri = water_refractive_index_iapws(actual_t.value, 997)
    mass = ppmv_to_mass_concentration(100, 40, temperature_K=actual_t.value, pressure_Pa=pressure)
    comparisons += [
        compare(eps, permittivity.permittivity_IAPWS(expected_t, 997)),
        compare(ri, refractivity.RI_IAPWS(expected_t, 997)),
        compare(mass, safety.ppmv_to_mgm3(100, 40, T=expected_t, P=pressure)),
    ]
    if scale != "ITS-90":
        assert ri != water_refractive_index_iapws(t, 997)
    assert raw.to_dict() == before and obs.channel("thermometer") == raw
    assert channel.sample_id == raw.sample_id and channel.sampled_at == raw.sampled_at
    assert channel.available_at == raw.available_at and not obs.errors()
    return {
        "checks": [{"passed": True}],
        "comparisons": comparisons,
        "phase": phase,
        "observation": obs.native_record(),
        "calibrated_input": actual_t.to_dict(),
    }


def fuel_unit(cas, formula, phase, water_phase, amount):
    directory = Path(chemicals.__file__).parent
    db = CaloricDatabase(directory)
    fuel = ComponentSpec(cas, formula, default_phase=phase, cas_number=cas)
    fuel = ComponentSpec.from_dict(json.loads(json.dumps(fuel.to_dict())))
    ids = {cas: phase, "124-38-9": "gas", "7782-44-7": "gas", "7732-18-5": water_phase}
    refs = {
        c: db.formation_reference(
            c, phase=p, enthalpy_method="CRC_THERMODYNAMIC", pressure_Pa=101325
        )
        for c, p in ids.items()
    }
    products = {"CO2": refs["124-38-9"], "O2": refs["7782-44-7"], "H2O": refs["7732-18-5"]}
    heat = combustion_reference_heat(fuel, refs[cas], products, Quantity(amount, "mol"))
    path = directory / db.source("CRC_THERMODYNAMIC").relative_path
    with path.open(encoding="utf8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    hf = {
        c: float(next(r for r in rows if r["CAS"] == c)["Hfg" if p == "gas" else "Hfl"])
        for c, p in ids.items()
    }
    atoms = parse_formula(formula)
    reference_stoich = combustion.combustion_stoichiometry(atoms)
    expected_hc = combustion.HHV_stoichiometry(
        reference_stoich,
        hf[cas],
        Hf_chemicals={"CO2": hf["124-38-9"], "O2": hf["7782-44-7"], "H2O": hf["7732-18-5"]},
    )
    comparisons = [compare(heat.value, amount * expected_hc, atol=1e-7)]
    limits = {}
    for limit, call in (("LFL", safety.Suzuki_LFL), ("UFL", safety.Suzuki_UFL)):
        limits[limit] = suzuki_flammability_limit(heat.value / amount, limit=limit)
        comparisons.append(compare(limits[limit], call(expected_hc)))
    co2 = combustion_stoichiometry(atoms)["CO2"] * amount
    mass = Quantity(co2 * molecular_weight(parse_formula("CO2")), "g")
    factor = AssessmentDatabase(directory).assessment("124-38-9", "GWP", method="IPCC (2021) 100yr")
    impact = emission_impact({"124-38-9": mass}, {"124-38-9": factor})
    expected_mass = reference_stoich["CO2"] * amount * molecular_weight({"C": 1, "O": 2}) / 1000
    comparisons.append(
        compare(
            impact.value, expected_mass * environment.GWP("124-38-9", method="IPCC (2021) 100yr")
        )
    )
    return {
        "checks": [{"passed": True}],
        "comparisons": comparisons,
        "heat": heat.to_dict(),
        "flammability": limits,
        "CO2_emission": mass.to_dict(),
        "impact": impact.to_dict(),
        "phase_references": {c: r.to_dict() for c, r in refs.items()},
    }


def consumer_units():
    for scale in ("ITS-90", "ITS-68", "ITS-48"):
        for t in (280, 300, 340):
            for pressure in (70000, 100000):
                yield (
                    f"thermal/{scale}/{t}/{pressure}",
                    dict(scale=scale, T=t, P=pressure),
                    (lambda s=scale, t=t, p=pressure: thermal_unit(s, t, p)),
                )
    for cas, formula, phase in (("74-82-8", "CH4", "gas"), ("64-17-5", "C2H6O", "liquid")):
        for wp in ("liquid", "gas"):
            for n in (0.1, 2):
                yield (
                    f"fuel/{cas}/{wp}/{n}",
                    dict(CAS=cas, formula=formula, phase=phase, water_phase=wp, amount_mol=n),
                    (lambda c=cas, f=formula, p=phase, w=wp, n=n: fuel_unit(c, f, p, w, n)),
                )


# Each group names the whole public contract, including exported scientific constants.
# Targets are existing native owners; this inventory is not numerical qualification.
GROUPS = {
    "solubility": {
        "Henry_constants dHenry_constants_dT d2Henry_constants_dT2": "physchem.henry:henry_matrix",
        "Henry_converter": "physchem.henry:henry_convert",
        "Henry_pressure": "physchem.henry:henry_pressure",
        "Henry_pressure_mixture": "physchem.henry:henry_pressure_mixture",
        "Tm_depression_eutectic": "physchem.solubility:freezing_point_depression",
        "solubility_eutectic": "physchem.solubility:solid_solubility",
        "solubility_parameter": "physchem.solubility:hildebrand_parameter",
        "hansen_delta_d hansen_delta_p hansen_delta_h": "physchem.solubility:HansenParameters",
        "hansen_delta_d_methods hansen_delta_p_methods hansen_delta_h_methods": "data.constants:ConstantDatabase.methods",
        "hansen_delta_d_all_methods hansen_delta_p_all_methods hansen_delta_h_all_methods": "physchem.solubility:HANSEN_SOURCES",
    },
    "permittivity": {
        "permittivity_CRC": "physchem.electro_optics:relative_permittivity_crc",
        "permittivity_IAPWS": "physchem.electro_optics:water_permittivity_iapws",
    },
    "refractivity": {
        "RI": "data.electro_optical:ElectroOpticalDatabase.refractive_index",
        "RI_methods": "data.constants:ConstantDatabase.methods",
        "RI_all_methods": "data.electro_optical:REFRACTIVE_INDEX_SOURCES",
        "RI_IAPWS": "physchem.electro_optics:water_refractive_index_iapws",
        "RI_from_molar_refractivity": "physchem.electro_optics:refractive_index_from_molar_refractivity",
        "RI_to_brix": "physchem.electro_optics:refractive_index_to_brix",
        "brix_to_RI": "physchem.electro_optics:brix_to_refractive_index",
        "TDE_RIXExpansion": "physchem.electro_optics:refractive_index_tde",
        "molar_refractivity_from_RI": "physchem.electro_optics:molar_refractivity",
        "polarizability_from_RI": "physchem.electro_optics:polarizability_volume",
    },
    "dipole": {
        "dipole_moment": "data.constants:ConstantDatabase.value",
        "dipole_moment_methods": "data.constants:ConstantDatabase.methods",
        "dipole_moment_all_methods": "data.electro_optical:DIPOLE_SOURCES",
    },
    "safety": {
        "LFL STEL TWA UFL Carcinogen Ceiling Skin T_autoignition T_flash": "data.assessments:AssessmentDatabase.assessment",
        "Carcinogen_methods Ceiling_methods STEL_methods Skin_methods TWA_methods "
        "T_autoignition_methods T_flash_methods": "data.assessments:AssessmentDatabase.assessment_methods",
        "LFL_methods UFL_methods": "data.assessments:AssessmentDatabase.flammability_methods",
        "Carcinogen_all_methods Ceiling_all_methods LFL_all_methods STEL_all_methods "
        "Skin_all_methods TWA_all_methods T_autoignition_all_methods T_flash_all_methods "
        "UFL_all_methods": "data.assessments:AssessmentDatabase.method_specs",
        "LFL_ISO_10156_2017": "physchem.safety_environment:iso_10156_lower_limit",
        "Crowl_Louvar_LFL Crowl_Louvar_UFL": "physchem.safety_environment:crowl_louvar_flammability_limit",
        "IARC_codes": "data.assessment_catalog:IARC_CLASSIFICATIONS",
        "NTP_codes": "data.assessment_catalog:NTP_CLASSIFICATIONS",
        "LFL_estimate UFL_estimate": "physchem.safety_environment:flammability_estimate",
        "NFPA_30_classification": "physchem.safety_environment:nfpa_30_classification",
        "Suzuki_LFL Suzuki_UFL": "physchem.safety_environment:suzuki_flammability_limit",
        "fire_mixing": "physchem.safety_environment:fire_mixture_limit",
        "mgm3_to_ppmv": "physchem.safety_environment:mass_concentration_to_ppmv",
        "ppmv_to_mgm3": "physchem.safety_environment:ppmv_to_mass_concentration",
    },
    "environment": {
        "GTP GWP ODP logP": "data.assessments:AssessmentDatabase.assessment",
        "GTP_methods GWP_methods ODP_methods logP_methods": "data.assessments:AssessmentDatabase.assessment_methods",
        "GTP_all_methods GWP_all_methods ODP_all_methods logP_all_methods": "data.assessments:AssessmentDatabase.method_specs",
    },
    "combustion": {
        "AKI octane_sensitivity": "physchem.combustion:octane_index",
        "MON RON ignition_delay": "data.fuels:FuelDatabase.fuel_value",
        "MON_methods RON_methods ignition_delay_methods": "data.fuels:FuelDatabase.fuel_methods",
        "ignition_delay_all_methods": "data.fuels:FUEL_METHODS",
        "CombustionData": "physchem.combustion:CombustionData",
        "HHV_Boie HHV_modified_Dulong": "physchem.combustion:heating_value_mass",
        "HHV_from_LHV LHV_from_HHV": "physchem.combustion:heating_value_convert",
        "HHV_stoichiometry": "physchem.combustion:combustion_enthalpy",
        "IDT_to_DCN Perez_Boehman_MON_from_ignition_delay Perez_Boehman_RON_from_ignition_delay": "physchem.combustion:ignition_delay_quality",
        "air_fuel_ratio_solver": "physchem.combustion_streams:air_fuel_ratio_solver",
        "combustion_data": "physchem.combustion:combustion_data",
        "combustion_products_mixture": "physchem.combustion:combustion_products_mixture",
        "combustion_spec_solver fuel_air_spec_solver": "physchem.combustion_streams:combustion_spec_solver",
        "combustion_stoichiometry": "physchem.combustion:combustion_stoichiometry",
    },
    "temperature": {
        "ITS90_68_difference": "physchem.temperature_scales:its90_68_difference",
        "T_converter": "physchem.temperature_scales:temperature_convert",
        "T_scales": "physchem.temperature_scales:TEMPERATURE_SCALES",
        "Ts_27 Ts_48 Ts_68 Ts_76 diffs_27 diffs_48 diffs_68 diffs_76": "physchem.temperature_scale_data:TEMPERATURE_SCALE_TABLES",
    },
}


def method_inventory(module, groups):
    mapping = {name: target for names, target in groups.items() for name in names.split()}
    upstream = importlib.import_module("chemicals." + module)
    assert set(mapping) == set(upstream.__all__), (
        set(upstream.__all__) - set(mapping),
        set(mapping) - set(upstream.__all__),
    )
    for target in mapping.values():
        mod, path = target.split(":")
        obj = importlib.import_module("chemworld." + mod)
        for attr in path.split("."):
            obj = getattr(obj, attr)
    return {
        "checks": [{"passed": True}],
        "public_symbols": len(mapping),
        "mapping": mapping,
        "coverage_kind": "inventory; numerical qualification remains F1-F5 fixed blocks",
    }


def source_inventory(db, source, count):
    records = db.records(source)
    assert len(records) == count
    return {
        "checks": [{"passed": True}],
        "source_id": source,
        "records": len(records),
        "relative_path": db.source(source).relative_path,
        "coverage_kind": "source inventory; original complete numerical qualification retained",
    }


def inventory_units():
    for mod, groups in GROUPS.items():
        yield "methods/" + mod, {"module": mod}, lambda m=mod, g=groups: method_inventory(m, g)
    root = Path(chemicals.__file__).parent
    seen, total = set(), 0
    source_groups = [
        (ConstantDatabase(root), {k: v[2] for k, v in F1_SOURCES.items()}),
        (ElectroOpticalDatabase(root), F2_COUNTS),
        (AssessmentDatabase(root), F3_COUNTS),
        (
            FuelDatabase(root),
            dict(
                zip(
                    [s.source_id for s in FUEL_SOURCES],
                    [58, 85, 255, 257, 532, 408, 278, 1066, 278],
                    strict=True,
                )
            ),
        ),
    ]
    for db, sources in source_groups:
        for source, count in sources.items():
            path = db.source(source).relative_path
            if path in seen:
                assert source == "WIKIDATA"
                continue
            seen.add(path)
            total += count
            yield (
                "source/" + source,
                {"source_id": source, "expected_records": count},
                (lambda d=db, s=source, n=count: source_inventory(d, s, n)),
            )
    assert len(seen) == 33 and total == 39548
