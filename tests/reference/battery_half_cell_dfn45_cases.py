"""Prospective Xu2019 half-cell DFN routes; no block is executed by importing this.

The native model is explicitly a constant-transport projection. Full released
Xu2019 and the named frozen-transport source variant remain separate routes.
Only the latter is a same-equation numerical comparison with native physics.
"""

from copy import deepcopy
from math import isfinite, sqrt

import numpy as np

PYBAMM_VERSION, CASADI_VERSION = "25.4.2", "3.6.7"
PYBAMMSOLVERS_VERSION = "0.3.3"
RTOL, ATOL, ROOT_TOL = 1e-8, 1e-10, 1e-8
F = 96485.33212
SOURCE_F_C_MOL = 96485.33212331001
CASE_IDS = tuple(f"H{i:02d}" for i in range(1, 7))
GRIDS = {
    "coarse": {"x": (6, 8), "r": 16},
    "middle": {"x": (12, 16), "r": 32},
    "fine": {"x": (24, 32), "r": 64},
}
SOURCE_VARIANTS = (
    "modular-full-Xu2019",
    "modular-constant-transport",
    "basic-full-raw-metal",
    "basic-full-pure-metal",
    "basic-constant-transport-pure-metal",
)
RESPONSE_ORDER = (
    "voltage_V",
    "current_A",
    "charge_C",
    "work_J",
    "metal_Li_mol",
    "positive_Li_mol",
    "electrolyte_salt_mol",
    "metal_boundary_ce_mol_m3",
    "metal_j0_A_m2",
    "metal_eta_V",
    "metal_ohmic_V",
)
PROFILE_COORDINATES = tuple(i / 10 for i in range(1, 10))
THRESHOLD_CANDIDATES = {
    "status": "prospective; coordinator must bind the final experiment note before GO",
    "same_equation_absolute_plus_relative": {
        "voltage_V": (2e-3, 2e-4),
        "current_A": (2e-5, 1e-2),
        "charge_C": (2e-3, 2e-3),
        "work_J": (1e-2, 2e-3),
        "surface_fraction": (2e-3, 2e-3),
        "radial_fraction": (2e-3, 2e-3),
        "salt_mol_m3": (2.0, 2e-3),
        "potential_V": (2e-3, 2e-4),
        "local_current_A": (2e-6, 1e-2),
        "metal_j0_A_m2": (2e-2, 2e-3),
        "metal_eta_V": (1e-3, 2e-3),
        "event_time_s": (1e-3, 1e-4),
    },
    "finest_pair_refinement_factor": 0.5,
    "independent_invariants": {
        "Faraday_C": 1e-6,
        "salt_mol": 1e-10,
        "solid_Li_mol": 1e-10,
        "hardware_current_A": 1e-8,
        "finite_CV_voltage_V": 1e-6,
        "event_surface_fraction": 1e-6,
        "metal_event_mol": 1e-10,
    },
    "full_Xu2019_vs_projection": {
        "status": "model-discrepancy reporting; not a same-equation pass/fail qualification",
        "retain": (
            "all full-source responses",
            "signed differences",
            "coefficient ranges",
            "every actual event and time",
            "all failures",
        ),
    },
}

# These are a declared projection of the released entries, not a second Xu2019
# parameter catalog. The source adapter preserves all 66 original entries/code.
XU2019_PROJECTION = {
    "area_m2": 0.01 * 0.0154,
    "metal_thickness_m": 0.0007,
    "metal_conductivity_S_m": 10776000.0,
    "metal_molar_volume_m3_mol": 1.3e-5,
    "lengths_m": (2.5e-5, 4.2e-5),
    "porosities": (0.39, 0.331),
    "bruggeman": (1.5, 1.5),
    "active_fraction": 0.518,
    "positive_sigma_material_S_m": 100.0,
    "positive_solid_bruggeman": 1.5,
    "radius_m": 5.3e-6,
    "particle_diffusivity_m2_s": 1e-14,
    "cmax_mol_m3": 48230.0,
    "initial_positive_ce_mol_m3": 4631.0,
    "ce_reference_mol_m3": 1000.0,
    "temperature_K": 298.15,
    "cation_transference": 0.38,
    "thermodynamic_factor": 1.0,
    "frozen_D_m2_s": 3.222722528605264e-10,
    "frozen_kappa_S_m": 1.194326363775062,
    "metal_salt_exponent": 0.3,
    "source_nominal_capacity_Ah": 0.0024,
    "ocv_polynomial_V": (4.3452, -1.6518, 1.6225, -2.0843, 3.5146, -2.2166),
    "ocv_exponential": (-0.5623e-4, 109.451, 100.006),
}


def _stage(duration, target, unit="A", samples=21, expected="complete"):
    return {
        "duration_s": float(duration),
        "target": float(target),
        "unit": unit,
        "samples": samples,
        "expected": expected,
    }


_CASES = {
    "H01": ("released-initial-rest", (_stage(60.0, 0.0, samples=11),)),
    "H02": ("released-one-C-forward", (_stage(600.0, 0.0024, samples=31),)),
    "H03": (
        "CC-rest-reverse-rest",
        (_stage(300.0, 0.0024), _stage(60.0, 0.0), _stage(300.0, -0.0012), _stage(60.0, 0.0)),
    ),
    "H04": (
        "CC-rest-finite-CV",
        (_stage(600.0, 0.0024), _stage(60.0, 0.0), _stage(300.0, 4.05, "V")),
    ),
    "H05": (
        "declared-thin-metal-Faraday-cutoff",
        (_stage(90.0, 0.0024, samples=41, expected="finite-metal-depleted"),),
    ),
    "H06": ("conserved-nonuniform-salt-CC-rest", (_stage(120.0, 0.0012), _stage(120.0, 0.0))),
}
_SCIENTIFIC_QUESTIONS = {
    "H01": "Released uniform rest: no physical drift; retain the original near-upper-cutoff t0",
    "H02": "Nominal-capacity CC: nonzero metal salt boundary and spatial particle/reaction fields",
    "H03": "Current/rest/reversal: signed Faraday and work, including internal redistribution",
    "H04": "One continuous CC/rest/finite-CV state history with declared hardware current limit",
    "H05": "Explicit thin fixed-geometry foil: finite-metal event versus known Faraday time",
    "H06": "Legal conserved-mean nonuniform salt: relaxation and full versus frozen transport",
}


def case_data(identity):
    if type(identity) is int:
        identity = f"H{identity:02d}"
    if identity not in _CASES:
        raise ValueError("unknown Xu2019 DFN half-cell unit")
    name, stages = _CASES[identity]
    return {
        "id": identity,
        "name": name,
        "purpose": "prospective-qualification",
        "scientific_question": _SCIENTIFIC_QUESTIONS[identity],
        "source_name": "Xu2019",
        "source_version": "PyBaMM-25.4.2",
        "temperature_K": 298.15,
        "ce_reference_mol_m3": 1000.0,
        "source_nominal_capacity_Ah": 0.0024,
        "maximum_current_A": 0.0048,
        "surface_margin": 0.01,
        "minimum_concentration_mol_m3": 1.0,
        "minimum_voltage_V": 3.5,
        "maximum_voltage_V": 4.2,
        "common_event_scale": 1e4,
        "metal_thickness_m": 1e-7 if identity == "H05" else 0.0007,
        "initial_salt_profile": "pore-mean-conserved-linear" if identity == "H06" else "uniform",
        "initial_salt_span_mol_m3": 500.0 if identity == "H06" else 0.0,
        "initial_salt_direction": -1 if identity == "H06" else 1,
        "stages": deepcopy(stages),
        "comparison_policy": {
            "same_equation_native_source": "modular-constant-transport",
            "same_equation_source_source": (
                ("modular-full-Xu2019", "basic-full-pure-metal"),
                ("modular-constant-transport", "basic-constant-transport-pure-metal"),
            ),
            "model_discrepancy_only": (
                "modular-full-Xu2019 versus native projection",
                "basic-full-raw-metal versus pure-metal convention",
            ),
            "original_ocp_and_endpoint_barrier_preserved": True,
            "common_interval_only_for_paired_fields": True,
            "actual_event_tails_and_t0_retained_separately": True,
            "profile_coordinates": PROFILE_COORDINATES,
        },
    }


def validate_case(data):
    if (
        data["source_name"] != "Xu2019"
        or data["source_version"] != "PyBaMM-25.4.2"
        or data["temperature_K"] != 298.15
        or data["ce_reference_mol_m3"] != 1000.0
    ):
        raise ValueError("half-cell source requires the released Xu2019 reference conditions")
    for key in (
        "maximum_current_A",
        "surface_margin",
        "minimum_concentration_mol_m3",
        "minimum_voltage_V",
        "maximum_voltage_V",
        "common_event_scale",
        "metal_thickness_m",
        "initial_salt_span_mol_m3",
    ):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("invalid half-cell source scenario/control declaration")
    if (
        data["maximum_current_A"] <= 0
        or data["metal_thickness_m"] <= 0
        or not 0 < data["surface_margin"] < 0.5
        or not 0 < data["minimum_concentration_mol_m3"] < 1000.0
        or data["minimum_voltage_V"] >= data["maximum_voltage_V"]
        or data["common_event_scale"] <= 0
        or data["initial_salt_profile"] not in {"uniform", "pore-mean-conserved-linear"}
        or not 0 <= data["initial_salt_span_mol_m3"] < 1000.0
        or type(data["initial_salt_direction"]) is not int
        or data["initial_salt_direction"] not in {-1, 1}
    ):
        raise ValueError("invalid physical half-cell source scenario")
    if not data["stages"]:
        raise ValueError("half-cell source needs explicit stages")
    for stage in data["stages"]:
        if (
            set(stage) != {"duration_s", "target", "unit", "samples", "expected"}
            or stage["unit"] not in {"A", "V"}
            or any(
                isinstance(stage[k], bool)
                or not isinstance(stage[k], (int, float))
                or not isfinite(stage[k])
                for k in ("duration_s", "target")
            )
            or stage["duration_s"] <= 0
            or type(stage["samples"]) is not int
            or stage["samples"] < 2
        ):
            raise ValueError("invalid half-cell source stage")
        if stage["unit"] == "A" and abs(stage["target"]) > data["maximum_current_A"]:
            raise ValueError("CC request exceeds the explicit finite source")
        if stage["unit"] == "V" and not (
            data["minimum_voltage_V"] <= stage["target"] <= data["maximum_voltage_V"]
        ):
            raise ValueError("CV request exceeds the physical voltage domain")
    return data


def validate_grid(grid):
    if (
        set(grid) != {"x", "r"}
        or len(grid["x"]) != 2
        or any(type(n) is not int or n < 2 for n in (*grid["x"], grid["r"]))
    ):
        raise ValueError("half-cell source grid requires two real x domains and radial cells")
    return grid


def stage_times(stage):
    return tuple(stage["duration_s"] * i / (stage["samples"] - 1) for i in range(stage["samples"]))


def initial_salt_profile(data, x_local_m):
    p = XU2019_PROJECTION
    lengths, eps = np.asarray(p["lengths_m"]), np.asarray(p["porosities"])
    starts = np.r_[0.0, np.cumsum(lengths)[:-1]]
    mean_x = np.sum(eps * lengths * (starts + lengths / 2)) / np.sum(eps * lengths)
    return 1000.0 + (
        data["initial_salt_direction"]
        * data["initial_salt_span_mol_m3"]
        * (np.asarray(x_local_m) - mean_x)
        / sum(lengths)
    )


def routes():
    result = []
    for identity in CASE_IDS:
        data = case_data(identity)
        for implementation in ("native", *SOURCE_VARIANTS):
            if implementation.startswith("basic") and any(s["unit"] == "V" for s in data["stages"]):
                continue
            for level, grid in GRIDS.items():
                result.append(
                    {
                        "unit": identity,
                        "implementation": implementation,
                        "grid_level": level,
                        "grid": deepcopy(grid),
                    }
                )
    return tuple(result)


def planned_denominators():
    histories = routes()
    return {
        "units": len(CASE_IDS),
        "histories": len(histories),
        "requested_stages": sum(len(case_data(r["unit"])["stages"]) for r in histories),
        "requested_sample_frames_before_events": sum(
            sum(s["samples"] for s in case_data(r["unit"])["stages"]) for r in histories
        ),
        "by_implementation": {
            name: sum(r["implementation"] == name for r in histories)
            for name in ("native", *SOURCE_VARIANTS)
        },
    }


def native_spec(data, grid):
    """Explicit finite-metal/constant-transport projection; not full Xu2019."""
    from chemworld.data.scientific_inputs import (
        InputSource,
        InputValue,
        QuantityColumn,
        ScientificRow,
        ScientificTable,
    )
    from chemworld.physchem.battery_half_cell import METAL_UNITS, LithiumMetalCounterElectrodeSpec
    from chemworld.physchem.battery_half_cell_dfn import (
        BatteryHalfCellDFNSpec,
        BatteryHalfCellElectrolyteSpec,
        LithiumMetalSaltKineticsSpec,
    )
    from chemworld.physchem.battery_materials import ElectrodeOCV, ElectrodeOCVEndpointPolicy
    from chemworld.physchem.battery_models import ParticleSpec
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    validate_case(data)
    validate_grid(grid)
    p = XU2019_PROJECTION
    source = InputSource(
        DataSourceProvenance(
            "PyBaMM25.4.2-Xu2019-half-cell-projection",
            0,
            "Released Xu2019.py/Valoen2005: explicitly frozen D,kappa at ce=1000,T=298.15; "
            "not a full correlated-transport or measured-cell qualification",
            "25.4.2",
            "BSD-3-Clause",
        ),
        "literature",
    )

    def table(identity, values, units):
        return ScientificTable(
            identity,
            "parameters",
            tuple(
                QuantityColumn(
                    k, units[k], "parameter", (0, 1e15), "Released equation/input derivation"
                )
                for k in values
            ),
            (source,),
            (
                ScientificRow(
                    "parameters",
                    source.provenance.source_id,
                    "literature",
                    "Li",
                    "parameters",
                    {
                        k: InputValue(
                            v,
                            units[k],
                            None,
                            "Source uncertainty not supplied; declared fixed-model projection",
                        )
                        for k, v in values.items()
                    },
                ),
            ),
        )

    metal_values = {
        "area_m2": p["area_m2"],
        "thickness_m": data["metal_thickness_m"],
        "conductivity_S_m": p["metal_conductivity_S_m"],
        "partial_molar_volume_m3_mol": p["metal_molar_volume_m3_mol"],
        "exchange_current_density_A_m2": 3.5e-8
        * SOURCE_F_C_MOL
        * (1 / p["metal_molar_volume_m3_mol"]) ** 0.7
        * 1000.0**0.3,
        "reference_temperature_K": 298.15,
        "electrolyte_concentration_mol_m3": 1000.0,
    }
    amplitude, exponent, offset = p["ocv_exponential"]
    curve = ElectrodeOCV(
        "Xu2019-released-positive-OCP",
        source,
        p["ocv_polynomial_V"],
        ((amplitude, exponent, offset / exponent),),
        endpoint_policy=ElectrodeOCVEndpointPolicy(source),
    )
    return BatteryHalfCellDFNSpec(
        LithiumMetalCounterElectrodeSpec(
            table("Xu2019-metal-projection", metal_values, METAL_UNITS), "parameters"
        ),
        ParticleSpec(
            radius_m=p["radius_m"],
            diffusivity_m2_s=p["particle_diffusivity_m2_s"],
            maximum_concentration_mol_m3=p["cmax_mol_m3"],
            active_volume_m3=p["area_m2"] * p["lengths_m"][1] * p["active_fraction"],
            exchange_current_scale_A_m2=5.76e-11 * SOURCE_F_C_MOL * sqrt(1000.0) * p["cmax_mol_m3"],
            cells=grid["r"],
            mesh_exponent=1.0,
        ),
        curve,
        LithiumMetalSaltKineticsSpec(
            table(
                "Xu2019-metal-salt-exponent",
                {"salt_exponent": 0.3},
                {"salt_exponent": "dimensionless"},
            ),
            "parameters",
        ),
        electrolyte=BatteryHalfCellElectrolyteSpec(
            area_m2=p["area_m2"],
            lengths_m=p["lengths_m"],
            porosities=p["porosities"],
            bruggeman=p["bruggeman"],
            cells=tuple(grid["x"]),
            diffusivity_m2_s=p["frozen_D_m2_s"],
            conductivity_S_m=p["frozen_kappa_S_m"],
            cation_transference=p["cation_transference"],
            thermodynamic_factor=p["thermodynamic_factor"],
            initial_concentration_mol_m3=1000.0,
            minimum_concentration_mol_m3=data["minimum_concentration_mol_m3"],
            positive_solid_conductivity_S_m=p["positive_sigma_material_S_m"]
            * p["active_fraction"] ** p["positive_solid_bruggeman"],
        ),
        initial_positive_fraction=p["initial_positive_ce_mol_m3"] / p["cmax_mol_m3"],
        working_reference_concentration_mol_m3=1000.0,
        temperature_K=298.15,
        surface_margin=data["surface_margin"],
        minimum_voltage_V=data["minimum_voltage_V"],
        maximum_voltage_V=data["maximum_voltage_V"],
        maximum_current_A=data["maximum_current_A"],
        provenance="Xu2019/PyBaMM25.4.2 explicit constant-transport projection",
    )


def native_initial_state(data, spec):
    from dataclasses import replace

    state = spec.initial_state()
    if data["initial_salt_profile"] == "uniform":
        return state
    e = spec.electrolyte
    amounts = {
        **state.amounts_mol,
        **dict(
            zip(
                e.keys,
                initial_salt_profile(data, e.mesh.centers_m) * e.diffusion.storage_volumes_m3,
                strict=True,
            )
        ),
    }
    return replace(state, amounts_mol=amounts)
