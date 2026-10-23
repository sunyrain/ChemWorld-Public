"""Copy outside the checkout and run against the installed wheel, without references."""

import json
import math
import sys

import chemworld
from chemworld.physchem import (
    ActivityModelSpec,
    CaloricReference,
    ComponentPropertyPackage,
    ComponentSpec,
    CubicEOSSpec,
    EOSComponentSpec,
    GibbsExcessMixingSpec,
    PhaseEvaluation,
    PhasePropertyModel,
    PhaseTransportSpec,
    PropertyCorrelation,
    evaluate_cubic_eos,
)


def main():
    packages = tuple(
        ComponentPropertyPackage(
            ComponentSpec(key, formula),
            (
                PropertyCorrelation(
                    key + "-Cp",
                    "cp_polynomial",
                    {"a": 100 + 20 * i},
                    {"temperature": "K"},
                    "J/(mol*K)",
                    property_id="liquid_heat_capacity",
                    source_note="Fixed installation input",
                ),
                PropertyCorrelation(
                    key + "-mu",
                    "andrade_viscosity",
                    {"A": 1e-5, "B": 1000 + 200 * i},
                    {"temperature": "K"},
                    "Pa*s",
                    property_id="liquid_viscosity",
                    source_note="Fixed installation input",
                ),
            ),
        )
        for i, (key, formula) in enumerate((("A", "C2H6"), ("B", "C3H8")))
    )
    datum = CaloricReference(
        phase="liquid",
        temperature_K=300,
        pressure_Pa=1e5,
        entropy_J_mol_K=100,
        source_note="Explicit installation datum",
    )
    activity = ActivityModelSpec(
        "installed-margules", ("A", "B"), "margules", {"A:A|B": 0.8, "A:B|A": 0.8}
    )
    model = PhasePropertyModel(
        packages,
        phase="liquid",
        references=dict.fromkeys(("A", "B"), datum),
        activity_model=activity,
        transport_models={
            "viscosity": PhaseTransportSpec(
                "viscosity",
                "log_mole_fraction",
                "Explicit installation method",
            )
        },
    )
    quantities = (
        "enthalpy",
        "cp",
        "entropy",
        "chemical_potential",
        "activity_coefficient",
        "viscosity",
    )
    state_args = {"temperature_K": 340, "pressure_Pa": 1e5, "quantities": quantities}
    result = model.evaluate({"A": 2, "B": 3}, **state_args)
    checks = {
        "public_phase_H": math.isclose(result.value("enthalpy"), 4480),
        "public_phase_Cp": math.isclose(result.value("cp"), 112),
        "activity_consumed": math.isclose(
            result.value("activity_coefficient", component="A"), math.exp(0.8 * 0.6**2)
        ),
        "transport_consumed": math.isclose(result.value("viscosity"), 1e-5 * math.exp(1120 / 340)),
        "G_sum_xmu": math.isclose(
            result.value("enthalpy") - 340 * result.value("entropy"),
            sum(
                result.composition[k] * result.value("chemical_potential", component=k)
                for k in ("A", "B")
            ),
        ),
        "model_JSON": PhasePropertyModel.from_dict(json.loads(json.dumps(model.to_dict())))
        .evaluate({"A": 2, "B": 3}, **state_args)
        .to_dict()
        == result.to_dict(),
        "snapshot_JSON": PhaseEvaluation.from_dict(
            json.loads(json.dumps(result.to_dict()))
        ).to_dict()
        == result.to_dict(),
    }
    pure = (
        packages[0]
        .phase_model(phase="liquid", reference=datum)
        .evaluate(
            {"A": 2},
            temperature_K=340,
            pressure_Pa=1e5,
            quantities=("enthalpy",),
        )
    )
    checks["package_phase_model"] = math.isclose(pure.value("enthalpy"), 4000)
    eos = CubicEOSSpec(
        "installed-HV",
        "peng_robinson",
        (
            EOSComponentSpec("A", 500, 4e6, 0.2),
            EOSComponentSpec("B", 650, 3e6, 0.6),
        ),
        ge_mixing=GibbsExcessMixingSpec("huron_vidal", activity),
    )
    state = evaluate_cubic_eos(
        eos, {"A": 0.4, "B": 0.6}, temperature_K=340, pressure_Pa=1e5, phase="vapor"
    )
    checks["GE_cubic_consumer"] = state.molar_volume_m3_mol > 0 and all(
        v > 0 for v in state.fugacity_coefficients.values()
    )
    checks["no_runtime_reference_backend"] = "thermo" not in sys.modules
    report = {
        "package_path": chemworld.__file__,
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
    }
    print(json.dumps(report, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
