"""Self-contained installed-package workflow; no source library or test fixtures."""

import json
import math

from chemworld.data import MultifluidCalibration, MultifluidFitParameter
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem import (
    ActivityModelSpec,
    AssociationSite,
    CanonicalSAFTAssociation,
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    HeatFlux,
    HelmholtzFluid,
    HelmholtzTerm,
    MaskedSAFTAssociation,
    MaterialPort,
    MolecularIsochoricThermodynamics,
    MolecularStream,
    MultifluidActivityModel,
    MultifluidAssociationModel,
    MultifluidBinaryPair,
    MultifluidModel,
    SAFTAssociation,
    molecular_composition_response,
    solve_molecular_critical,
    trace_molecular_critical,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.solver_backend import ODESolverPolicy

R = 8.31446261815324


def authored_model():
    """Polynomial EOS with analytic Tc=sqrt(3)*Tr, rhoc=rhor/sqrt(3)."""
    fluids = tuple(
        HelmholtzFluid(
            key,
            str(i + 1),
            0.03 + 0.01 * i,
            R,
            200.0,
            1000.0,
            100.0,
            1000.0,
            1e8,
            (
                HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 0.0, "a2": 0.0}),
                HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 2.5}),
            ),
            (
                HelmholtzTerm(
                    "ResidualHelmholtzPower",
                    {"n": [-3.0, 0.5], "d": [1.0, 2.0], "t": [1.0, 0.0], "l": [0.0, 0.0]},
                ),
            ),
            "authored analytical polynomial; not a material fit",
        )
        for i, key in enumerate(("A", "B"))
    )
    return MultifluidModel(
        fluids,
        (MultifluidBinaryPair(("A", "B"), 1.0, 1.0, 1.0, 1.0, 0.0, (), "authored pair"),),
        R,
        "source_equal",
        "authored polynomial mixture",
    )


def fit_model(model):
    columns = tuple(
        QuantityColumn(k, unit, role, bounds, "authored install probe")
        for k, unit, role, bounds in (
            ("temperature", "K", "condition", (100.0, 1000.0)),
            ("molar_density", "mol/m^3", "condition", (0.0, 5000.0)),
            ("x:A", "dimensionless", "condition", (0.0, 1.0)),
            ("x:B", "dimensionless", "condition", (0.0, 1.0)),
            ("pressure_Pa", "Pa", "observation", (0.0, 1e8)),
        )
    )
    rows = []
    for i, (t, rho, x) in enumerate(
        ((400.0, 150.0, 0.3), (450.0, 250.0, 0.6), (420.0, 200.0, 0.5))
    ):
        rows.append(
            ScientificRow(
                str(i),
                "authored",
                str(i),
                str(i),
                "train" if i < 2 else "validation",
                {
                    "temperature": InputValue(t, "K", 0.0, "exact"),
                    "molar_density": InputValue(rho, "mol/m^3", 0.0, "exact"),
                    "x:A": InputValue(x, "dimensionless", 0.0, "exact"),
                    "x:B": InputValue(1 - x, "dimensionless", 0.0, "exact"),
                    "pressure_Pa": InputValue(
                        model.state(t, rho, (x, 1 - x)).pressure_Pa,
                        "Pa",
                        1.0,
                        "synthetic known sigma",
                    ),
                },
            )
        )
    table = ScientificTable(
        "install-fit",
        "measurements",
        columns,
        (
            InputSource(
                DataSourceProvenance(
                    "authored", 0, "Synthetic install probe", version="1", license="MIT"
                ),
                "synthetic",
            ),
        ),
        tuple(rows),
    )
    parameter = MultifluidFitParameter("binary", ("A", "B"), "gamma_T", (0.9, 1.1), 1.04)
    report = MultifluidCalibration(
        table, model, (parameter,), fitted_source="installation synthetic recovery"
    ).fit(max_nfev=20)
    assert report["status"] == "fitted", report["status"]
    assert abs(report["estimate"][parameter.parameter_id] - 1) < 1e-8
    assert report["validation"]["weighted_rmse"] < 1e-5
    return MultifluidModel.from_dict(json.loads(json.dumps(report["fitted_model"])))


def coupled_models(base):
    b = (3e-5, 4e-5)
    activity = ActivityModelSpec(
        "authored Wilson",
        base.component_ids,
        "wilson",
        {"lambda_a:A|B": math.log(b[1] / b[0]) - 0.2, "lambda_a:B|A": math.log(b[0] / b[1]) - 0.2},
    )
    policy = "explicit_joint_parameterization"
    yield MultifluidActivityModel(
        base, activity, b, 1.17, "wilson_volume_residual", "authored joint model", policy
    )
    sites = (AssociationSite("A", "e", 1), AssociationSite("B", "H", 1))
    assoc = SAFTAssociation(
        sites, ((400.0, 400.0), (400.0, 400.0)), ((1e-28, 1e-28), (1e-28, 1e-28)), "authored sites"
    )
    kernel = CanonicalSAFTAssociation(
        base.component_ids, b, assoc, "authored contact", "Carnahan-Starling"
    )
    yield MultifluidAssociationModel(
        base,
        MaskedSAFTAssociation(kernel, ((False, True), (True, False))),
        "authored joint model",
        policy,
    )


def probe():
    base = authored_model()
    critical = solve_molecular_critical(
        base,
        (0.5, 0.5),
        initial_temperature_K=350.0,
        initial_density_mol_m3=600.0,
        temperature_bounds_K=(300.0, 400.0),
        density_bounds_mol_m3=(400.0, 800.0),
    )
    assert critical.success, critical.to_dict()
    assert abs(critical.response["temperature_K"] - 200 * math.sqrt(3)) < 1e-5
    assert abs(critical.response["molar_density_mol_m3"] - 1000 / math.sqrt(3)) < 1e-5
    path = trace_molecular_critical(
        base,
        ((0.3, 0.7), (0.5, 0.5), (0.7, 0.3)),
        initial_temperature_K=350.0,
        initial_density_mol_m3=600.0,
        temperature_bounds_K=(300.0, 400.0),
        density_bounds_mol_m3=(400.0, 800.0),
    )
    assert path["success"] and path["completed_stations"] == 3, path
    models = (fit_model(base), *coupled_models(base))
    for model in models:
        response = molecular_composition_response(model, 400.0, (50.0, 50.0))
        assert response["locally_convex"] and response["density_susceptibility_mol2_J_m3"]
        thermal = MolecularIsochoricThermodynamics(model.caloric_model(), 10.0, 5.0)
        restored = MolecularIsochoricThermodynamics.from_dict(
            json.loads(json.dumps(thermal.to_dict()))
        )
        assert restored == thermal
        for mode in ("heat", "feed"):
            start = CoupledState({"A": 0.5, "B": 0.5}, 400.0, 10.0)
            options = {
                "policy": ODESolverPolicy(
                    "multifluid27-install", method="BDF", rtol=1e-10, atol=1e-12
                )
            }
            if mode == "heat":
                options["heat_flux"] = lambda _: HeatFlux(5.0)
                expected_n, expected_energy = dict(start.amounts_mol), 0.5
            else:
                stream = MolecularStream(restored.model.state(450.0, 100.0, (0.3, 0.7)), 0.01)
                contribution = stream.contribution({"A": "feed-A", "B": "feed-B"})
                options["processes"] = (
                    CoupledProcess("feed", lambda _, contribution=contribution: contribution),
                )
                options["ports"] = tuple(
                    MaterialPort("feed-" + k, k, "source", "vessel", "in") for k in ("A", "B")
                )
                expected_n = {"A": 0.5003, "B": 0.5007}
                expected_energy = 0.1 * stream.enthalpy_rate_W
            result = CoupledSystem(restored, **options).advance(start, 0.1)
            assert all(
                abs(result.final_state.amounts_mol[k] - v) < 1e-11 for k, v in expected_n.items()
            )
            delta = restored.internal_energy_j(result.final_state) - restored.internal_energy_j(
                start
            )
            assert abs(delta - expected_energy) < 2e-5
            assert abs(result.energy_balance_residual_J) < 2e-5
    return {
        "critical_roots": 1,
        "critical_path_stations": 3,
        "fits": 1,
        "response_and_codec_models": 3,
        "finite_inventory_advances": 6,
    }


if __name__ == "__main__":
    print(json.dumps(probe()))
