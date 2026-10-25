"""Fixed parent-53 closure cases, shared only by tests/reference orchestration."""

from dataclasses import replace

from chemworld.data.calibration_uncertainty import KNOWN_INDEPENDENT_SIGMA
from chemworld.data.catalysis_calibration import CatalyticEnergyCalibration
from chemworld.examples.catalytic_energy_tables import FRAME, energy_calibration

FIT_CASES = ("scaling", "scaling_units", "scaling_fixed", "scaling_log", "bep", "bep_units")
BOUNDARY_CASES = ("rank", "bound", "optimizer_budget", "forward_budget")


def rebuild(original, *, table=None, parameters=None, **options):
    return CatalyticEnergyCalibration(
        original.table if table is None else table,
        original.model,
        original.parameters if parameters is None else parameters,
        fitted_source=original.fitted_source,
        reference_frame=FRAME,
        energy_kind=original.energy_kind,
        noise_model=KNOWN_INDEPENDENT_SIGMA,
        **options,
    )


def case(name):
    calibration = energy_calibration(
        bep=name.startswith("bep"),
        mixed_units=name.endswith("units"),
        confounded=name == "rank",
        max_forward_groups=1 if name == "forward_budget" else 6000,
    )
    options = {"max_nfev": 1 if name == "optimizer_budget" else 60}
    if name == "scaling_fixed":
        options["fixed_parameters"] = {"intercept_J_mol": -5000.0}
    if name in {"bound", "scaling_log"}:
        parameters = list(calibration.parameters)
        parameters[1] = replace(
            parameters[1],
            bounds=(-1.0, 0.2) if name == "bound" else (0.05, 2.0),
            initial=0.1 if name == "bound" else 0.4,
            coordinate="linear" if name == "bound" else "log",
        )
        calibration = rebuild(calibration, parameters=parameters)
    return calibration, options
