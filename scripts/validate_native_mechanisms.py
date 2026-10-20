"""Fixed development block in NATIVE_MECHANISM_ACCEPTANCE.md.

Run with the optional cantera==3.1.0 dependency and --output <summary.json>.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from chemworld.physchem.coupled_system import CoupledState, HeatFlux
from chemworld.physchem.gas_mechanism_batch import gas_batch_system
from chemworld.physchem.mechanism_import import import_mechanism
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.solver_backend import ODESolverPolicy

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/mechanisms"
TEMPERATURES = (600, 900, 1200, 1500, 1800)
PRESSURES = (1e4, 1e5, 1e6)


def run_block():
    import cantera as ct  # Validation only; never an imported native backend.

    if ct.__version__ != "3.1.0":
        raise RuntimeError("validation requires Cantera 3.1.0")
    rows = []
    started = time.monotonic()
    total = 57  # 45 rates + 9 heated trajectories + 3 equilibrium limits.

    def record(label, action):
        try:
            result = action()
            rows.append({"case": label, **result})
        except Exception as error:
            rows.append(
                {"case": label, "passed": False, "error": f"{type(error).__name__}: {error}"}
            )
        print(
            f"{len(rows)}/{total}: {label}: {'PASS' if rows[-1]['passed'] else 'FAIL'}; "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )

    for name in ("pressure_log", "chebyshev", "cantera-3.1.0/h2o2"):
        hydrogen = "h2o2" in name
        path = FIXTURES / f"{name}.yaml"
        network = import_mechanism(path, phase_name="ohmech" if hydrogen else None)
        gas = ct.Solution(str(path), "ohmech" if hydrogen else network.network_id)
        if gas.n_species != len(network.species) or gas.n_reactions != len(network.reactions):
            raise RuntimeError("import changed species/reaction count")
        for temperature in TEMPERATURES:
            for pressure in PRESSURES:

                def rate_case(gas=gas, network=network, temperature=temperature, pressure=pressure):
                    gas.TPX = temperature, pressure, np.ones(gas.n_species)
                    concentrations = dict(zip(gas.species_names, gas.concentrations, strict=True))
                    native = np.array(
                        list(
                            network.reaction_rates(
                                concentrations, volume_L=1, temperature_K=temperature
                            ).values()
                        )
                    )
                    reference = gas.net_rates_of_progress
                    error = np.abs(native - reference)
                    threshold = 1e-12 + 1e-5 * np.abs(reference)
                    return {
                        "passed": bool(np.all(error <= threshold)),
                        "T_K": temperature,
                        "P_Pa": pressure,
                        "max_scaled_error": float(np.max(error / threshold)),
                        "max_absolute_error_mol_L_s": float(np.max(error)),
                        "reaction_count": gas.n_reactions,
                    }

                record(f"{name}:rates:{temperature}:{pressure}", rate_case)
        initial_temperatures = (1000, 1200, 1400) if hydrogen else (700, 1100, 1600)
        for initial_temperature in initial_temperatures:

            def trajectory_case(
                gas=gas, network=network, initial_temperature=initial_temperature, hydrogen=hydrogen
            ):
                gas.TPX = (
                    initial_temperature,
                    1e5,
                    {"H2": 2, "O2": 1, "AR": 7} if hydrogen else {"A": 1},
                )
                amounts = dict(zip(gas.species_names, gas.concentrations, strict=True))
                start = CoupledState(amounts, initial_temperature, 1.0)
                duration = 0.002 if hydrogen else 0.5
                times = np.linspace(0, duration, 21)
                policy = ODESolverPolicy(
                    "finite-gas-reference-bdf", method="BDF", rtol=1e-9, atol=1e-13
                )
                system = gas_batch_system(
                    network, heat_flux=lambda state: HeatFlux(5), policy=policy
                )
                system = replace(
                    system,
                    inventory_scales_mol=dict.fromkeys(amounts, sum(amounts.values())),
                    temperature_scale_K=initial_temperature,
                )
                native = system.advance(start, duration, evaluation_times_s=times)
                # Reconstruct from the serialized native mechanism;
                # no reference trajectory is fed back.
                replay_network = ReactionNetworkSpec.from_dict(
                    json.loads(json.dumps(network.to_dict()))
                )
                replay_system = gas_batch_system(
                    replay_network, heat_flux=lambda state: HeatFlux(5), policy=policy
                )
                replay_system = replace(
                    replay_system,
                    inventory_scales_mol=dict.fromkeys(amounts, sum(amounts.values())),
                    temperature_scale_K=initial_temperature,
                )
                replay = replay_system.advance(start, duration, evaluation_times_s=times)
                reactor = ct.IdealGasReactor(gas, energy="on", volume=0.001)
                reservoir = ct.Reservoir(gas)
                wall = ct.Wall(reservoir, reactor, A=1, Q=5)
                reference_solver = ct.ReactorNet([reactor])
                reference_solver.rtol, reference_solver.atol = 1e-11, 1e-20
                reference = []
                for t in times:
                    if t > 0:
                        reference_solver.advance(t)
                    reference.append([*reactor.thermo.concentrations, reactor.T])
                reference = np.array(reference)
                observed = np.array(
                    [
                        [*point.amounts_mol.values(), point.temperature_K]
                        for point in native.trajectory
                    ]
                )
                scale = np.array(
                    [max(float(np.max(reference[:, :-1])), 1e-15)] * gas.n_species
                    + [float(np.max(reference[:, -1]))]
                )
                error = float(np.max(np.abs(observed - reference) / scale))
                elements = max(
                    abs(value) for value in native.conservation_residuals.values()
                ) / sum(amounts.values())
                energy = abs(native.energy_balance_residual_J) / max(
                    abs(system.thermodynamics.internal_energy_j(start)), 5 * duration, 1
                )
                exact = native.trajectory == replay.trajectory
                return {
                    "passed": error <= 1e-3 and elements <= 1e-8 and energy <= 1e-6 and exact,
                    "initial_T_K": initial_temperature,
                    "duration_s": duration,
                    "evaluation_points": len(times),
                    "normalized_max_error": error,
                    "element_relative_residual": elements,
                    "energy_relative_residual": energy,
                    "serialized_replay_exact": exact,
                    "final_T_native_K": native.final_state.temperature_K,
                    "final_T_reference_K": reactor.T,
                    "boundary_heat_J": native.final_state.boundary_heat_J,
                    "source_species": gas.n_species,
                    "source_reactions": gas.n_reactions,
                    "wall_area_m2": wall.area,
                }

            record(f"{name}:heated-batch:{initial_temperature}", trajectory_case)
        if hydrogen:
            for temperature in (1000, 1200, 1400):

                def equilibrium_case(gas=gas, network=network, temperature=temperature):
                    gas.TPX = temperature, 1e5, {"H2": 2, "O2": 1, "AR": 7}
                    gas.equilibrate("TP", rtol=1e-12)
                    concentrations = dict(zip(gas.species_names, gas.concentrations, strict=True))
                    native = np.array(
                        list(
                            network.reaction_rates(
                                concentrations, volume_L=1, temperature_K=temperature
                            ).values()
                        )
                    )
                    scale = np.maximum(
                        np.maximum(gas.forward_rates_of_progress, gas.reverse_rates_of_progress),
                        1e-12,
                    )
                    error = float(np.max(np.abs(native) / scale))
                    return {
                        "passed": error <= 1e-5,
                        "max_relative_net_rate": error,
                        "temperature_K": temperature,
                    }

                record(f"h2o2:equilibrium:{temperature}", equilibrium_case)
    return {
        "status": "development",
        "reference": f"Cantera {ct.__version__}",
        "source": "tests/fixtures/mechanisms/README.md",
        "planned": total,
        "executed": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "failed": sum(not row["passed"] for row in rows),
        "skipped": 0,
        "elapsed_s": time.monotonic() - started,
        "cases": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_block()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["failed"] == 0 and report["executed"] == report["planned"] else 1)
