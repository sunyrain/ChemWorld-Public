"""Fixed finite-column trajectories and independently assembled balances."""

import time
from itertools import product

import numpy as np
import pytest

from chemworld.examples.native_distillation_research import (
    initial_column_state,
    synthetic_column_spec,
)
from chemworld.physchem.dynamic_distillation import advance_dynamic_column
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec

GRID = tuple(product((3, 5), (0, 1, 3), (5, 10)))


def progress_counter(label):
    last = [time.monotonic()]

    def progress(t, evaluations):
        now = time.monotonic()
        if now - last[0] >= 30:
            print(f"column {label}: RHS evaluations={evaluations}, trial_time={t:.3f}s", flush=True)
            last[0] = now

    return progress


def dynamic_result(trays, reflux, power):
    spec = synthetic_column_spec(trays)
    initial = initial_column_state(spec)
    heater = ThermalDeviceSpec(maximum_power_W=power, setpoint_K=430)
    result = advance_dynamic_column(
        spec,
        initial,
        heater=heater,
        duration_s=120,
        reflux_ratio=reflux,
        evaluation_times_s=np.linspace(0, 120, 61),
        progress=progress_counter((trays, reflux, power)),
    )
    final = result.final_state
    for state in result.trajectory:
        for i, (n, h) in enumerate(zip(state.amounts_mol, state.enthalpies_J, strict=True)):
            phase = spec.at_ph(i, n, h)
            if phase is not None:
                assert 250 - 1e-7 <= phase.temperature_K <= 430 + 1e-7
    assert 0 < result.actual_duration_s <= 120
    for key in spec.properties.component_ids:
        assert sum(n[key] for n in final.amounts_mol) == pytest.approx(
            sum(n[key] for n in initial.amounts_mol), rel=1e-8, abs=1e-12
        )

    # Independently evaluate the declared polynomial/latent fixture on actual
    # phase amounts, adding only the authored hardware heat capacities.
    def energy(state):
        value = 0
        for index, (n, h) in enumerate(zip(state.amounts_mol, state.enthalpies_J, strict=True)):
            phase = spec.at_ph(index, n, h)
            if phase is None:
                continue
            value += spec.capacities_j_k[index] * (phase.temperature_K - 298.15)
            for key, cp, latent in (("light", 80, 30000), ("heavy", 100, 40000)):
                value += (
                    n[key] * cp * (phase.temperature_K - 298.15)
                    - phase.liquid_amounts_mol[key] * latent
                )
        return value

    delta_h = energy(final) - energy(initial)
    actual_heat = final.heater_energy_J - final.condenser_energy_J
    assert delta_h == pytest.approx(actual_heat, rel=1e-6, abs=1e-6)
    for key in spec.properties.component_ids:
        assert sum(part["amounts_mol"][key] for part in result.fractions) == pytest.approx(
            final.amounts_mol[-1][key], abs=1e-10
        )

    return result


@pytest.mark.parametrize("trays,reflux,power", GRID)
def test_fixed_dynamic_grid_conserves_actual_component_and_heat_ports(trays, reflux, power):
    dynamic_result(trays, reflux, power)
