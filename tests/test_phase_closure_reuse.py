"""Cell reuse is opt-in; clock-dependent callbacks retain their sequence."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from chemworld.physchem.phase_inventory import PhaseInventoryRates, integrate_phase_inventory


@dataclass(frozen=True)
class State:
    amounts_mol: tuple
    enthalpies_J: tuple
    time_s: float = 0.0
    heater_energy_J: float = 0.0
    condenser_energy_J: float = 0.0


def advance(reuse=None):
    calls, times = [], []

    def closure(index, amounts, enthalpy):
        calls.append((index, dict(amounts), enthalpy))
        return len(calls)

    def rates(state, flashes):
        times.append(state.time_s)
        return PhaseInventoryRates(np.zeros((2, 1)), np.zeros(2), 0, 0, ())

    spec = SimpleNamespace(properties=SimpleNamespace(component_ids=("a",)), at_ph=closure)
    result = integrate_phase_inventory(
        spec, State(({"a": 0.1}, {"a": 0.2}), (0.0, 0.0)), duration_s=1.0,
        rates=rates, closure_indices=(0, 1), stop_functions=(),
        enthalpy_bounds=lambda i, n: (-1, 1),
        **({} if reuse is None else {"reuse_cell_closures": reuse}))
    return result, calls, times


def test_custom_provider_default_keeps_all_calls_and_opt_in_keeps_time_callbacks():
    default, calls, times = advance()
    explicit, original_calls, original_times = advance(False)
    reused, fewer_calls, reused_times = advance(True)
    assert calls == original_calls
    assert times == original_times == reused_times
    assert default == explicit == reused
    assert len(fewer_calls) < len(calls)
