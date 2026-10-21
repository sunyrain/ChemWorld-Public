"""Read-only impedance of the actual isothermal native battery residual.

No second transport or electrical law is implemented here. The retained state
contains inventories, charge/work coordinates and the existing algebraic circuit.
Fixed temperature and receipt-only energy counters have no feedback in this
explicit isothermal/no-side-reaction domain and are exactly omitted.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse

from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC, battery_system
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_frequency import FrequencyResponse, residual_frequency_response
from chemworld.physchem.residual_solver import mass_matrix_system
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualEquation,
    ResidualVariable,
)


def battery_impedance_system(spec, state):
    """Assemble the same CC residual, with current setpoint as a small-signal input."""
    if spec.thermal is not None or spec.sei is not None:
        raise ValueError(
            "impedance currently requires an isothermal battery without side reactions"
        )
    # Only the mathematical form changes; the model and physical inventories do not.
    coupled = battery_system(
        replace(spec, numerical_form="residual"), ScalarControlProgram(((0.0, 0.0),), "A")
    )
    material = coupled.thermodynamics.inventory_keys
    coordinates = tuple(item.name for item in coupled.coordinates)
    if set(state.amounts_mol) != set(material) or set(state.coordinates) != set(coordinates):
        raise ValueError(
            "impedance state must match the declared battery inventory and coordinates"
        )
    if any(value < 0 for value in state.amounts_mol.values()):
        raise ResidualDomainError("negative battery inventory")
    for item in coupled.coordinates:
        item.validate(state.coordinates[item.name])
    for process in coupled.processes:
        if process.validate_state is not None:
            process.validate_state(state)
    closure = coupled.algebraic
    names = tuple(v.name for v in closure.variables)
    variables = tuple(
        ResidualVariable(k, "mol", coupled.inventory_scales_mol[k], lower=0) for k in material
    )
    variables += tuple(
        ResidualVariable(item.name, item.unit, item.scale) for item in coupled.coordinates
    )
    count = len(variables)
    equations = tuple(ResidualEquation(v.name, v.unit + "/s", v.scale) for v in variables)
    variables += closure.variables
    equations += closure.equations
    identities = tuple(v.name for v in variables)
    guessed = closure.initial_values(state)
    initial = np.array(
        [
            *(state.amounts_mol[k] for k in material),
            *(state.coordinates[k] for k in coordinates),
            *(guessed[k] for k in names),
        ]
    )

    def view(t, y):
        return CoupledStateView(
            dict(zip(material, y[: len(material)], strict=True)),
            state.temperature_K,
            state.volume_L,
            t,
            coordinates=dict(zip(coordinates, y[len(material) : count], strict=True)),
            algebraic_values=dict(zip(names, y[count:], strict=True)),
        )

    base = view(state.time_s, initial)
    for event in coupled.all_events:
        if event.terminal and event.value(base) <= 0:
            raise ResidualDomainError(
                "impedance point is on/outside a battery stopping boundary: " + event.event_id
            )
    control = next(i for i, e in enumerate(equations) if e.name == "battery:control")
    input_derivative = np.zeros((len(variables), 1))
    input_derivative[control, 0] = 1 / spec.maximum_current_A

    def rhs(t, y, p):
        candidate = view(t, y)
        rates = assemble_coupled_rates(
            candidate,
            processes=coupled.processes,
            thermodynamics=coupled.thermodynamics,
            heat_flux=coupled.heat_flux,
            ports=coupled.ports,
            energy_ports=coupled.energy_ports,
            isothermal=True,
        )
        result = np.r_[
            [rates.amount_rates_mol_s[k] for k in material],
            [rates.coordinate_rates[k] for k in coordinates],
            -np.asarray(closure.residual(candidate)),
        ]
        result[control] += p[0] / spec.maximum_current_A
        return result

    system = mass_matrix_system(
        "battery:rest-impedance",
        variables,
        equations,
        sparse.diags(np.r_[np.ones(count), np.zeros(len(names))]),
        rhs,
        derivatives="hybrid",
        rhs_jacobian=None
        if coupled.residual_rhs_jacobian is None
        else lambda t, y, p: coupled.residual_rhs_jacobian(view(t, y), identities),
        rhs_parameter_jacobian=lambda t, y, p: input_derivative.copy(),
        parameters=(
            ResidualVariable(
                "current_setpoint_A",
                "A",
                spec.maximum_current_A,
                "parameter",
                -spec.maximum_current_A,
                spec.maximum_current_A,
            ),
        ),
        parameter_values=(0.0,),
    )
    initial.setflags(write=False)
    return system, initial


@dataclass(frozen=True)
class BatteryImpedance:
    response: FrequencyResponse
    voltage_index: int

    @property
    def success(self):
        return self.response.success

    @property
    def impedance_ohm(self):
        return tuple(
            None if p.response is None else -p.response[self.voltage_index, 0]
            for p in self.response.points
        )

    def to_dict(self):
        return {
            "success": self.success,
            "unit": "ohm",
            "definition": "Z = -delta_V / delta_I",
            "current_convention": "positive discharge",
            "phasor_convention": "exp(+i*2*pi*f*t)",
            "scope": "rest, isothermal, no side reactions or double-layer capacitance",
            "points": [
                {
                    "frequency_Hz": p.frequency_Hz,
                    "success": p.success,
                    "real_ohm": None if z is None else float(z.real),
                    "imag_ohm": None if z is None else float(z.imag),
                    "failure_reason": p.failure_reason,
                    "relative_linear_residual": p.relative_linear_residual,
                }
                for p, z in zip(self.response.points, self.impedance_ohm, strict=True)
            ],
            "diagnostics": dict(self.response.diagnostics),
        }


def battery_impedance(spec, frequencies_Hz, *, state=None, stationary_tolerance=1e-9):
    """Compute a spectrum without altering state, time, observations or receipts."""
    state = spec.initial_state() if state is None else state
    system, initial = battery_impedance_system(spec, state)
    response = residual_frequency_response(
        system,
        initial,
        frequencies_Hz,
        time_s=state.time_s,
        stationary_tolerance=stationary_tolerance,
    )
    index = next(i for i, v in enumerate(system.variables) if v.name == BATTERY_ALGEBRAIC[1])
    return BatteryImpedance(response, index)


__all__ = ["BatteryImpedance", "battery_impedance", "battery_impedance_system"]
