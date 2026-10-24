"""Stationary small-signal analysis of the actual empirical circuit residual.

The original RC and two-node thermal rates and algebraic terminal are reused.
Only six receipt-only integral coordinates are omitted: they have no feedback,
and absolute throughput is not differentiable at zero current. No device state
is relaxed, clipped, or replaced and no sinusoidal acquisition is simulated.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from chemworld.physchem.battery_thevenin import (
    CELL_T,
    JIG_T,
    SOC,
    VOLTAGE,
    rc_key,
    thevenin_system,
)
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_frequency import FrequencyResponse, residual_frequency_response
from chemworld.physchem.residual_solver import mass_matrix_system
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualEquation,
    ResidualVariable,
)


def thevenin_impedance_system(spec, state):
    """Declare the common residual and current input at the supplied rest point."""
    coupled = thevenin_system(spec, ScalarControlProgram(((0., 0.),), "A"))
    declared = {c.name for c in coupled.coordinates}
    if state.amounts_mol or set(state.coordinates) != declared:
        raise ValueError("Thevenin impedance requires the model's exact coordinate ownership")
    for coordinate in coupled.coordinates:
        coordinate.validate(state.coordinates[coordinate.name])
    for process in coupled.processes:
        if process.validate_state is not None:
            process.validate_state(state)
    # Check genuine nondifferentiability first, then keep the numerical SOC
    # stencil in its current interpolation segment. A stationary point close
    # to a knot has a unique local slope: a central difference crossing that
    # knot would incorrectly average two different physical responses.
    soc = state.coordinates[SOC]
    spec.ocv.gradient({k: soc for k, _, _ in spec.ocv.axes})
    soc_segment = None
    for nodes in spec.ocv._points:
        segment = min(max(int(np.searchsorted(nodes, soc, side="right"))-1, 0), len(nodes)-2)
        soc_segment = (nodes[segment], nodes[segment+1])
    surface_segment = None
    surface = soc
    surface_gradient = {}
    if spec.diffusion is not None:
        surface = spec.diffusion.surface_soc(soc, state.coordinates)
        spec.ocv.gradient({k: surface for k, _, _ in spec.ocv.axes})
        for nodes in spec.ocv._points:
            segment = min(max(int(np.searchsorted(nodes, surface, side="right"))-1, 0),
                          len(nodes)-2)
            surface_segment = (nodes[segment], nodes[segment+1])
        surface_gradient = dict(zip(spec.diffusion.coordinate_keys,
                                    spec.diffusion.surface_deviation_gradient, strict=True))
    # Actual body temperatures, not the formal reference temperature, remain
    # physical small-signal coordinates. Integral receipts have no feedback.
    # An installed external-controller q is parked under this explicit CC rest
    # programme. Keep its actual saved value, exclude its perturbation, and do
    # not invent a closed-loop impedance or a new zero-frequency controller mode.
    retained = {SOC, CELL_T, JIG_T,
                *(rc_key(k) for k in range(len(spec.rc_elements))), *surface_gradient}
    coordinates = tuple(c for c in coupled.coordinates if c.name in retained)
    count, closure = len(coordinates), coupled.algebraic
    names = tuple(c.name for c in coordinates)
    algebraic_names = tuple(v.name for v in closure.variables)
    def local_variable(c):
        lower, upper = c.minimum, c.maximum
        intervals = []
        if c.name == SOC and soc_segment:
            intervals.append(soc_segment)
        coefficient = 1. if c.name == SOC else surface_gradient.get(c.name, 0.)
        if coefficient != 0. and surface_segment:
            value = state.coordinates[c.name]
            intervals.append(tuple(sorted(value+(edge-surface)/coefficient
                                          for edge in surface_segment)))
        for lo, hi in intervals:
            lower = lo if lower is None else max(lower, lo)
            upper = hi if upper is None else min(upper, hi)
        return ResidualVariable(c.name, c.unit, c.scale, lower=lower, upper=upper)

    variables = tuple(local_variable(c) for c in coordinates) + closure.variables
    equations = tuple(ResidualEquation(c.name, c.unit+"/s", c.scale) for c in coordinates)
    equations += closure.equations
    guess = closure.initial_values(state)
    initial = np.array([*(state.coordinates[k] for k in names),
                        *(guess[k] for k in algebraic_names)])

    def view(time_s, values):
        return CoupledStateView({}, state.temperature_K, state.volume_L, time_s,
            coordinates={**state.coordinates, **dict(zip(names, values[:count], strict=True))},
            algebraic_values=dict(zip(algebraic_names, values[count:], strict=True)))

    base = view(state.time_s, initial)
    for event in coupled.all_events:
        if event.terminal and event.value(base) <= 0:
            raise ResidualDomainError("impedance point is on/outside a stopping boundary: "
                                      + event.event_id)
    inputs = {"soc": state.coordinates[SOC], "temperature_K": state.coordinates[CELL_T],
              "current_A": 0.}
    for k, pair in enumerate(spec.rc_elements):
        if state.coordinates[rc_key(k)] != 0.:
            # A nonzero relaxing polarization multiplies the parameter
            # derivatives. At exactly zero it does not: continuity suffices.
            for law in pair:
                law.gradient({name: inputs[name] for name, _, _ in law.axes})
    control = next(i for i, equation in enumerate(equations)
                   if equation.name == "thevenin:control")
    limit = spec.native_parameters["maximum_current_A"]
    forcing = np.zeros((len(variables), 1))
    forcing[control, 0] = 1/limit

    def rhs(time_s, values, parameters):
        candidate = view(time_s, values)
        rates = assemble_coupled_rates(candidate, processes=coupled.processes,
            thermodynamics=coupled.thermodynamics, heat_flux=coupled.heat_flux,
            ports=coupled.ports, energy_ports=coupled.energy_ports, isothermal=True)
        result = np.r_[[rates.coordinate_rates[k] for k in names],
                       -np.asarray(closure.residual(candidate))]
        result[control] += parameters[0]/limit
        return result

    system = mass_matrix_system("thevenin:rest-impedance", variables, equations,
        sparse.diags(np.r_[np.ones(count), np.zeros(len(algebraic_names))]), rhs,
        derivatives="hybrid", rhs_parameter_jacobian=lambda t, y, p: forcing.copy(),
        parameters=(ResidualVariable("current_setpoint_A", "A", limit, "parameter",
                                     -limit, limit),),
        parameter_values=(0.,))
    initial.setflags(write=False)
    return system, initial


@dataclass(frozen=True)
class TheveninImpedance:
    response: FrequencyResponse
    voltage_index: int

    @property
    def success(self):
        return self.response.success

    @property
    def impedance_ohm(self):
        return tuple(None if p.response is None else -p.response[self.voltage_index, 0]
                     for p in self.response.points)

    def to_dict(self):
        return {"success": self.success, "unit": "ohm", "definition": "Z = -delta_V / delta_I",
            "current_convention": "positive discharge", "phasor_convention": "exp(+i*2*pi*f*t)",
            "scope": ("stationary RC/optional SOC diffusion/two-node circuit; frozen boundary; "
                      "not a sinusoidal acquisition"),
            "points": [{"frequency_Hz": p.frequency_Hz, "success": p.success,
                "real_ohm": None if z is None else float(z.real),
                "imag_ohm": None if z is None else float(z.imag),
                "relative_linear_residual": p.relative_linear_residual,
                "failure_reason": p.failure_reason}
                for p, z in zip(self.response.points, self.impedance_ohm, strict=True)],
            "diagnostics": dict(self.response.diagnostics)}


def thevenin_impedance(spec, frequencies_Hz, *, state=None, stationary_tolerance=1e-9):
    """Read-only local port response; actual nonstationary body states reject."""
    state = spec.initial_state() if state is None else state
    system, initial = thevenin_impedance_system(spec, state)
    response = residual_frequency_response(system, initial, frequencies_Hz, time_s=state.time_s,
                                           stationary_tolerance=stationary_tolerance)
    index = next(i for i, variable in enumerate(system.variables) if variable.name == VOLTAGE)
    return TheveninImpedance(response, index)


__all__ = ["TheveninImpedance", "thevenin_impedance", "thevenin_impedance_system"]
