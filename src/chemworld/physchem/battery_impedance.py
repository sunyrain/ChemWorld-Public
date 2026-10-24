"""Read-only impedance of the actual isothermal native battery residual.

No second transport or electrical law is implemented here. The retained state
contains inventories, charge/work coordinates and the existing algebraic circuit.
Fixed temperature and receipt-only energy counters have no feedback in this
explicit isothermal/no-side-reaction domain and are exactly omitted.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse

from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC, F, battery_system
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
    """Assemble the same differentiable CC residual at a stationary rest state.

    Current-sigmoid OCP is evaluated inside that residual, including its actual
    current derivative. Axen hysteresis has distinct left/right current
    derivatives at rest, so there is no ordinary two-sided EIS in this scope.
    """
    if getattr(spec, "plating", None) is not None:
        raise ValueError("lithium plating requires a side-reaction impedance closure; unsupported")
    if getattr(spec, "model_family", None) == "lead-acid-loqs-isothermal-v1":
        raise ValueError("lead-acid LOQS impedance is unsupported")
    if getattr(spec, "model_family", None) == "spm-spatial-thermal-v1":
        raise ValueError("spatial thermal impedance is unsupported")
    if spec.thermal is not None or spec.sei is not None:
        raise ValueError(
            "impedance currently requires an isothermal battery without side reactions"
        )
    for electrode in ("negative", "positive"):
        ocp = getattr(spec, electrode + "_ocp", None)
        if ocp is not None and ocp.mode == "axen":
            raise ValueError(
                "Axen hysteresis has no unique two-sided rest-current derivative; "
                "ordinary rest impedance is unsupported"
            )
    composite = getattr(spec, "model_family", None) == "composite-spm-isothermal-v1"
    half_cell = getattr(spec, "model_family", None) == "half-cell-spm-isothermal-v1"
    msmr = getattr(spec, "model_family", None) == "msmr-spm-isothermal-v1"
    newman_tobias = getattr(spec, "model_family", None) == "newman-tobias-isothermal-v1"
    # These specifications already declare residual form as a read-only
    # property. Other models retain the original mathematical-form conversion.
    coupled = battery_system(
        spec if composite or half_cell or msmr or newman_tobias
        else replace(spec, numerical_form="residual"),
        ScalarControlProgram(((0.0, 0.0),), "A"),
    )
    material = coupled.thermodynamics.inventory_keys
    coordinates = tuple(item.name for item in coupled.coordinates)
    if set(state.amounts_mol) != set(material) or set(state.coordinates) != set(coordinates):
        raise ValueError(
            "impedance state must match the declared battery inventory and coordinates"
        )
    if getattr(spec, "model_family", None) == "split-ocvr-isothermal-v1":
        # Directional time-domain events allow recovery from a fraction face.
        # Ordinary two-sided EIS instead requires an interior operating point.
        for electrode in ("negative", "positive"):
            capacity_mol = getattr(spec, electrode + "_capacity_C") / F
            if not (
                getattr(spec, electrode + "_minimum_fraction") * capacity_mol
                < state.amounts_mol[electrode + ":Li"]
                < getattr(spec, electrode + "_maximum_fraction") * capacity_mol
            ):
                raise ResidualDomainError(
                    "impedance point is on/outside a battery stopping boundary: "
                    + electrode + " fraction"
                )
    if any(value < 0 for value in state.amounts_mol.values()):
        raise ResidualDomainError("negative battery inventory")
    for item in coupled.coordinates:
        item.validate(state.coordinates[item.name])
    for process in coupled.processes:
        if process.validate_state is not None:
            process.validate_state(state)
    if newman_tobias:
        # The mean-inventory time-domain events permit rest or inward recovery
        # on a stopping face. A two-sided sinusoid instead needs both actual
        # finite stocks strictly inside their declared mol intervals.
        for electrode in ("negative", "positive"):
            capacity = getattr(spec, electrode).capacity_mol
            if not (
                capacity * spec.surface_margin
                < state.amounts_mol[electrode + ":Li"]
                < capacity * (1 - spec.surface_margin)
            ):
                raise ResidualDomainError(
                    "impedance point is on/outside a battery stopping boundary: "
                    + electrode + ":Li"
                )
    if msmr:
        # Every original radial Li stock must be inside the stopping domain.
        # Compare in mol before potential inversion; no constitutive trial
        # extension or x(U) round trip can admit an exact inventory boundary.
        for electrode in ("negative", "positive"):
            particle = getattr(spec, electrode)
            for key, volume in zip(
                particle.keys(electrode), particle.mesh.cell_volumes_m3, strict=True
            ):
                capacity = volume * particle.maximum_concentration_mol_m3
                if not (
                    capacity * spec.surface_margin
                    < state.amounts_mol[key]
                    < capacity * (1 - spec.surface_margin)
                ):
                    raise ResidualDomainError(
                        "impedance point is on/outside a battery stopping boundary: " + key
                    )
    if half_cell:
        # A depleted metal reservoir remains valid for rest/replating in time,
        # but cannot support the two directions of an ordinary sinusoidal input.
        if state.amounts_mol["metal:Li"] <= 0:
            raise ResidualDomainError(
                "half-cell impedance requires strictly positive metal inventory"
            )
        # At CC rest the original working-surface flux correction is exactly
        # zero. Compare the same outer-shell amount in mol, avoiding a fraction
        # round trip that can move an exact stopping face into the interior.
        particle = spec.positive
        capacity = (
            particle.mesh.cell_volumes_m3[-1] * particle.maximum_concentration_mol_m3
        )
        if not (
            capacity * spec.surface_margin
            < state.amounts_mol[particle.keys("positive")[-1]]
            < capacity * (1 - spec.surface_margin)
        ):
            raise ResidualDomainError(
                "impedance point is on/outside a battery stopping boundary: working surface"
            )
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
    if composite:
        from chemworld.physchem.battery_composite import composite_fields

        # Solve only the original electrical closure at the supplied inventory.
        # Zero terminal current may still drive internal material transfer; the
        # complete stationary residual below must reject that state. Discard any
        # algebraic values from a prior control, including caller-supplied views.
        consistent = composite_fields(
            spec, replace(state, algebraic_values={}), current_A=0.0
        )
        guessed = consistent["algebraic_values"]
        for (_, _, particle, keys, identity), current in zip(
            spec.particle_bins, consistent["reaction_current_A"], strict=True
        ):
            # The same surface reconstruction in mol avoids a divide/multiply
            # round trip admitting an exact stopping face as an interior point.
            volume = particle.mesh.cell_volumes_m3[-1]
            surface_amount = state.amounts_mol[keys[-1]] - (
                (current / F) * particle.mesh.spherical_surface_distance_m * volume
                / (particle.area_m2 * particle.diffusivity_m2_s)
            )
            capacity = volume * particle.maximum_concentration_mol_m3
            if not (
                capacity * spec.surface_margin
                < surface_amount
                < capacity * (1 - spec.surface_margin)
            ):
                raise ResidualDomainError(
                    "impedance point is on/outside a battery stopping boundary: " + identity
                )
    else:
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
    control_name = (
        "msmr:control" if msmr else "half-cell:control" if half_cell else "battery:control"
    )
    control = next(i for i, e in enumerate(equations) if e.name == control_name)
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
