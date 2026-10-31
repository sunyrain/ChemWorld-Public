"""One FV / shared-potential electrical kernel for particle electrodes.

MPM and composite material owners supply inventories and OCP derivatives.
No second physical circuit, time integrator or prescribed current split exists.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from math import isfinite
from typing import TYPE_CHECKING, Any, NotRequired, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from chemworld.physchem.battery_models import (
    BATTERY_ALGEBRAIC,
    BatteryThermodynamics,
    F,
    ParticleSpec,
    R,
    particle_surface_fraction,
)
from chemworld.physchem.battery_ocp import (
    hysteresis_coordinates,
    hysteresis_key,
    hysteresis_rates,
    validate_hysteresis,
    validate_ocp_cv,
)
from chemworld.physchem.coupled_system import (
    CoupledAlgebraic,
    CoupledCoordinate,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    ProcessContribution,
)
from chemworld.physchem.finite_volume import ConservativeDiffusion
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import (
    ResidualDerivatives,
    ResidualEquation,
    ResidualNumericalError,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    solve_steady_residual,
)

if TYPE_CHECKING:
    from chemworld.physchem.battery_composite import BatteryCompositeSPMSpec
    from chemworld.physchem.battery_mpm import BatteryMPMSpec
    from chemworld.physchem.battery_plating import _PlatingSideReaction, _PlatingSPMAdapter

    ParticleOwner: TypeAlias = BatteryCompositeSPMSpec | BatteryMPMSpec | _PlatingSPMAdapter

Population = tuple[str, int, ParticleSpec, tuple[str, ...], str]
CurrentCoordinates = tuple[tuple[int, tuple[int, ...], tuple[float, ...]], ...]


class ParticleFields(TypedDict):
    """Derived fields at the shared particle-electrode electrical solution."""

    algebraic_values: dict[str, float]
    current_A: float
    voltage_V: float
    reaction_current_A: NDArray[np.float64]
    surface_fraction: NDArray[np.float64]
    average_fraction: NDArray[np.float64]
    kinetic_overpotential_V: NDArray[np.float64]
    area_m2: NDArray[np.float64]
    solid_potential_V: NDArray[np.float64]
    electrolyte_potential_V: float
    bv_residual_V: NDArray[np.float64]
    closure_diagnostic: NotRequired[dict[str, Any]]
    stripping_current_A: NotRequired[float]
    plated_mol: NotRequired[float]
    dead_mol: NotRequired[float]
    negative_electrode_potential_V: NotRequired[float]
    deactivation_rate_mol_s: NotRequired[float]


def current_coordinates(spec: ParticleOwner) -> CurrentCoordinates:
    """One largest-area balance pivot per electrode, never a physical split."""
    result = []
    for name in ("negative", "positive"):
        areas = np.array([p.area_m2 for n, _, p, _, _ in spec.particle_bins if n == name])
        pivot = int(np.argmax(areas))
        free = tuple(i for i in range(len(areas)) if i != pivot)
        scales = tuple(float(spec.maximum_current_A * areas[i] / areas.sum()) for i in free)
        result.append((pivot, free, scales))
    return tuple(result)


def reaction_map(spec: ParticleOwner) -> NDArray[np.float64]:
    result = np.zeros((sum(spec.counts), len(spec.algebraic_names)))
    row, column = 0, 0
    for count, sign, (pivot, free, _) in zip(
        spec.counts, (1, -1), spec.current_coordinates, strict=True
    ):
        for j, i in enumerate(free):
            result[row + i, column + j] = 1
            result[row + pivot, column + j] = -1
        result[row + pivot, -2] = sign
        row += count
        column += count - 1
    side: _PlatingSideReaction | None = getattr(spec, "particle_side_reaction", None)
    if side is not None:
        # A side reaction shares the negative electrode's potential and actual
        # inventory. Its current is independent of the external source bound.
        result[spec.current_coordinates[0][0], spec.algebraic_names.index(side.current_key)] = -1
    result.setflags(write=False)
    return result


def _fields(
    spec: ParticleOwner, state: CoupledStateView, values: Mapping[str, float],
) -> ParticleFields:
    current, voltage = (values[key] for key in BATTERY_ALGEBRAIC)
    q = spec.reaction_map @ np.array([values[key] for key in spec.algebraic_names])
    surface, average, eta, area, ocv = [], [], [], [], []
    for value, (name, i, particle, keys, identity) in zip(q, spec.particle_bins, strict=True):
        theta = particle_surface_fraction(particle, identity, state, value / F)
        # Analytic finite trial continuation; accepted states use true surfaces
        # and every size has its own stopping boundary. No inventory is clipped.
        trial = np.clip(theta, spec.surface_margin / 2, 1 - spec.surface_margin / 2)
        j0 = particle.exchange_current_scale_A_m2 * np.sqrt(trial * (1 - trial))
        eta.append(
            2 * R * state.temperature_K / F * np.arcsinh(value / (2 * particle.area_m2 * j0))
        )
        surface.append(theta)
        average.append(sum(state.amounts_mol[k] for k in keys) / particle.capacity_mol)
        area.append(particle.area_m2)
        ocv.append(spec.population_ocp_derivatives((name, i, particle, keys, identity),
                                                   theta, current, state)[0])
    solid = np.r_[
        np.zeros(spec.counts[0]),
        np.full(spec.counts[1], voltage + current * spec.series_resistance_ohm),
    ]
    result: ParticleFields = {
        "algebraic_values": dict(values),
        "current_A": current,
        "voltage_V": voltage,
        "reaction_current_A": q,
        "surface_fraction": np.array(surface),
        "average_fraction": np.array(average),
        "kinetic_overpotential_V": np.array(eta),
        "area_m2": np.array(area),
        "solid_potential_V": solid,
        "electrolyte_potential_V": values[spec.electrolyte_potential_key],
        "bv_residual_V": (solid - values[spec.electrolyte_potential_key]
                          - np.array(ocv) - np.array(eta)),
    }
    side: _PlatingSideReaction | None = getattr(spec, "particle_side_reaction", None)
    if side is not None:
        result.update(side.fields(state, values, spec.electrolyte_potential_key))
    return result



def particle_algebraic(spec: ParticleOwner, program: ScalarControlProgram) -> CoupledAlgebraic:
    if program.unit == "V":
        validate_ocp_cv(spec)
    if program.unit not in {"A", "V"} or any(
        abs(v) > spec.maximum_current_A
        if program.unit == "A"
        else not spec.minimum_voltage_V <= v <= spec.maximum_voltage_V
        for _, v in program.setpoints
    ):
        raise ValueError("MPM control programme is outside the declared device domain")
    side: _PlatingSideReaction | None = getattr(spec, "particle_side_reaction", None)
    variables = (
        *(
            ResidualVariable(k, "A", scale, "algebraic")
            for k, scale in zip(
                spec.reaction_keys,
                (scale for _, _, scales in spec.current_coordinates for scale in scales),
                strict=True,
            )
        ),
        *((side.variable,) if side is not None else ()),
        ResidualVariable(spec.electrolyte_potential_key, "V", role="algebraic"),
        ResidualVariable(BATTERY_ALGEBRAIC[0], "A", spec.maximum_current_A, "algebraic"),
        ResidualVariable(BATTERY_ALGEBRAIC[1], "V", role="algebraic"),
    )
    equations = (
        *(
            ResidualEquation(key, "V") for key in spec.population_equations
        ),
        *((side.equation,) if side is not None else ()),
        ResidualEquation("battery:control", "1"),
    )

    def initial_values(state: CoupledStateView) -> dict[str, float]:
        current = program.value_at(state.time_s) if program.unit == "A" else 0.0
        currents: list[float] = []
        means: dict[str, float] = {}
        for name, sign, (_, free, _) in zip(
            ("negative", "positive"), (1, -1), spec.current_coordinates, strict=True
        ):
            bins = [b for b in spec.particle_bins if b[0] == name]
            areas = np.array([b[2].area_m2 for b in bins])
            guesses = sign * current * areas / areas.sum()
            currents.extend(guesses[i] for i in free)
            means[name] = spec.electrode_initial_potential(name, state, current)
        return dict(
            zip(
                spec.algebraic_names,
                [*currents, *((0.0,) if side is not None else ()),
                 -means["negative"], current, means["positive"] - means["negative"]],
                strict=True,
            )
        )

    def residual(state: CoupledStateView) -> NDArray[np.float64]:
        f = _fields(spec, state, state.algebraic_values)
        current, voltage, target = f["current_A"], f["voltage_V"], program.value_at(state.time_s)
        requested = current + spec.maximum_current_A * (voltage - target)
        control = (
            (current - target) / spec.maximum_current_A
            if program.unit == "A"
            else (current - np.clip(requested, -spec.maximum_current_A, spec.maximum_current_A))
            / spec.maximum_current_A
        )
        return np.r_[f["bv_residual_V"],
                     [] if side is None else [side.residual(state, f)], control]

    return CoupledAlgebraic(variables, equations, initial_values, residual)



def _closure_jacobian(
    spec: ParticleOwner, state: CoupledStateView, program: ScalarControlProgram,
    identities: Sequence[str],
) -> sparse.csc_matrix:
    index = {name: i for i, name in enumerate(identities)}
    count = sum(spec.counts)
    side: _PlatingSideReaction | None = getattr(spec, "particle_side_reaction", None)
    jac = np.zeros((count + 1 + int(side is not None), len(identities)))
    columns = [index[key] for key in spec.algebraic_names]
    phi_column = index[spec.electrolyte_potential_key]
    jac[:count, phi_column] = -1
    jac[spec.counts[0] : count, index[BATTERY_ALGEBRAIC[1]]] = 1
    jac[spec.counts[0] : count, index[BATTERY_ALGEBRAIC[0]]] = spec.series_resistance_ohm
    f = _fields(spec, state, state.algebraic_values)
    coefficient = 2 * R * state.temperature_K / F
    for row, population in enumerate(spec.particle_bins):
        _, _, particle, keys, _ = population
        theta = f["surface_fraction"][row]
        trial = np.clip(theta, spec.surface_margin / 2, 1 - spec.surface_margin / 2)
        j0 = particle.exchange_current_scale_A_m2 * np.sqrt(trial * (1 - trial))
        z = f["reaction_current_A"][row] / (2 * particle.area_m2 * j0)
        factor = coefficient / np.sqrt(1 + z * z)
        eta_q = factor / (2 * particle.area_m2 * j0)
        eta_theta = (
            -factor * z * (1 - 2 * trial) / (2 * trial * (1 - trial)) if theta == trial else 0.0
        )
        _, slope, di, coordinate_derivatives = spec.population_ocp_derivatives(
            population, theta, f["current_A"], state)
        theta_q = -particle.mesh.spherical_surface_distance_m / (
            F * particle.area_m2 * particle.diffusivity_m2_s * particle.maximum_concentration_mol_m3
        )
        jac[row, columns] -= ((slope + eta_theta) * theta_q + eta_q) * spec.reaction_map[row]
        jac[row, index[BATTERY_ALGEBRAIC[0]]] -= di
        for key, derivative in coordinate_derivatives.items():
            if key in index:
                jac[row, index[key]] -= derivative
        if keys[-1] in index:
            jac[row, index[keys[-1]]] = -(slope + eta_theta) / (
                particle.mesh.cell_volumes_m3[-1] * particle.maximum_concentration_mol_m3
            )
        if "temperature" in index:
            jac[row, index["temperature"]] = (
                -f["kinetic_overpotential_V"][row] / state.temperature_K
            )
    if side is not None:
        jac[count] = side.closure_jacobian(state, f, identities, spec.electrolyte_potential_key)
    current, voltage = f["current_A"], f["voltage_V"]
    requested = current + spec.maximum_current_A * (voltage - program.value_at(state.time_s))
    if program.unit == "A" or abs(requested) >= spec.maximum_current_A:
        jac[-1, index[BATTERY_ALGEBRAIC[0]]] = 1 / spec.maximum_current_A
    else:
        jac[-1, index[BATTERY_ALGEBRAIC[1]]] = -1
    return sparse.csc_matrix(jac)



def particle_fields(
    spec: ParticleOwner, state: CoupledStateView, *, current_A: float | None = None,
    program: ScalarControlProgram | None = None,
) -> ParticleFields:
    if isinstance(state, CoupledState):
        validate_hysteresis(spec, state)
    if (current_A is None) == (program is None):
        raise ValueError("supply either a current or a control programme")
    if current_A is not None:
        if not isfinite(current_A) or abs(current_A) > spec.maximum_current_A:
            raise ValueError("current is outside battery capability")
        program = ScalarControlProgram(((0.0, current_A),), "A")
    # The exclusive-input check above guarantees a programme on either path.
    program = cast(ScalarControlProgram, program)
    closure = particle_algebraic(spec, program)
    names = spec.algebraic_names
    if all(k in state.algebraic_values for k in names) and (
        current_A is None or state.algebraic_values[BATTERY_ALGEBRAIC[0]] == current_A
    ):
        return _fields(spec, state, state.algebraic_values)

    def view(y: NDArray[np.float64]) -> CoupledStateView:
        return CoupledStateView(
            state.amounts_mol,
            state.temperature_K,
            state.volume_L,
            state.time_s,
            coordinates=state.coordinates,
            algebraic_values=dict(zip(names, y, strict=True)),
        )

    system = ResidualSystem(
        spec.closure_id,
        closure.variables,
        closure.equations,
        lambda t, y, rate, p: closure.residual(view(y)),
        ResidualDerivatives(
            "hybrid", state=lambda t, y, rate, p: _closure_jacobian(spec, view(y), program, names)
        ),
    )
    guess = closure.initial_values(state)
    answer = solve_steady_residual(
        system, [guess[k] for k in names], policy=RootPolicy(tolerance=1e-12)
    )
    if not answer.root.success:
        raise ResidualNumericalError("MPM electrical closure:" + answer.root.failure_reason)
    f = _fields(spec, state, dict(zip(names, answer.root.value, strict=True)))
    if abs(f["current_A"]) > spec.maximum_current_A * (1 + 1e-10):
        raise ResidualNumericalError("MPM circuit solution exceeds its actual source bound")
    return {**f, "closure_diagnostic": answer.root.diagnostics}



def particle_system(
    spec: ParticleOwner, program: ScalarControlProgram, *, thermal_device: object | None = None,
    control_events: Sequence[CoupledEvent] = (),
) -> CoupledSystem:
    if thermal_device is not None:
        raise ValueError("MPM currently requires explicit isothermal physics, no thermal device")
    closure = particle_algebraic(spec, program)
    side: _PlatingSideReaction | None = getattr(spec, "particle_side_reaction", None)
    processes = [
        ConservativeDiffusion(p.mesh, p.diffusivity_m2_s)
        .couple(keys, process_id=f"battery:{identity}", owner_id="battery")
        .process
        for _, _, p, keys, identity in spec.particle_bins
    ]

    def fields(state: CoupledStateView) -> ParticleFields:
        return particle_fields(spec, state, program=program)

    def source(state: CoupledStateView) -> ProcessContribution:
        f = fields(state)
        amounts = {
            keys[-1]: -q / F
            for q, (_, _, _, keys, _) in zip(
                f["reaction_current_A"], spec.particle_bins, strict=True
            )
        }
        if side is not None:
            amounts.update(side.amount_rates(f))
        reaction_currents = None
        if side is not None and spec.hysteresis_keys:
            reaction_currents = {
                name: sum(q for q, population in zip(
                    f["reaction_current_A"], spec.particle_bins, strict=True)
                    if population[0] == name)
                for name in ("negative", "positive")
            }
        return ProcessContribution(
            amount_rates_mol_s=amounts,
            coordinate_rates={
                "discharged_charge_C": f["current_A"],
                "delivered_work_J": f["current_A"] * f["voltage_V"],
                **hysteresis_rates(spec, state, f["current_A"],
                                   reaction_currents_A=reaction_currents),
            },
        )

    def validate(state: CoupledStateView) -> None:
        spec.validate_particle_owner(state)
        validate_hysteresis(spec, state)
        if (
            abs(state.temperature_K - spec.temperature_K) > 1e-8
            or abs(state.volume_L - spec.volume_l) > 1e-12
        ):
            raise ValueError("MPM requires its declared isothermal volume")
        for _, _, p, keys, _ in spec.particle_bins:
            if any(
                state.amounts_mol[k] > v * p.maximum_concentration_mol_m3 * (1 + 1e-8)
                for k, v in zip(keys, p.mesh.cell_volumes_m3, strict=True)
            ):
                raise ValueError("MPM particle inventory exceeds its finite capacity")

    processes.append(CoupledProcess(spec.intercalation_id, source, validate_state=validate))
    events = spec.particle_events(fields)
    scales = {k: p.capacity_mol / p.cells for _, _, p, keys, _ in spec.particle_bins for k in keys}
    nkeys = [k for n, _, _, keys, _ in spec.particle_bins if n == "negative" for k in keys]
    pkeys = [k for n, _, _, keys, _ in spec.particle_bins if n == "positive" for k in keys]
    if side is not None:
        scales.update(side.inventory_scales)
        nkeys.extend(side.inventory_scales)

    @lru_cache(maxsize=8)
    def linear_jacobian(identities: tuple[str, ...]) -> sparse.csc_matrix:
        index = {k: i for i, k in enumerate(identities)}
        result = sparse.lil_matrix((len(identities), len(identities)))
        acols = [index[k] for k in spec.algebraic_names]
        for row, (_, _, p, keys, _) in enumerate(spec.particle_bins):
            cols = [index[k] for k in keys]
            result[np.ix_(cols, cols)] = ConservativeDiffusion(
                p.mesh, p.diffusivity_m2_s
            ).inventory_jacobian_s_inv
            result[index[keys[-1]], acols] = -spec.reaction_map[row] / F
        result[index["discharged_charge_C"], index[BATTERY_ALGEBRAIC[0]]] = 1.0
        if side is not None:
            for key, derivative in side.amount_current_derivatives.items():
                result[index[key], index[side.current_key]] = derivative
            for (row, column), derivative in getattr(side, "amount_state_derivatives", {}).items():
                result[index[row], index[column]] += derivative
        return result.tocsc()

    def jacobian(state: CoupledStateView, identities: tuple[str, ...]) -> sparse.csc_matrix:
        index = {k: i for i, k in enumerate(identities)}
        rows = np.array([index[k] for k in spec.algebraic_names])
        closure_j = -_closure_jacobian(spec, state, program, identities).tocoo()
        work = index["delivered_work_J"]
        data = np.r_[
            closure_j.data,
            state.algebraic_values[BATTERY_ALGEBRAIC[1]],
            state.algebraic_values[BATTERY_ALGEBRAIC[0]],
        ]
        rr = np.r_[rows[closure_j.row], work, work]
        cc = np.r_[closure_j.col, index[BATTERY_ALGEBRAIC[0]], index[BATTERY_ALGEBRAIC[1]]]
        result = linear_jacobian(tuple(identities)) + sparse.csc_matrix(
            (data, (rr, cc)), shape=(len(identities), len(identities))
        )
        extra = sparse.lil_matrix(result.shape)
        for name in ("negative", "positive"):
            key = hysteresis_key(name)
            if key in index:
                if side is not None:
                    reaction_row = np.sum([
                        spec.reaction_map[i] for i, population in enumerate(spec.particle_bins)
                        if population[0] == name
                    ], axis=0)
                    reaction_current = reaction_row @ np.array([
                        state.algebraic_values[k] for k in spec.algebraic_names])
                    _, dq, dh = getattr(spec, name + "_ocp").reaction_rate_derivatives(
                        reaction_current, state.coordinates[key], getattr(spec, name).capacity_mol)
                    extra[index[key], [index[k] for k in spec.algebraic_names]] = dq * reaction_row
                    extra[index[key], index[key]] = dh
                    continue
                _, di, dh = getattr(spec, name + "_ocp").rate_derivatives(
                    name, state.algebraic_values[BATTERY_ALGEBRAIC[0]],
                    state.coordinates[key], getattr(spec, name).capacity_mol)
                extra[index[key], index[BATTERY_ALGEBRAIC[0]]] = di
                extra[index[key], index[key]] = dh
        return result + extra.tocsc()

    return CoupledSystem(
        BatteryThermodynamics(spec),
        tuple(processes),
        isothermal=True,
        events=(*events, *control_events),
        inventory_scales_mol=scales,
        coordinates=(
            CoupledCoordinate("discharged_charge_C", "C", spec.usable_charge_c),
            CoupledCoordinate("delivered_work_J", "J", spec.usable_charge_c * 4),
            *hysteresis_coordinates(spec),
        ),
        conservation_laws={
            "lithium_mol": dict.fromkeys(spec.inventory_keys, 1),
            "negative_faradaic_C": {**dict.fromkeys(nkeys, F), "discharged_charge_C": 1},
            "positive_faradaic_C": {**dict.fromkeys(pkeys, F), "discharged_charge_C": -1},
        },
        numerical_form="residual",
        algebraic=closure,
        residual_rhs_jacobian=jacobian,
    )



def legacy_particle_events(
    spec: ParticleOwner, fields: Callable[[CoupledStateView], ParticleFields],
) -> list[CoupledEvent]:
    events: list[CoupledEvent] = []

    def boundary(selection: slice, lower: bool) -> Callable[[CoupledStateView], float]:
        def value(state: CoupledStateView) -> float:
            surface = fields(state)["surface_fraction"][selection]
            if lower:
                return float(np.min(surface)) - spec.surface_margin
            return 1 - spec.surface_margin - float(np.max(surface))
        return value

    first = 0
    for name, count in zip(("negative", "positive"), spec.counts, strict=True):
        selection = slice(first, first + count)
        events.extend(
            (
                CoupledEvent(
                    f"battery:{name}:depleted",
                    boundary(selection, True),
                ),
                CoupledEvent(
                    f"battery:{name}:saturated",
                    boundary(selection, False),
                ),
            )
        )
        first += count
    events.extend(
        (
            CoupledEvent(
                "battery:minimum-voltage", lambda s: fields(s)["voltage_V"] - spec.minimum_voltage_V
            ),
            CoupledEvent(
                "battery:maximum-voltage", lambda s: spec.maximum_voltage_V - fields(s)["voltage_V"]
            ),
        )
    )
    return events
