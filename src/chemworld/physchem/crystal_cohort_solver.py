"""Geometry coordinates for the shared crystal inventory/temperature equations.

Diameters and composition ratios have explicit numerical units. They are not
invented molar stocks: every candidate reconstructs its actual cohort molecules
before using the common CoupledSystem rate assembler and ODE backend.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import isfinite
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix

from chemworld.physchem import solver_backend
from chemworld.physchem.coupled_system import (
    CoupledAdvanceResult,
    CoupledEvent,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    assemble_coupled_rates,
)
from chemworld.physchem.solver_backend import ODESolverPolicy

if TYPE_CHECKING:
    from chemworld.physchem.crystal_thermal_kinetics import (
        CrystalThermalSpec,
        CrystalThermodynamics,
        KineticTerms,
        LiquidKeys,
        SolidKeys,
        ThermalCrystalCohort,
    )
    from chemworld.physchem.legacy_crystal_enthalpy import LegacyCrystalEnthalpy


class _TerminalEvent(Protocol):
    terminal: bool
    direction: int

    def __call__(self, time: float, scaled: NDArray[np.float64]) -> float: ...


def solve_crystal_interval(
    system: CoupledSystem,
    initial: CoupledState,
    duration_s: float,
    model: CrystalThermalSpec,
    liquid_keys: LiquidKeys,
    solid_keys: SolidKeys,
    cohorts: Sequence[ThermalCrystalCohort | None],
    *,
    kinetic_terms: KineticTerms,
    evaluation_times_s: Sequence[float] = (),
    inventory_independent_heat_flux: bool | None = False,
) -> CoupledAdvanceResult:
    # The crystal layout supplies species-resolved calorics (or the retained
    # constant-enthalpy law); generic CoupledSystem has a broader thermal API.
    thermal = cast("CrystalThermodynamics", system.thermodynamics)
    keys = system.thermodynamics.inventory_keys
    position = {key: i for i, key in enumerate(keys)}
    count = len(keys)
    vector = np.array(
        [
            *(initial.amounts_mol[k] for k in keys),
            initial.temperature_K,
            initial.boundary_heat_J,
            initial.chemical_energy_change_J,
            initial.heat_loss_J,
            initial.work_J,
        ]
    )
    scales = np.ones(count + 5)
    for key, index in position.items():
        candidates = [
            sum(w * initial.amounts_mol[k] for k, w in law.items())
            for law in system.conservation_laws.values()
            if key in law
        ]
        # Dissolved fractions can approach zero while their actual component
        # remains in the solid. Resolve errors against that conserved stock,
        # rather than requiring relative accuracy of an exhausted subpool.
        stock = min((v for v in candidates if v > 0), default=1e-8)
        scales[index] = stock
    scales[count] = 350
    seed_stock = sum(
        initial.amounts_mol[key] for key in system.conservation_laws["seed-provenance"]
    )
    impurity_stock = (
        0.0
        if model.impurity_component is None
        else sum(
            initial.amounts_mol[key] for key in system.conservation_laws[model.impurity_component]
        )
    )
    geometry = []
    for index, (fields, cohort) in enumerate(zip(solid_keys, cohorts, strict=True)):
        if cohort is None:
            continue
        seed, grown = position[fields["seed"]], position[fields["grown"]]
        impurity = position[fields["impurity"]] if "impurity" in fields else None
        n = vector[seed] + vector[grown]
        if n <= 0:
            raise ValueError("an existing cohort must start with actual positive host mass")
        vector[grown] = vector[seed] / n
        vector[seed] = (n / cohort.particle_count / model.particle_moles(1)) ** (1 / 3)
        if impurity is not None:
            vector[impurity] /= n
        # A fixed absolute length resolution prevents microscopic, vanishing
        # radii from demanding arbitrarily precise relative geometry. With the
        # declared atol this is 1e-15 m, far tighter than the 1e-9 m benchmark.
        scales[seed] = max(vector[seed], 1e-2)
        scales[grown] = max(1e-300, min(1e300, seed_stock / n)) if seed_stock > 0 else 1.0
        if impurity is not None:
            scales[impurity] = (
                max(1e-300, min(1e300, impurity_stock / n)) if impurity_stock > 0 else 1.0
            )
        geometry.append((index, seed, grown, impurity, cohort.particle_count))
    geometry_positions = {p for _, a, b, c, _ in geometry for p in (a, b, c) if p is not None}
    molecular_positions = tuple((key, index) for key, index in position.items()
                                if index not in geometry_positions)
    from chemworld.physchem.crystal_thermal_kinetics import CrystalThermalSpec

    if type(model) is CrystalThermalSpec:
        # Preserve particle_moles' multiplication/division order exactly.
        from math import pi
        particle_factor = model.kinetics.crystal_density_kg_m3 * pi / 6
        molecular_weight = model.kinetics.target_molecular_weight_kg_mol

        def particle_moles(diameter_m: float) -> float:
            return particle_factor * diameter_m**3 / molecular_weight
    else:
        particle_moles = model.particle_moles
    inactive: set[int] = set()
    corrections: dict[str, float] = {}
    deaths = []
    diagnostics = []
    points: list[CoupledState] = []
    elapsed = 0.0
    stopping = None

    def amounts(raw: NDArray[np.float64], *, accepted: bool = False) -> dict[str, float]:
        # Keep the canonical key order while avoiding float conversions that
        # would immediately be overwritten by the geometry reconstruction.
        # Molecular and geometry positions partition all keys; every slot is
        # filled below before the resulting inventory is read or returned.
        result = cast(dict[str, float], dict.fromkeys(keys))
        for key, index in molecular_positions:
            result[key] = float(raw[index])
        for index, diameter, seed_fraction, impurity_ratio, particles in geometry:
            d = float(raw[diameter])
            theta = float(raw[seed_fraction])
            loading = 0 if impurity_ratio is None else float(raw[impurity_ratio])
            n = 0 if index in inactive else particles * particle_moles(max(d, 0))
            if accepted and (
                d < -1e-12
                or max(-theta, theta - 1, 0) * n > 1e-24 + 1e-10 * seed_stock
                or max(-loading, 0) * n > 1e-24 + 1e-10 * impurity_stock
            ):
                raise RuntimeError(
                    f"crystal geometry/composition left its molar physical domain: cohort={index}, "
                    f"d={d!r}, seed_fraction={theta!r}, loading={loading!r}, host_mol={n!r}, "
                    f"seed_stock={seed_stock!r}, impurity_stock={impurity_stock!r}"
                )
            if accepted and not 0 <= theta <= 1:
                corrections[solid_keys[index]["seed"]] = max(
                    corrections.get(solid_keys[index]["seed"], 0),
                    abs(theta - min(max(theta, 0), 1)) * n,
                )
            if accepted and loading < 0 and impurity_ratio is not None:
                corrections[solid_keys[index]["impurity"]] = max(
                    corrections.get(solid_keys[index]["impurity"], 0), -loading * n
                )
            theta, loading = min(max(theta, 0), 1), max(loading, 0)
            fields = solid_keys[index]
            result[fields["seed"]] = theta * n
            result[fields["grown"]] = (1 - theta) * n
            if impurity_ratio is not None:
                result[fields["impurity"]] = loading * n
        for key, index in molecular_positions:
            value = result[key]
            tolerance = 1e-10 * scales[index] + 1e-24
            if not isfinite(value) or (accepted and value < -tolerance):
                raise RuntimeError(
                    f"crystal molecular stock outside nonnegative domain: {key}={value!r}; "
                    f"tolerance={tolerance!r}"
                )
            if value < 0:
                if accepted:
                    corrections[key] = max(corrections.get(key, 0), -value)
                trial_impurity = (
                    not accepted
                    and model.impurity_component is not None
                    and key == liquid_keys[model.impurity_component][0]
                )
                if not trial_impurity:
                    result[key] = 0.0
        return result

    def view(
        time: float, raw: NDArray[np.float64], *, accepted: bool = False,
    ) -> CoupledStateView:
        return CoupledStateView(
            amounts(raw, accepted=accepted),
            float(raw[count]),
            initial.volume_L,
            initial.time_s + time,
        )

    def snapshot(time: float, raw: NDArray[np.float64]) -> CoupledState:
        state = view(time, raw, accepted=True)
        return CoupledState(
            state.amounts_mol,
            state.temperature_K,
            state.volume_L,
            state.time_s,
            boundary_heat_J=float(raw[count + 1]),
            chemical_energy_change_J=float(raw[count + 2]),
            heat_loss_J=float(raw[count + 3]),
            work_J=float(raw[count + 4]),
        )

    def rhs(time: float, scaled: NDArray[np.float64]) -> NDArray[np.float64]:
        raw = scaled * scales
        state = view(time, raw)
        rates = assemble_coupled_rates(
            state,
            processes=system.processes,
            thermodynamics=system.thermodynamics,
            heat_flux=system.heat_flux,
            isothermal=system.isothermal,
        )
        derivative = np.array(
            [
                *(rates.amount_rates_mol_s[k] for k in keys),
                rates.temperature_rate_K_s,
                rates.heater_W,
                rates.chemical_energy_W,
                rates.loss_W,
                rates.work_W,
            ]
        )
        velocity, _, capture = kinetic_terms(model, state, liquid_keys)
        total = sum(state.amounts_mol[k] for k in liquid_keys[model.target_component])
        fluid_seed = (
            state.amounts_mol[liquid_keys[model.target_component][0]] / total if total else 0
        )
        for index, diameter, seed_fraction, impurity_ratio, _ in geometry:
            if index in inactive:
                derivative[diameter] = derivative[seed_fraction] = 0
                if impurity_ratio is not None:
                    derivative[impurity_ratio] = 0
                continue
            derivative[diameter] = velocity
            # Pure recession preserves each actual molecular fraction exactly;
            # its mass coordinates have a singular derivative at disappearance.
            growth = 3 * max(velocity, 0) / float(raw[diameter]) if raw[diameter] > 0 else 0.0
            derivative[seed_fraction] = growth * (fluid_seed - float(raw[seed_fraction]))
            if impurity_ratio is not None:
                derivative[impurity_ratio] = growth * (capture - float(raw[impurity_ratio]))
        return derivative / scales

    def jacobian(time: float, scaled: NDArray[np.float64]) -> csc_matrix:
        """Local cohort columns are analytic; shared C/T columns use differences.

        Liquid balance rows are dense but are sums of independent cohort
        transfers. Differencing every complete population separately for each
        cohort column needlessly costs O(cohorts**2) physical evaluations.
        This derivative changes no physical rate or convergence tolerance.
        """
        raw = scaled * scales
        state = view(time, raw)
        base = rhs(time, scaled)
        # Each cohort column touches only its local coordinates and the shared
        # liquid/energy rows. Store those entries directly: assigning a dense
        # vector to a LIL column scans the entire population for every cohort.
        rows, columns, values = [], [], []

        def append_column(column: int, entries: Iterable[tuple[int, float]]) -> None:
            for row, value in entries:
                if value != 0.0:
                    rows.append(row)
                    columns.append(column)
                    values.append(value)

        for global_column in (*[i for i in range(count) if i not in geometry_positions], count):
            step = np.sqrt(np.finfo(float).eps) * max(abs(scaled[global_column]), 1.0)
            if (
                model.impurity_component is not None
                and global_column == position[liquid_keys[model.impurity_component][0]]
                and impurity_stock > 0
            ):
                step = min(
                    step,
                    0.1 * model.impurity_resolution_fraction * impurity_stock
                    / scales[global_column],
                )
            shifted = scaled.copy()
            shifted[global_column] += step
            difference = (rhs(time, shifted) - base) / step
            append_column(
                global_column, ((row, difference[row]) for row in np.flatnonzero(difference)),
            )

        velocity, _, capture = kinetic_terms(model, state, liquid_keys)
        fluid_target = liquid_keys[model.target_component]
        fluid_seed, fluid_grown = (position[k] for k in fluid_target)
        free_target = sum(state.amounts_mol[k] for k in fluid_target)
        theta_fluid = state.amounts_mol[fluid_target[0]] / free_target if free_target else 0.0
        capacity = system.thermodynamics.heat_capacity_j_k(state)
        cv_cache: dict[int, float] = {}

        def cv(key: str) -> float:
            prop = thermal.species[key]
            identity = id(prop)
            if identity not in cv_cache:
                if isinstance(prop, (int, float)):
                    value = 0.0  # explicit legacy constant molecular href
                elif hasattr(prop, "cv_J_mol_K"):
                    value = prop.cv_J_mol_K
                else:
                    value = prop.evaluate(state.temperature_K, ("cv",))["cv"].value
                cv_cache[identity] = value
            return cv_cache[identity]

        for index, diameter, seed_fraction, impurity_ratio, particles in geometry:
            if index in inactive or raw[diameter] <= 0:
                continue
            d = raw[diameter]
            theta = min(max(raw[seed_fraction], 0.0), 1.0)
            loading = max(raw[impurity_ratio], 0.0) if impurity_ratio is not None else 0.0
            fields = solid_keys[index]
            alpha = particles * model.particle_moles(1.0)
            host = alpha * d**3
            rate = 3 * alpha * d**2 * velocity
            radial_rate_derivative = 6 * alpha * d * velocity
            growth = 3 * max(velocity, 0.0) / d
            source_seed = theta_fluid if velocity >= 0 else theta
            source_impurity = capture if velocity >= 0 else loading
            target_delta = thermal.species_energy_j_mol(
                fields["seed"], state.temperature_K
            ) - thermal.species_energy_j_mol(fluid_target[0], state.temperature_K)
            impurity_delta = 0.0
            if impurity_ratio is not None:
                impurity_delta = thermal.species_energy_j_mol(
                    fields["impurity"], state.temperature_K
                ) - thermal.species_energy_j_mol(
                    liquid_keys[cast(str, model.impurity_component)][0], state.temperature_K
                )
            for column in (diameter, seed_fraction, impurity_ratio):
                if column is None:
                    continue
                derivative = {}
                host_rate, seed_rate, impurity_rate, capacity_derivative = 0.0, 0.0, 0.0, 0.0
                if column == diameter:
                    host_rate = radial_rate_derivative
                    seed_rate = host_rate * source_seed
                    impurity_rate = host_rate * source_impurity
                    derivative[seed_fraction] = -growth / d * (theta_fluid - raw[seed_fraction])
                    if impurity_ratio is not None:
                        derivative[impurity_ratio] = -growth / d * (capture - raw[impurity_ratio])
                    capacity_derivative = (
                        3
                        * alpha
                        * d**2
                        * (
                            cv(fields["seed"])
                            + (
                                loading * cv(fields["impurity"])
                                if impurity_ratio is not None
                                else 0.0
                            )
                        )
                    )
                elif column == seed_fraction:
                    derivative[seed_fraction] = -growth
                    if velocity < 0 and 0 < raw[seed_fraction] < 1:
                        seed_rate = rate
                else:
                    # The remaining coordinate is the non-None impurity ratio.
                    derivative[column] = -growth
                    if velocity < 0 and raw[column] > 0:
                        impurity_rate = rate
                    capacity_derivative = host * cv(fields["impurity"])
                derivative[fluid_seed] = -seed_rate
                derivative[fluid_grown] = -(host_rate - seed_rate)
                if impurity_ratio is not None:
                    derivative[
                        position[liquid_keys[cast(str, model.impurity_component)][0]]
                    ] = -impurity_rate
                chemical = target_delta * host_rate + impurity_delta * impurity_rate
                derivative[count] = (
                    0.0
                    if system.isothermal
                    else -chemical / capacity
                    - base[count] * scales[count] * capacity_derivative / capacity
                )
                derivative[count + 1] = chemical if system.isothermal else 0.0
                derivative[count + 2] = chemical
                append_column(
                    column,
                    (
                        (row, value * scales[column] / scales[row])
                        for row, value in derivative.items()
                    ),
                )
        return csc_matrix((values, (rows, columns)), shape=(len(scaled), len(scaled)))

    while elapsed < duration_s:
        # Several numerically coincident disappearance roots can be reported
        # as one terminal event. Retire every already nonpositive host before
        # restarting; otherwise a second host can start past its zero root.
        for index, diameter, _, _, _ in geometry:
            if index not in inactive and vector[diameter] <= 0:
                vector[diameter] = 0.0
                inactive.add(index)
                deaths.append(
                    {
                        "cohort": index,
                        "time_s": initial.time_s + elapsed,
                        "numerical_coordinate": "diameter_m",
                    }
                )
        callbacks, names = [], []
        event_candidate: NDArray[np.float64] | None = None
        event_time: float | None = None
        event_view: CoupledStateView | None = None

        def event_state(time: float, scaled: NDArray[np.float64]) -> CoupledStateView:
            nonlocal event_candidate, event_time, event_view
            # All event callbacks at one candidate inspect the same immutable
            # state. Reset at every topology restart; never cache callbacks.
            if event_time != time or not np.array_equal(
                cast(NDArray[np.float64], event_candidate), scaled,
            ):
                event_candidate, event_time = scaled.copy(), time
                event_view = view(time, scaled * scales)
            return cast(CoupledStateView, event_view)

        for spec in system.all_events:

            def root(
                time: float, scaled: NDArray[np.float64], spec: CoupledEvent = spec,
            ) -> float:
                return spec.value(event_state(time, scaled))

            event_callback = cast(_TerminalEvent, root)
            event_callback.terminal, event_callback.direction = True, spec.direction
            callbacks.append(event_callback)
            names.append(spec.event_id)
        death_indices = {}
        for index, diameter, _, _, _ in geometry:
            if index in inactive:
                continue

            def death(
                time: float, scaled: NDArray[np.float64], diameter: int = diameter,
            ) -> float:
                return scaled[diameter] * scales[diameter]

            death_callback = cast(_TerminalEvent, death)
            death_callback.terminal, death_callback.direction = True, -1
            death_indices[len(callbacks)] = (index, diameter)
            callbacks.append(death_callback)
            names.append(f"cohort_disappearance:{index}")
        for callback, name in zip(callbacks, names, strict=True):
            if callback(elapsed, vector / scales) <= 0 and not name.startswith(
                "cohort_disappearance:"
            ):
                stopping = name
                break
        if stopping is not None:
            break
        grid = tuple(
            sorted(
                {
                    elapsed,
                    duration_s,
                    *(t for t in evaluation_times_s if elapsed <= t <= duration_s),
                }
            )
        )
        report = solver_backend.solve_ode(
            rhs,
            vector / scales,
            time_span_s=(elapsed, duration_s),
            evaluation_times_s=grid,
            events=callbacks or None,
            jac_sparsity=(None if inventory_independent_heat_flux else system.jacobian_sparsity()),
            jac=jacobian if inventory_independent_heat_flux else None,
            policy=ODESolverPolicy(
                "chemworld_crystal_geometry_radau_v1",
                "Radau",
                1e-11,
                1e-13,
                event_names=tuple(names),
            ),
        )
        report.raise_for_failure("shared crystal geometry advance")
        diagnostics.append(report.diagnostic.to_dict())
        for t, values in zip(report.t, report.y.T, strict=True):
            point = snapshot(float(t), values * scales)
            if not points or point.time_s > points[-1].time_s:
                points.append(point)
        if report.terminal_point is None:
            elapsed = duration_s
            vector = report.y[:, -1] * scales
            break
        time, values, event_index = report.terminal_point
        vector = values * scales
        if event_index in death_indices:
            index, diameter = death_indices[event_index]
            if abs(vector[diameter]) > 1e-12:
                raise RuntimeError("crystal disappearance root has unresolved radius")
            vector[diameter] = 0.0
            for other, other_diameter, _, _, _ in geometry:
                if other != index and other not in inactive and vector[other_diameter] <= 0:
                    vector[other_diameter] = 0.0
                    inactive.add(other)
                    deaths.append(
                        {
                            "cohort": other,
                            "time_s": initial.time_s + time,
                            "numerical_coordinate": "diameter_m",
                        }
                    )
            point = snapshot(time, vector)
            deaths.append(
                {"cohort": index, "time_s": point.time_s, "numerical_coordinate": "diameter_m"}
            )
            inactive.add(index)
        else:
            stopping = names[event_index]
            point = snapshot(time, vector)
        if points and abs(points[-1].time_s - point.time_s) < 1e-12:
            points[-1] = point
        else:
            points.append(point)
        if stopping is not None:
            elapsed = time
            break
        if time < elapsed:
            raise RuntimeError("crystal topology event reversed physical time")
        elapsed = time
    final = snapshot(elapsed, vector)
    if not points:
        points = [initial, final]
    elif points[-1].time_s != final.time_s:
        points.append(final)
    else:
        points[-1] = final
    residuals = {
        name: sum(w * (final.amounts_mol[k] - initial.amounts_mol[k]) for k, w in law.items())
        for name, law in system.conservation_laws.items()
    }
    for name, law in system.conservation_laws.items():
        stock = sum(abs(w * initial.amounts_mol[k]) for k, w in law.items())
        if abs(residuals[name]) > 1e-8 * max(stock, 1e-12):
            raise RuntimeError(f"crystal geometry component/provenance balance failed: {name}")

    def energy(state: CoupledStateView) -> float:
        value = thermal.internal_energy_j(state)
        return (cast("LegacyCrystalEnthalpy", thermal).control_volume_enthalpy_j(state)
                if value is None else value)

    energy_residual = (
        energy(final)
        - energy(initial)
        - final.boundary_heat_J
        + initial.boundary_heat_J
        + final.heat_loss_J
        - initial.heat_loss_J
        - final.work_J
        + initial.work_J
    )
    if abs(energy_residual) > 1e-6 * max(1, abs(energy(initial)), abs(final.boundary_heat_J)):
        raise RuntimeError("crystal geometry energy balance failed")
    return CoupledAdvanceResult(
        initial,
        final,
        tuple(points),
        "stopped" if stopping else "success",
        stopping,
        {
            "solver_calls": len(diagnostics),
            "jacobian_strategy": (
                "analytic_cohort_numeric_global"
                if inventory_independent_heat_flux
                else "finite_difference"
            ),
            "segments": diagnostics,
            **{k: sum(int(cast(int, d.get(k, 0) or 0)) for d in diagnostics)
               for k in ("nfev", "njev", "nlu")},
            "cohort_disappearances": deaths,
            "coordinate_units": {
                "liquid_and_birth_stock": "mol",
                "cohort_diameter": "m",
                "cohort_seed_fraction": "mol/mol",
                "cohort_impurity_loading": "mol/mol",
                "temperature": "K",
                "cumulative_energy": "J",
            },
        },
        energy_residual,
        residuals,
        corrections,
    )
