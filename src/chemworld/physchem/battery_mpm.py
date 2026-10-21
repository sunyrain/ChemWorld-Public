"""Many-particle electrodes: independent size inventories, shared BV potentials.

The ideal uniform electrolyte is the declared MPM reduction. No private time
integrator, prescribed current fractions or averaged independent cells are used.
"""

from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from math import isfinite

import numpy as np
from scipy import sparse

from chemworld.physchem.battery_models import (
    BATTERY_ALGEBRAIC,
    BATTERY_COORDINATES,
    BatteryReadout,
    BatterySPMSpec,
    BatteryThermodynamics,
    F,
    R,
    particle_surface_fraction,
)
from chemworld.physchem.battery_particles import ParticleSizeDistribution
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

MPM_ID = "mpm-isothermal-v1"
PHI_E = "battery:mpm:electrolyte-potential_V"


@dataclass(frozen=True)
class BatteryMPMSpec(BatterySPMSpec):
    negative_distribution: ParticleSizeDistribution | None = None
    positive_distribution: ParticleSizeDistribution | None = None
    model_family: str = MPM_ID
    provenance: str = "synthetic-MPM-v1"

    def __post_init__(self):
        super().__post_init__()
        if (
            self.model_family != MPM_ID
            or self.numerical_form != "residual"
            or self.electrolyte is not None
            or self.thermal is not None
            or self.sei is not None
        ):
            raise ValueError(
                "MPM requires its explicit isothermal uniform-electrolyte residual model"
            )
        for name in ("negative_distribution", "positive_distribution"):
            value = getattr(self, name)
            if isinstance(value, dict):
                value = ParticleSizeDistribution.from_dict(value)
                object.__setattr__(self, name, value)
            if value is not None and not isinstance(value, ParticleSizeDistribution):
                raise ValueError("MPM requires a declared particle size distribution")

    @property
    def model_id(self):
        return MPM_ID

    @cached_property
    def particle_bins(self):
        bins = []
        for name in ("negative", "positive"):
            original = getattr(self, name)
            distribution = getattr(self, name + "_distribution")
            radii = (original.radius_m,) if distribution is None else distribution.radii_m
            weights = (1.0,) if distribution is None else distribution.volume_weights
            for i, (radius, weight) in enumerate(zip(radii, weights, strict=True)):
                particle = replace(
                    original, radius_m=radius, active_volume_m3=original.active_volume_m3 * weight
                )
                identity = f"{name}:size:{i}"
                bins.append((name, i, particle, particle.keys(identity), identity))
        return tuple(bins)

    @cached_property
    def inventory_species(self):
        return {key: "Li" for _, _, _, keys, _ in self.particle_bins for key in keys}

    @cached_property
    def counts(self):
        return tuple(
            sum(name == electrode for name, *_ in self.particle_bins)
            for electrode in ("negative", "positive")
        )

    @cached_property
    def current_coordinates(self):
        """Free bin indices/scales and one well-conditioned balance pivot per electrode.

        Eliminating a tiny tail current subtracts O(I) quantities to recover a
        very small current. Instead eliminate the largest-area bin and scale
        each independent current by its own area share. This is a coordinate
        choice only: no bin or BV equation is removed from the physical model.
        """
        result = []
        for name in ("negative", "positive"):
            areas = np.array([p.area_m2 for n, _, p, _, _ in self.particle_bins if n == name])
            pivot = int(np.argmax(areas))
            free = tuple(i for i in range(len(areas)) if i != pivot)
            scales = tuple(float(self.maximum_current_A * areas[i] / areas.sum()) for i in free)
            result.append((pivot, free, scales))
        return tuple(result)

    @cached_property
    def reaction_keys(self):
        return tuple(
            f"battery:{name}:size-current:{i}_A"
            for name, (_, free, _) in zip(
                ("negative", "positive"), self.current_coordinates, strict=True
            )
            for i in free
        )

    @cached_property
    def algebraic_names(self):
        return (*self.reaction_keys, PHI_E, *BATTERY_ALGEBRAIC)

    @cached_property
    def reaction_map(self):
        """Eliminate each electrode's largest-area current by Faradaic balance."""
        result = np.zeros((sum(self.counts), len(self.algebraic_names)))
        row, column = 0, 0
        for count, sign, (pivot, free, _) in zip(
            self.counts, (1, -1), self.current_coordinates, strict=True
        ):
            for j, i in enumerate(free):
                result[row + i, column + j] = 1
                result[row + pivot, column + j] = -1
            result[row + pivot, -2] = sign
            row += count
            column += count - 1
        result.setflags(write=False)
        return result

    def initial_state(self):
        amounts = {}
        for name, _, particle, keys, _ in self.particle_bins:
            theta = getattr(self, "initial_" + name + "_fraction")
            amounts.update(
                zip(
                    keys,
                    particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3 * theta,
                    strict=True,
                )
            )
        return CoupledState(
            amounts,
            self.temperature_K,
            self.volume_l,
            coordinates=dict.fromkeys(BATTERY_COORDINATES, 0.0),
        )

    def to_dict(self):
        result = super().to_dict()
        for name in ("negative_distribution", "positive_distribution"):
            distribution = getattr(self, name)
            result[name] = None if distribution is None else distribution.to_dict()
        return result


def _fields(spec, state, values):
    current, voltage = (values[key] for key in BATTERY_ALGEBRAIC)
    q = spec.reaction_map @ np.array([values[key] for key in spec.algebraic_names])
    surface, average, eta, area, ocv = [], [], [], [], []
    for value, (name, _, particle, keys, identity) in zip(q, spec.particle_bins, strict=True):
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
        ocv.append(spec.ocv(name, theta))
    solid = np.r_[
        np.zeros(spec.counts[0]),
        np.full(spec.counts[1], voltage + current * spec.series_resistance_ohm),
    ]
    return {
        "algebraic_values": dict(values),
        "current_A": current,
        "voltage_V": voltage,
        "reaction_current_A": q,
        "surface_fraction": np.array(surface),
        "average_fraction": np.array(average),
        "kinetic_overpotential_V": np.array(eta),
        "area_m2": np.array(area),
        "solid_potential_V": solid,
        "electrolyte_potential_V": values[PHI_E],
        "bv_residual_V": solid - values[PHI_E] - np.array(ocv) - np.array(eta),
    }


def mpm_algebraic(spec, program):
    if program.unit not in {"A", "V"} or any(
        abs(v) > spec.maximum_current_A
        if program.unit == "A"
        else not spec.minimum_voltage_V <= v <= spec.maximum_voltage_V
        for _, v in program.setpoints
    ):
        raise ValueError("MPM control programme is outside the declared device domain")
    variables = (
        *(
            ResidualVariable(k, "A", scale, "algebraic")
            for k, scale in zip(
                spec.reaction_keys,
                (scale for _, _, scales in spec.current_coordinates for scale in scales),
                strict=True,
            )
        ),
        ResidualVariable(PHI_E, "V", role="algebraic"),
        ResidualVariable(BATTERY_ALGEBRAIC[0], "A", spec.maximum_current_A, "algebraic"),
        ResidualVariable(BATTERY_ALGEBRAIC[1], "V", role="algebraic"),
    )
    equations = (
        *(
            ResidualEquation(f"battery:bv:{name}:size:{i}", "V")
            for name, i, *_ in spec.particle_bins
        ),
        ResidualEquation("battery:control", "1"),
    )

    def initial_values(state):
        current = program.value_at(state.time_s) if program.unit == "A" else 0.0
        currents, means = [], {}
        for name, sign, (_, free, _) in zip(
            ("negative", "positive"), (1, -1), spec.current_coordinates, strict=True
        ):
            bins = [b for b in spec.particle_bins if b[0] == name]
            areas = np.array([b[2].area_m2 for b in bins])
            guesses = sign * current * areas / areas.sum()
            currents.extend(guesses[i] for i in free)
            total = sum(state.amounts_mol[k] for _, _, _, keys, _ in bins for k in keys)
            means[name] = spec.ocv(name, total / getattr(spec, name).capacity_mol)
        return dict(
            zip(
                spec.algebraic_names,
                [*currents, -means["negative"], current, means["positive"] - means["negative"]],
                strict=True,
            )
        )

    def residual(state):
        f = _fields(spec, state, state.algebraic_values)
        current, voltage, target = f["current_A"], f["voltage_V"], program.value_at(state.time_s)
        requested = current + spec.maximum_current_A * (voltage - target)
        control = (
            (current - target) / spec.maximum_current_A
            if program.unit == "A"
            else (current - np.clip(requested, -spec.maximum_current_A, spec.maximum_current_A))
            / spec.maximum_current_A
        )
        return np.r_[f["bv_residual_V"], control]

    return CoupledAlgebraic(variables, equations, initial_values, residual)


def _closure_jacobian(spec, state, program, identities):
    index = {name: i for i, name in enumerate(identities)}
    count = sum(spec.counts)
    jac = np.zeros((count + 1, len(identities)))
    columns = [index[key] for key in spec.algebraic_names]
    phi_column = index[PHI_E]
    jac[:count, phi_column] = -1
    jac[spec.counts[0] : count, index[BATTERY_ALGEBRAIC[1]]] = 1
    jac[spec.counts[0] : count, index[BATTERY_ALGEBRAIC[0]]] = spec.series_resistance_ohm
    f = _fields(spec, state, state.algebraic_values)
    coefficient = 2 * R * state.temperature_K / F
    for row, (name, _, particle, keys, _) in enumerate(spec.particle_bins):
        theta = f["surface_fraction"][row]
        trial = np.clip(theta, spec.surface_margin / 2, 1 - spec.surface_margin / 2)
        j0 = particle.exchange_current_scale_A_m2 * np.sqrt(trial * (1 - trial))
        z = f["reaction_current_A"][row] / (2 * particle.area_m2 * j0)
        factor = coefficient / np.sqrt(1 + z * z)
        eta_q = factor / (2 * particle.area_m2 * j0)
        eta_theta = (
            -factor * z * (1 - 2 * trial) / (2 * trial * (1 - trial)) if theta == trial else 0.0
        )
        slope = spec.ocv(name, theta, derivative=True)
        theta_q = -particle.mesh.spherical_surface_distance_m / (
            F * particle.area_m2 * particle.diffusivity_m2_s * particle.maximum_concentration_mol_m3
        )
        jac[row, columns] -= ((slope + eta_theta) * theta_q + eta_q) * spec.reaction_map[row]
        if keys[-1] in index:
            jac[row, index[keys[-1]]] = -(slope + eta_theta) / (
                particle.mesh.cell_volumes_m3[-1] * particle.maximum_concentration_mol_m3
            )
        if "temperature" in index:
            jac[row, index["temperature"]] = (
                -f["kinetic_overpotential_V"][row] / state.temperature_K
            )
    current, voltage = f["current_A"], f["voltage_V"]
    requested = current + spec.maximum_current_A * (voltage - program.value_at(state.time_s))
    if program.unit == "A" or abs(requested) >= spec.maximum_current_A:
        jac[-1, index[BATTERY_ALGEBRAIC[0]]] = 1 / spec.maximum_current_A
    else:
        jac[-1, index[BATTERY_ALGEBRAIC[1]]] = -1
    return sparse.csc_matrix(jac)


def mpm_fields(spec, state, *, current_A=None, program=None):
    if (current_A is None) == (program is None):
        raise ValueError("supply either a current or a control programme")
    if current_A is not None:
        if not isfinite(current_A) or abs(current_A) > spec.maximum_current_A:
            raise ValueError("current is outside battery capability")
        program = ScalarControlProgram(((0.0, current_A),), "A")
    closure = mpm_algebraic(spec, program)
    names = spec.algebraic_names
    if all(k in state.algebraic_values for k in names) and (
        current_A is None or state.algebraic_values[BATTERY_ALGEBRAIC[0]] == current_A
    ):
        return _fields(spec, state, state.algebraic_values)

    def view(y):
        return CoupledStateView(
            state.amounts_mol,
            state.temperature_K,
            state.volume_L,
            state.time_s,
            coordinates=state.coordinates,
            algebraic_values=dict(zip(names, y, strict=True)),
        )

    system = ResidualSystem(
        "battery:mpm-closure",
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


def mpm_readout(spec, state, current_A):
    f = mpm_fields(spec, state, current_A=current_A)
    surface, average, eta = [], [], []
    first = 0
    for name, count in zip(("negative", "positive"), spec.counts, strict=True):
        selection = slice(first, first + count)
        areas = f["area_m2"][selection]
        capacities = np.array([p.capacity_mol for n, _, p, _, _ in spec.particle_bins if n == name])
        surface.append(float(np.average(f["surface_fraction"][selection], weights=areas)))
        average.append(float(np.average(f["average_fraction"][selection], weights=capacities)))
        eta.append(float(np.average(f["kinetic_overpotential_V"][selection], weights=areas)))
        first += count
    return BatteryReadout(
        f["voltage_V"], f["current_A"], *surface, *average, *eta, f["current_A"], 0.0, 0.0
    )


def mpm_dissipation_w(spec, state, current_A):
    from chemworld.physchem.battery_models import battery_chemical_potentials

    f = mpm_fields(spec, state, current_A=current_A)
    potentials = battery_chemical_potentials(spec, state)
    interior, boundary = 0.0, 0.0
    for q, (name, _, particle, keys, identity) in zip(
        f["reaction_current_A"], spec.particle_bins, strict=True
    ):
        rates = ConservativeDiffusion(particle.mesh, particle.diffusivity_m2_s).evaluate(
            [state.amounts_mol[key] for key in keys]
        )
        interior += sum(
            float(rate) * (potentials[left] - potentials[right])
            for rate, left, right in zip(
                rates.face_rates_mol_s[1:-1], keys[:-1], keys[1:], strict=True
            )
        )
        surface = particle_surface_fraction(particle, identity, state, q / F)
        boundary += q / F * (potentials[keys[-1]] + F * spec.ocv(name, surface))
    return {
        "kinetic_W": float(np.dot(f["reaction_current_A"], f["kinetic_overpotential_V"])),
        "ohmic_W": current_A**2 * spec.series_resistance_ohm,
        "interior_diffusion_W": interior,
        "surface_diffusion_W": boundary,
        "sei_reaction_W": 0.0,
    }


def mpm_system(spec, program, *, thermal_device=None, control_events=()):
    if thermal_device is not None:
        raise ValueError("MPM currently requires explicit isothermal physics, no thermal device")
    closure = mpm_algebraic(spec, program)
    processes = [
        ConservativeDiffusion(p.mesh, p.diffusivity_m2_s)
        .couple(keys, process_id=f"battery:{identity}", owner_id="battery")
        .process
        for _, _, p, keys, identity in spec.particle_bins
    ]

    def fields(state):
        return mpm_fields(spec, state, program=program)

    def source(state):
        f = fields(state)
        amounts = {
            keys[-1]: -q / F
            for q, (_, _, _, keys, _) in zip(
                f["reaction_current_A"], spec.particle_bins, strict=True
            )
        }
        return ProcessContribution(
            amount_rates_mol_s=amounts,
            coordinate_rates={
                "discharged_charge_C": f["current_A"],
                "delivered_work_J": f["current_A"] * f["voltage_V"],
            },
        )

    def validate(state):
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

    processes.append(CoupledProcess("battery:mpm-intercalation", source, validate_state=validate))
    events = []
    first = 0
    for name, count in zip(("negative", "positive"), spec.counts, strict=True):
        selection = slice(first, first + count)
        events.extend(
            (
                CoupledEvent(
                    f"battery:{name}:depleted",
                    lambda s, sel=selection: (
                        float(np.min(fields(s)["surface_fraction"][sel])) - spec.surface_margin
                    ),
                ),
                CoupledEvent(
                    f"battery:{name}:saturated",
                    lambda s, sel=selection: (
                        1 - spec.surface_margin - float(np.max(fields(s)["surface_fraction"][sel]))
                    ),
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
    scales = {k: p.capacity_mol / p.cells for _, _, p, keys, _ in spec.particle_bins for k in keys}
    nkeys = [k for n, _, _, keys, _ in spec.particle_bins if n == "negative" for k in keys]
    pkeys = [k for n, _, _, keys, _ in spec.particle_bins if n == "positive" for k in keys]

    @lru_cache(maxsize=8)
    def linear_jacobian(identities):
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
        return result.tocsc()

    def jacobian(state, identities):
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
        return linear_jacobian(tuple(identities)) + sparse.csc_matrix(
            (data, (rr, cc)), shape=(len(identities), len(identities))
        )

    return CoupledSystem(
        BatteryThermodynamics(spec),
        tuple(processes),
        isothermal=True,
        events=(*events, *control_events),
        inventory_scales_mol=scales,
        coordinates=(
            CoupledCoordinate("discharged_charge_C", "C", spec.usable_charge_c),
            CoupledCoordinate("delivered_work_J", "J", spec.usable_charge_c * 4),
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
