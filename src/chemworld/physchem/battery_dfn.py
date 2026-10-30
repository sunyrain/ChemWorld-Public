"""Isothermal DFN: spatial particles, electrolyte and one coupled electrical root.

Independent implementation of the constant-property PyBaMM 25.4.2 DFN equations.
The finite-volume charge equations are integrated exactly for cellwise-constant
reaction rates. All local reactions and the external circuit remain simultaneous
algebraic unknowns; neither a prescribed reaction profile nor a private solver is
used. This module owns no persistent physical or numerical state.
"""

from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache
from math import isfinite

import numpy as np
from scipy import sparse

from chemworld.physchem import battery_porous_electrical as porous_electrical
from chemworld.physchem.battery_electrolyte import ELECTROLYTE_SPECIES, BatteryElectrolyteSpec
from chemworld.physchem.battery_models import (
    BATTERY_ALGEBRAIC,
    BATTERY_COORDINATES,
    BatteryReadout,
    BatterySPMSpec,
    BatteryThermodynamics,
    F,
    ParticleSpec,
    R,
    particle_surface_fraction,
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

PHI_E = "battery:electrolyte-offset_V"
DFN_ID = "dfn-isothermal-v1"


@dataclass(frozen=True)
class BatteryDFNSpec(BatterySPMSpec):
    """Planar porous electrodes; mesh defaults match the synthetic 17-unit block.

    Other parameters and coarser meshes require their own convergence checks.
    Explicit stored meshes are never silently upgraded during deserialization.
    """

    negative: ParticleSpec = field(default_factory=lambda: ParticleSpec(cells=64))
    positive: ParticleSpec = field(default_factory=lambda: ParticleSpec(cells=64))
    electrolyte: BatteryElectrolyteSpec = field(
        default_factory=lambda: BatteryElectrolyteSpec(cells=(24, 24, 24)))
    model_family: str = DFN_ID
    provenance: str = "synthetic-DFN-v1"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.model_family != DFN_ID or self.electrolyte is None:
            raise ValueError("DFN requires its explicit model identity and electrolyte")
        if self.numerical_form != "residual":
            raise ValueError("DFN requires the common coupled residual formulation")

    @property
    def model_id(self):
        return DFN_ID

    @cached_property
    def particle_bins(self):
        bins = []
        for name, region in (("negative", 0), ("positive", 2)):
            original = getattr(self, name)
            count = self.electrolyte.cells[region]
            particle = replace(original, active_volume_m3=original.active_volume_m3 / count)
            for i in range(count):
                identity = f"{name}:x:{i}"
                bins.append((name, i, particle, particle.keys(identity), identity))
        return tuple(bins)

    @cached_property
    def inventory_species(self):
        return {
            **{key: "Li" for _, _, _, keys, _ in self.particle_bins for key in keys},
            **dict.fromkeys(self.electrolyte.keys, ELECTROLYTE_SPECIES),
        }

    @cached_property
    def reaction_keys(self):
        # Last cell in each electrode is eliminated with sum(q_n)=I, sum(q_p)=-I.
        # This enforces exact charge conservation even at numerical trial points.
        return tuple(f"battery:{name}:reaction:{i}_A"
                     for name, region in (("negative", 0), ("positive", 2))
                     for i in range(self.electrolyte.cells[region] - 1))

    @cached_property
    def algebraic_names(self):
        return (*self.reaction_keys, PHI_E, *BATTERY_ALGEBRAIC)

    @cached_property
    def electrical_maps(self):
        # Unit columns of an exactly linear constitutive map, not differencing
        # nonlinear physics. They depend on declared geometry only, not state.
        return porous_electrical.electrical_maps(
            self.electrolyte, self.reaction_keys, self.series_resistance_ohm,
            self.algebraic_names)

    def initial_state(self):
        amounts = self.electrolyte.initial_amounts()
        for name, _, particle, keys, _ in self.particle_bins:
            values = (particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3
                      * getattr(self, f"initial_{name}_fraction"))
            amounts.update(zip(keys, values, strict=True))
        return CoupledState(amounts, self.temperature_K, self.volume_l,
                            coordinates=dict.fromkeys(BATTERY_COORDINATES, 0.0))


def _reaction_currents(spec, values):
    return porous_electrical.reaction_currents(spec.electrolyte, spec.reaction_keys, values)


def _integral_centres(face_values, widths, coefficients):
    """Integral of a linear face field, from the left boundary to each centre."""
    return porous_electrical.integral_centres(face_values, widths, coefficients)


def _linear_electrical(spec, values):
    return porous_electrical.linear_electrical(
        spec.electrolyte, spec.reaction_keys, spec.series_resistance_ohm, values)


def _fields(spec, state, values):
    e = spec.electrolyte
    current, voltage = (values[key] for key in BATTERY_ALGEBRAIC)
    nn, _, np_ = e.cells
    q, ionic_faces, phi_e, phi_s, solid_faces = _linear_electrical(spec, values)
    ohmic = float(np.mean(phi_e[-np_:]) - np.mean(phi_e[:nn]))
    c = e.concentrations(state)
    # Finite trial extension only; accepted states/events use the signed true c.
    trial_c = np.maximum(c, e.minimum_concentration_mol_m3 / 2)
    coefficient = 2 * R * state.temperature_K * (1 - e.cation_transference) / F
    phi_e += coefficient * e.thermodynamic_factor * np.log(
        trial_c / e.initial_concentration_mol_m3)
    surfaces, average, eta, areas = [], [], [], []
    electrode_c = np.r_[trial_c[:nn], trial_c[-np_:]]
    electrode_phi_e = np.r_[phi_e[:nn], phi_e[-np_:]]
    ocv = []
    for k, (name, _, particle, keys, identity) in enumerate(spec.particle_bins):
        theta = particle_surface_fraction(particle, identity, state, q[k] / F)
        trial = np.clip(theta, spec.surface_margin / 2, 1 - spec.surface_margin / 2)
        j0 = (particle.exchange_current_scale_A_m2
              * np.sqrt(electrode_c[k] / e.initial_concentration_mol_m3 * trial * (1 - trial)))
        eta.append(2 * R * state.temperature_K / F * np.arcsinh(q[k] / particle.area_m2 / (2 * j0)))
        surfaces.append(theta)
        average.append(sum(state.amounts_mol[key] for key in keys) / particle.capacity_mol)
        areas.append(particle.area_m2)
        ocv.append(spec.ocv(name, theta))
    return {
        "current_A": current, "voltage_V": voltage, "reaction_current_A": q,
        "interfacial_current_A_m2": q / np.array(areas),
        "electrolyte_concentration_mol_m3": c, "electrolyte_potential_V": phi_e,
        "electrolyte_face_current_A_m2": ionic_faces,
        "negative_solid_face_current_A_m2": solid_faces[0],
        "positive_solid_face_current_A_m2": solid_faces[1],
        "solid_potential_V": phi_s, "surface_fraction": np.array(surfaces),
        "average_fraction": np.array(average), "kinetic_overpotential_V": np.array(eta),
        "electrolyte_ohmic_V": ohmic,
        "concentration_overpotential_V": float(np.mean(phi_e[-np_:])
                                                - np.mean(phi_e[:nn]) - ohmic),
        "solid_ohmic_V": float(voltage + current * spec.series_resistance_ohm
                                - np.mean(phi_s[nn:]) + np.mean(phi_s[:nn])),
        "bv_residual_V": phi_s - electrode_phi_e - np.array(ocv) - np.array(eta),
    }


def dfn_algebraic(spec, program):
    """All local inverse-BV equations plus one finite-source CC/CV constraint."""
    if program.unit not in {"A", "V"} or any(
        abs(v) > spec.maximum_current_A if program.unit == "A" else
        not spec.minimum_voltage_V <= v <= spec.maximum_voltage_V
        for _, v in program.setpoints
    ):
        raise ValueError("DFN control programme is outside the device domain")
    names = spec.algebraic_names
    variables = (
        *(ResidualVariable(key, "A", spec.maximum_current_A, "algebraic")
          for key in spec.reaction_keys),
        ResidualVariable(PHI_E, "V", 1, "algebraic"),
        # The circuit equation itself imposes the exact source bounds. Newton
        # trials may cross them; projecting a trial before resolving BV can
        # destroy the coupled descent direction at a saturated source.
        ResidualVariable(BATTERY_ALGEBRAIC[0], "A", spec.maximum_current_A, "algebraic"),
        ResidualVariable(BATTERY_ALGEBRAIC[1], "V", 1, "algebraic"),
    )
    equations = (*(ResidualEquation(f"battery:bv:{name}:{i}", "V")
                   for name, i, _, _, _ in spec.particle_bins),
        ResidualEquation("battery:control", "1"),)

    def initial_values(state):
        current = program.value_at(state.time_s) if program.unit == "A" else 0.0
        nn, _, np_ = spec.electrolyte.cells
        q = np.r_[np.full(nn - 1, current / nn), np.full(np_ - 1, -current / np_)]
        mean = {}
        for name in ("negative", "positive"):
            total = sum(state.amounts_mol[k] for n, _, _, keys, _ in spec.particle_bins
                        if n == name for k in keys)
            theta = total / getattr(spec, name).capacity_mol
            mean[name] = spec.ocv(name, theta)
        return dict(zip(names, [*q, -mean["negative"], current,
                                mean["positive"] - mean["negative"]], strict=True))

    def residual(state):
        fields = _fields(spec, state, state.algebraic_values)
        current, voltage = (fields[key] for key in ("current_A", "voltage_V"))
        target = program.value_at(state.time_s)
        if program.unit == "A":
            control = (current - target) / spec.maximum_current_A
        else:
            # Projection is the actual finite-source complementarity law, not
            # clipping a predicted voltage or repairing any physical inventory.
            requested = current + spec.maximum_current_A * (voltage - target)
            control = (current - np.clip(requested, -spec.maximum_current_A,
                                          spec.maximum_current_A)) / spec.maximum_current_A
        return np.r_[fields["bv_residual_V"], control]

    return CoupledAlgebraic(variables, equations, initial_values, residual)


def _closure_jacobian(spec, state, program, identities):
    """Analytic local inverse-BV/circuit derivatives in physical coordinates."""
    index = {key: i for i, key in enumerate(identities)}
    e = spec.electrolyte
    na = len(spec.algebraic_names)
    nn, ns, _ = e.cells
    qmap, electrical = spec.electrical_maps
    jac = np.zeros((na, len(identities)))
    algebraic_columns = [index[key] for key in spec.algebraic_names]
    jac[:-1, algebraic_columns] = electrical
    f = _fields(spec, state, state.algebraic_values)
    coefficient = 2 * R * state.temperature_K / F
    for row, (name, i, particle, keys, _) in enumerate(spec.particle_bins):
        cx = i if name == "negative" else nn + ns + i
        c = f["electrolyte_concentration_mol_m3"][cx]
        trial_c = max(c, e.minimum_concentration_mol_m3 / 2)
        theta = f["surface_fraction"][row]
        trial = np.clip(theta, spec.surface_margin / 2, 1 - spec.surface_margin / 2)
        j0 = particle.exchange_current_scale_A_m2 * np.sqrt(
            trial_c / e.initial_concentration_mol_m3 * trial * (1 - trial))
        q = f["reaction_current_A"][row]
        z = q / (2 * particle.area_m2 * j0)
        factor = coefficient / np.sqrt(1 + z * z)
        eta_q = factor / (2 * particle.area_m2 * j0)
        eta_theta = (-factor * z * (1 - 2 * trial) / (2 * trial * (1 - trial))
                     if trial == theta else 0.0)
        slope = spec.ocv(name, theta, derivative=True)
        surface_q = -particle.mesh.spherical_surface_distance_m / (
            F * particle.area_m2 * particle.diffusivity_m2_s
            * particle.maximum_concentration_mol_m3)
        jac[row, algebraic_columns] -= ((slope + eta_theta) * surface_q + eta_q) * qmap[row]
        if keys[-1] in index:
            jac[row, index[keys[-1]]] = -(slope + eta_theta) / (
                particle.mesh.cell_volumes_m3[-1] * particle.maximum_concentration_mol_m3)
        if e.keys[cx] in index and c == trial_c:
            jac[row, index[e.keys[cx]]] = (
                -coefficient * (1 - e.cation_transference) * e.thermodynamic_factor / c
                + factor * z / (2 * c)
            ) / e.diffusion.storage_volumes_m3[cx]
        if "temperature" in index:
            jac[row, index["temperature"]] = (
                -coefficient * (1 - e.cation_transference) * e.thermodynamic_factor
                * np.log(trial_c / e.initial_concentration_mol_m3)
                - f["kinetic_overpotential_V"][row]
            ) / state.temperature_K
    current, voltage = (f[key] for key in ("current_A", "voltage_V"))
    requested = current + spec.maximum_current_A * (voltage - program.value_at(state.time_s))
    if program.unit == "A" or abs(requested) >= spec.maximum_current_A:
        jac[-1, index[BATTERY_ALGEBRAIC[0]]] = 1 / spec.maximum_current_A
    else:
        jac[-1, index[BATTERY_ALGEBRAIC[1]]] = -1.0
    return sparse.csc_matrix(jac)


def dfn_fields(spec, state, *, current_A=None, program=None):
    """Derived private fields; restart a shared algebraic solve from inventory."""
    if (current_A is None) == (program is None):
        raise ValueError("supply either a current or a control programme")
    if current_A is not None:
        if not isfinite(current_A) or abs(current_A) > spec.maximum_current_A:
            raise ValueError("current is outside battery capability")
        program = ScalarControlProgram(((0, current_A),), "A")
    closure = dfn_algebraic(spec, program)
    names = tuple(v.name for v in closure.variables)
    if all(key in state.algebraic_values for key in names) and (
        current_A is None or state.algebraic_values[BATTERY_ALGEBRAIC[0]] == current_A
    ):
        return _fields(spec, state, state.algebraic_values)

    def residual(t, y, rate, p):
        candidate = CoupledStateView(
            state.amounts_mol, state.temperature_K, state.volume_L, state.time_s,
            coordinates=state.coordinates, algebraic_values=dict(zip(names, y, strict=True)))
        return closure.residual(candidate)

    def jacobian(t, y, rate, p):
        candidate = CoupledStateView(
            state.amounts_mol, state.temperature_K, state.volume_L, state.time_s,
            coordinates=state.coordinates, algebraic_values=dict(zip(names, y, strict=True)))
        return _closure_jacobian(spec, candidate, program, names)

    system = ResidualSystem("battery:dfn-closure", closure.variables, closure.equations,
                            residual, ResidualDerivatives("hybrid", state=jacobian))
    guess = closure.initial_values(state)
    answer = solve_steady_residual(system, [guess[name] for name in names],
                                   policy=RootPolicy(tolerance=1e-12))
    if not answer.root.success:
        raise ResidualNumericalError("DFN electrical closure:" + answer.root.failure_reason)
    fields = _fields(spec, state, dict(zip(names, answer.root.value, strict=True)))
    if abs(fields["current_A"]) > spec.maximum_current_A * (1 + 1e-10):
        raise ResidualNumericalError("DFN circuit solution exceeds its actual source bound")
    return {**fields, "closure_diagnostic": answer.root.diagnostics}


def dfn_readout(spec, state, current_A):
    fields = dfn_fields(spec, state, current_A=current_A)
    nn = spec.electrolyte.cells[0]
    return BatteryReadout(
        fields["voltage_V"], fields["current_A"],
        float(np.mean(fields["surface_fraction"][:nn])),
        float(np.mean(fields["surface_fraction"][nn:])),
        float(np.mean(fields["average_fraction"][:nn])),
        float(np.mean(fields["average_fraction"][nn:])),
        float(np.mean(fields["kinetic_overpotential_V"][:nn])),
        float(np.mean(fields["kinetic_overpotential_V"][nn:])),
        fields["current_A"], 0.0, 0.0,
        fields["concentration_overpotential_V"], fields["electrolyte_ohmic_V"],
        fields["solid_ohmic_V"],
    )


def dfn_system(spec, program, *, thermal_device=None, control_events=()):
    if thermal_device is not None:
        raise ValueError("DFN-v1 is isothermal; thermal coupling is a separate model")
    closure = dfn_algebraic(spec, program)
    e = spec.electrolyte
    processes = [ConservativeDiffusion(p.mesh, p.diffusivity_m2_s).couple(
        keys, process_id=f"battery:{identity}", owner_id="battery").process
        for _, _, p, keys, identity in spec.particle_bins]
    processes.append(e.diffusion.couple(e.keys, process_id="battery:electrolyte-diffusion",
                                        owner_id="battery").process)

    def fields(state):
        return dfn_fields(spec, state, program=program)

    def source(state):
        f = fields(state)
        q = f["reaction_current_A"]
        amounts = {keys[-1]: -value / F
                   for value, (_, _, _, keys, _) in zip(q, spec.particle_bins, strict=True)}
        nn, ns, _ = e.cells
        salt = np.r_[q[:nn], np.zeros(ns), q[nn:]] * (1 - e.cation_transference) / F
        amounts.update(zip(e.keys, salt, strict=True))
        return ProcessContribution(amount_rates_mol_s=amounts, coordinate_rates={
            "discharged_charge_C": f["current_A"],
            "delivered_work_J": f["current_A"] * f["voltage_V"],
        })

    def validate(state):
        if abs(state.temperature_K - spec.temperature_K) > 1e-8 or abs(
            state.volume_L - spec.volume_l
        ) > 1e-12:
            raise ValueError("DFN requires its declared isothermal volume")
        for _, _, particle, keys, _ in spec.particle_bins:
            if any(state.amounts_mol[k] > volume * particle.maximum_concentration_mol_m3
                   * (1 + 1e-8) for k, volume in
                   zip(keys, particle.mesh.cell_volumes_m3, strict=True)):
                raise ValueError("particle inventory exceeds its finite capacity")
        if e.minimum_resolved_concentration(state) < e.minimum_concentration_mol_m3 * (1 - 1e-7):
            raise ValueError("electrolyte is outside its positive-concentration domain")

    processes.append(CoupledProcess("battery:dfn-intercalation", source, validate_state=validate))
    events = [CoupledEvent("battery:electrolyte:depleted", lambda s:
                           e.minimum_resolved_concentration(s) - e.minimum_concentration_mol_m3)]
    for name, region in (("negative", 0), ("positive", 2)):
        selection = slice(0, e.cells[0]) if region == 0 else slice(e.cells[0], None)
        events.extend((
            CoupledEvent(f"battery:{name}:depleted", lambda s, sel=selection:
                         float(np.min(fields(s)["surface_fraction"][sel])) - spec.surface_margin),
            CoupledEvent(f"battery:{name}:saturated", lambda s, sel=selection:
                         1 - spec.surface_margin
                         - float(np.max(fields(s)["surface_fraction"][sel]))),
        ))
    events.extend((
        CoupledEvent("battery:minimum-voltage", lambda s:
                     fields(s)["voltage_V"] - spec.minimum_voltage_V),
        CoupledEvent("battery:maximum-voltage", lambda s:
                     spec.maximum_voltage_V - fields(s)["voltage_V"]),
    ))
    nkeys = [k for name, _, _, keys, _ in spec.particle_bins if name == "negative" for k in keys]
    pkeys = [k for name, _, _, keys, _ in spec.particle_bins if name == "positive" for k in keys]
    scales = {k: p.capacity_mol / p.cells for _, _, p, keys, _ in spec.particle_bins for k in keys}
    scales.update(e.initial_amounts())

    @lru_cache(maxsize=8)
    def linear_jacobian(identities):
        """Geometry/ordering only; never cache physical state or a solved field."""
        index = {key: i for i, key in enumerate(identities)}
        rows, columns, data = [], [], []

        def add(row_ids, column_ids, matrix):
            coo = sparse.coo_matrix(matrix)
            rows.extend(np.asarray(row_ids)[coo.row])
            columns.extend(np.asarray(column_ids)[coo.col])
            data.extend(coo.data)

        algebraic_columns = [index[key] for key in spec.algebraic_names]
        qmap, _ = spec.electrical_maps
        for row, (_, _, particle, keys, _) in enumerate(spec.particle_bins):
            cols = [index[key] for key in keys]
            add(cols, cols, ConservativeDiffusion(
                particle.mesh, particle.diffusivity_m2_s).inventory_jacobian_s_inv)
            add([index[keys[-1]]], algebraic_columns, [-qmap[row] / F])
        salt_columns = [index[key] for key in e.keys]
        add(salt_columns, salt_columns, e.diffusion.inventory_jacobian_s_inv)
        nn, ns, _ = e.cells
        salt_source = np.vstack((qmap[:nn], np.zeros((ns, len(algebraic_columns))), qmap[nn:]))
        add(salt_columns, algebraic_columns, salt_source * (1 - e.cation_transference) / F)
        add([index["discharged_charge_C"]], [index[BATTERY_ALGEBRAIC[0]]], [[1.0]])
        return sparse.csc_matrix((data, (rows, columns)),
                                 shape=(len(identities), len(identities)))

    def jacobian(state, identities):
        index = {key: i for i, key in enumerate(identities)}
        algebraic_rows = np.array([index[key] for key in spec.algebraic_names])
        closure_coo = -_closure_jacobian(spec, state, program, identities).tocoo()
        ci, cv = (index[key] for key in BATTERY_ALGEBRAIC)
        work = index["delivered_work_J"]
        data = np.r_[closure_coo.data, state.algebraic_values[BATTERY_ALGEBRAIC[1]],
                     state.algebraic_values[BATTERY_ALGEBRAIC[0]]]
        rows = np.r_[algebraic_rows[closure_coo.row], work, work]
        columns = np.r_[closure_coo.col, ci, cv]
        dynamic = sparse.csc_matrix((data, (rows, columns)),
                                    shape=(len(identities), len(identities)))
        return linear_jacobian(tuple(identities)) + dynamic

    return CoupledSystem(
        BatteryThermodynamics(spec), tuple(processes), isothermal=True,
        events=(*events, *control_events), inventory_scales_mol=scales,
        coordinates=(CoupledCoordinate("discharged_charge_C", "C", spec.usable_charge_c),
                     CoupledCoordinate("delivered_work_J", "J", spec.usable_charge_c * 4)),
        conservation_laws={
            "lithium_mol": dict.fromkeys(spec.inventory_keys, 1),
            "electrolyte_salt_mol": dict.fromkeys(e.keys, 1),
            "negative_faradaic_C": {**dict.fromkeys(nkeys, F), "discharged_charge_C": 1},
            "positive_faradaic_C": {**dict.fromkeys(pkeys, F), "discharged_charge_C": -1},
        }, numerical_form="residual", algebraic=closure, residual_rhs_jacobian=jacobian,
    )
