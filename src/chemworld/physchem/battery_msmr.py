"""Isothermal MSMR SPM using unique radial Li and the common coupled residual.

Uniform spherical finite volumes implement N=M grad(U), with harmonic face M.
The source convention extrapolates surface potential, not a boundary Li stock.
Only terminal electrical work is reported; no thermal or chemical-energy closure
is implied by this bounded model family. Stationary impedance differentiates
this same residual at an interior rest state; reaction rows remain shared
occupancy/kinetic contributions, not separate lithium inventories.
"""

from dataclasses import asdict, dataclass, replace
from math import fsum, isclose, isfinite

import numpy as np
from scipy import sparse

from chemworld.data.scientific_inputs import _text
from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC, BATTERY_COORDINATES, ParticleSpec
from chemworld.physchem.battery_msmr_materials import MSMRMaterialSpec
from chemworld.physchem.coupled_system import (
    CoupledAlgebraic,
    CoupledCoordinate,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    InternalTransfer,
    ProcessContribution,
)
from chemworld.physchem.electrochemistry import FARADAY_C_PER_MOL as F
from chemworld.physchem.electrochemistry import R_J_PER_MOL_K as R
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

MSMR_FAMILY = "msmr-spm-isothermal-v1"


@dataclass(frozen=True)
class BatteryMSMRSPMSpec:
    negative: ParticleSpec
    positive: ParticleSpec
    negative_material: MSMRMaterialSpec
    positive_material: MSMRMaterialSpec
    initial_negative_fraction: float = 0.7
    initial_positive_fraction: float = 0.3
    temperature_K: float = 298.15
    electrolyte_concentration_mol_m3: float = 1000.0
    surface_margin: float = 0.01
    minimum_voltage_V: float = 2.5
    maximum_voltage_V: float = 4.5
    maximum_current_A: float = 2.0
    resistance_ohm: float = 0.0
    provenance: str = "declared-msmr-spm-v1"

    def __post_init__(self) -> None:
        for name in ("negative", "positive"):
            particle, material = getattr(self, name), getattr(self, name + "_material")
            if isinstance(particle, dict):
                particle = ParticleSpec(**particle)
                object.__setattr__(self, name, particle)
            if isinstance(material, dict):
                material = MSMRMaterialSpec.from_dict(material)
                object.__setattr__(self, name + "_material", material)
            if not isinstance(particle, ParticleSpec) or not isinstance(material, MSMRMaterialSpec):
                raise ValueError("MSMR requires explicit particles and sourced reaction materials")
            if particle.mesh_exponent != 1 or particle.cells < 2:
                raise ValueError(
                    "MSMR source surface closure requires uniform grids with >=2 cells"
                )
            if any(not isfinite(v) or v <= 0 for v in (particle.area_m2, particle.capacity_mol)):
                raise ValueError("MSMR derived particle area/capacity must be positive and finite")
        values = (
            self.initial_negative_fraction,
            self.initial_positive_fraction,
            self.temperature_K,
            self.electrolyte_concentration_mol_m3,
            self.surface_margin,
            self.minimum_voltage_V,
            self.maximum_voltage_V,
            self.maximum_current_A,
            self.resistance_ohm,
        )
        if (
            any(isinstance(v, bool) or not isfinite(v) for v in values)
            or self.temperature_K <= 0
            or self.electrolyte_concentration_mol_m3 <= 0
            or not 0 < self.surface_margin < 0.5
            or not all(
                self.surface_margin < v < 1 - self.surface_margin
                for v in (self.initial_negative_fraction, self.initial_positive_fraction)
            )
            or self.minimum_voltage_V >= self.maximum_voltage_V
            or self.maximum_current_A <= 0
            or self.resistance_ohm < 0
        ):
            raise ValueError("invalid MSMR operating domain")
        _text(self.provenance, "MSMR provenance")
        if not isfinite(self.usable_charge_c) or self.usable_charge_c <= 0:
            raise ValueError("MSMR usable charge must be positive and finite")

    @property
    def model_family(self):
        return MSMR_FAMILY

    @property
    def model_id(self):
        return MSMR_FAMILY

    @property
    def numerical_form(self):
        return "residual"

    @property
    def thermal(self) -> None:
        return None

    @property
    def sei(self) -> None:
        return None

    @property
    def electrolyte(self) -> None:
        return None

    @property
    def has_directional_ocp(self):
        return False

    @property
    def hysteresis_keys(self):
        return ()

    @property
    def coordinate_keys(self):
        return BATTERY_COORDINATES

    @property
    def inventory_keys(self):
        return (*self.negative.keys("negative"), *self.positive.keys("positive"))

    @property
    def inventory_species(self):
        return dict.fromkeys(self.inventory_keys, "Li")

    @property
    def series_resistance_ohm(self):
        return self.resistance_ohm

    @property
    def volume_l(self):
        return (self.negative.active_volume_m3 + self.positive.active_volume_m3) * 1000

    @property
    def usable_charge_c(self):
        return F * min(
            self.negative.capacity_mol * (self.initial_negative_fraction - self.surface_margin),
            self.positive.capacity_mol * (1 - self.surface_margin - self.initial_positive_fraction),
        )

    def current_at_c_rate(self, rate):
        if not isfinite(rate):
            raise ValueError("MSMR C rate must be finite")
        return rate * self.usable_charge_c / 3600

    def initial_coordinates(self):
        return dict.fromkeys(self.coordinate_keys, 0.0)

    def initial_state(self):
        amounts = {}
        for name in ("negative", "positive"):
            p = getattr(self, name)
            amounts.update(
                zip(
                    p.keys(name),
                    p.mesh.cell_volumes_m3
                    * p.maximum_concentration_mol_m3
                    * getattr(self, "initial_" + name + "_fraction"),
                    strict=True,
                )
            )
        return CoupledState(
            amounts, self.temperature_K, self.volume_l, coordinates=self.initial_coordinates()
        )

    def to_dict(self):
        return {
            "model_family": MSMR_FAMILY,
            **{
                key: (
                    asdict(value)
                    if isinstance(value, ParticleSpec)
                    else value.to_dict()
                    if isinstance(value, MSMRMaterialSpec)
                    else value
                )
                for key, value in self.__dict__.items()
            },
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("model_family", None) != MSMR_FAMILY:
            raise ValueError("unsupported MSMR family")
        return cls(**value)


@dataclass(frozen=True)
class MSMRReadout:
    voltage_V: float
    current_A: float
    negative_surface_fraction: float
    positive_surface_fraction: float
    negative_average_fraction: float
    positive_average_fraction: float
    negative_potential_V: float
    positive_potential_V: float
    negative_overpotential_V: float
    positive_overpotential_V: float
    negative_surface_occupancies: tuple[float, ...]
    positive_surface_occupancies: tuple[float, ...]
    negative_source_surface_occupancies: tuple[float, ...]
    positive_source_surface_occupancies: tuple[float, ...]
    negative_reaction_current_densities_A_m2: tuple[float, ...]
    positive_reaction_current_densities_A_m2: tuple[float, ...]


def _trial_fractions(raw):
    # No legal interior value is changed. This extension is for rejected
    # endpoint/outside residual trials only; it is never an accepted inventory.
    return np.where(raw <= 0.0, 1e-12, np.where(raw >= 1.0, 1 - 1e-12, raw))


def _particle_values(spec, state, name):
    p, material = getattr(spec, name), getattr(spec, name + "_material")
    capacity = p.maximum_concentration_mol_m3 * p.mesh.cell_volumes_m3
    raw = np.array([state.amounts_mol[k] for k in p.keys(name)]) / capacity
    # Only residual trial states use this representable extension. Accepted
    # inventories are separately validated, and events inspect the real values.
    fractions = _trial_fractions(raw)
    potentials = np.array([material.potential(float(x), spec.temperature_K) for x in fractions])
    slopes = np.array(
        [material.stoichiometry_derivatives(float(u), spec.temperature_K)[1] for u in potentials]
    )
    if np.any(~np.isfinite(slopes)) or np.any(slopes >= 0):
        raise ValueError("MSMR particle inverse derivative is not representable")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        derivative = 1 / (slopes * capacity)
    derivative[raw != fractions] = 0.0
    if np.any(~np.isfinite(derivative)):
        raise ValueError("MSMR inventory potential derivative is not representable")
    return raw, potentials, derivative, capacity


def msmr_particle_transport(spec, state, name):
    """One evaluation per face, returning conservative mol/s and its analytic Jacobian."""
    p = getattr(spec, name)
    raw, potentials, du, capacity = _particle_values(spec, state, name)
    x = _trial_fractions(raw)
    coefficient = p.maximum_concentration_mol_m3 * F / (R * spec.temperature_K) * p.diffusivity_m2_s
    mobility = coefficient * x * (1 - x)
    dm = coefficient * (1 - 2 * x) / capacity
    dm[raw != x] = 0.0
    left, right = mobility[:-1], mobility[1:]
    total = left + right
    # Avoid squaring a tiny legal mobility before dividing by its scale.
    harmonic = 2 * (left / total) * right
    geometry = p.mesh.face_areas_m2[1:-1] / np.diff(p.mesh.centers_m)
    difference = np.diff(potentials)
    flux = geometry * harmonic * difference
    dl = geometry * (2 * (right / total) ** 2 * dm[:-1] * difference - harmonic * du[:-1])
    dr = geometry * (2 * (left / total) ** 2 * dm[1:] * difference + harmonic * du[1:])
    rates = np.zeros(p.cells)
    rates[:-1] -= flux
    rates[1:] += flux
    jac = np.zeros((p.cells, p.cells))
    i = np.arange(p.cells - 1)
    jac[i, i] -= dl
    jac[i, i + 1] -= dr
    jac[i + 1, i] += dl
    jac[i + 1, i + 1] += dr
    return rates, jac, flux


def _port(spec, state, current):
    if not isfinite(current) or abs(current) > spec.maximum_current_A * (1 + 1e-14):
        raise ValueError("MSMR current outside source capability")
    result, derivatives, arrays = {"current_A": current}, {}, {}
    voltage, vi = -current * spec.resistance_ohm, -spec.resistance_ohm
    for name, polarity in (("negative", 1.0), ("positive", -1.0)):
        p, material = getattr(spec, name), getattr(spec, name + "_material")
        raw, u, du, _ = _particle_values(spec, state, name)
        surface_u = 1.5 * u[-1] - 0.5 * u[-2]
        eta, currents, eta_j, eta_u = material.kinetic_response(
            float(surface_u),
            polarity * current / p.area_m2,
            spec.temperature_K,
            spec.electrolyte_concentration_mol_m3,
        )
        occupancies = material.occupancies(float(surface_u), spec.temperature_K)
        source_occupancies = tuple(
            1.5 * a - 0.5 * b
            for a, b in zip(
                material.occupancies(float(u[-1]), spec.temperature_K),
                material.occupancies(float(u[-2]), spec.temperature_K),
                strict=True,
            )
        )
        result.update(
            {
                name + "_potential_V": float(surface_u),
                name + "_overpotential_V": eta,
                name + "_surface_fraction": fsum(occupancies),
                name + "_average_fraction": fsum(state.amounts_mol[k] for k in p.keys(name))
                / p.capacity_mol,
                name + "_surface_occupancies": occupancies,
                name + "_source_surface_occupancies": source_occupancies,
                name + "_reaction_current_densities_A_m2": currents,
            }
        )
        arrays[name + "_particle_potentials_V"] = tuple(u)
        arrays[name + "_particle_fractions"] = tuple(raw)
        voltage -= polarity * (surface_u + eta)
        vi -= eta_j / p.area_m2
        for k, weight, value in zip(p.keys(name)[-2:], (-0.5, 1.5), du[-2:], strict=True):
            derivatives[k] = -polarity * (1 + eta_u) * weight * value
    result["voltage_V"] = float(voltage)
    derivatives[BATTERY_ALGEBRAIC[0]] = vi
    return MSMRReadout(**result), derivatives, arrays


def validate_msmr_state(spec, state):
    if set(state.amounts_mol) != set(spec.inventory_keys) or set(state.coordinates) != set(
        spec.coordinate_keys
    ):
        raise ValueError("MSMR state needs its unique radial Li and charge/work coordinates")
    if abs(state.temperature_K - spec.temperature_K) > 1e-8 or not isclose(
        state.volume_L, spec.volume_l, rel_tol=1e-12, abs_tol=1e-14
    ):
        raise ValueError("MSMR requires its declared isothermal fixed volume")
    charge = state.coordinates["discharged_charge_C"]
    for name, polarity in (("negative", 1.0), ("positive", -1.0)):
        p = getattr(spec, name)
        amounts = [state.amounts_mol[k] for k in p.keys(name)]
        if any(
            not 0 < n < capacity
            for n, capacity in zip(
                amounts, p.maximum_concentration_mol_m3 * p.mesh.cell_volumes_m3, strict=True
            )
        ):
            raise ValueError("MSMR accepted radial inventory must be strictly inside capacity")
        initial = p.capacity_mol * getattr(spec, "initial_" + name + "_fraction")
        if not isclose(
            F * fsum(amounts) + polarity * charge, F * initial, rel_tol=1e-8, abs_tol=1e-8
        ):
            raise ValueError("MSMR radial Li and charge violate the same Faraday balance")


def msmr_readout(spec, state, current_A):
    if isinstance(state, CoupledState):
        validate_msmr_state(spec, state)
    return _port(spec, state, current_A)[0]


def msmr_control_mode(spec, state, program):
    target = program.value_at(state.time_s)
    if program.unit == "A":
        return "constant_current", target
    if program.unit != "V":
        raise ValueError("MSMR supports current and voltage programs only")
    limit = spec.maximum_current_A
    if _port(spec, state, -limit)[0].voltage_V < target:
        return "negative_current_limit", -limit
    if _port(spec, state, limit)[0].voltage_V > target:
        return "positive_current_limit", limit
    return "constant_voltage", target


def msmr_algebraic(spec, program):
    ik, vk = BATTERY_ALGEBRAIC
    variables = (
        ResidualVariable(
            ik,
            "A",
            spec.maximum_current_A,
            "algebraic",
            -spec.maximum_current_A,
            spec.maximum_current_A,
        ),
        ResidualVariable(vk, "V", 1.0, "algebraic"),
    )
    equations = (
        ResidualEquation("msmr:control", "1"),
        ResidualEquation("msmr:terminal-voltage", "V"),
    )

    def initial(state):
        mode, target = msmr_control_mode(spec, state, program)
        current = 0.0 if mode == "constant_voltage" else target
        return {ik: current, vk: _port(spec, state, current)[0].voltage_V}

    def residual(state):
        current, voltage = (state.algebraic_values[k] for k in BATTERY_ALGEBRAIC)
        mode, target = msmr_control_mode(spec, state, program)
        return [
            voltage - target
            if mode == "constant_voltage"
            else (current - target) / spec.maximum_current_A,
            voltage - _port(spec, state, current)[0].voltage_V,
        ]

    return CoupledAlgebraic(variables, equations, initial, residual)


def msmr_control_current(spec, state, program):
    if BATTERY_ALGEBRAIC[0] in state.algebraic_values:
        return state.algebraic_values[BATTERY_ALGEBRAIC[0]]
    mode, target = msmr_control_mode(spec, state, program)
    if mode != "constant_voltage":
        if abs(target) > spec.maximum_current_A:
            raise ValueError("MSMR current is outside source capability")
        return target
    closure = msmr_algebraic(spec, program)
    guess = closure.initial_values(state)

    def residual(t, y, rate, params):
        view = CoupledStateView(
            state.amounts_mol,
            state.temperature_K,
            state.volume_L,
            state.time_s,
            coordinates=state.coordinates,
            algebraic_values=dict(zip(BATTERY_ALGEBRAIC, y, strict=True)),
        )
        return closure.residual(view)

    def jacobian(t, y, rate, params):
        return [[0.0, 1.0], [-_port(spec, state, y[0])[1][BATTERY_ALGEBRAIC[0]], 1.0]]

    system = ResidualSystem(
        "msmr:electrical-closure",
        closure.variables,
        closure.equations,
        residual,
        ResidualDerivatives("hybrid", state=jacobian),
    )
    answer = solve_steady_residual(
        system, [guess[k] for k in BATTERY_ALGEBRAIC], policy=RootPolicy(tolerance=1e-12)
    )
    if not answer.root.success:
        raise ResidualNumericalError("MSMR electrical closure:" + answer.root.failure_reason)
    return float(answer.root.value[0])


def msmr_fields(spec, state, *, current_A=None, program=None):
    if isinstance(state, CoupledState):
        validate_msmr_state(spec, state)
        state = replace(state, algebraic_values={})
    if (current_A is None) == (program is None):
        raise ValueError("MSMR fields require exactly one current or control program")
    current = current_A if program is None else msmr_control_current(spec, state, program)
    reading, _, arrays = _port(spec, state, current)
    return {**asdict(reading), **arrays}


class _MSMRThermodynamics:
    def __init__(self, spec):
        self.inventory_keys = spec.inventory_keys

    def heat_capacity_j_k(self, state):
        return 1.0

    def internal_energy_j(self, state):
        return None

    def chemical_energy_rate_w(self, state, rates, contributions):
        return 0.0


def _msmr_rhs_jacobian(spec, state, program, identities):
    index = {k: i for i, k in enumerate(identities)}
    jac = np.zeros((len(identities), len(identities)))
    ik, vk = BATTERY_ALGEBRAIC
    current = state.algebraic_values[ik]
    for name, polarity in (("negative", 1.0), ("positive", -1.0)):
        keys = getattr(spec, name).keys(name)
        cols = [index[k] for k in keys]
        jac[np.ix_(cols, cols)] = msmr_particle_transport(spec, state, name)[1]
        jac[index[keys[-1]], index[ik]] = -polarity / F
    read, derivatives, _ = _port(spec, state, current)
    dv = np.zeros(len(identities))
    for key, value in derivatives.items():
        dv[index[key]] = value
    jac[index["discharged_charge_C"], index[ik]] = 1.0
    jac[index["delivered_work_J"]] = current * dv
    jac[index["delivered_work_J"], index[ik]] += read.voltage_V
    mode, _ = msmr_control_mode(spec, state, program)
    if mode == "constant_voltage":
        jac[index[ik], index[vk]] = -1.0
    else:
        jac[index[ik], index[ik]] = -1 / spec.maximum_current_A
    jac[index[vk]] = dv
    jac[index[vk], index[vk]] -= 1.0
    return sparse.csc_matrix(jac)


def msmr_system(spec, program, *, thermal_device=None, control_events=()):
    if not isinstance(spec, BatteryMSMRSPMSpec):
        raise ValueError("MSMR requires its explicit specification")
    if thermal_device is not None:
        raise ValueError("MSMR thermal and chemical-energy closure are not implemented")
    if (
        not isinstance(program, ScalarControlProgram)
        or program.unit not in {"A", "V"}
        or any(
            (
                abs(v) > spec.maximum_current_A
                if program.unit == "A"
                else not spec.minimum_voltage_V <= v <= spec.maximum_voltage_V
            )
            for _, v in program.setpoints
        )
    ):
        raise ValueError("MSMR requires an in-domain CC or CV program")

    def evaluate(state):
        transfers = []
        for name in ("negative", "positive"):
            keys = getattr(spec, name).keys(name)
            flux = msmr_particle_transport(spec, state, name)[2]
            for i, rate in enumerate(flux):
                source, target = (keys[i], keys[i + 1]) if rate >= 0 else (keys[i + 1], keys[i])
                transfers.append(
                    InternalTransfer(f"msmr:{name}:face:{i}", source, target, abs(rate))
                )
        current = msmr_control_current(spec, state, program)
        read = _port(spec, state, current)[0]
        nkey, pkey = spec.negative.keys("negative")[-1], spec.positive.keys("positive")[-1]
        source, target = (nkey, pkey) if current >= 0 else (pkey, nkey)
        transfers.append(InternalTransfer("msmr:interelectrode", source, target, abs(current) / F))
        return ProcessContribution(
            internal_transfers=tuple(transfers),
            coordinate_rates={
                "discharged_charge_C": current,
                "delivered_work_J": current * read.voltage_V,
            },
        )

    def reading(state):
        return _port(spec, state, msmr_control_current(spec, state, program))[0]

    events = []
    for name in ("negative", "positive"):
        events.extend(
            (
                CoupledEvent(
                    f"msmr:{name}:depleted",
                    lambda s, name=name: (
                        getattr(reading(s), name + "_surface_fraction") - spec.surface_margin
                    ),
                ),
                CoupledEvent(
                    f"msmr:{name}:saturated",
                    lambda s, name=name: (
                        1 - spec.surface_margin - getattr(reading(s), name + "_surface_fraction")
                    ),
                ),
                CoupledEvent(
                    f"msmr:{name}:local-depleted",
                    lambda s, name=name: (
                        min(_particle_values(spec, s, name)[0]) - spec.surface_margin
                    ),
                ),
                CoupledEvent(
                    f"msmr:{name}:local-saturated",
                    lambda s, name=name: (
                        1 - spec.surface_margin - max(_particle_values(spec, s, name)[0])
                    ),
                ),
            )
        )
    events.extend(
        (
            CoupledEvent(
                "msmr:minimum-voltage", lambda s: reading(s).voltage_V - spec.minimum_voltage_V
            ),
            CoupledEvent(
                "msmr:maximum-voltage", lambda s: spec.maximum_voltage_V - reading(s).voltage_V
            ),
        )
    )
    if program.unit == "V":
        events.extend(
            CoupledEvent(
                f"msmr:cv:{label}-limit",
                lambda s, value=value: (
                    _port(spec, s, value)[0].voltage_V - program.value_at(s.time_s)
                ),
                direction=0,
                terminal=False,
            )
            for label, value in (
                ("negative", -spec.maximum_current_A),
                ("positive", spec.maximum_current_A),
            )
        )
    return CoupledSystem(
        _MSMRThermodynamics(spec),
        (
            CoupledProcess(
                "msmr:particle-and-interface",
                evaluate,
                validate_state=lambda s: validate_msmr_state(spec, s),
            ),
        ),
        isothermal=True,
        numerical_form="residual",
        events=(*events, *control_events),
        coordinates=(
            CoupledCoordinate("discharged_charge_C", "C", spec.usable_charge_c),
            CoupledCoordinate("delivered_work_J", "J", spec.usable_charge_c * 4),
        ),
        inventory_scales_mol={
            k: getattr(spec, name).capacity_mol / getattr(spec, name).cells
            for name in ("negative", "positive")
            for k in getattr(spec, name).keys(name)
        },
        conservation_laws={
            "lithium_mol": dict.fromkeys(spec.inventory_keys, 1.0),
            "negative_faradaic_C": {
                **dict.fromkeys(spec.negative.keys("negative"), F),
                "discharged_charge_C": 1.0,
            },
            "positive_faradaic_C": {
                **dict.fromkeys(spec.positive.keys("positive"), F),
                "discharged_charge_C": -1.0,
            },
        },
        algebraic=msmr_algebraic(spec, program),
        residual_rhs_jacobian=lambda s, ids: _msmr_rhs_jacobian(spec, s, program, ids),
    )


def advance_msmr(
    spec,
    state,
    program,
    duration_s,
    *,
    evaluation_times_s=None,
    thermal_device=None,
    control_events=(),
):
    result = msmr_system(
        spec, program, thermal_device=thermal_device, control_events=control_events
    ).advance(state, duration_s, evaluation_times_s=evaluation_times_s)
    return replace(
        result,
        solver_diagnostic={
            **result.solver_diagnostic,
            "battery_control_mode": msmr_control_mode(spec, result.final_state, program)[0],
            "energy_scope": "isothermal; terminal electrical work only; no MSMR heat closure",
        },
    )
