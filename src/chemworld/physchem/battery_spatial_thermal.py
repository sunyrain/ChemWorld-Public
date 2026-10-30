"""SPM plus genuine pouch-thickness temperature field on one coupled residual.

Finite scope: constant sourced electrode OCV, zero entropy, constant solid D
and concentration-dependent BV exchange scale; ideal current collectors and
uniform electrolyte. Thermal nodes are not independent electrochemical cells.
"""

from dataclasses import dataclass, replace
from functools import cached_property
from math import asinh, isclose, isfinite, sqrt

import numpy as np
from scipy import sparse

from chemworld.physchem.battery_heat_stack import BatteryHeatStackSpec
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import (
    BATTERY_ALGEBRAIC,
    BATTERY_COORDINATES,
    BatteryReadout,
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
    EnergyPort,
    HeatFlux,
    InternalTransfer,
    ProcessContribution,
)
from chemworld.physchem.finite_volume import ConservativeDiffusion
from chemworld.physchem.residual_system import (
    ResidualDerivatives,
    ResidualEquation,
    ResidualNumericalError,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    solve_steady_residual,
)

SPATIAL_THERMAL_FAMILY = "spm-spatial-thermal-v1"
ENERGY_SCOPE = (
    "constant sourced OCV, zero entropy and fixed D/BV scale; "
    "G plus thickness-field sensible energy, terminal work and ambient boundary heat"
)


@dataclass(frozen=True)
class _ThermalMetadata:
    """Compatibility readout only; ambient need not lie inside stopping bounds."""

    heat_capacity_J_K: float
    ambient_temperature_K: float
    heat_transfer_W_K: float
    minimum_temperature_K: float
    maximum_temperature_K: float


@dataclass(frozen=True)
class BatterySpatialThermalSPMSpec:
    negative: ParticleSpec
    positive: ParticleSpec
    heat_stack: BatteryHeatStackSpec
    negative_ocv: ElectrodeOCV
    positive_ocv: ElectrodeOCV
    initial_negative_fraction: float = 0.7
    initial_positive_fraction: float = 0.3
    temperature_K: float = 298.15
    initial_temperatures_K: tuple[float, ...] | None = None
    surface_margin: float = 0.01
    minimum_voltage_V: float = 2.5
    maximum_voltage_V: float = 4.5
    maximum_current_A: float = 2.0
    provenance: str = "declared-constant-OCV-spatial-thermal-SPM-v1"

    def __post_init__(self) -> None:
        for name, cls in (
            ("negative", ParticleSpec),
            ("positive", ParticleSpec),
            ("heat_stack", BatteryHeatStackSpec),
            ("negative_ocv", ElectrodeOCV),
            ("positive_ocv", ElectrodeOCV),
        ):
            value = getattr(self, name)
            if isinstance(value, dict):
                value = cls(**value) if cls is ParticleSpec else cls.from_dict(value)
                object.__setattr__(self, name, value)
            if not isinstance(value, cls):
                raise ValueError(
                    "spatial thermal SPM requires explicit particles, heat stack and OCV"
                )
        for name in (
            "initial_negative_fraction",
            "initial_positive_fraction",
            "temperature_K",
            "surface_margin",
            "minimum_voltage_V",
            "maximum_voltage_V",
            "maximum_current_A",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError("spatial battery operating declarations must be finite")
            object.__setattr__(self, name, float(value))
        if (
            not 0 < self.surface_margin < 0.5
            or self.maximum_current_A <= 0
            or self.minimum_voltage_V >= self.maximum_voltage_V
            or not isinstance(self.provenance, str)
            or not self.provenance.strip()
        ):
            raise ValueError("invalid spatial battery operating domain")
        for name in ("negative", "positive"):
            curve, particle = getattr(self, name + "_ocv"), getattr(self, name)
            if (
                curve.endpoint_policy is not None
                or any(curve.polynomial_V[1:])
                or any(a * k != 0 for a, k, _ in curve.exponential_terms)
                or any(a * k != 0 for a, k, _ in curve.tanh_terms)
                or "SPM" not in curve.supported_models
                or curve.stoichiometry_domain[0] > self.surface_margin
                or curve.stoichiometry_domain[1] < 1 - self.surface_margin
            ):
                raise ValueError(
                    "spatial thermal scope requires sourced constant net electrode OCV"
                )
            if (
                not self.surface_margin
                < getattr(self, "initial_" + name + "_fraction")
                < (1 - self.surface_margin)
            ):
                raise ValueError("initial particle fractions must be inside stopping surfaces")
            if particle.active_volume_m3 > self.heat_stack.area_m2 * getattr(
                self.heat_stack, name
            ).length_m * (1 + 1e-12):
                raise ValueError("active solid volume exceeds its declared thermal electrode")
        if not isfinite(self.usable_charge_c) or self.usable_charge_c <= 0:
            raise ValueError("derived charge scale must remain positive and finite")
        if self.initial_temperatures_K is not None:
            initial = tuple(float(v) for v in self.initial_temperatures_K)
            mean, _ = self.heat_stack.pack_temperatures(initial)
            object.__setattr__(self, "initial_temperatures_K", initial)
            object.__setattr__(self, "temperature_K", mean)
        values = self.initial_temperature_field
        if np.any(values <= self.thermal.minimum_temperature_K) or np.any(
            values >= self.thermal.maximum_temperature_K
        ):
            raise ValueError("every initial local temperature must be inside its stopping bounds")

    @property
    def model_family(self):
        return SPATIAL_THERMAL_FAMILY

    model_id = model_family

    @property
    def numerical_form(self) -> str:
        return "residual"

    @property
    def series_resistance_ohm(self) -> float:
        return 0.0

    @property
    def sei(self) -> None:
        return None

    electrolyte = negative_ocp = positive_ocp = sei

    @property
    def has_directional_ocp(self) -> bool:
        return False

    @property
    def hysteresis_keys(self):
        return ()

    @cached_property
    def thermal(self):
        h = self.heat_stack
        return _ThermalMetadata(
            h.total_heat_capacity_J_K,
            h.ambient_temperature_K,
            0,
            h.minimum_temperature_K,
            h.maximum_temperature_K,
        )

    @property
    def thermal_coordinate_keys(self):
        return self.heat_stack.coordinate_keys

    @property
    def coordinate_keys(self):
        return (*BATTERY_COORDINATES, *self.thermal_coordinate_keys)

    @property
    def initial_temperature_field(self):
        return np.array(
            self.initial_temperatures_K
            if self.initial_temperatures_K is not None
            else [self.temperature_K] * len(self.heat_stack.node_ids)
        )

    def initial_coordinates(self):
        _, differences = self.heat_stack.pack_temperatures(self.initial_temperature_field)
        return {**dict.fromkeys(BATTERY_COORDINATES, 0.0), **differences}

    @property
    def inventory_species(self):
        return dict.fromkeys(
            (*self.negative.keys("negative"), *self.positive.keys("positive")), "Li"
        )

    @property
    def inventory_keys(self):
        return tuple(self.inventory_species)

    @property
    def volume_l(self):
        return 1000 * float(np.sum(self.heat_stack.volumes_m3))

    @property
    def usable_charge_c(self):
        return F * min(
            self.negative.capacity_mol * self.initial_negative_fraction,
            self.positive.capacity_mol * (1 - self.initial_positive_fraction),
        )

    def current_at_c_rate(self, rate):
        if not isfinite(rate):
            raise ValueError("C-rate must be finite")
        return rate * self.usable_charge_c / 3600

    def ocv(self, electrode, theta, *, derivative=False):
        if electrode not in {"negative", "positive"}:
            raise ValueError("unknown electrode")
        return 0.0 if derivative else getattr(self, electrode + "_ocv").value(0.5)

    def initial_state(self):
        amounts = {}
        for name in ("negative", "positive"):
            particle = getattr(self, name)
            amounts.update(
                zip(
                    particle.keys(name),
                    particle.mesh.cell_volumes_m3
                    * particle.maximum_concentration_mol_m3
                    * getattr(self, "initial_" + name + "_fraction"),
                    strict=True,
                )
            )
        return CoupledState(
            amounts, self.temperature_K, self.volume_l, coordinates=self.initial_coordinates()
        )

    def to_dict(self):
        from dataclasses import asdict

        return {
            "model_family": self.model_family,
            "negative": asdict(self.negative),
            "positive": asdict(self.positive),
            **{
                k: getattr(self, k).to_dict()
                for k in ("heat_stack", "negative_ocv", "positive_ocv")
            },
            **{
                k: getattr(self, k)
                for k in (
                    "initial_negative_fraction",
                    "initial_positive_fraction",
                    "temperature_K",
                    "surface_margin",
                    "minimum_voltage_V",
                    "maximum_voltage_V",
                    "maximum_current_A",
                    "provenance",
                )
            },
            "initial_temperatures_K": (
                None if self.initial_temperatures_K is None else list(self.initial_temperatures_K)
            ),
        }

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        if data.pop("model_family") != SPATIAL_THERMAL_FAMILY:
            raise ValueError("unknown spatial battery family")
        return cls(**data)


def validate_spatial_thermal_state(spec, state):
    if set(state.amounts_mol) != set(spec.inventory_keys) or set(state.coordinates) != set(
        spec.coordinate_keys
    ):
        raise ValueError("spatial battery state must retain every Li and thermal coordinate")
    if not isclose(state.volume_L, spec.volume_l, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("spatial battery must preserve its original stack geometry")
    temperatures = spec.heat_stack.temperatures(state.temperature_K, state.coordinates)
    if (
        not np.all(np.isfinite(temperatures))
        or np.min(temperatures) < spec.thermal.minimum_temperature_K - 1e-7
        or np.max(temperatures) > spec.thermal.maximum_temperature_K + 1e-7
    ):
        raise ValueError("local temperature is outside the caloric stopping domain")
    for name, sign in (("negative", 1), ("positive", -1)):
        particle = getattr(spec, name)
        values = np.array([state.amounts_mol[k] for k in particle.keys(name)])
        if (
            not np.all(np.isfinite(values))
            or np.any(values < 0)
            or np.any(
                values
                > particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3 * (1 + 1e-8)
            )
        ):
            raise ValueError("particle inventory is outside finite capacity")
        charge = state.coordinates["discharged_charge_C"]
        expected = F * particle.capacity_mol * getattr(spec, "initial_" + name + "_fraction")
        if not isclose(F * sum(values) + sign * charge, expected, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("spatial battery inventories disagree with original Faradaic charge")


def _electrical(spec, state, current):
    """Local inverse BV coefficient and derivatives; temperature remains a field."""
    if not isfinite(current) or abs(current) > spec.maximum_current_A * (1 + 1e-12):
        raise ValueError("current is outside the declared source capability")
    temperatures = spec.heat_stack.temperatures(state.temperature_K, state.coordinates)
    data = {}
    for name, sign in (("negative", 1), ("positive", -1)):
        particle = getattr(spec, name)
        theta = particle_surface_fraction(particle, name, state, sign * current / F)
        trial = float(np.clip(theta, spec.surface_margin / 2, 1 - spec.surface_margin / 2))
        j0 = particle.exchange_current_scale_A_m2 * sqrt(trial * (1 - trial))
        z = sign * current / (2 * particle.area_m2 * j0)
        factor = 2 * R / F / sqrt(1 + z * z)
        alpha = 2 * R / F * asinh(z)
        alpha_theta = (
            -factor * z * (1 - 2 * trial) / (2 * trial * (1 - trial)) if theta == trial else 0.0
        )
        beta = particle.mesh.spherical_surface_distance_m / (
            F * particle.area_m2 * particle.diffusivity_m2_s * particle.maximum_concentration_mol_m3
        )
        data[name] = {
            "theta": theta,
            "alpha": alpha,
            "alpha_i": factor * sign / (2 * particle.area_m2 * j0) - alpha_theta * sign * beta,
            "alpha_amount": alpha_theta
            / (particle.mesh.cell_volumes_m3[-1] * particle.maximum_concentration_mol_m3),
            "mean_temperature": float(np.mean(temperatures[spec.heat_stack.electrode_slice(name)])),
            "average_fraction": sum(state.amounts_mol[k] for k in particle.keys(name))
            / particle.capacity_mol,
        }
    return temperatures, data


def spatial_thermal_readout(spec, state, current_A):
    if isinstance(state, CoupledState):
        validate_spatial_thermal_state(spec, state)
    _, data = _electrical(spec, state, current_A)
    n, p = data["negative"], data["positive"]
    en, ep = n["alpha"] * n["mean_temperature"], p["alpha"] * p["mean_temperature"]
    voltage = spec.ocv("positive", 0.5) - spec.ocv("negative", 0.5) + ep - en
    return BatteryReadout(
        voltage,
        current_A,
        n["theta"],
        p["theta"],
        n["average_fraction"],
        p["average_fraction"],
        en,
        ep,
        current_A,
        0.0,
        0.0,
    )


def spatial_thermal_control_mode(spec, state, program):
    target = program.value_at(state.time_s)
    if program.unit == "A":
        return "constant_current", target
    if program.unit != "V":
        raise ValueError("spatial battery control requires A or V")
    limit = spec.maximum_current_A
    if spatial_thermal_readout(spec, state, -limit).voltage_V < target:
        return "negative_current_limit", -limit
    if spatial_thermal_readout(spec, state, limit).voltage_V > target:
        return "positive_current_limit", limit
    return "constant_voltage", target


def spatial_thermal_algebraic(spec, program):
    if program.unit not in {"A", "V"} or any(
        abs(v) > spec.maximum_current_A
        if program.unit == "A"
        else not spec.minimum_voltage_V <= v <= spec.maximum_voltage_V
        for _, v in program.setpoints
    ):
        raise ValueError("spatial battery programme is outside device capability")
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
        ResidualVariable(vk, "V", role="algebraic"),
    )
    equations = (
        ResidualEquation("battery:control", "1"),
        ResidualEquation("battery:terminal-voltage", "V"),
    )

    def initial(state):
        i = program.value_at(state.time_s) if program.unit == "A" else 0.0
        return {ik: i, vk: spatial_thermal_readout(spec, state, i).voltage_V}

    def residual(state):
        i, v = (state.algebraic_values[k] for k in BATTERY_ALGEBRAIC)
        mode, target = spatial_thermal_control_mode(spec, state, program)
        return [
            v - target if mode == "constant_voltage" else (i - target) / spec.maximum_current_A,
            v - spatial_thermal_readout(spec, state, i).voltage_V,
        ]

    return CoupledAlgebraic(variables, equations, initial, residual)


def spatial_thermal_control_current(spec, state, program):
    if BATTERY_ALGEBRAIC[0] in state.algebraic_values:
        return state.algebraic_values[BATTERY_ALGEBRAIC[0]]
    closure = spatial_thermal_algebraic(spec, program)
    if program.unit == "A":
        return program.value_at(state.time_s)
    guess = closure.initial_values(state)

    def residual(t, y, rate, parameters):
        view = CoupledStateView(
            state.amounts_mol,
            state.temperature_K,
            state.volume_L,
            state.time_s,
            coordinates=state.coordinates,
            algebraic_values=dict(zip(BATTERY_ALGEBRAIC, y, strict=True)),
        )
        return closure.residual(view)

    system = ResidualSystem(
        "battery:spatial-electrical-closure",
        closure.variables,
        closure.equations,
        residual,
        ResidualDerivatives("finite_difference"),
    )
    answer = solve_steady_residual(
        system, [guess[k] for k in BATTERY_ALGEBRAIC], policy=RootPolicy(tolerance=1e-12)
    )
    if not answer.root.success:
        raise ResidualNumericalError("spatial battery control:" + answer.root.failure_reason)
    return float(answer.root.value[0])


def spatial_thermal_fields(spec, state, *, current_A=None, program=None):
    if (current_A is None) == (program is None):
        raise ValueError("supply exactly one current or programme")
    if program is not None:
        current_A = spatial_thermal_control_current(spec, state, program)
    reading = spatial_thermal_readout(spec, state, current_A)
    stack = spec.heat_stack
    temperatures, data = _electrical(spec, state, current_A)
    generated = np.zeros(len(stack.node_ids))
    for name, sign in (("negative", 1), ("positive", -1)):
        sel = stack.electrode_slice(name)
        generated[sel] = (
            sign * current_A * data[name]["alpha"] * temperatures[sel] / getattr(stack, name).cells
        )
    losses = stack.loss_conductance_W_K * (temperatures - stack.ambient_temperature_K)
    local_rates = (
        stack.conductive_heat_W(state.coordinates) + generated - losses
    ) / stack.heat_capacities_J_K
    mean_rate = float((sum(generated) - sum(losses)) / stack.total_heat_capacity_J_K)
    return {
        "temperature_K": temperatures,
        "node_ids": stack.node_ids,
        "heat_capacity_mean_temperature_K": state.temperature_K,
        "volume_mean_temperature_K": float(stack.volumes_m3 @ temperatures / sum(stack.volumes_m3)),
        "minimum_temperature_K": float(min(temperatures)),
        "maximum_temperature_K": float(max(temperatures)),
        "negative_mean_temperature_K": data["negative"]["mean_temperature"],
        "positive_mean_temperature_K": data["positive"]["mean_temperature"],
        "generated_heat_W": generated,
        "boundary_heat_loss_W": losses,
        "local_temperature_rates_K_s": local_rates,
        "mean_temperature_rate_K_s": mean_rate,
        "voltage_V": reading.voltage_V,
        "current_A": current_A,
    }


def spatial_thermal_field_ledger(spec, state, *, current_A=0.0, program=None):
    """Return a JSON-safe read-only thermal/electrical field snapshot.

    The snapshot is derived from the accepted coupled state and never feeds a
    corrected temperature or heat term back into the solver. Local fields,
    ambient losses, electrode means and the battery readout stay together so
    event endpoints can be audited without NumPy objects.
    """
    validate_spatial_thermal_state(spec, state)
    fields = spatial_thermal_fields(spec, state, current_A=current_A, program=program)
    temperatures = tuple(float(v) for v in fields["temperature_K"])
    local_rates = tuple(float(v) for v in fields["local_temperature_rates_K_s"])
    return {
        "time_s": float(state.time_s),
        "temperature_K": temperatures,
        "local_temperature_rates_K_s": local_rates,
        "heat_capacity_mean_temperature_K": float(fields["heat_capacity_mean_temperature_K"]),
        "volume_mean_temperature_K": float(fields["volume_mean_temperature_K"]),
        "minimum_temperature_K": float(fields["minimum_temperature_K"]),
        "maximum_temperature_K": float(fields["maximum_temperature_K"]),
        "ambient_temperature_K": float(spec.heat_stack.ambient_temperature_K),
        "generated_heat_W": float(np.sum(fields["generated_heat_W"])),
        "boundary_heat_loss_W": float(np.sum(fields["boundary_heat_loss_W"])),
        "mean_temperature_rate_K_s": float(fields["mean_temperature_rate_K_s"]),
        "internal_energy_J": float(spatial_thermal_internal_energy_j(spec, state)),
        "current_A": float(fields["current_A"]),
        "voltage_V": float(fields["voltage_V"]),
        "negative_mean_temperature_K": float(fields["negative_mean_temperature_K"]),
        "positive_mean_temperature_K": float(fields["positive_mean_temperature_K"]),
    }


def spatial_thermal_chemical_potentials(spec, state):
    return {
        k: -F * spec.ocv(name, 0.5)
        for name in ("negative", "positive")
        for k in getattr(spec, name).keys(name)
    }


def spatial_thermal_chemical_energy_j(spec, state):
    mu = spatial_thermal_chemical_potentials(spec, state)
    return sum(mu[k] * state.amounts_mol[k] for k in spec.inventory_keys)


def spatial_thermal_internal_energy_j(spec, state):
    return spatial_thermal_chemical_energy_j(
        spec, state
    ) + spec.heat_stack.total_heat_capacity_J_K * (state.temperature_K - spec.temperature_K)


class _SpatialThermodynamics:
    def __init__(self, spec):
        self.spec, self.inventory_keys = spec, spec.inventory_keys

    def heat_capacity_j_k(self, state):
        return self.spec.heat_stack.total_heat_capacity_J_K

    def internal_energy_j(self, state):
        return spatial_thermal_internal_energy_j(self.spec, state)

    def chemical_energy_rate_w(self, state, derivatives, contributions):
        if any(c.legacy_reaction_energy_W is not None for c in contributions):
            raise ValueError("spatial battery cannot double-count chemical heat")
        mu = spatial_thermal_chemical_potentials(self.spec, state)
        return sum(mu[k] * derivatives[k] for k in self.inventory_keys)


def _rhs_jacobian(spec, state, program, identities):
    index = {k: i for i, k in enumerate(identities)}
    size = len(identities)
    ik, vk = BATTERY_ALGEBRAIC
    current = state.algebraic_values[ik]
    stack = spec.heat_stack
    temperatures, data = _electrical(spec, state, current)
    dtemp = np.zeros((len(temperatures), size))
    tcols = [index[k] for k in ("temperature", *spec.thermal_coordinate_keys)]
    dtemp[:, tcols] = stack.temperature_matrix
    jac, dq, dv = np.zeros((size, size)), np.zeros_like(dtemp), np.zeros(size)
    for name, sign in (("negative", 1), ("positive", -1)):
        particle, d = getattr(spec, name), data[name]
        keys = particle.keys(name)
        cols = [index[k] for k in keys]
        jac[np.ix_(cols, cols)] = ConservativeDiffusion(
            particle.mesh, particle.diffusivity_m2_s
        ).inventory_jacobian_s_inv.toarray()
        jac[index[keys[-1]], index[ik]] = -sign / F
        da = np.zeros(size)
        da[index[ik]], da[index[keys[-1]]] = d["alpha_i"], d["alpha_amount"]
        sel, cells = stack.electrode_slice(name), getattr(stack, name).cells
        dv -= sign * (d["mean_temperature"] * da + d["alpha"] * np.mean(dtemp[sel], axis=0))
        dq[sel] += sign * current * d["alpha"] / cells * dtemp[sel]
        dq[sel] += sign * current / cells * np.outer(temperatures[sel], da)
        dq[sel, index[ik]] += sign * d["alpha"] / cells * temperatures[sel]
    # Pair the same face flux derivatives used in the RHS. The mean column
    # cancels exactly before multiplication by large thermal conductances.
    face_jac = stack.conductance_W_K[:, None] * np.diff(dtemp, axis=0)
    conduction_jac = np.zeros_like(dtemp)
    conduction_jac[:-1] += face_jac
    conduction_jac[1:] -= face_jac
    dl = (
        conduction_jac + dq - stack.loss_conductance_W_K[:, None] * dtemp
    ) / stack.heat_capacities_J_K[:, None]
    dm = (np.sum(dq, axis=0) - stack.loss_conductance_W_K @ dtemp) / stack.total_heat_capacity_J_K
    jac[index["temperature"]] = dm
    jac[[index[k] for k in spec.thermal_coordinate_keys]] = dl[:-1] - dm
    jac[index["discharged_charge_C"], index[ik]] = 1
    jac[index["delivered_work_J"]] = current * dv
    jac[index["delivered_work_J"], index[ik]] += spatial_thermal_readout(
        spec, state, current
    ).voltage_V
    # The shared residual also packs quadrature receipts. Their derivatives are
    # part of the same Newton system even though none is a new physical state.
    if "chemical_energy" in index:
        jac[index["chemical_energy"], index[ik]] = spec.ocv("negative", 0.5) - spec.ocv(
            "positive", 0.5
        )
    if "heat_loss" in index:
        jac[index["heat_loss"]] = stack.loss_conductance_W_K @ dtemp
    if "energy:battery:terminal" in index:
        jac[index["energy:battery:terminal"]] = -jac[index["delivered_work_J"]]
    mode, _ = spatial_thermal_control_mode(spec, state, program)
    if mode == "constant_voltage":
        jac[index[ik], index[vk]] = -1
    else:
        jac[index[ik], index[ik]] = -1 / spec.maximum_current_A
    jac[index[vk]] = dv
    jac[index[vk], index[vk]] -= 1
    return sparse.csc_matrix(jac)


def spatial_thermal_system(spec, program, *, thermal_device=None, control_events=()):
    if thermal_device is not None:
        raise ValueError("spatial thermal battery has no declared external thermal contact")
    closure = spatial_thermal_algebraic(spec, program)
    processes = [
        ConservativeDiffusion(p.mesh, p.diffusivity_m2_s)
        .couple(p.keys(name), process_id=f"battery:{name}", owner_id=name)
        .process
        for name, p in (("negative", spec.negative), ("positive", spec.positive))
    ]
    nk, pk = spec.negative.keys("negative"), spec.positive.keys("positive")

    def fields(state):
        return spatial_thermal_fields(spec, state, program=program)

    def transfer(state):
        f = fields(state)
        i, v = f["current_A"], f["voltage_V"]
        source, target = (nk[-1], pk[-1]) if i >= 0 else (pk[-1], nk[-1])
        return ProcessContribution(
            internal_transfers=(InternalTransfer("battery:Li", source, target, abs(i) / F),),
            coordinate_rates={
                "discharged_charge_C": i,
                "delivered_work_J": i * v,
                **dict(
                    zip(
                        spec.thermal_coordinate_keys,
                        f["local_temperature_rates_K_s"][:-1] - f["mean_temperature_rate_K_s"],
                        strict=True,
                    )
                ),
            },
            energy_port_rates_W={"battery:terminal": -i * v},
        )

    processes.append(
        CoupledProcess(
            "battery:spatial-thermal-intercalation",
            transfer,
            validate_state=lambda state: validate_spatial_thermal_state(spec, state),
        )
    )
    events = []
    for name, sign in (("negative", 1), ("positive", -1)):

        def surface(state, n=name, sg=sign):
            return particle_surface_fraction(
                getattr(spec, n),
                n,
                state,
                sg * spatial_thermal_control_current(spec, state, program) / F,
            )

        events.extend(
            (
                CoupledEvent(
                    f"battery:{name}:depleted", lambda s, f=surface: f(s) - spec.surface_margin
                ),
                CoupledEvent(
                    f"battery:{name}:saturated", lambda s, f=surface: 1 - spec.surface_margin - f(s)
                ),
            )
        )
    for j, name in enumerate(spec.heat_stack.node_ids):
        events.extend(
            (
                CoupledEvent(
                    f"battery:thermal:{name}:minimum-temperature",
                    lambda s, i=j: (
                        spec.heat_stack.temperatures(s.temperature_K, s.coordinates)[i]
                        - spec.thermal.minimum_temperature_K
                    ),
                ),
                CoupledEvent(
                    f"battery:thermal:{name}:maximum-temperature",
                    lambda s, i=j: (
                        spec.thermal.maximum_temperature_K
                        - spec.heat_stack.temperatures(s.temperature_K, s.coordinates)[i]
                    ),
                ),
            )
        )
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
    return CoupledSystem(
        _SpatialThermodynamics(spec),
        tuple(processes),
        isothermal=False,
        heat_flux=lambda s: HeatFlux(loss_W=float(sum(fields(s)["boundary_heat_loss_W"]))),
        energy_ports=(EnergyPort("battery:terminal", "external_source", "battery", "in"),),
        events=(*events, *control_events),
        coordinates=(
            CoupledCoordinate("discharged_charge_C", "C", spec.usable_charge_c),
            CoupledCoordinate("delivered_work_J", "J", spec.usable_charge_c * 4),
            *(CoupledCoordinate(k, "K", 10, minimum=None) for k in spec.thermal_coordinate_keys),
        ),
        inventory_scales_mol={
            k: p.capacity_mol / p.cells
            for name, p in (("negative", spec.negative), ("positive", spec.positive))
            for k in p.keys(name)
        },
        conservation_laws={
            "lithium_mol": dict.fromkeys(spec.inventory_keys, 1),
            "negative_faradaic_C": {**dict.fromkeys(nk, F), "discharged_charge_C": 1},
            "positive_faradaic_C": {**dict.fromkeys(pk, F), "discharged_charge_C": -1},
        },
        numerical_form="residual",
        algebraic=closure,
        residual_rhs_jacobian=lambda s, ids: _rhs_jacobian(spec, s, program, ids),
    )


def advance_spatial_thermal(
    spec,
    state,
    program,
    duration_s,
    *,
    evaluation_times_s=None,
    thermal_device=None,
    control_events=(),
):
    result = spatial_thermal_system(
        spec, program, thermal_device=thermal_device, control_events=control_events
    ).advance(state, duration_s, evaluation_times_s=evaluation_times_s)
    f = spatial_thermal_fields(spec, result.final_state, program=program)
    return replace(
        result,
        solver_diagnostic={
            **result.solver_diagnostic,
            "energy_scope": ENERGY_SCOPE,
            "battery_control_mode": spatial_thermal_control_mode(spec, result.final_state, program)[
                0
            ],
            **{
                k: f[k]
                for k in (
                    "minimum_temperature_K",
                    "maximum_temperature_K",
                    "heat_capacity_mean_temperature_K",
                    "volume_mean_temperature_K",
                )
            },
        },
    )
