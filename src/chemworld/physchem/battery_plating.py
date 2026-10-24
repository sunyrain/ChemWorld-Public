"""Finite metallic lithium coupled to the original particle-electrode circuit.

The reversible/irreversible kinetics follow the actual PyBaMM 25.4.2 plating
submodel and the constant-rate O'Kane parameter form. Positive side current
strips metal. No state is inferred from SOC, and no independent solver or
post-step loss correction is used. The first scope is isothermal single-valued
SPM or MPM with a fixed electrolyte concentration and fixed negative geometry.
"""

from dataclasses import dataclass, field, replace
from functools import cached_property
from math import exp, isclose, isfinite
from types import MappingProxyType

import numpy as np

from chemworld.data.scientific_inputs import ScientificTable, _keys, _number, _text
from chemworld.foundation.units import convert_value
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.electrochemistry import FARADAY_C_PER_MOL as F
from chemworld.physchem.electrochemistry import R_J_PER_MOL_K as R
from chemworld.physchem.residual_system import ResidualEquation, ResidualVariable

PLATING_VERSION = "pybamm-25.4.2-spm-plating-v1"
PARTIAL_PLATING_VERSION = "pybamm-25.4.2-constant-decay-plating-v1"
PLATING_SPECIES = {"plating:Li": "Li", "dead:Li": "Li"}
PLATING_UNITS = {
    "k_m_s": "m/s",
    "alpha": "dimensionless",
    "ce_mol_m3": "mol/m^3",
    "negative_electrode_volume_m3": "m^3",
}
PARTIAL_PLATING_UNITS = {**PLATING_UNITS, "gamma_s_inv": "1/s"}
STRIPPING_CURRENT = "battery:plating:stripping-current_A"
PLATING_PHI_E = "battery:plating:electrolyte-potential_V"


@dataclass(frozen=True)
class LithiumPlatingSpec:
    parameters: ScientificTable
    parameter_row_id: str
    mode: str = "reversible"
    initial_plated_mol: float = 0.0
    initial_dead_mol: float = 0.0
    _native: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.mode not in {"reversible", "irreversible", "partially reversible"}:
            raise ValueError("unknown plating kinetic mode")
        if not isinstance(self.parameters, ScientificTable) or self.parameters.kind != "parameters":
            raise ValueError("plating requires a sourced parameter table")
        _text(self.parameter_row_id, "parameter_row_id")
        rows = {row.row_id: row for row in self.parameters.rows}
        if self.parameter_row_id not in rows:
            raise ValueError("plating parameter row does not exist")
        cells = rows[self.parameter_row_id].values
        units = PARTIAL_PLATING_UNITS if self.mode == "partially reversible" else PLATING_UNITS
        if cells.keys() != units.keys():
            raise ValueError("plating parameters must match the declared mode's parameter columns")
        if any(v.quality != "valid" or v.value is None for v in cells.values()):
            raise ValueError("plating requires valid finite parameters")
        values = {key: convert_value(cells[key].value, cells[key].unit, unit)
                  for key, unit in units.items()}
        if (any(not isfinite(v) for v in values.values())
                or values["k_m_s"] < 0 or not 0 < values["alpha"] < 1
                or values["ce_mol_m3"] <= 0 or values["negative_electrode_volume_m3"] <= 0):
            raise ValueError("invalid plating rate, transfer coefficient, concentration or volume")
        if values.get("gamma_s_inv", 0.0) < 0:
            raise ValueError("constant metallic lithium decay rate must be nonnegative")
        for name in ("initial_plated_mol", "initial_dead_mol"):
            value = _number(getattr(self, name), name)
            if value < 0:
                raise ValueError("initial metallic lithium amounts must be nonnegative")
            object.__setattr__(self, name, value)
        if not isfinite(self.initial_plated_mol / values["negative_electrode_volume_m3"]):
            raise ValueError("initial plated concentration is not representable")
        if not isfinite(values.get("gamma_s_inv", 0.0) * self.initial_plated_mol):
            raise ValueError("initial metallic lithium deactivation rate is not representable")
        object.__setattr__(self, "_native", MappingProxyType(values))

    @property
    def native_parameters(self):
        return self._native

    def initial_amounts(self):
        return {"plating:Li": self.initial_plated_mol, "dead:Li": self.initial_dead_mol}

    @property
    def decay_rate_s_inv(self):
        return self._native.get("gamma_s_inv", 0.0)

    def deactivation_rate_mol_s(self, plated_mol):
        """Declared constant-rate metal transfer, independent of electron current."""
        if not isfinite(plated_mol) or plated_mol < 0:
            raise ValueError("metal deactivation requires finite nonnegative plated lithium")
        rate = self.decay_rate_s_inv * plated_mol
        if not isfinite(rate):
            raise ValueError("metallic lithium deactivation rate is not representable")
        return rate

    def current_derivatives(self, potential_V, plated_mol, temperature_K, area_m2):
        """Return q_strip and partials w.r.t. potential, plated mol and T.

        The concentration denominator is the entire negative electrode's fixed
        volume. The multiplier is active-particle surface area, never collector
        area. Invalid/overflowing evaluations are refused, not clipped.
        """
        if (not all(isfinite(v) for v in (potential_V, plated_mol, temperature_K, area_m2))
                or plated_mol < 0 or temperature_K <= 0 or area_m2 <= 0):
            raise ValueError("plating kinetics requires finite potential and physical inventory")
        k, alpha, ce, volume = (self._native[name] for name in PLATING_UNITS)
        if k == 0:
            return 0.0, 0.0, 0.0, 0.0
        factor = F / (R * temperature_K)
        gain = area_m2 * F * k
        try:
            strip_exp = (exp((1 - alpha) * factor * potential_V)
                         if self.mode in {"reversible", "partially reversible"} else 0.0)
            plate_exp = exp(-alpha * factor * potential_V)
            strip = gain * plated_mol / volume * strip_exp
            plate = gain * ce * plate_exp
            value = strip - plate
            dpotential = factor * ((1 - alpha) * strip + alpha * plate)
            dplated = gain / volume * strip_exp
            dtemperature = -potential_V / temperature_K * dpotential
        except OverflowError as exc:
            raise ValueError(
                "plating kinetic response is outside the representable domain") from exc
        if not all(isfinite(v) for v in (value, dpotential, dplated, dtemperature)):
            raise ValueError("plating kinetic response is outside the representable domain")
        return value, dpotential, dplated, dtemperature

    def to_dict(self):
        version = (PARTIAL_PLATING_VERSION if self.mode == "partially reversible"
                   else PLATING_VERSION)
        return {"model_version": version, "parameters": self.parameters.to_dict(),
                "parameter_row_id": self.parameter_row_id, "mode": self.mode,
                "initial_plated_mol": self.initial_plated_mol,
                "initial_dead_mol": self.initial_dead_mol}

    @classmethod
    def from_dict(cls, value):
        _keys(value, ("model_version", "parameters", "parameter_row_id", "mode",
                      "initial_plated_mol", "initial_dead_mol"))
        version = (PARTIAL_PLATING_VERSION if value["mode"] == "partially reversible"
                   else PLATING_VERSION)
        if value["model_version"] != version:
            raise ValueError("plating equation version does not match its declared mode")
        return cls(ScientificTable.from_dict(value["parameters"]), value["parameter_row_id"],
                   value["mode"], value["initial_plated_mol"], value["initial_dead_mol"])


def _nonincreasing_ocv(spec, electrode):
    curve = getattr(spec, electrode + "_ocv")
    if curve is None:
        return getattr(spec, electrode + "_ocv_slope_V") <= 0
    if (not any(curve.polynomial_V[1:])
            and all(a * k == 0 for a, k, _ in curve.exponential_terms)
            and all(a * k == 0 for a, k, _ in curve.tanh_terms)):
        return True
    lo, hi = curve.stoichiometry_domain
    return all(curve.enclosure(lo+(hi-lo)*i/64, lo+(hi-lo)*(i+1)/64,
                               derivative=True).upper <= 0 for i in range(64))


def validate_plating_configuration(spec):
    """Declare the actual supported owner and certify the passive closure."""
    from chemworld.physchem.battery_models import BatterySPMSpec
    from chemworld.physchem.battery_mpm import BatteryMPMSpec

    if (type(spec) not in (BatterySPMSpec, BatteryMPMSpec)
            or not isinstance(spec.plating, LithiumPlatingSpec)
            or spec.electrolyte is not None or spec.thermal is not None or spec.sei is not None
            or spec.numerical_form != "residual"):
        raise ValueError("plating requires isothermal no-SEI residual SPM or MPM")
    if spec.has_directional_ocp and (
            type(spec) is not BatterySPMSpec or spec.plating.mode != "reversible"
            or any(c is not None and c.mode not in {"single", "axen"}
                   for c in (spec.negative_ocp, spec.positive_ocp))):
        raise ValueError("directional plating requires SPM Axen with reversible plating")
    if spec.plating.mode == "partially reversible" and type(spec) is not BatterySPMSpec:
        raise ValueError("constant-decay partially reversible plating currently requires SPM")
    volume = spec.plating.native_parameters["negative_electrode_volume_m3"]
    if volume < spec.negative.active_volume_m3:
        raise ValueError("negative active volume cannot exceed plating electrode geometry")
    if (not isfinite((volume + spec.positive.active_volume_m3) * 1000)
            or not isfinite(spec.usable_charge_c) or spec.usable_charge_c <= 0):
        raise ValueError("plating geometry and charge scale must be finite and positive")
    total_negative = (spec.negative.capacity_mol * spec.initial_negative_fraction
                      + sum(spec.plating.initial_amounts().values()))
    if not isfinite(F * total_negative):
        raise ValueError("plating initial Faradaic inventory is not representable")
    for name in ("negative", "positive"):
        particles = [p for n, _, p, _, _ in _plating_bins(spec) if n == name]
        if (any(not isfinite(p.area_m2) or p.area_m2 <= 0
                or not isfinite(p.capacity_mol) or p.capacity_mol <= 0 for p in particles)
                or not isfinite(sum(p.area_m2 for p in particles))):
            raise ValueError("plating requires representable particle area and capacity")
        ocp = getattr(spec, name + "_ocp")
        monotone = (ocp.cv_monotone_certified if ocp is not None and ocp.mode == "axen"
                    else _nonincreasing_ocv(spec, name))
        if not monotone:
            raise ValueError("plating current closure requires certified nonincreasing OCV")


def _plating_bins(spec):
    if getattr(spec, "model_family", None) == "mpm-isothermal-v1":
        return spec.particle_bins
    return tuple((name, 0, getattr(spec, name), getattr(spec, name).keys(name), name)
                 for name in ("negative", "positive"))


def validate_plating_state(spec, state):
    if (set(state.amounts_mol) != set(spec.inventory_keys)
            or set(state.coordinates) != set(spec.coordinate_keys)):
        raise ValueError("plating state must retain both metal inventories, both particles and Q/W")
    if (not isclose(state.temperature_K, spec.temperature_K, rel_tol=0, abs_tol=1e-8)
            or not isclose(state.volume_L, spec.volume_l, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("plating state requires its fixed temperature and declared volume")
    if any(not isfinite(v) or v < 0 for v in state.amounts_mol.values()):
        raise ValueError("plating inventories must be finite and nonnegative")
    if any(not isfinite(v) for v in state.coordinates.values()):
        raise ValueError("plating charge/work must be finite")
    if spec.has_directional_ocp:
        from chemworld.physchem.battery_ocp import validate_hysteresis

        validate_hysteresis(spec, state)
    charge = state.coordinates["discharged_charge_C"]
    for name, sign in (("negative", 1), ("positive", -1)):
        total = 0.0
        for n, _, particle, keys, _ in _plating_bins(spec):
            if n != name:
                continue
            amounts = [state.amounts_mol[key] for key in keys]
            if any(v > cap*(1+1e-8) for v, cap in zip(
                    amounts, particle.mesh.cell_volumes_m3*particle.maximum_concentration_mol_m3,
                    strict=True)):
                raise ValueError("plating particle inventory exceeds finite capacity")
            total += sum(amounts)
        initial = getattr(spec, name).capacity_mol * getattr(spec, "initial_"+name+"_fraction")
        if name == "negative":
            total += sum(state.amounts_mol[k] for k in PLATING_SPECIES)
            initial += sum(spec.plating.initial_amounts().values())
        if not isclose(F*total+sign*charge, F*initial, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("plating inventories disagree with original Faradaic charge")
    if spec.plating.mode == "partially reversible" and spec.plating.decay_rate_s_inv > 0:
        initial_dead = spec.plating.initial_dead_mol
        if state.amounts_mol["dead:Li"] < initial_dead - (1e-14 + 1e-10*initial_dead):
            raise ValueError("inactive lithium cannot fall below its declared initial inventory")
        spec.plating.deactivation_rate_mol_s(state.amounts_mol["plating:Li"])
        return
    fixed = ("dead:Li" if spec.plating.mode in {"reversible", "partially reversible"}
             else "plating:Li")
    if not isclose(state.amounts_mol[fixed], spec.plating.initial_amounts()[fixed],
                   rel_tol=1e-10, abs_tol=1e-14):
        raise ValueError("metal inventory changes in an unsupported plating mode")


class _PlatingSideReaction:
    current_key = STRIPPING_CURRENT

    def __init__(self, spec):
        self.spec = spec

    @property
    def variable(self):
        return ResidualVariable(self.current_key, "A", self.spec.maximum_current_A, "algebraic")

    @property
    def equation(self):
        return ResidualEquation("battery:plating:kinetics", "dimensionless")

    @property
    def inventory_scales(self):
        return dict.fromkeys(PLATING_SPECIES, self.spec.negative.capacity_mol)

    @property
    def amount_current_derivatives(self):
        return {"plating:Li": -1/F if self.spec.plating.mode in {
                    "reversible", "partially reversible"} else 0.0,
                "dead:Li": -1/F if self.spec.plating.mode == "irreversible" else 0.0}

    @property
    def amount_state_derivatives(self):
        gamma = self.spec.plating.decay_rate_s_inv
        if gamma == 0:
            return {}
        return {("plating:Li", "plating:Li"): -gamma,
                ("dead:Li", "plating:Li"): gamma}

    def amount_rates(self, fields):
        rates = {key: value*fields["stripping_current_A"]
                 for key, value in self.amount_current_derivatives.items()}
        if self.spec.plating.decay_rate_s_inv != 0:
            rate = self.spec.plating.deactivation_rate_mol_s(fields["plated_mol"])
            rates["plating:Li"] -= rate
            rates["dead:Li"] += rate
        return rates

    def fields(self, state, values, potential_key):
        result = {"stripping_current_A": values[self.current_key],
                "plated_mol": state.amounts_mol["plating:Li"],
                "dead_mol": state.amounts_mol["dead:Li"],
                "negative_electrode_potential_V": -values[potential_key]}
        if self.spec.plating.mode == "partially reversible":
            result["deactivation_rate_mol_s"] = self.spec.plating.deactivation_rate_mol_s(
                result["plated_mol"])
        return result

    def derivatives(self, state, fields):
        return self.spec.plating.current_derivatives(
            fields["negative_electrode_potential_V"], state.amounts_mol["plating:Li"],
            state.temperature_K,
            sum(p.area_m2 for n, _, p, _, _ in _plating_bins(self.spec) if n == "negative"))

    def residual(self, state, fields):
        value = self.derivatives(state, fields)[0]
        return (fields["stripping_current_A"]-value) / self.spec.maximum_current_A

    def closure_jacobian(self, state, fields, identities, potential_key):
        _, du, dn, dt = self.derivatives(state, fields)
        index = {key: i for i, key in enumerate(identities)}
        row = np.zeros(len(identities))
        row[index[self.current_key]] = 1 / self.spec.maximum_current_A
        row[index[potential_key]] = du / self.spec.maximum_current_A
        if "plating:Li" in index:
            row[index["plating:Li"]] = -dn / self.spec.maximum_current_A
        if "temperature" in index:
            row[index["temperature"]] = -dt / self.spec.maximum_current_A
        return row


class _PlatingSPMAdapter:
    """A single bin per original SPM particle, not a new persistent model."""

    def __init__(self, spec):
        validate_plating_configuration(spec)
        self.spec = spec

    def __getattr__(self, name):
        return getattr(self.spec, name)

    @cached_property
    def particle_bins(self):
        return tuple((name, 0, getattr(self.spec, name), getattr(self.spec, name).keys(name), name)
                     for name in ("negative", "positive"))

    counts = (1, 1)
    current_coordinates = ((0, (), ()), (0, (), ()))
    reaction_keys = ()
    electrolyte_potential_key = PLATING_PHI_E
    closure_id = "battery:plating-closure"
    intercalation_id = "battery:plating-intercalation"
    population_equations = ("battery:bv:negative", "battery:bv:positive")

    @cached_property
    def algebraic_names(self):
        from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC

        return (STRIPPING_CURRENT, PLATING_PHI_E, *BATTERY_ALGEBRAIC)

    @cached_property
    def particle_side_reaction(self):
        return _PlatingSideReaction(self.spec)

    @cached_property
    def reaction_map(self):
        from chemworld.physchem import battery_particle_electrodes as shared

        return shared.reaction_map(self)

    def population_ocp_derivatives(self, population, theta, current, state):
        if self.spec.has_directional_ocp:
            from chemworld.physchem.battery_ocp import hysteresis_key

            name = population[0]
            value, slope, di, dh = self.spec.ocp_derivatives(name, theta, current, state)
            coordinates = ({hysteresis_key(name): dh}
                           if hysteresis_key(name) in self.spec.hysteresis_keys else {})
            return value, slope, di, coordinates
        return (self.spec.ocv(population[0], theta),
                self.spec.ocv(population[0], theta, derivative=True), 0.0, {})

    def electrode_initial_potential(self, name, state, current):
        particle = getattr(self.spec, name)
        theta = sum(state.amounts_mol[k] for k in particle.keys(name)) / particle.capacity_mol
        if self.spec.has_directional_ocp:
            return self.spec.operational_ocp(name, theta, current, state)
        return self.spec.ocv(name, theta)

    def validate_particle_owner(self, state):
        validate_plating_state(self.spec, state)

    def particle_events(self, fields):
        from chemworld.physchem import battery_particle_electrodes as shared

        # Zero plated/dead Li is an admissible state, not a stopping event.
        return shared.legacy_particle_events(self, fields)


@dataclass(frozen=True)
class PlatingReadout:
    voltage_V: float
    current_A: float
    negative_surface_fraction: float
    positive_surface_fraction: float
    negative_average_fraction: float
    positive_average_fraction: float
    negative_overpotential_V: float
    positive_overpotential_V: float
    stripping_current_A: float
    negative_intercalation_current_A: float
    plated_mol: float
    dead_mol: float
    negative_electrode_potential_V: float


def _plating_owner(spec):
    validate_plating_configuration(spec)
    if getattr(spec, "model_family", None) == "mpm-isothermal-v1":
        return spec
    return _PlatingSPMAdapter(spec)


def plating_fields(spec, state, *, current_A=None, program=None):
    from chemworld.physchem import battery_particle_electrodes as shared

    if isinstance(state, CoupledState):
        validate_plating_state(spec, state)
    fields = shared.particle_fields(_plating_owner(spec), state,
                                    current_A=current_A, program=program)
    bins = _plating_bins(spec)
    names = tuple(b[0] for b in bins)
    volumes = np.array([p.active_volume_m3 / getattr(spec, n).active_volume_m3
                        for n, _, p, _, _ in bins])
    area_weights = np.empty(len(bins))
    for name in ("negative", "positive"):
        mask = np.array([n == name for n in names])
        area_weights[mask] = fields["area_m2"][mask] / sum(fields["area_m2"][mask])
    negative = np.array([n == "negative" for n in names])
    return {**fields,
            "negative_intercalation_current_A": float(sum(fields["reaction_current_A"][negative])),
            "particle_electrodes": names,
            "particle_radii_m": np.array([p.radius_m for _, _, p, _, _ in bins]),
            "particle_volume_weights": volumes, "particle_area_weights": area_weights,
            "negative_active_area_m2": float(sum(fields["area_m2"][negative]))}


def plating_readout(spec, state, current_A):
    f = plating_fields(spec, state, current_A=current_A)
    if getattr(spec, "model_family", None) != "mpm-isothermal-v1":
        # Retain the established one-particle readout arithmetic exactly.
        return PlatingReadout(f["voltage_V"], f["current_A"], *map(float, f["surface_fraction"]),
                             *map(float, f["average_fraction"]),
                             *map(float, f["kinetic_overpotential_V"]), f["stripping_current_A"],
                             f["negative_intercalation_current_A"], f["plated_mol"], f["dead_mol"],
                             f["negative_electrode_potential_V"])
    surface, average, eta = [], [], []
    bins = _plating_bins(spec)
    for name in ("negative", "positive"):
        mask = np.array([n == name for n, *_ in bins])
        areas = f["area_m2"][mask]
        capacities = [p.capacity_mol for n, _, p, _, _ in bins if n == name]
        surface.append(float(np.average(f["surface_fraction"][mask], weights=areas)))
        average.append(float(np.average(f["average_fraction"][mask], weights=capacities)))
        eta.append(float(np.average(f["kinetic_overpotential_V"][mask], weights=areas)))
    return PlatingReadout(f["voltage_V"], f["current_A"], *surface, *average, *eta,
                         f["stripping_current_A"],
                         f["negative_intercalation_current_A"], f["plated_mol"], f["dead_mol"],
                         f["negative_electrode_potential_V"])


def plating_particle_flux(spec, state, name, current_A):
    """Total outward mol/s for an electrode; individual bin fluxes remain in fields."""
    if name not in {"negative", "positive"}:
        raise ValueError("unknown plating electrode")
    f = plating_fields(spec, state, current_A=current_A)
    currents = zip(f["reaction_current_A"], f["particle_electrodes"], strict=True)
    return float(sum(q for q, n in currents if n == name)) / F


def plating_algebraic(spec, program):
    from chemworld.physchem import battery_particle_electrodes as shared

    return shared.particle_algebraic(_plating_owner(spec), program)


def plating_control_current(spec, state, program):
    return float(plating_fields(spec, state, program=program)["current_A"])


def plating_control_mode(spec, state, program):
    target = program.value_at(state.time_s)
    f = plating_fields(spec, state, program=program)
    if program.unit == "A":
        return "constant_current", target
    if f["voltage_V"] < target-1e-9:
        return "negative_current_limit", -spec.maximum_current_A
    if f["voltage_V"] > target+1e-9:
        return "positive_current_limit", spec.maximum_current_A
    return "constant_voltage", target


def plating_system(spec, program, *, thermal_device=None, control_events=()):
    from chemworld.physchem import battery_particle_electrodes as shared

    return shared.particle_system(_plating_owner(spec), program,
                                  thermal_device=thermal_device, control_events=control_events)


def advance_plating(spec, state, program, duration_s, *, evaluation_times_s=None,
                    thermal_device=None, control_events=()):
    result = plating_system(spec, program, thermal_device=thermal_device,
                             control_events=control_events).advance(
        state, duration_s, evaluation_times_s=evaluation_times_s)
    mode, _ = plating_control_mode(spec, result.final_state, program)
    return replace(result, solver_diagnostic={**result.solver_diagnostic,
        "battery_control_mode": mode, "plating_mode": spec.plating.mode,
        "energy_scope": "isothermal; terminal electrical work only; plating energy is undeclared"})
