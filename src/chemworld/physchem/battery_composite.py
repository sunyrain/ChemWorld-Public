"""Two-material negative / single-material positive isothermal SPM.

The phase-resolved surface formulation follows PyBaMM 25.4.2 SPM (BSD-3-Clause).
Material inputs retain their own sources; no complete Chen composite material or
hysteresis model is implied. The original MPM particle-electrode residual owns
all diffusion, BV current sharing and time advancement.
"""

import re
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from math import isclose, isfinite

import numpy as np

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem import battery_particle_electrodes as shared
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import (
    BATTERY_ALGEBRAIC,
    BATTERY_COORDINATES,
    F,
    ParticleSpec,
)
from chemworld.physchem.coupled_system import CoupledEvent, CoupledState
from chemworld.physchem.residual_system import ResidualDomainError

COMPOSITE_FAMILY = "composite-spm-isothermal-v1"
PHASE_VERSION = "single-valued-active-material-phase-v1"


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _monotone(curve, lower, upper):
    # Exact constant expressions avoid outward roundoff being mistaken for a
    # physically positive slope; all nonconstant expressions use interval bounds.
    constant = (not any(curve.polynomial_V[1:])
                and all(a*k == 0 for a, k, _ in curve.exponential_terms)
                and all(a*k == 0 for a, k, _ in curve.tanh_terms))
    return constant or all(
        curve.enclosure(lower+(upper-lower)*i/64, lower+(upper-lower)*(i+1)/64,
                        derivative=True).upper <= 0 for i in range(64))


@dataclass(frozen=True)
class ActiveMaterialPhaseSpec:
    phase_id: str
    particle: ParticleSpec
    ocv: ElectrodeOCV
    initial_fraction: float
    source: InputSource

    def __post_init__(self):
        if not isinstance(self.phase_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}",
                                                                self.phase_id):
            raise ValueError("phase identity must be a stable nonempty identifier")
        if not isinstance(self.particle, ParticleSpec) or not isinstance(self.ocv, ElectrodeOCV):
            raise ValueError("each material phase requires its own ParticleSpec and ElectrodeOCV")
        if not isinstance(self.source, InputSource):
            raise ValueError("phase parameters require an explicit InputSource")
        fraction = _finite(self.initial_fraction, "initial material fraction")
        if not 0 < fraction < 1:
            raise ValueError("initial material fraction must be inside (0,1)")
        object.__setattr__(self, "initial_fraction", fraction)

    def to_dict(self):
        return {"model_version": PHASE_VERSION, "phase_id": self.phase_id,
                "particle": asdict(self.particle), "ocv": self.ocv.to_dict(),
                "initial_fraction": self.initial_fraction, "source": self.source.to_dict()}

    @classmethod
    def from_dict(cls, value):
        if (set(value) != {"model_version", "phase_id", "particle", "ocv",
                           "initial_fraction", "source"}
                or value["model_version"] != PHASE_VERSION):
            raise ValueError("unknown active-material phase fields or version")
        return cls(value["phase_id"], ParticleSpec(**value["particle"]),
                   ElectrodeOCV.from_dict(value["ocv"]), value["initial_fraction"],
                   InputSource.from_dict(value["source"]))


@dataclass(frozen=True)
class BatteryCompositeSPMSpec:
    negative_phases: tuple[ActiveMaterialPhaseSpec, ActiveMaterialPhaseSpec]
    positive_phase: ActiveMaterialPhaseSpec
    temperature_K: float = 298.15
    surface_margin: float = 0.01
    minimum_voltage_V: float = 2.5
    maximum_voltage_V: float = 4.5
    maximum_current_A: float = 2.0
    series_resistance_ohm: float = 0.0
    provenance: str = "declared-single-valued-composite-SPM-v1"

    def __post_init__(self):
        phases = tuple(ActiveMaterialPhaseSpec.from_dict(p) if isinstance(p, dict) else p
                       for p in self.negative_phases)
        positive = (ActiveMaterialPhaseSpec.from_dict(self.positive_phase)
                    if isinstance(self.positive_phase, dict) else self.positive_phase)
        if (len(phases) != 2 or not all(isinstance(p, ActiveMaterialPhaseSpec)
                                      for p in (*phases, positive))):
            raise ValueError("composite SPM requires exactly two negative and one positive phase")
        if phases[0].phase_id == phases[1].phase_id:
            raise ValueError("negative material phase identities must be unique")
        object.__setattr__(self, "negative_phases", phases)
        object.__setattr__(self, "positive_phase", positive)
        for name in ("temperature_K", "surface_margin", "minimum_voltage_V", "maximum_voltage_V",
                     "maximum_current_A", "series_resistance_ohm"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if (not 0 < self.surface_margin < .5 or self.temperature_K <= 0
                or self.maximum_current_A <= 0 or self.series_resistance_ohm < 0
                or self.minimum_voltage_V >= self.maximum_voltage_V
                or not isinstance(self.provenance, str) or not self.provenance.strip()):
            raise ValueError("invalid composite battery operating domain")
        for phase in self.phases:
            if ("SPM" not in phase.ocv.supported_models
                    or phase.ocv.stoichiometry_domain[0] > self.surface_margin
                    or phase.ocv.stoichiometry_domain[1] < 1-self.surface_margin
                    or not self.surface_margin < phase.initial_fraction < 1-self.surface_margin):
                raise ValueError("phase OCV/initial state must cover the declared SPM surfaces")
            if not _monotone(phase.ocv, self.surface_margin, 1-self.surface_margin):
                raise ValueError("phase OCV monotonicity is uncertified for unique current sharing")
        if not all(isfinite(p.particle.area_m2) and p.particle.area_m2 > 0
                   and isfinite(p.particle.capacity_mol) and p.particle.capacity_mol > 0
                   for p in self.phases):
            raise ValueError("derived phase area/capacity must be finite and positive")
        if not isfinite(self.usable_charge_c) or self.usable_charge_c <= 0:
            raise ValueError("derived composite charge scale must be finite and positive")

    @property
    def phases(self):
        return (*self.negative_phases, self.positive_phase)

    @property
    def model_family(self):
        return COMPOSITE_FAMILY

    model_id = model_family

    @property
    def numerical_form(self):
        return "residual"

    @property
    def thermal(self):
        return None

    sei = electrolyte = negative_ocp = positive_ocp = thermal

    @property
    def has_directional_ocp(self):
        return False

    @property
    def hysteresis_keys(self):
        return ()

    @property
    def coordinate_keys(self):
        return BATTERY_COORDINATES

    def initial_coordinates(self):
        return dict.fromkeys(BATTERY_COORDINATES, 0.0)

    @property
    def volume_l(self):
        return 1000*sum(p.particle.active_volume_m3 for p in self.phases)

    @property
    def usable_charge_c(self):
        return F*min(sum(p.particle.capacity_mol*p.initial_fraction for p in self.negative_phases),
                     self.positive_phase.particle.capacity_mol*(1-self.positive_phase.initial_fraction))

    def current_at_c_rate(self, rate):
        return _finite(rate, "C-rate")*self.usable_charge_c/3600

    @cached_property
    def particle_bins(self):
        return tuple((name, i, phase.particle,
                      phase.particle.keys(f"{name}:phase:{phase.phase_id}"),
                      f"{name}:phase:{phase.phase_id}")
                     for name, phases in (("negative", self.negative_phases),
                                          ("positive", (self.positive_phase,)))
                     for i, phase in enumerate(phases))

    @cached_property
    def inventory_species(self):
        return {k: "Li" for _, _, _, keys, _ in self.particle_bins for k in keys}

    @property
    def inventory_keys(self):
        return tuple(self.inventory_species)

    @property
    def counts(self):
        return (2, 1)

    @cached_property
    def current_coordinates(self):
        return shared.current_coordinates(self)

    @cached_property
    def reaction_keys(self):
        return tuple(f"battery:{name}:phase-current:{phases[i].phase_id}_A"
                     for name, phases, (_, free, _) in zip(
                         ("negative", "positive"), (self.negative_phases, (self.positive_phase,)),
                         self.current_coordinates, strict=True) for i in free)

    @property
    def electrolyte_potential_key(self):
        return "battery:composite:electrolyte-potential_V"

    @cached_property
    def algebraic_names(self):
        return (*self.reaction_keys, self.electrolyte_potential_key, *BATTERY_ALGEBRAIC)

    @cached_property
    def reaction_map(self):
        return shared.reaction_map(self)

    @property
    def population_equations(self):
        return tuple(f"battery:bv:{identity}" for *_, identity in self.particle_bins)

    @property
    def closure_id(self):
        return "battery:composite-closure"

    @property
    def intercalation_id(self):
        return "battery:composite-intercalation"

    def population_ocp_derivatives(self, population, theta, current, state):
        name, i, *_ = population
        phase = self.negative_phases[i] if name == "negative" else self.positive_phase
        return (phase.ocv.value(theta, trial=True),
                phase.ocv.derivatives(theta, trial=True)[1], 0.0, {})

    def electrode_initial_potential(self, name, state, current):
        bins = [p for p in self.particle_bins if p[0] == name]
        return sum(p.area_m2*self.population_ocp_derivatives(
            b, sum(state.amounts_mol[k] for k in keys)/p.capacity_mol, current, state)[0]
            for b in bins for _, _, p, keys, _ in (b,))/sum(b[2].area_m2 for b in bins)

    def initial_state(self):
        amounts = {}
        for phase, (_, _, particle, keys, _) in zip(self.phases, self.particle_bins, strict=True):
            amounts.update(zip(keys, particle.mesh.cell_volumes_m3
                               * particle.maximum_concentration_mol_m3*phase.initial_fraction,
                               strict=True))
        return CoupledState(amounts, self.temperature_K, self.volume_l,
                            coordinates=self.initial_coordinates())

    def validate_particle_owner(self, state):
        validate_composite_state(self, state)

    def particle_events(self, fields):
        events = []
        for i, (*_, identity) in enumerate(self.particle_bins):
            events.extend((
                CoupledEvent(f"battery:{identity}:depleted",
                    lambda s, j=i: fields(s)["surface_fraction"][j]-self.surface_margin),
                CoupledEvent(f"battery:{identity}:saturated",
                    lambda s, j=i: 1-self.surface_margin-fields(s)["surface_fraction"][j]),
            ))
        events.extend((
            CoupledEvent("battery:minimum-voltage",
                         lambda s: fields(s)["voltage_V"]-self.minimum_voltage_V),
            CoupledEvent("battery:maximum-voltage",
                         lambda s: self.maximum_voltage_V-fields(s)["voltage_V"]),
        ))
        return events

    def to_dict(self):
        return {"model_family": COMPOSITE_FAMILY,
                "negative_phases": [p.to_dict() for p in self.negative_phases],
                "positive_phase": self.positive_phase.to_dict(),
                **{key: getattr(self, key) for key in (
                    "temperature_K", "surface_margin", "minimum_voltage_V", "maximum_voltage_V",
                    "maximum_current_A", "series_resistance_ohm", "provenance")}}

    @classmethod
    def from_dict(cls, value):
        fields = {"model_family", "negative_phases", "positive_phase", "temperature_K",
                  "surface_margin", "minimum_voltage_V", "maximum_voltage_V", "maximum_current_A",
                  "series_resistance_ohm", "provenance"}
        if set(value) != fields or value["model_family"] != COMPOSITE_FAMILY:
            raise ValueError("unknown composite SPM fields or family")
        return cls(**{k: v for k, v in value.items() if k != "model_family"})


@dataclass(frozen=True)
class PhaseReadout:
    electrode: str
    phase_id: str
    surface_fraction: float
    average_fraction: float
    reaction_current_A: float
    inventory_mol: float
    kinetic_overpotential_V: float


@dataclass(frozen=True)
class CompositeReadout:
    voltage_V: float
    current_A: float
    phase_readouts: tuple[PhaseReadout, ...]


def validate_composite_state(spec, state):
    if (set(state.amounts_mol) != set(spec.inventory_keys)
            or set(state.coordinates) != set(spec.coordinate_keys)):
        raise ValueError("composite state must retain every phase inventory and original Q/W")
    if (not isclose(state.temperature_K, spec.temperature_K, abs_tol=1e-8, rel_tol=0)
            or not isclose(state.volume_L, spec.volume_l, abs_tol=1e-12, rel_tol=0)):
        raise ValueError("composite state must preserve declared isothermal volume")
    if any(not isfinite(v) for v in state.coordinates.values()):
        raise ValueError("composite charge/work must be finite")
    totals = {"negative": 0.0, "positive": 0.0}
    initial = {"negative": 0.0, "positive": 0.0}
    for phase, (name, _, particle, keys, _) in zip(spec.phases, spec.particle_bins, strict=True):
        for key, volume in zip(keys, particle.mesh.cell_volumes_m3, strict=True):
            value = state.amounts_mol[key]
            if not isfinite(value) or not 0 <= value <= (
                    volume*particle.maximum_concentration_mol_m3*(1+1e-8)):
                raise ValueError("phase inventory is outside finite material capacity")
        totals[name] += sum(state.amounts_mol[k] for k in keys)
        initial[name] += particle.capacity_mol*phase.initial_fraction
    charge = state.coordinates["discharged_charge_C"]
    for name, sign in (("negative", 1), ("positive", -1)):
        if not isclose(F*totals[name]+sign*charge, F*initial[name], rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("composite phase inventories disagree with original Faradaic charge")


def composite_fields(spec, state, *, current_A=None, program=None):
    if isinstance(state, CoupledState):
        validate_composite_state(spec, state)
        # A newly requested control must not reuse a previous stage's algebraic solution.
        state = replace(state, algebraic_values={})
    fields = shared.particle_fields(spec, state, current_A=current_A, program=program)
    if np.any(fields["surface_fraction"] < spec.surface_margin-1e-8) or np.any(
            fields["surface_fraction"] > 1-spec.surface_margin+1e-8):
        raise ResidualDomainError("composite electrical root is outside a material surface domain")
    return fields


def composite_readout(spec, state, current_A):
    fields = composite_fields(spec, state, current_A=current_A)
    return CompositeReadout(fields["voltage_V"], fields["current_A"], tuple(
        PhaseReadout(name, phase.phase_id, float(fields["surface_fraction"][i]),
                     float(fields["average_fraction"][i]), float(fields["reaction_current_A"][i]),
                     sum(state.amounts_mol[k] for k in keys),
                     float(fields["kinetic_overpotential_V"][i]))
        for i, (phase, (name, _, _, keys, _)) in enumerate(
            zip(spec.phases, spec.particle_bins, strict=True))))


def composite_algebraic(spec, program):
    return shared.particle_algebraic(spec, program)


def composite_control_current(spec, state, program):
    return float(composite_fields(spec, state, program=program)["current_A"])


def composite_control_mode(spec, state, program):
    target = program.value_at(state.time_s)
    fields = composite_fields(spec, state, program=program)
    if program.unit == "A":
        return "constant_current", target
    if fields["voltage_V"] < target-1e-9:
        return "negative_current_limit", -spec.maximum_current_A
    if fields["voltage_V"] > target+1e-9:
        return "positive_current_limit", spec.maximum_current_A
    return "constant_voltage", target


def composite_system(spec, program, *, thermal_device=None, control_events=()):
    return shared.particle_system(spec, program, thermal_device=thermal_device,
                                  control_events=control_events)


def advance_composite(spec, state, program, duration_s, *, evaluation_times_s=None,
                      thermal_device=None, control_events=()):
    result = composite_system(spec, program, thermal_device=thermal_device,
                               control_events=control_events).advance(
        state, duration_s, evaluation_times_s=evaluation_times_s)
    # Failures of existence/domain do not become a clipped phase trajectory.
    mode, _ = composite_control_mode(spec, result.final_state, program)
    return replace(result, solver_diagnostic={**result.solver_diagnostic,
                   "battery_control_mode": mode, "material_scope": "two-negative-one-positive",
                   "energy_scope": "isothermal; composite chemical energy is not declared"})
