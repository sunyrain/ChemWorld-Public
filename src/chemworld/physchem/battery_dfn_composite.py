"""Isothermal 2+1-phase DFN with spatial particles and one electrical residual.

The selected source is PyBaMM 25.4.2 modular DFN with single-valued OCPs,
not BasicDFNComposite's directional secondary OCP. Every phase/x/r inventory is
unique; cell-summed phase currents drive the original porous electrical and salt
transport kernels. No phase or cell owns a private root, clock, or integrator.
"""

from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache
from math import isclose, isfinite

import numpy as np
from scipy import sparse

from chemworld.physchem.battery_composite import (
    ActiveMaterialPhaseSpec,
    PhaseReadout,
    _finite,
    _monotone,
)
from chemworld.physchem.battery_electrolyte import ELECTROLYTE_SPECIES, BatteryElectrolyteSpec
from chemworld.physchem.battery_models import (
    BATTERY_ALGEBRAIC,
    BATTERY_COORDINATES,
    BatteryThermodynamics,
    F,
    R,
    particle_surface_fraction,
)
from chemworld.physchem.battery_porous_electrical import PHI_E, electrical_from_cell_currents
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

COMPOSITE_DFN_FAMILY = "composite-dfn-isothermal-v1"


@dataclass(frozen=True)
class BatteryCompositeDFNSpec:
    negative_phases: tuple[ActiveMaterialPhaseSpec, ActiveMaterialPhaseSpec]
    positive_phase: ActiveMaterialPhaseSpec
    electrolyte: BatteryElectrolyteSpec = field(default_factory=BatteryElectrolyteSpec)
    temperature_K: float = 298.15
    surface_margin: float = 0.01
    minimum_voltage_V: float = 2.5
    maximum_voltage_V: float = 4.5
    maximum_current_A: float = 2.0
    series_resistance_ohm: float = 0.0
    provenance: str = "declared-single-valued-composite-DFN-v1"

    def __post_init__(self) -> None:
        phases = tuple(
            ActiveMaterialPhaseSpec.from_dict(p) if isinstance(p, dict) else p
            for p in self.negative_phases
        )
        positive = (
            ActiveMaterialPhaseSpec.from_dict(self.positive_phase)
            if isinstance(self.positive_phase, dict)
            else self.positive_phase
        )
        raw_electrolyte: object = self.electrolyte
        electrolyte = (
            BatteryElectrolyteSpec(**raw_electrolyte)
            if isinstance(raw_electrolyte, dict)
            else raw_electrolyte
        )
        if len(phases) != 2 or not all(
            isinstance(p, ActiveMaterialPhaseSpec) for p in (*phases, positive)
        ):
            raise ValueError("composite DFN requires exactly two negative and one positive phase")
        if phases[0].phase_id == phases[1].phase_id:
            raise ValueError("negative material phase identities must be unique")
        if not isinstance(electrolyte, BatteryElectrolyteSpec):
            raise ValueError("composite DFN requires its declared electrolyte geometry")
        object.__setattr__(self, "negative_phases", phases)
        object.__setattr__(self, "positive_phase", positive)
        object.__setattr__(self, "electrolyte", electrolyte)
        for name in (
            "temperature_K",
            "surface_margin",
            "minimum_voltage_V",
            "maximum_voltage_V",
            "maximum_current_A",
            "series_resistance_ohm",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if (
            not 0 < self.surface_margin < 0.5
            or self.temperature_K <= 0
            or self.maximum_current_A <= 0
            or self.series_resistance_ohm < 0
            or self.minimum_voltage_V >= self.maximum_voltage_V
            or not isinstance(self.provenance, str)
            or not self.provenance.strip()
        ):
            raise ValueError("invalid composite DFN operating domain")
        for phase in self.phases:
            if (
                "DFN" not in phase.ocv.supported_models
                or phase.ocv.stoichiometry_domain[0] > self.surface_margin
                or phase.ocv.stoichiometry_domain[1] < 1 - self.surface_margin
                or not self.surface_margin < phase.initial_fraction < 1 - self.surface_margin
            ):
                raise ValueError("phase OCV/initial state must cover the declared DFN surfaces")
            if not _monotone(phase.ocv, self.surface_margin, 1 - self.surface_margin):
                raise ValueError("phase OCV monotonicity is uncertified for unique current sharing")
            if any(
                not isfinite(v) or v <= 0
                for v in (phase.particle.area_m2, phase.particle.capacity_mol)
            ):
                raise ValueError("derived phase area/capacity must be finite and positive")
        for region, materials in ((0, phases), (2, (positive,))):
            allowed = (
                electrolyte.area_m2
                * electrolyte.lengths_m[region]
                * (1 - electrolyte.porosities[region])
            )
            active = sum(p.particle.active_volume_m3 for p in materials)
            # A zero-binder electrode is legal; tolerate only arithmetic roundoff
            # in the separately evaluated geometrical product and phase sum.
            if active > allowed and not isclose(
                active, allowed, rel_tol=8 * np.finfo(float).eps, abs_tol=0.0
            ):
                raise ValueError("active phase volumes exceed the non-pore electrode geometry")
        if any(not isfinite(v) or v <= 0 for v in (self.usable_charge_c, self.volume_l)):
            raise ValueError(
                "derived composite DFN charge/volume scale must be finite and positive"
            )

    @property
    def phases(self):
        return (*self.negative_phases, self.positive_phase)

    @property
    def model_family(self):
        return COMPOSITE_DFN_FAMILY

    model_id = model_family

    @property
    def numerical_form(self):
        return "residual"

    @property
    def thermal(self) -> None:
        return None

    sei = plating = negative_ocp = positive_ocp = thermal

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
        return 1000 * self.electrolyte.area_m2 * sum(self.electrolyte.lengths_m)

    @property
    def usable_charge_c(self):
        return F * min(
            sum(p.particle.capacity_mol * p.initial_fraction for p in self.negative_phases),
            self.positive_phase.particle.capacity_mol * (1 - self.positive_phase.initial_fraction),
        )

    def current_at_c_rate(self, rate):
        return _finite(rate, "C-rate") * self.usable_charge_c / 3600

    @cached_property
    def particle_bins(self):
        bins = []
        for name, region, phases in (
            ("negative", 0, self.negative_phases),
            ("positive", 2, (self.positive_phase,)),
        ):
            count = self.electrolyte.cells[region]
            for phase in phases:
                particle = replace(
                    phase.particle, active_volume_m3=phase.particle.active_volume_m3 / count
                )
                for x in range(count):
                    identity = f"{name}:phase:{phase.phase_id}:x:{x}"
                    bins.append((name, x, particle, particle.keys(identity), identity))
        return tuple(bins)

    @cached_property
    def bin_phases(self):
        nn, _, np_ = self.electrolyte.cells
        return (
            (self.negative_phases[0],) * nn
            + (self.negative_phases[1],) * nn
            + (self.positive_phase,) * np_
        )

    @cached_property
    def bin_cell_indices(self):
        nn, _, _ = self.electrolyte.cells
        return tuple(x if name == "negative" else nn + x for name, x, *_ in self.particle_bins)

    @cached_property
    def bin_electrolyte_indices(self):
        nn, ns, _ = self.electrolyte.cells
        return tuple(x if name == "negative" else nn + ns + x for name, x, *_ in self.particle_bins)

    @cached_property
    def inventory_species(self):
        return {
            **{key: "Li" for _, _, _, keys, _ in self.particle_bins for key in keys},
            **dict.fromkeys(self.electrolyte.keys, ELECTROLYTE_SPECIES),
        }

    @property
    def inventory_keys(self):
        return tuple(self.inventory_species)

    @cached_property
    def free_bin_indices(self):
        nn, _, np_ = self.electrolyte.cells
        return (*range(2 * nn - 1), *range(2 * nn, 2 * nn + np_ - 1))

    @cached_property
    def reaction_keys(self):
        return tuple(
            f"battery:{self.particle_bins[i][-1]}:reaction_A" for i in self.free_bin_indices
        )

    @cached_property
    def algebraic_names(self):
        return (*self.reaction_keys, PHI_E, *BATTERY_ALGEBRAIC)

    @cached_property
    def reaction_map(self):
        nn, _, np_ = self.electrolyte.cells
        matrix = np.zeros((2 * nn + np_, len(self.algebraic_names)))
        for column, row in enumerate(self.free_bin_indices):
            matrix[row, column] = 1
            matrix[2 * nn - 1 if row < 2 * nn else 2 * nn + np_ - 1, column] = -1
        matrix[2 * nn - 1, -2] = 1
        matrix[-1, -2] = -1
        matrix.flags.writeable = False
        return matrix

    @cached_property
    def cell_sum_map(self):
        nn, _, np_ = self.electrolyte.cells
        matrix = np.zeros((nn + np_, len(self.particle_bins)))
        matrix[self.bin_cell_indices, np.arange(len(self.particle_bins))] = 1
        matrix.flags.writeable = False
        return matrix

    @cached_property
    def electrical_maps(self):
        columns = []
        for column in np.eye(len(self.algebraic_names)).T:
            q = self.cell_sum_map @ (self.reaction_map @ column)
            _, _, phi_e, phi_s, _ = electrical_from_cell_currents(
                self.electrolyte, q, column[-2], column[-1], column[-3], self.series_resistance_ohm
            )
            drops = (
                phi_s[np.asarray(self.bin_cell_indices)]
                - phi_e[np.asarray(self.bin_electrolyte_indices)]
            )
            columns.append(drops)
        matrix = np.asarray(columns).T
        matrix.flags.writeable = False
        return self.reaction_map, matrix

    def initial_state(self):
        amounts = self.electrolyte.initial_amounts()
        for phase, (_, _, particle, keys, _) in zip(
            self.bin_phases, self.particle_bins, strict=True
        ):
            amounts.update(
                zip(
                    keys,
                    particle.mesh.cell_volumes_m3
                    * particle.maximum_concentration_mol_m3
                    * phase.initial_fraction,
                    strict=True,
                )
            )
        return CoupledState(
            amounts, self.temperature_K, self.volume_l, coordinates=self.initial_coordinates()
        )

    def to_dict(self):
        return {
            "model_family": self.model_family,
            "negative_phases": [p.to_dict() for p in self.negative_phases],
            "positive_phase": self.positive_phase.to_dict(),
            "electrolyte": self.electrolyte.to_dict(),
            **{
                key: getattr(self, key)
                for key in (
                    "temperature_K",
                    "surface_margin",
                    "minimum_voltage_V",
                    "maximum_voltage_V",
                    "maximum_current_A",
                    "series_resistance_ohm",
                    "provenance",
                )
            },
        }

    @classmethod
    def from_dict(cls, value):
        fields = {
            "model_family",
            "negative_phases",
            "positive_phase",
            "electrolyte",
            "temperature_K",
            "surface_margin",
            "minimum_voltage_V",
            "maximum_voltage_V",
            "maximum_current_A",
            "series_resistance_ohm",
            "provenance",
        }
        if set(value) != fields or value["model_family"] != COMPOSITE_DFN_FAMILY:
            raise ValueError("unknown composite DFN fields or family")
        return cls(**{k: v for k, v in value.items() if k != "model_family"})


@dataclass(frozen=True)
class CompositeDFNReadout:
    voltage_V: float
    current_A: float
    phase_readouts: tuple[PhaseReadout, ...]
    concentration_overpotential_V: float
    electrolyte_ohmic_V: float
    solid_ohmic_V: float

    def to_dict(self):
        """Return the strict JSON readout contract for the three-phase model."""

        return {
            "model": "composite-dfn-readout-v1",
            "voltage_V": self.voltage_V,
            "current_A": self.current_A,
            "phase_readouts": [
                {
                    "electrode": item.electrode,
                    "phase_id": item.phase_id,
                    "surface_fraction": item.surface_fraction,
                    "average_fraction": item.average_fraction,
                    "reaction_current_A": item.reaction_current_A,
                    "inventory_mol": item.inventory_mol,
                    "kinetic_overpotential_V": item.kinetic_overpotential_V,
                }
                for item in self.phase_readouts
            ],
            "concentration_overpotential_V": self.concentration_overpotential_V,
            "electrolyte_ohmic_V": self.electrolyte_ohmic_V,
            "solid_ohmic_V": self.solid_ohmic_V,
        }

    @classmethod
    def from_dict(cls, value):
        """Restore a readout while rejecting unknown fields and invalid phase sets."""

        fields = {
            "model",
            "voltage_V",
            "current_A",
            "phase_readouts",
            "concentration_overpotential_V",
            "electrolyte_ohmic_V",
            "solid_ohmic_V",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("unknown composite DFN readout fields")
        if value["model"] != "composite-dfn-readout-v1":
            raise ValueError("unknown composite DFN readout model")
        if not isinstance(value["phase_readouts"], list) or len(value["phase_readouts"]) != 3:
            raise ValueError("composite DFN readout requires exactly three phase records")
        records = []
        seen = set()
        for item in value["phase_readouts"]:
            if not isinstance(item, dict) or set(item) != {
                "electrode", "phase_id", "surface_fraction", "average_fraction",
                "reaction_current_A", "inventory_mol", "kinetic_overpotential_V",
            }:
                raise ValueError("unknown composite DFN phase readout fields")
            identity = (item["electrode"], item["phase_id"])
            if identity in seen or item["electrode"] not in ("negative", "positive"):
                raise ValueError("composite DFN phase readout identities must be unique")
            seen.add(identity)
            if any(
                isinstance(item[key], bool) or not isfinite(item[key])
                for key in (
                    "surface_fraction", "average_fraction", "reaction_current_A",
                    "inventory_mol", "kinetic_overpotential_V",
                )
            ) or not 0 <= item["surface_fraction"] <= 1 or not 0 <= item[
                "average_fraction"
            ] <= 1 or item["inventory_mol"] < 0:
                raise ValueError("composite DFN phase readout contains invalid values")
            records.append(PhaseReadout(**{key: item[key] for key in (
                "electrode", "phase_id", "surface_fraction", "average_fraction",
                "reaction_current_A", "inventory_mol", "kinetic_overpotential_V",
            )}))
        if sum(item.electrode == "negative" for item in records) != 2 or sum(
            item.electrode == "positive" for item in records
        ) != 1:
            raise ValueError("composite DFN readout requires two negative and one positive phase")
        scalars = (
            "voltage_V", "current_A", "concentration_overpotential_V",
            "electrolyte_ohmic_V", "solid_ohmic_V",
        )
        if any(isinstance(value[key], bool) or not isfinite(value[key]) for key in scalars):
            raise ValueError("composite DFN readout scalar values must be finite")
        return cls(
            value["voltage_V"],
            value["current_A"],
            tuple(records),
            value["concentration_overpotential_V"],
            value["electrolyte_ohmic_V"],
            value["solid_ohmic_V"],
        )


def validate_composite_dfn_state(spec, state):
    if set(state.amounts_mol) != set(spec.inventory_keys) or set(state.coordinates) != set(
        spec.coordinate_keys
    ):
        raise ValueError("composite DFN state requires every phase/salt inventory and original Q/W")
    if not isclose(state.temperature_K, spec.temperature_K, abs_tol=1e-8, rel_tol=0) or not isclose(
        state.volume_L, spec.volume_l, abs_tol=1e-12, rel_tol=0
    ):
        raise ValueError("composite DFN requires its declared isothermal volume")
    if any(not isfinite(v) for v in (*state.amounts_mol.values(), *state.coordinates.values())):
        raise ValueError("composite DFN inventory and coordinates must be finite")
    totals = {"negative": 0.0, "positive": 0.0}
    initial = {"negative": 0.0, "positive": 0.0}
    for phase, (name, _, particle, keys, _) in zip(
        spec.bin_phases, spec.particle_bins, strict=True
    ):
        for key, volume in zip(keys, particle.mesh.cell_volumes_m3, strict=True):
            if (
                not 0
                <= state.amounts_mol[key]
                <= (volume * particle.maximum_concentration_mol_m3 * (1 + 1e-8))
            ):
                raise ValueError("composite DFN particle inventory exceeds finite capacity")
        totals[name] += sum(state.amounts_mol[k] for k in keys)
        initial[name] += particle.capacity_mol * phase.initial_fraction
    charge = state.coordinates["discharged_charge_C"]
    for name, sign in (("negative", 1), ("positive", -1)):
        if not isclose(
            F * totals[name] + sign * charge, F * initial[name], rel_tol=1e-8, abs_tol=1e-8
        ):
            raise ValueError("composite DFN inventories disagree with original Faradaic charge")
    e = spec.electrolyte
    if e.minimum_resolved_concentration(state) < e.minimum_concentration_mol_m3 * (1 - 1e-7):
        raise ValueError("electrolyte is outside its positive-concentration domain")
    if not isclose(
        sum(state.amounts_mol[k] for k in e.keys),
        sum(e.initial_amounts().values()),
        rel_tol=1e-8,
        abs_tol=1e-12,
    ):
        raise ValueError("composite DFN salt inventory disagrees with the closed cell")


def _fields(spec, state, values):
    e = spec.electrolyte
    current, voltage = (values[key] for key in BATTERY_ALGEBRAIC)
    q = spec.reaction_map @ np.asarray([values[k] for k in spec.algebraic_names])
    cell_q = spec.cell_sum_map @ q
    _, ionic_faces, phi_e, phi_s, solid_faces = electrical_from_cell_currents(
        e, cell_q, current, voltage, values[PHI_E], spec.series_resistance_ohm
    )
    nn, _, np_ = e.cells
    ohmic = float(np.mean(phi_e[-np_:]) - np.mean(phi_e[:nn]))
    c = e.concentrations(state)
    # Only a residual trial extension. Accepted inventory and events use true c/theta.
    trial_c = np.maximum(c, e.minimum_concentration_mol_m3 / 2)
    coefficient = 2 * R * state.temperature_K * (1 - e.cation_transference) / F
    phi_e += coefficient * e.thermodynamic_factor * np.log(trial_c / e.initial_concentration_mol_m3)
    surfaces, averages, eta, areas, ocv = [], [], [], [], []
    for row, (phase, (_, _, p, keys, identity), cx) in enumerate(
        zip(spec.bin_phases, spec.particle_bins, spec.bin_electrolyte_indices, strict=True)
    ):
        theta = particle_surface_fraction(p, identity, state, q[row] / F)
        trial = np.clip(theta, spec.surface_margin / 2, 1 - spec.surface_margin / 2)
        j0 = p.exchange_current_scale_A_m2 * np.sqrt(
            trial_c[cx] / e.initial_concentration_mol_m3 * trial * (1 - trial)
        )
        eta.append(2 * R * state.temperature_K / F * np.arcsinh(q[row] / p.area_m2 / (2 * j0)))
        surfaces.append(theta)
        averages.append(sum(state.amounts_mol[k] for k in keys) / p.capacity_mol)
        areas.append(p.area_m2)
        ocv.append(phase.ocv.value(theta, trial=True))
    return {
        "current_A": current,
        "voltage_V": voltage,
        "reaction_current_A": q,
        "cell_reaction_current_A": cell_q,
        "interfacial_current_A_m2": q / np.asarray(areas),
        "electrolyte_concentration_mol_m3": c,
        "electrolyte_potential_V": phi_e,
        "electrolyte_face_current_A_m2": ionic_faces,
        "negative_solid_face_current_A_m2": solid_faces[0],
        "positive_solid_face_current_A_m2": solid_faces[1],
        "solid_potential_V": phi_s,
        "surface_fraction": np.asarray(surfaces),
        "average_fraction": np.asarray(averages),
        "kinetic_overpotential_V": np.asarray(eta),
        "open_circuit_potential_V": np.asarray(ocv),
        "electrolyte_ohmic_V": ohmic,
        "concentration_overpotential_V": float(np.mean(phi_e[-np_:]) - np.mean(phi_e[:nn]) - ohmic),
        "solid_ohmic_V": float(
            voltage
            + current * spec.series_resistance_ohm
            - np.mean(phi_s[nn:])
            + np.mean(phi_s[:nn])
        ),
        "bv_residual_V": phi_s[np.asarray(spec.bin_cell_indices)]
        - phi_e[np.asarray(spec.bin_electrolyte_indices)]
        - np.asarray(ocv)
        - eta,
    }


def composite_dfn_algebraic(spec, program):
    if program.unit not in {"A", "V"} or any(
        abs(v) > spec.maximum_current_A
        if program.unit == "A"
        else not spec.minimum_voltage_V <= v <= spec.maximum_voltage_V
        for _, v in program.setpoints
    ):
        raise ValueError("composite DFN control programme is outside the device domain")
    names = spec.algebraic_names
    variables = (
        *(
            ResidualVariable(k, "A", spec.maximum_current_A, "algebraic")
            for k in spec.reaction_keys
        ),
        ResidualVariable(PHI_E, "V", 1, "algebraic"),
        ResidualVariable(BATTERY_ALGEBRAIC[0], "A", spec.maximum_current_A, "algebraic"),
        ResidualVariable(BATTERY_ALGEBRAIC[1], "V", 1, "algebraic"),
    )
    equations = (
        *(ResidualEquation(f"battery:bv:{identity}", "V") for *_, identity in spec.particle_bins),
        ResidualEquation("battery:control", "1"),
    )

    def initial_values(state):
        current = program.value_at(state.time_s) if program.unit == "A" else 0.0
        q, means = [], {}
        for name, sign in (("negative", 1), ("positive", -1)):
            selected = [
                (phase, b)
                for phase, b in zip(spec.bin_phases, spec.particle_bins, strict=True)
                if b[0] == name
            ]
            area = sum(b[2].area_m2 for _, b in selected)
            means[name] = (
                sum(
                    b[2].area_m2
                    * phase.ocv.value(
                        sum(state.amounts_mol[k] for k in b[3]) / b[2].capacity_mol, trial=True
                    )
                    for phase, b in selected
                )
                / area
            )
            q.extend(sign * current * b[2].area_m2 / area for _, b in selected)
        values = [
            *(q[i] for i in spec.free_bin_indices),
            -means["negative"],
            current,
            means["positive"] - means["negative"],
        ]
        return dict(zip(names, values, strict=True))

    def residual(state):
        fields = _fields(spec, state, state.algebraic_values)
        current, voltage = (fields[k] for k in ("current_A", "voltage_V"))
        target = program.value_at(state.time_s)
        requested = current + spec.maximum_current_A * (voltage - target)
        control = (
            (current - target)
            if program.unit == "A"
            else current - np.clip(requested, -spec.maximum_current_A, spec.maximum_current_A)
        )
        return np.r_[fields["bv_residual_V"], control / spec.maximum_current_A]

    return CoupledAlgebraic(variables, equations, initial_values, residual)


def _closure_jacobian(spec, state, program, identities):
    index = {k: i for i, k in enumerate(identities)}
    jac = np.zeros((len(spec.algebraic_names), len(identities)))
    qmap, electrical = spec.electrical_maps
    columns = [index[k] for k in spec.algebraic_names]
    jac[:-1, columns] = electrical
    e = spec.electrolyte
    f = _fields(spec, state, state.algebraic_values)
    coefficient = 2 * R * state.temperature_K / F
    for row, (phase, (_, _, p, keys, _), cx) in enumerate(
        zip(spec.bin_phases, spec.particle_bins, spec.bin_electrolyte_indices, strict=True)
    ):
        c = f["electrolyte_concentration_mol_m3"][cx]
        trial_c = max(c, e.minimum_concentration_mol_m3 / 2)
        theta = f["surface_fraction"][row]
        trial = np.clip(theta, spec.surface_margin / 2, 1 - spec.surface_margin / 2)
        j0 = p.exchange_current_scale_A_m2 * np.sqrt(
            trial_c / e.initial_concentration_mol_m3 * trial * (1 - trial)
        )
        z = f["reaction_current_A"][row] / (2 * p.area_m2 * j0)
        factor = coefficient / np.sqrt(1 + z * z)
        eta_q = factor / (2 * p.area_m2 * j0)
        eta_theta = (
            -factor * z * (1 - 2 * trial) / (2 * trial * (1 - trial)) if trial == theta else 0.0
        )
        slope = phase.ocv.derivatives(theta, trial=True)[1]
        surface_q = -p.mesh.spherical_surface_distance_m / (
            F * p.area_m2 * p.diffusivity_m2_s * p.maximum_concentration_mol_m3
        )
        jac[row, columns] -= ((slope + eta_theta) * surface_q + eta_q) * qmap[row]
        if keys[-1] in index:
            jac[row, index[keys[-1]]] = -(slope + eta_theta) / (
                p.mesh.cell_volumes_m3[-1] * p.maximum_concentration_mol_m3
            )
        if e.keys[cx] in index and c == trial_c:
            jac[row, index[e.keys[cx]]] = (
                -coefficient * (1 - e.cation_transference) * e.thermodynamic_factor / c
                + factor * z / (2 * c)
            ) / e.diffusion.storage_volumes_m3[cx]
        if "temperature" in index:
            jac[row, index["temperature"]] = (
                -coefficient
                * (1 - e.cation_transference)
                * e.thermodynamic_factor
                * np.log(trial_c / e.initial_concentration_mol_m3)
                - f["kinetic_overpotential_V"][row]
            ) / state.temperature_K
    current, voltage = (f[k] for k in ("current_A", "voltage_V"))
    requested = current + spec.maximum_current_A * (voltage - program.value_at(state.time_s))
    if program.unit == "A" or abs(requested) >= spec.maximum_current_A:
        jac[-1, index[BATTERY_ALGEBRAIC[0]]] = 1 / spec.maximum_current_A
    else:
        jac[-1, index[BATTERY_ALGEBRAIC[1]]] = -1.0
    return sparse.csc_matrix(jac)


def composite_dfn_fields(spec, state, *, current_A=None, program=None):
    if isinstance(state, CoupledState):
        validate_composite_dfn_state(spec, state)
    if (current_A is None) == (program is None):
        raise ValueError("supply either a current or a control programme")
    if current_A is not None:
        if not isfinite(current_A) or abs(current_A) > spec.maximum_current_A:
            raise ValueError("current is outside composite DFN capability")
        program = ScalarControlProgram(((0.0, current_A),), "A")
    closure = composite_dfn_algebraic(spec, program)
    names = spec.algebraic_names
    # Only the solver view consumes its present algebraic iterate. Public reads
    # re-close from the same inventories, including when a previous control is stale.
    if isinstance(state, CoupledStateView) and all(k in state.algebraic_values for k in names):
        return _fields(spec, state, state.algebraic_values)

    def candidate(vector):
        return CoupledStateView(
            state.amounts_mol,
            state.temperature_K,
            state.volume_L,
            state.time_s,
            coordinates=state.coordinates,
            algebraic_values=dict(zip(names, vector, strict=True)),
        )

    def residual(t, y, rate, parameters):
        return closure.residual(candidate(y))

    def jacobian(t, y, rate, parameters):
        return _closure_jacobian(spec, candidate(y), program, names)

    system = ResidualSystem(
        "battery:composite-dfn-closure",
        closure.variables,
        closure.equations,
        residual,
        ResidualDerivatives(
            "analytic",
            state=jacobian,
            rate=lambda t, y, rate, p: sparse.csc_matrix((len(y), len(y))),
        ),
    )
    guess = closure.initial_values(state)
    answer = solve_steady_residual(
        system, [guess[k] for k in names], policy=RootPolicy(tolerance=1e-12)
    )
    if not answer.root.success:
        raise ResidualNumericalError(
            "composite DFN electrical closure:" + answer.root.failure_reason
        )
    fields = _fields(spec, state, dict(zip(names, answer.root.value, strict=True)))
    if abs(fields["current_A"]) > spec.maximum_current_A * (1 + 1e-10):
        raise ResidualNumericalError("composite DFN circuit exceeds its actual source bound")
    return {**fields, "closure_diagnostic": answer.root.diagnostics}


def composite_dfn_readout(spec, state, current_A):
    f = composite_dfn_fields(spec, state, current_A=current_A)
    records = []
    for name, phases in (("negative", spec.negative_phases), ("positive", (spec.positive_phase,))):
        for phase in phases:
            rows = [
                i
                for i, (p, b) in enumerate(zip(spec.bin_phases, spec.particle_bins, strict=True))
                if b[0] == name and p.phase_id == phase.phase_id
            ]
            records.append(
                PhaseReadout(
                    name,
                    phase.phase_id,
                    float(np.mean(f["surface_fraction"][rows])),
                    float(np.mean(f["average_fraction"][rows])),
                    float(sum(f["reaction_current_A"][rows])),
                    sum(state.amounts_mol[k] for i in rows for k in spec.particle_bins[i][3]),
                    float(np.mean(f["kinetic_overpotential_V"][rows])),
                )
            )
    return CompositeDFNReadout(
        f["voltage_V"],
        f["current_A"],
        tuple(records),
        f["concentration_overpotential_V"],
        f["electrolyte_ohmic_V"],
        f["solid_ohmic_V"],
    )


def composite_dfn_control_current(spec, state, program):
    return composite_dfn_fields(spec, state, program=program)["current_A"]


def composite_dfn_control_mode(spec, state, program):
    target = program.value_at(state.time_s)
    f = composite_dfn_fields(spec, state, program=program)
    if program.unit == "A":
        return "constant_current", target
    if f["voltage_V"] < target - 1e-9:
        return "negative_current_limit", -spec.maximum_current_A
    if f["voltage_V"] > target + 1e-9:
        return "positive_current_limit", spec.maximum_current_A
    return "constant_voltage", target


def composite_dfn_system(spec, program, *, thermal_device=None, control_events=()):
    if thermal_device is not None:
        raise ValueError("composite DFN is isothermal; thermal coupling is unsupported")
    closure = composite_dfn_algebraic(spec, program)
    e = spec.electrolyte
    processes = [
        ConservativeDiffusion(p.mesh, p.diffusivity_m2_s)
        .couple(keys, process_id=f"battery:{identity}", owner_id="battery")
        .process
        for _, _, p, keys, identity in spec.particle_bins
    ]
    processes.append(
        e.diffusion.couple(
            e.keys, process_id="battery:electrolyte-diffusion", owner_id="battery"
        ).process
    )

    def fields(state):
        return composite_dfn_fields(spec, state, program=program)

    def source(state):
        f = fields(state)
        amounts = {
            keys[-1]: -q / F
            for q, (_, _, _, keys, _) in zip(
                f["reaction_current_A"], spec.particle_bins, strict=True
            )
        }
        nn, ns, _ = e.cells
        q = f["cell_reaction_current_A"]
        salt = np.r_[q[:nn], np.zeros(ns), q[nn:]] * (1 - e.cation_transference) / F
        amounts.update(zip(e.keys, salt, strict=True))
        return ProcessContribution(
            amount_rates_mol_s=amounts,
            coordinate_rates={
                "discharged_charge_C": f["current_A"],
                "delivered_work_J": f["current_A"] * f["voltage_V"],
            },
        )

    processes.append(
        CoupledProcess(
            "battery:composite-dfn-intercalation",
            source,
            validate_state=lambda s: validate_composite_dfn_state(spec, s),
        )
    )
    events = [
        CoupledEvent(
            "battery:electrolyte:depleted",
            lambda s: e.minimum_resolved_concentration(s) - e.minimum_concentration_mol_m3,
        )
    ]
    for i, (*_, identity) in enumerate(spec.particle_bins):
        events.extend(
            (
                CoupledEvent(
                    f"battery:{identity}:depleted",
                    lambda s, j=i: fields(s)["surface_fraction"][j] - spec.surface_margin,
                ),
                CoupledEvent(
                    f"battery:{identity}:saturated",
                    lambda s, j=i: 1 - spec.surface_margin - fields(s)["surface_fraction"][j],
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
    nkeys = [k for name, _, _, keys, _ in spec.particle_bins if name == "negative" for k in keys]
    pkeys = [k for name, _, _, keys, _ in spec.particle_bins if name == "positive" for k in keys]
    scales = {k: p.capacity_mol / p.cells for _, _, p, keys, _ in spec.particle_bins for k in keys}
    scales.update(e.initial_amounts())

    @lru_cache(maxsize=8)
    def linear_jacobian(identities):
        index = {k: i for i, k in enumerate(identities)}
        rows, columns, data = [], [], []

        def add(row_ids, column_ids, matrix):
            coo = sparse.coo_matrix(matrix)
            rows.extend(np.asarray(row_ids)[coo.row])
            columns.extend(np.asarray(column_ids)[coo.col])
            data.extend(coo.data)

        acols = [index[k] for k in spec.algebraic_names]
        qmap = spec.reaction_map
        for row, (_, _, p, keys, _) in enumerate(spec.particle_bins):
            cols = [index[k] for k in keys]
            add(
                cols,
                cols,
                ConservativeDiffusion(p.mesh, p.diffusivity_m2_s).inventory_jacobian_s_inv,
            )
            add([index[keys[-1]]], acols, [-qmap[row] / F])
        salt_cols = [index[k] for k in e.keys]
        add(salt_cols, salt_cols, e.diffusion.inventory_jacobian_s_inv)
        nn, ns, _ = e.cells
        cell_map = spec.cell_sum_map @ qmap
        salt_source = np.vstack((cell_map[:nn], np.zeros((ns, len(acols))), cell_map[nn:]))
        add(salt_cols, acols, salt_source * (1 - e.cation_transference) / F)
        add([index["discharged_charge_C"]], [index[BATTERY_ALGEBRAIC[0]]], [[1.0]])
        return sparse.csc_matrix((data, (rows, columns)), shape=(len(identities), len(identities)))

    def jacobian(state, identities):
        index = {k: i for i, k in enumerate(identities)}
        arows = np.asarray([index[k] for k in spec.algebraic_names])
        closure_coo = -_closure_jacobian(spec, state, program, identities).tocoo()
        ci, cv = (index[k] for k in BATTERY_ALGEBRAIC)
        work = index["delivered_work_J"]
        data = np.r_[
            closure_coo.data,
            state.algebraic_values[BATTERY_ALGEBRAIC[1]],
            state.algebraic_values[BATTERY_ALGEBRAIC[0]],
        ]
        rows = np.r_[arows[closure_coo.row], work, work]
        cols = np.r_[closure_coo.col, ci, cv]
        return linear_jacobian(tuple(identities)) + sparse.csc_matrix(
            (data, (rows, cols)), shape=(len(identities), len(identities))
        )

    return CoupledSystem(
        BatteryThermodynamics(spec),
        tuple(processes),
        isothermal=True,
        numerical_form="residual",
        algebraic=closure,
        residual_rhs_jacobian=jacobian,
        events=(*events, *control_events),
        inventory_scales_mol=scales,
        coordinates=(
            CoupledCoordinate("discharged_charge_C", "C", spec.usable_charge_c),
            CoupledCoordinate("delivered_work_J", "J", spec.usable_charge_c * 4),
        ),
        conservation_laws={
            "lithium_mol": dict.fromkeys(spec.inventory_keys, 1),
            "electrolyte_salt_mol": dict.fromkeys(e.keys, 1),
            "negative_faradaic_C": {**dict.fromkeys(nkeys, F), "discharged_charge_C": 1},
            "positive_faradaic_C": {**dict.fromkeys(pkeys, F), "discharged_charge_C": -1},
        },
    )


def advance_composite_dfn(
    spec,
    state,
    program,
    duration_s,
    *,
    evaluation_times_s=None,
    thermal_device=None,
    control_events=(),
):
    result = composite_dfn_system(
        spec, program, thermal_device=thermal_device, control_events=control_events
    ).advance(state, duration_s, evaluation_times_s=evaluation_times_s)
    return replace(
        result,
        solver_diagnostic={
            **result.solver_diagnostic,
            "battery_control_mode": composite_dfn_control_mode(spec, result.final_state, program)[
                0
            ],
            "material_scope": "two-negative-one-positive; resolved thickness/phase/radius",
            "energy_scope": (
                "isothermal; terminal electrical work only; no chemical/thermal closure"
            ),
        },
    )
