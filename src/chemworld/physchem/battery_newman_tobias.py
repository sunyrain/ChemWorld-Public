"""Newman-Tobias: two mean Li inventories and resolved porous charge transport.

Independent constant-property, isothermal realization of PyBaMM 25.4.2's
uniform-profile NewmanTobias equations. No radial or electrolyte-salt state is
invented. Geometry integration is shared with DFN; time and roots use the common
coupled residual. Rest impedance differentiates this same residual without adding
diffusion or double-layer states. Terminal work is not chemical/thermal closure.
"""

from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from math import isclose, isfinite

import numpy as np
from scipy import sparse

from chemworld.data.scientific_inputs import _text
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import (
    BATTERY_ALGEBRAIC,
    BATTERY_COORDINATES,
    BatteryReadout,
    BatteryThermodynamics,
    F,
    R,
)
from chemworld.physchem.battery_porous_electrical import (
    PHI_E,
    PorousElectrodeGeometrySpec,
    electrical_maps,
    linear_electrical,
    reaction_keys,
)
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

NT_FAMILY = "newman-tobias-isothermal-v1"


@dataclass(frozen=True)
class AveragedElectrodeSpec:
    """One material's finite mean inventory, with no radial diffusivity or grid.

    j0 scale is at the explicitly specified reference electrolyte concentration.
    Spherical radius enters reaction area only: 3 Vactive / R.
    """

    radius_m: float = 1e-5
    maximum_concentration_mol_m3: float = 30000.0
    active_volume_m3: float = 1e-6
    exchange_current_scale_A_m2: float = 4.0

    def __post_init__(self) -> None:
        if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("averaged electrode requires positive finite parameters")
        if any(not isfinite(v) or v <= 0 for v in (self.area_m2, self.capacity_mol)):
            raise ValueError("derived electrode area/capacity must be representable")

    @property
    def capacity_mol(self):
        return self.maximum_concentration_mol_m3 * self.active_volume_m3

    @property
    def area_m2(self):
        return 3 * self.active_volume_m3 / self.radius_m


@dataclass(frozen=True)
class BatteryNewmanTobiasSpec:
    negative: AveragedElectrodeSpec = field(default_factory=AveragedElectrodeSpec)
    positive: AveragedElectrodeSpec = field(default_factory=AveragedElectrodeSpec)
    geometry: PorousElectrodeGeometrySpec = field(default_factory=PorousElectrodeGeometrySpec)
    negative_ocv: ElectrodeOCV | None = None
    positive_ocv: ElectrodeOCV | None = None
    negative_ocv_intercept_V: float = 0.2
    negative_ocv_slope_V: float = -0.1
    positive_ocv_intercept_V: float = 4.2
    positive_ocv_slope_V: float = -0.8
    initial_negative_fraction: float = 0.7
    initial_positive_fraction: float = 0.3
    temperature_K: float = 298.15
    electrolyte_concentration_mol_m3: float = 1000.0
    reference_electrolyte_concentration_mol_m3: float = 1000.0
    surface_margin: float = 0.01
    minimum_voltage_V: float = 2.5
    maximum_voltage_V: float = 4.5
    maximum_current_A: float = 2.0
    resistance_ohm: float = 0.0
    provenance: str = "synthetic-Newman-Tobias-v1"

    def __post_init__(self) -> None:
        for name, cls in (
            ("negative", AveragedElectrodeSpec),
            ("positive", AveragedElectrodeSpec),
            ("geometry", PorousElectrodeGeometrySpec),
        ):
            value = getattr(self, name)
            if isinstance(value, dict):
                value = cls(**value)
                object.__setattr__(self, name, value)
            if not isinstance(value, cls):
                raise ValueError("NT requires explicit mean electrodes and porous geometry")
        for name in ("negative_ocv", "positive_ocv"):
            curve = getattr(self, name)
            if isinstance(curve, dict):
                curve = ElectrodeOCV.from_dict(curve)
                object.__setattr__(self, name, curve)
            if curve is not None and not isinstance(curve, ElectrodeOCV):
                raise ValueError("NT OCV must be an explicit sourced expression")
            if curve is not None and "NewmanTobias" not in curve.supported_models:
                raise ValueError("OCV must explicitly support NewmanTobias")
        numbers = [value for value in self.__dict__.values() if isinstance(value, (int, float))]
        if any(isinstance(v, bool) or not isfinite(v) for v in numbers):
            raise ValueError("NT parameters must be finite numbers")
        if (
            not 0 < self.surface_margin < 0.5
            or self.temperature_K <= 0
            or self.maximum_current_A <= 0
            or self.resistance_ohm < 0
            or self.electrolyte_concentration_mol_m3 <= 0
            or self.reference_electrolyte_concentration_mol_m3 <= 0
            or self.minimum_voltage_V >= self.maximum_voltage_V
        ):
            raise ValueError("invalid NT operating domain")
        for name, region in (("negative", 0), ("positive", 2)):
            theta = getattr(self, "initial_" + name + "_fraction")
            if not self.surface_margin < theta < 1 - self.surface_margin:
                raise ValueError("initial NT stoichiometry must lie inside its event limits")
            active = getattr(self, name).active_volume_m3
            available = (
                self.geometry.area_m2
                * self.geometry.lengths_m[region]
                * (1 - self.geometry.porosities[region])
            )
            if active > available * (1 + 1e-12):
                raise ValueError("active material and electrolyte cannot exceed electrode volume")
            curve = getattr(self, name + "_ocv")
            if curve is not None and not (
                curve.stoichiometry_domain[0] <= self.surface_margin
                and curve.stoichiometry_domain[1] >= 1 - self.surface_margin
            ):
                raise ValueError("OCV domain must cover the declared NT operating interval")
        if not isfinite(self.usable_charge_c) or self.usable_charge_c <= 0:
            raise ValueError("NT usable charge must be positive and finite")
        if not isfinite(self.volume_l) or self.volume_l <= 0:
            raise ValueError("NT geometry volume must be positive and finite")
        _text(self.provenance, "NT provenance")

    @property
    def model_family(self):
        return NT_FAMILY

    @property
    def model_id(self):
        return NT_FAMILY

    @property
    def numerical_form(self) -> str:
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
    def has_directional_ocp(self) -> bool:
        return False

    @property
    def hysteresis_keys(self):
        return ()

    @property
    def coordinate_keys(self):
        return BATTERY_COORDINATES

    @property
    def inventory_keys(self) -> tuple[str, str]:
        return ("negative:Li", "positive:Li")

    @property
    def inventory_species(self):
        return dict.fromkeys(self.inventory_keys, "Li")

    @property
    def series_resistance_ohm(self):
        return self.resistance_ohm

    @property
    def volume_l(self):
        return self.geometry.area_m2 * sum(self.geometry.lengths_m) * 1000

    @property
    def usable_charge_c(self):
        return F * min(
            self.negative.capacity_mol * (self.initial_negative_fraction - self.surface_margin),
            self.positive.capacity_mol * (1 - self.surface_margin - self.initial_positive_fraction),
        )

    def current_at_c_rate(self, rate):
        if not isfinite(rate):
            raise ValueError("C rate must be finite")
        return rate * self.usable_charge_c / 3600

    @cached_property
    def reaction_keys(self):
        return reaction_keys(self.geometry)

    @cached_property
    def algebraic_names(self):
        return (*self.reaction_keys, PHI_E, *BATTERY_ALGEBRAIC)

    @cached_property
    def electrical_maps(self):
        return electrical_maps(
            self.geometry, self.reaction_keys, self.series_resistance_ohm, self.algebraic_names
        )

    def initial_coordinates(self):
        return dict.fromkeys(self.coordinate_keys, 0.0)

    def initial_state(self):
        return CoupledState(
            {
                name + ":Li": getattr(self, name).capacity_mol
                * getattr(self, "initial_" + name + "_fraction")
                for name in ("negative", "positive")
            },
            self.temperature_K,
            self.volume_l,
            coordinates=self.initial_coordinates(),
        )

    def ocv(self, name, theta, *, derivative=False):
        if name not in {"negative", "positive"}:
            raise ValueError("unknown NT electrode")
        curve = getattr(self, name + "_ocv")
        if curve is not None:
            return (
                curve.derivatives(theta, trial=True)[1]
                if derivative
                else curve.value(theta, trial=True)
            )
        slope = getattr(self, name + "_ocv_slope_V")
        return slope if derivative else getattr(self, name + "_ocv_intercept_V") + slope * theta

    def to_dict(self):
        result = {"model_family": NT_FAMILY}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            result[key] = (
                asdict(value)
                if isinstance(value, AveragedElectrodeSpec)
                else value.to_dict()
                if hasattr(value, "to_dict")
                else value
            )
        return result

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("model_family", None) != NT_FAMILY:
            raise ValueError("unsupported Newman-Tobias family")
        return cls(**value)


def _fields(spec, state, values):
    g = spec.geometry
    current, voltage = (values[key] for key in BATTERY_ALGEBRAIC)
    q, ionic_faces, phi_e, phi_s, solid_faces = linear_electrical(
        g, spec.reaction_keys, spec.series_resistance_ohm, values
    )
    nn, _, np_ = g.cells
    theta, j0, areas, ocv = [], [], [], []
    for name, count in (("negative", nn), ("positive", np_)):
        electrode = getattr(spec, name)
        raw = state.amounts_mol[name + ":Li"] / electrode.capacity_mol
        # Only outside-domain residual trials are extended; accepted inner
        # inventories are never clipped, even near a declared small margin.
        trial = (
            raw
            if 0 < raw < 1
            else (spec.surface_margin / 2 if raw <= 0 else 1 - spec.surface_margin / 2)
        )
        theta.extend([raw] * count)
        j0.extend(
            [
                electrode.exchange_current_scale_A_m2
                * np.sqrt(
                    spec.electrolyte_concentration_mol_m3
                    / spec.reference_electrolyte_concentration_mol_m3
                    * trial
                    * (1 - trial)
                )
            ]
            * count
        )
        areas.extend([electrode.area_m2 / count] * count)
        ocv.extend([spec.ocv(name, raw)] * count)
    theta, j0, areas, ocv = map(np.asarray, (theta, j0, areas, ocv))
    if np.any(~np.isfinite(j0 * areas)) or np.any(j0 * areas <= 0):
        raise ValueError("NT kinetic scale is not representable")
    eta = 2 * R * state.temperature_K / F * np.arcsinh(q / (2 * areas * j0))
    return {
        "current_A": current,
        "voltage_V": voltage,
        "reaction_current_A": q,
        "interfacial_current_A_m2": q / areas,
        "electrolyte_potential_V": phi_e,
        "solid_potential_V": phi_s,
        "electrolyte_face_current_A_m2": ionic_faces,
        "negative_solid_face_current_A_m2": solid_faces[0],
        "positive_solid_face_current_A_m2": solid_faces[1],
        "surface_fraction": theta,
        "average_fraction": theta.copy(),
        "negative_average_fraction": float(theta[0]),
        "positive_average_fraction": float(theta[-1]),
        "kinetic_overpotential_V": eta,
        "exchange_current_A_m2": j0,
        "reaction_area_m2": areas,
        "concentration_overpotential_V": 0.0,
        "electrolyte_ohmic_V": float(np.mean(phi_e[-np_:]) - np.mean(phi_e[:nn])),
        "solid_ohmic_V": float(
            voltage
            + current * spec.series_resistance_ohm
            - np.mean(phi_s[nn:])
            + np.mean(phi_s[:nn])
        ),
        "bv_residual_V": phi_s - np.r_[phi_e[:nn], phi_e[-np_:]] - ocv - eta,
    }


def _check_program(spec, program):
    if (
        not isinstance(program, ScalarControlProgram)
        or program.unit not in {"A", "V"}
        or any(
            abs(v) > spec.maximum_current_A
            if program.unit == "A"
            else not spec.minimum_voltage_V <= v <= spec.maximum_voltage_V
            for _, v in program.setpoints
        )
    ):
        raise ValueError("NT requires an in-domain CC or finite-source CV program")


def newman_tobias_algebraic(spec, program):
    _check_program(spec, program)
    variables = (
        *(
            ResidualVariable(key, "A", spec.maximum_current_A, "algebraic")
            for key in spec.reaction_keys
        ),
        ResidualVariable(PHI_E, "V", 1, "algebraic"),
        ResidualVariable(BATTERY_ALGEBRAIC[0], "A", spec.maximum_current_A, "algebraic"),
        ResidualVariable(BATTERY_ALGEBRAIC[1], "V", 1, "algebraic"),
    )
    equations = (
        *(
            ResidualEquation(f"battery:bv:{name}:{i}", "V")
            for name, region in (("negative", 0), ("positive", 2))
            for i in range(spec.geometry.cells[region])
        ),
        ResidualEquation("battery:control", "1"),
    )

    def initial_values(state):
        current = program.value_at(state.time_s) if program.unit == "A" else 0.0
        nn, _, np_ = spec.geometry.cells
        q = np.r_[np.full(nn - 1, current / nn), np.full(np_ - 1, -current / np_)]
        un, up = (
            spec.ocv(name, state.amounts_mol[name + ":Li"] / getattr(spec, name).capacity_mol)
            for name in ("negative", "positive")
        )
        return dict(zip(spec.algebraic_names, [*q, -un, current, up - un], strict=True))

    def residual(state):
        f = _fields(spec, state, state.algebraic_values)
        current, voltage = f["current_A"], f["voltage_V"]
        target = program.value_at(state.time_s)
        requested = current + spec.maximum_current_A * (voltage - target)
        control = (
            (current - target)
            if program.unit == "A"
            else current - np.clip(requested, -spec.maximum_current_A, spec.maximum_current_A)
        )
        return np.r_[f["bv_residual_V"], control / spec.maximum_current_A]

    return CoupledAlgebraic(variables, equations, initial_values, residual)


def _closure_jacobian(spec, state, program, identities):
    index = {key: i for i, key in enumerate(identities)}
    qmap, electrical = spec.electrical_maps
    jac = np.zeros((len(spec.algebraic_names), len(identities)))
    columns = [index[key] for key in spec.algebraic_names]
    jac[:-1, columns] = electrical
    f = _fields(spec, state, state.algebraic_values)
    coefficient = 2 * R * state.temperature_K / F
    nn = spec.geometry.cells[0]
    for row, q in enumerate(f["reaction_current_A"]):
        name = "negative" if row < nn else "positive"
        electrode = getattr(spec, name)
        raw = f["surface_fraction"][row]
        z = q / (2 * f["reaction_area_m2"][row] * f["exchange_current_A_m2"][row])
        factor = coefficient / np.hypot(1, z)
        eta_q = factor / (2 * f["reaction_area_m2"][row] * f["exchange_current_A_m2"][row])
        jac[row, columns] -= eta_q * qmap[row]
        if name + ":Li" in index:
            eta_theta = -factor * z * (1 - 2 * raw) / (2 * raw * (1 - raw)) if 0 < raw < 1 else 0
            jac[row, index[name + ":Li"]] = (
                -(spec.ocv(name, raw, derivative=True) + eta_theta) / electrode.capacity_mol
            )
        if "temperature" in index:
            jac[row, index["temperature"]] = (
                -f["kinetic_overpotential_V"][row] / state.temperature_K
            )
    requested = f["current_A"] + spec.maximum_current_A * (
        f["voltage_V"] - program.value_at(state.time_s)
    )
    if program.unit == "A" or abs(requested) >= spec.maximum_current_A:
        jac[-1, index[BATTERY_ALGEBRAIC[0]]] = 1 / spec.maximum_current_A
    else:
        jac[-1, index[BATTERY_ALGEBRAIC[1]]] = -1
    return sparse.csc_matrix(jac)


def validate_newman_tobias_state(spec, state):
    if set(state.amounts_mol) != set(spec.inventory_keys) or set(state.coordinates) != set(
        spec.coordinate_keys
    ):
        raise ValueError("NT requires exactly two mean Li inventories and Q/W coordinates")
    if abs(state.temperature_K - spec.temperature_K) > 1e-8 or not isclose(
        state.volume_L, spec.volume_l, rel_tol=1e-12, abs_tol=1e-14
    ):
        raise ValueError("NT requires its fixed isothermal volume")
    if any(not isfinite(v) for v in state.coordinates.values()):
        raise ValueError("NT coordinates must be finite")
    for name, sign in (("negative", 1), ("positive", -1)):
        capacity = getattr(spec, name).capacity_mol
        amount = state.amounts_mol[name + ":Li"]
        if not 0 < amount < capacity:
            raise ValueError("NT mean Li must lie strictly inside finite capacity")
        initial = capacity * getattr(spec, "initial_" + name + "_fraction")
        if not isclose(
            F * amount + sign * state.coordinates["discharged_charge_C"],
            F * initial,
            rel_tol=1e-8,
            abs_tol=1e-8,
        ):
            raise ValueError("NT mean Li and Q violate the common Faraday balance")


def newman_tobias_fields(spec, state, *, current_A=None, program=None):
    if (current_A is None) == (program is None):
        raise ValueError("supply either current or a control program")
    if current_A is not None:
        if (
            isinstance(current_A, bool)
            or not isfinite(current_A)
            or abs(current_A) > spec.maximum_current_A
        ):
            raise ValueError("NT current exceeds its finite source")
        program = ScalarControlProgram(((0.0, current_A),), "A")
    closure = newman_tobias_algebraic(spec, program)
    names = spec.algebraic_names
    if isinstance(state, CoupledState):
        validate_newman_tobias_state(spec, state)
    elif all(key in state.algebraic_values for key in names) and (
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
        "battery:NT-closure",
        closure.variables,
        closure.equations,
        lambda t, y, rate, p: closure.residual(view(y)),
        ResidualDerivatives(
            "analytic",
            state=lambda t, y, rate, p: _closure_jacobian(spec, view(y), program, names),
            rate=lambda t, y, rate, p: sparse.csc_matrix((len(names), len(names))),
        ),
    )
    guess = closure.initial_values(state)
    answer = solve_steady_residual(
        system, [guess[k] for k in names], policy=RootPolicy(tolerance=1e-12)
    )
    if not answer.root.success:
        raise ResidualNumericalError("NT electrical closure:" + answer.root.failure_reason)
    f = _fields(spec, state, dict(zip(names, answer.root.value, strict=True)))
    if abs(f["current_A"]) > spec.maximum_current_A * (1 + 1e-10):
        raise ResidualNumericalError("NT closure exceeded its actual source bound")
    return {**f, "closure_diagnostic": answer.root.diagnostics}


def newman_tobias_spatial_profile(spec, state, *, current_A=None, program=None):
    """Return a JSON-safe read-only profile of the NT porous electrical state.

    The two mean lithium stocks remain the only persistent inventories. This
    projection labels the cell-integrated reaction currents and shared
    centre/face electrical fields without creating a second state or solver.
    """
    if current_A is None and program is None:
        current_A = 0.0
    fields = newman_tobias_fields(spec, state, current_A=current_A, program=program)
    nn, _, np_ = spec.geometry.cells
    labels = tuple(
        f"{electrode}:cell:{index}"
        for electrode, count in (("negative", nn), ("positive", np_))
        for index in range(count)
    )

    def vector(values):
        return [float(value) for value in np.asarray(values)]

    q = np.asarray(fields["reaction_current_A"])
    return {
        "model_family": NT_FAMILY,
        "cell_labels": list(labels),
        "thickness_cells": list(spec.geometry.cells),
        "current_A": float(fields["current_A"]),
        "voltage_V": float(fields["voltage_V"]),
        "reaction_current_A": {
            label: float(value) for label, value in zip(labels, q, strict=True)
        },
        "interfacial_current_A_m2": vector(fields["interfacial_current_A_m2"]),
        "surface_fraction": vector(fields["surface_fraction"]),
        "electrolyte_potential_V": vector(fields["electrolyte_potential_V"]),
        "solid_potential_V": vector(fields["solid_potential_V"]),
        "electrolyte_face_current_A_m2": vector(
            fields["electrolyte_face_current_A_m2"]
        ),
        "solid_face_current_A_m2": {
            "negative": vector(fields["negative_solid_face_current_A_m2"]),
            "positive": vector(fields["positive_solid_face_current_A_m2"]),
        },
        "charge_balance_A": {
            "negative_reaction_sum_A": float(np.sum(q[:nn])),
            "positive_reaction_sum_A": float(np.sum(q[nn:])),
            "negative_expected_A": float(fields["current_A"]),
            "positive_expected_A": float(-fields["current_A"]),
        },
    }


def newman_tobias_readout(spec, state, current_A):
    f = newman_tobias_fields(spec, state, current_A=current_A)
    nn = spec.geometry.cells[0]
    return BatteryReadout(
        f["voltage_V"],
        f["current_A"],
        f["negative_average_fraction"],
        f["positive_average_fraction"],
        f["negative_average_fraction"],
        f["positive_average_fraction"],
        float(np.mean(f["kinetic_overpotential_V"][:nn])),
        float(np.mean(f["kinetic_overpotential_V"][nn:])),
        f["current_A"],
        0.0,
        0.0,
        0.0,
        f["electrolyte_ohmic_V"],
        f["solid_ohmic_V"],
    )


def newman_tobias_control_current(spec, state, program):
    return newman_tobias_fields(spec, state, program=program)["current_A"]


def newman_tobias_control_mode(spec, state, program):
    _check_program(spec, program)
    target = program.value_at(state.time_s)
    if program.unit == "A":
        return "constant_current", target
    for current, mode, sign in (
        (-spec.maximum_current_A, "negative_current_limit", -1),
        (spec.maximum_current_A, "positive_current_limit", 1),
    ):
        voltage = newman_tobias_fields(spec, state, current_A=current)["voltage_V"]
        if sign * (voltage - target) > 0:
            return mode, current
    return "constant_voltage", target


def newman_tobias_system(spec, program, *, thermal_device=None, control_events=()):
    if thermal_device is not None:
        raise ValueError("Newman-Tobias is isothermal; external thermal coupling is unsupported")
    closure = newman_tobias_algebraic(spec, program)

    def fields(state):
        return newman_tobias_fields(spec, state, program=program)

    def evaluate(state):
        f = fields(state)
        current, voltage = f["current_A"], f["voltage_V"]
        source, destination = spec.inventory_keys if current >= 0 else spec.inventory_keys[::-1]
        return ProcessContribution(
            internal_transfers=(
                InternalTransfer("NT:interelectrode", source, destination, abs(current) / F),
            ),
            coordinate_rates={
                "discharged_charge_C": current,
                "delivered_work_J": current * voltage,
            },
        )

    events = []
    for name in ("negative", "positive"):
        capacity = getattr(spec, name).capacity_mol
        sign = 1 if name == "negative" else -1
        events.extend(
            (
                # A boundary state may rest or reverse into its legal interval.
                # Only the stopping test is switched; no inventory/rate is clipped.
                CoupledEvent(
                    f"battery:{name}:depleted",
                    lambda s, n=name, c=capacity, d=sign: (
                        s.amounts_mol[n + ":Li"] / c - spec.surface_margin
                        if d * fields(s)["current_A"] > 0
                        else 1.0
                    ),
                ),
                CoupledEvent(
                    f"battery:{name}:saturated",
                    lambda s, n=name, c=capacity, d=sign: (
                        1 - spec.surface_margin - s.amounts_mol[n + ":Li"] / c
                        if d * fields(s)["current_A"] < 0
                        else 1.0
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

    def jacobian(state, identities):
        index = {key: i for i, key in enumerate(identities)}
        jac = np.zeros((len(identities), len(identities)))
        ik, vk = BATTERY_ALGEBRAIC
        jac[index["negative:Li"], index[ik]] = -1 / F
        jac[index["positive:Li"], index[ik]] = 1 / F
        jac[index["discharged_charge_C"], index[ik]] = 1
        jac[index["delivered_work_J"], index[ik]] = state.algebraic_values[vk]
        jac[index["delivered_work_J"], index[vk]] = state.algebraic_values[ik]
        rows = [index[k] for k in spec.algebraic_names]
        jac[rows, :] = -_closure_jacobian(spec, state, program, identities).toarray()
        return sparse.csc_matrix(jac)

    return CoupledSystem(
        BatteryThermodynamics(spec),
        (
            CoupledProcess(
                "NT:interelectrode",
                evaluate,
                validate_state=lambda s: validate_newman_tobias_state(spec, s),
            ),
        ),
        isothermal=True,
        numerical_form="residual",
        algebraic=closure,
        residual_rhs_jacobian=jacobian,
        events=(*events, *control_events),
        inventory_scales_mol={
            name + ":Li": getattr(spec, name).capacity_mol for name in ("negative", "positive")
        },
        coordinates=(
            CoupledCoordinate("discharged_charge_C", "C", spec.usable_charge_c),
            CoupledCoordinate("delivered_work_J", "J", spec.usable_charge_c * 4),
        ),
        conservation_laws={
            "lithium_mol": dict.fromkeys(spec.inventory_keys, 1),
            "negative_faradaic_C": {"negative:Li": F, "discharged_charge_C": 1},
            "positive_faradaic_C": {"positive:Li": F, "discharged_charge_C": -1},
        },
    )


def advance_newman_tobias(
    spec,
    state,
    program,
    duration_s,
    *,
    evaluation_times_s=None,
    thermal_device=None,
    control_events=(),
):
    result = newman_tobias_system(
        spec, program, thermal_device=thermal_device, control_events=control_events
    ).advance(state, duration_s, evaluation_times_s=evaluation_times_s)
    return replace(
        result,
        solver_diagnostic={
            **result.solver_diagnostic,
            "battery_control_mode": newman_tobias_control_mode(spec, result.final_state, program)[
                0
            ],
            "energy_scope": "isothermal; terminal electrical work only; no NT heat closure",
        },
    )
