"""Closed, saturated, volume-compatible isothermal lead-acid LOQS.

The six actual species inventories obey Pb + PbO2 + 2 H2SO4 <->
2 PbSO4 + 2 H2O. Bode (1977) molality OCPs and the two-electron,
leading-order Butler-Volmer equations follow PyBaMM 25.4.2/Sulzer2019.
The default molar volumes are explicitly synthetic and volume-compatible;
they are NOT the Sulzer material set. No hidden solvent or porosity state,
ohmic transport, thermal/chemical energy model, or private integrator exists.
"""

from dataclasses import asdict, dataclass, field, replace
from math import isclose, isfinite

import numpy as np
from scipy import sparse

from chemworld.data.scientific_inputs import _text
from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC, BATTERY_COORDINATES, F, R
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

LOQS_FAMILY = "lead-acid-loqs-isothermal-v1"
LOQS_KEYS = (
    "negative:Pb",
    "negative:PbSO4",
    "positive:PbO2",
    "positive:PbSO4",
    "electrolyte:H2SO4",
    "electrolyte:H2O",
)
LOQS_STOICHIOMETRY = (-1, 1, -1, 1, -2, 2)
_COMPOUNDS = ("Pb", "PbSO4", "PbO2", "PbSO4", "H2SO4", "H2O")
_ELEMENTS = {
    "Pb": (1, 1, 1, 1, 0, 0),
    "S": (0, 1, 0, 1, 1, 0),
    "H": (0, 0, 0, 0, 2, 2),
    "O": (0, 4, 2, 4, 4, 1),
}


def _numbers(values, name, *, positive=True):
    if any(
        isinstance(v, (bool, np.bool_)) or not isfinite(v) or (v <= 0 if positive else v < 0)
        for v in values
    ):
        raise ValueError(
            f"{name} requires finite {'positive' if positive else 'nonnegative'} values"
        )


@dataclass(frozen=True)
class LeadAcidGeometrySpec:
    area_m2: float = 0.01
    lengths_m: tuple[float, float, float] = (0.001, 0.0005, 0.001)
    inactive_solid_volumes_m3: tuple[float, float] = (2e-6, 2e-6)
    separator_porosity: float = 0.6
    reaction_area_densities_m_inv: tuple[float, float] = (2e5, 2e5)

    def __post_init__(self):
        for name, count in (
            ("lengths_m", 3),
            ("inactive_solid_volumes_m3", 2),
            ("reaction_area_densities_m_inv", 2),
        ):
            value = tuple(getattr(self, name))
            if len(value) != count:
                raise ValueError("LOQS geometry has fixed n/s/p layer identities")
            _numbers(value, name, positive=name != "inactive_solid_volumes_m3")
            object.__setattr__(self, name, value)
        _numbers((self.area_m2, self.separator_porosity), "LOQS geometry")
        if self.separator_porosity >= 1:
            raise ValueError("separator porosity must be between zero and one")
        _numbers((*self.layer_volumes_m3, *self.reaction_areas_m2), "derived LOQS geometry")
        if any(
            v >= self.layer_volumes_m3[i]
            for v, i in zip(self.inactive_solid_volumes_m3, (0, 2), strict=True)
        ):
            raise ValueError("inactive solid cannot fill an entire electrode")

    @property
    def layer_volumes_m3(self):
        return tuple(self.area_m2 * length for length in self.lengths_m)

    @property
    def reaction_areas_m2(self):
        return tuple(
            self.layer_volumes_m3[i] * a
            for i, a in zip((0, 2), self.reaction_area_densities_m_inv, strict=True)
        )

    def to_dict(self):
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class LeadAcidMaterialSpec:
    # Pb, PbO2, PbSO4, H2SO4, H2O. These satisfy the CLOSED volume constraint.
    molar_volumes_m3_mol: tuple[float, float, float, float, float] = (
        18e-6,
        26e-6,
        48e-6,
        44e-6,
        18e-6,
    )
    reference_acid_concentration_mol_m3: float = 4000.0
    negative_exchange_current_density_A_m2: float = 0.06
    positive_exchange_current_density_A_m2: float = 0.004
    water_molar_mass_kg_mol: float = 0.01801528
    provenance: str = "synthetic compatible volumes; Bode1977 OCP; Sulzer2019 kinetic form"

    def __post_init__(self):
        volumes = tuple(self.molar_volumes_m3_mol)
        if len(volumes) != 5:
            raise ValueError("LOQS material requires Pb/PbO2/PbSO4/acid/water molar volumes")
        object.__setattr__(self, "molar_volumes_m3_mol", volumes)
        _numbers(
            (
                *volumes,
                self.reference_acid_concentration_mol_m3,
                self.negative_exchange_current_density_A_m2,
                self.positive_exchange_current_density_A_m2,
                self.water_molar_mass_kg_mol,
            ),
            "LOQS material",
        )
        pb, pbo2, sulfate, acid, water = volumes
        scale = max(volumes)
        if abs(pb + pbo2 - 2 * sulfate - 2 * (water - acid)) > 1e-12 * scale:
            raise ValueError("closed LOQS molar volumes violate saturated volume compatibility")
        if self.reference_acid_concentration_mol_m3 * acid >= 1:
            raise ValueError("reference electrolyte has no positive water volume")
        _text(self.provenance, "LOQS material provenance")

    @property
    def reference_water_concentration_mol_m3(self):
        _, _, _, acid, water = self.molar_volumes_m3_mol
        return (1 - acid * self.reference_acid_concentration_mol_m3) / water

    def to_dict(self):
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class BatteryLeadAcidSpec:
    geometry: LeadAcidGeometrySpec = field(default_factory=LeadAcidGeometrySpec)
    material: LeadAcidMaterialSpec = field(default_factory=LeadAcidMaterialSpec)
    initial_amounts_mol: tuple[float, ...] | None = None
    temperature_K: float = 298.15
    minimum_acid_concentration_mol_m3: float = 10.0
    minimum_water_concentration_mol_m3: float = 1.0
    minimum_porosity: float = 0.01
    maximum_porosity: float = 0.99
    minimum_voltage_V: float = 1.0
    maximum_voltage_V: float = 3.0
    maximum_current_A: float = 2.0
    provenance: str = "synthetic closed LOQS; default initial eps=.5/.6/.5 and acid=4000mol/m3"

    def __post_init__(self):
        for name, cls in (("geometry", LeadAcidGeometrySpec), ("material", LeadAcidMaterialSpec)):
            value = getattr(self, name)
            if isinstance(value, dict):
                value = cls(**value)
                object.__setattr__(self, name, value)
            if not isinstance(value, cls):
                raise ValueError("LOQS requires explicit material and geometry declarations")
        _numbers(
            (
                self.temperature_K,
                self.minimum_acid_concentration_mol_m3,
                self.minimum_water_concentration_mol_m3,
                self.maximum_current_A,
            ),
            "LOQS domain",
        )
        if any(
            isinstance(v, bool) or not isfinite(v)
            for v in (
                self.minimum_porosity,
                self.maximum_porosity,
                self.minimum_voltage_V,
                self.maximum_voltage_V,
            )
        ) or not (
            0 <= self.minimum_porosity < self.maximum_porosity <= 1
            and self.minimum_voltage_V < self.maximum_voltage_V
        ):
            raise ValueError("invalid LOQS porosity or voltage domain")
        amounts = self.initial_amounts_mol
        if amounts is None:
            if self.geometry != LeadAcidGeometrySpec() or self.material != LeadAcidMaterialSpec():
                raise ValueError(
                    "custom LOQS geometry/material requires six explicit initial amounts"
                )
            vn, vs, vp = self.geometry.layer_volumes_m3
            inert_n, inert_p = self.geometry.inactive_solid_volumes_m3
            pb, pbo2, _, acid, water = self.material.molar_volumes_m3_mol
            pore = 0.5 * vn + self.geometry.separator_porosity * vs + 0.5 * vp
            na = 4000 * pore
            amounts = (
                (0.5 * vn - inert_n) / pb,
                0.0,
                (0.5 * vp - inert_p) / pbo2,
                0.0,
                na,
                (pore - acid * na) / water,
            )
        amounts = tuple(amounts)
        if len(amounts) != 6:
            raise ValueError("LOQS requires exactly six initial material inventories")
        _numbers(amounts, "LOQS initial amounts", positive=False)
        object.__setattr__(self, "initial_amounts_mol", amounts)
        _text(self.provenance, "LOQS provenance")
        _numbers((self.volume_l, self.usable_charge_c), "derived LOQS scales")
        validate_lead_acid_state(self, self.initial_state())
        f = _composition(self, self.initial_state())
        if not (
            f["acid_concentration_mol_m3"] > self.minimum_acid_concentration_mol_m3
            and f["water_concentration_mol_m3"] > self.minimum_water_concentration_mol_m3
            and all(self.minimum_porosity < e < self.maximum_porosity for e in f["porosities"])
        ):
            raise ValueError("initial LOQS electrolyte and pores must lie inside event limits")

    @property
    def model_family(self):
        return LOQS_FAMILY

    @property
    def model_id(self):
        return LOQS_FAMILY

    @property
    def numerical_form(self):
        return "residual"

    @property
    def inventory_keys(self):
        return LOQS_KEYS

    @property
    def inventory_species(self):
        return {
            key: "leadacid:" + species for key, species in zip(LOQS_KEYS, _COMPOUNDS, strict=True)
        }

    @property
    def species_formulas(self):
        return {"leadacid:" + species: species for species in _COMPOUNDS}

    @property
    def coordinate_keys(self):
        return BATTERY_COORDINATES

    @property
    def hysteresis_keys(self):
        return ()

    @property
    def thermal(self):
        return None

    @property
    def electrolyte(self):
        return None

    @property
    def sei(self):
        return None

    @property
    def has_directional_ocp(self):
        return False

    @property
    def series_resistance_ohm(self):
        return 0.0

    @property
    def volume_l(self):
        return 1000 * sum(self.geometry.layer_volumes_m3)

    @property
    def usable_charge_c(self):
        # A numerical scale, not an assertion that pore/voltage cutoffs are later.
        n = self.initial_amounts_mol
        return 2 * F * max(min(n[0], n[2], n[4] / 2), min(n[1], n[3], n[5] / 2))

    def current_at_c_rate(self, rate):
        if isinstance(rate, bool) or not isfinite(rate):
            raise ValueError("LOQS C rate must be finite")
        return rate * self.usable_charge_c / 3600

    def initial_coordinates(self):
        return dict.fromkeys(self.coordinate_keys, 0.0)

    def initial_state(self):
        return CoupledState(
            dict(zip(LOQS_KEYS, self.initial_amounts_mol, strict=True)),
            self.temperature_K,
            self.volume_l,
            coordinates=self.initial_coordinates(),
        )

    def to_dict(self):
        return {
            "model_family": LOQS_FAMILY,
            **{
                key: value.to_dict()
                if hasattr(value, "to_dict")
                else list(value)
                if isinstance(value, tuple)
                else value
                for key in self.__dataclass_fields__
                for value in (getattr(self, key),)
            },
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("model_family", None) != LOQS_FAMILY:
            raise ValueError("unsupported lead-acid LOQS family")
        return cls(**value)


def _composition(spec, state):
    n = np.asarray([state.amounts_mol[k] for k in LOQS_KEYS])
    pb, pbo2, sulfate, _, _ = spec.material.molar_volumes_m3_mol
    volumes = np.asarray(spec.geometry.layer_volumes_m3)
    inert_n, inert_p = spec.geometry.inactive_solid_volumes_m3
    pores = np.array(
        [
            volumes[0] - inert_n - pb * n[0] - sulfate * n[1],
            volumes[1] * spec.geometry.separator_porosity,
            volumes[2] - inert_p - pbo2 * n[2] - sulfate * n[3],
        ]
    )
    pore = float(sum(pores))
    # Only residual trials outside the logarithm domain use an extension.
    # Every physically valid positive inventory/volume is evaluated unchanged.
    v = pore if pore > 0 else spec.volume_l / 1000 * 1e-6
    acid = n[4] if n[4] > 0 else spec.minimum_acid_concentration_mol_m3 * v / 2
    water = n[5] if n[5] > 0 else spec.minimum_water_concentration_mol_m3 * v / 2
    dc = -acid / v**2 * np.array([-pb, -sulfate, -pbo2, -sulfate, 0.0, 0.0])
    dw = -water / v**2 * np.array([-pb, -sulfate, -pbo2, -sulfate, 0.0, 0.0])
    if pore <= 0:
        dc[:] = dw[:] = 0
    dc[4] = 1 / v if n[4] > 0 else 0
    dw[5] = 1 / v if n[5] > 0 else 0
    if n[4] <= 0:
        dc[:] = 0
    if n[5] <= 0:
        dw[:] = 0
    m = acid / (water * spec.material.water_molar_mass_kg_mol)
    dlogm = dc / (acid / v) - dw / (water / v)
    return {
        "amounts_mol": dict(state.amounts_mol),
        "porosities": tuple(pores / volumes),
        "pore_volumes_m3": tuple(pores),
        "total_pore_volume_m3": pore,
        "acid_concentration_mol_m3": acid / v,
        "water_concentration_mol_m3": water / v,
        "molality_mol_kg": m,
        "dc_dn": dc,
        "dw_dn": dw,
        "dlogm_dn": dlogm,
    }


def _voltage(spec, state, current):
    f = _composition(spec, state)
    ca, cw, m = (
        f[k] for k in ("acid_concentration_mol_m3", "water_concentration_mol_m3", "molality_mol_kg")
    )
    if any(not isfinite(v) or v <= 0 for v in (ca, cw, m)):
        raise ValueError("LOQS concentrations and molality must be representable and positive")
    logm = np.log10(m)
    un = -0.294 - 0.074 * logm - 0.030 * logm**2 - 0.031 * logm**3 - 0.012 * logm**4
    up = 1.628 + 0.074 * logm + 0.033 * logm**2 + 0.043 * logm**3 + 0.022 * logm**4
    ddu_dlogm = 0.148 + 0.126 * logm + 0.222 * logm**2 + 0.136 * logm**3
    material = spec.material
    cn = material.reference_acid_concentration_mol_m3
    wn = material.reference_water_concentration_mol_m3
    j0n = material.negative_exchange_current_density_A_m2 * ca / cn
    j0p = material.positive_exchange_current_density_A_m2 * (ca / cn) ** 2 * cw / wn
    areas = spec.geometry.reaction_areas_m2
    if any(not isfinite(v) or v <= 0 for v in (j0n, j0p, areas[0] * j0n, areas[1] * j0p)):
        raise ValueError("LOQS reaction kinetic scale is not representable")
    coefficient = R * state.temperature_K / F  # TWO-electron symmetric BV.
    zn, zp = current / (2 * areas[0] * j0n), -current / (2 * areas[1] * j0p)
    etan, etap = coefficient * np.arcsinh(zn), coefficient * np.arcsinh(zp)
    hn, hp = np.hypot(1, zn), np.hypot(1, zp)
    dlgn = f["dc_dn"] / ca
    dlgp = 2 * dlgn + f["dw_dn"] / cw
    dvn = ddu_dlogm / np.log(10) * f["dlogm_dn"]
    dvn += coefficient * (zn / hn * dlgn - zp / hp * dlgp)
    dvi = -coefficient * (1 / (2 * areas[0] * j0n * hn) + 1 / (2 * areas[1] * j0p * hp))
    voltage = up - un + etap - etan
    if not isfinite(voltage) or not np.all(np.isfinite(dvn)) or not isfinite(dvi) or dvi >= 0:
        raise ValueError("LOQS voltage/monotone derivative is not representable")
    return {
        **{k: v for k, v in f.items() if not k.startswith("d")},
        "current_A": float(current),
        "voltage_V": float(voltage),
        "negative_ocp_V": float(un),
        "positive_ocp_V": float(up),
        "negative_overpotential_V": float(etan),
        "positive_overpotential_V": float(etap),
        "negative_exchange_current_A_m2": float(j0n),
        "positive_exchange_current_A_m2": float(j0p),
        "negative_interfacial_current_A_m2": current / areas[0],
        "positive_interfacial_current_A_m2": -current / areas[1],
    }, (dvn, dvi, (etap - etan) / state.temperature_K)


def validate_lead_acid_state(spec, state):
    if set(state.amounts_mol) != set(LOQS_KEYS) or set(state.coordinates) != set(
        BATTERY_COORDINATES
    ):
        raise ValueError("LOQS requires exactly six true inventories and Q/W coordinates")
    if not isclose(state.temperature_K, spec.temperature_K, rel_tol=0, abs_tol=1e-8) or not isclose(
        state.volume_L, spec.volume_l, rel_tol=1e-12, abs_tol=1e-14
    ):
        raise ValueError("LOQS requires its fixed isothermal geometry")
    _numbers(state.amounts_mol.values(), "LOQS accepted inventories", positive=False)
    if any(not isfinite(v) for v in state.coordinates.values()):
        raise ValueError("LOQS coordinates must be finite")
    f = _composition(spec, state)
    if (
        state.amounts_mol[LOQS_KEYS[4]] <= 0
        or state.amounts_mol[LOQS_KEYS[5]] <= 0
        or not (f["total_pore_volume_m3"] > 0 and all(0 <= e <= 1 for e in f["porosities"]))
    ):
        raise ValueError("LOQS requires positive real acid/water and admissible pores")
    _, _, _, va, vw = spec.material.molar_volumes_m3_mol
    liquid = va * state.amounts_mol[LOQS_KEYS[4]] + vw * state.amounts_mol[LOQS_KEYS[5]]
    if not isclose(
        liquid, f["total_pore_volume_m3"], rel_tol=1e-8, abs_tol=spec.volume_l / 1000 * 1e-10
    ):
        raise ValueError("LOQS acid/water volume does not fill the actual pore volume")
    q = state.coordinates[BATTERY_COORDINATES[0]]
    for key, n0, nu in zip(LOQS_KEYS, spec.initial_amounts_mol, LOQS_STOICHIOMETRY, strict=True):
        if not isclose(
            2 * F * (state.amounts_mol[key] - n0),
            nu * q,
            rel_tol=1e-8,
            abs_tol=max(spec.usable_charge_c, 1) * 1e-10,
        ):
            raise ValueError("LOQS true material inventory and Q violate Faraday balance")


def _check_program(spec, program):
    if (
        not isinstance(program, ScalarControlProgram)
        or program.unit not in {"A", "V"}
        or any(
            abs(value) > spec.maximum_current_A
            if program.unit == "A"
            else not spec.minimum_voltage_V <= value <= spec.maximum_voltage_V
            for _, value in program.setpoints
        )
    ):
        raise ValueError("LOQS requires an in-domain CC or finite-source CV program")


def lead_acid_control_mode(spec, state, program):
    _check_program(spec, program)
    target = program.value_at(state.time_s)
    if program.unit == "A":
        return "constant_current", target
    if _voltage(spec, state, spec.maximum_current_A)[0]["voltage_V"] >= target:
        return "positive_current_limit", spec.maximum_current_A
    if _voltage(spec, state, -spec.maximum_current_A)[0]["voltage_V"] <= target:
        return "negative_current_limit", -spec.maximum_current_A
    return "constant_voltage", target


def lead_acid_algebraic(spec, program):
    _check_program(spec, program)
    ik, vk = BATTERY_ALGEBRAIC

    def initial(state):
        mode, target = lead_acid_control_mode(spec, state, program)
        current = 0.0 if mode == "constant_voltage" else target
        return {ik: current, vk: _voltage(spec, state, current)[0]["voltage_V"]}

    def residual(state):
        current, voltage = (state.algebraic_values[k] for k in BATTERY_ALGEBRAIC)
        physical = _voltage(spec, state, current)[0]["voltage_V"]
        mode, target = lead_acid_control_mode(spec, state, program)
        return np.array(
            [
                voltage - physical,
                voltage - target
                if mode == "constant_voltage"
                else (current - target) / spec.maximum_current_A,
            ]
        )

    return CoupledAlgebraic(
        (
            ResidualVariable(ik, "A", spec.maximum_current_A, "algebraic"),
            ResidualVariable(vk, "V", 1.0, "algebraic"),
        ),
        (
            ResidualEquation("LOQS:terminal-voltage", "V"),
            ResidualEquation("LOQS:control", "dimensionless"),
        ),
        initial,
        residual,
    )


def _closure_jacobian(spec, state, program, identities):
    index = {key: i for i, key in enumerate(identities)}
    ik, vk = BATTERY_ALGEBRAIC
    _, (dvn, dvi, dvt) = _voltage(spec, state, state.algebraic_values[ik])
    jac = np.zeros((2, len(identities)))
    for key, value in zip(LOQS_KEYS, dvn, strict=True):
        if key in index:
            jac[0, index[key]] = -value
    if "temperature" in index:
        jac[0, index["temperature"]] = -dvt
    jac[0, index[ik]], jac[0, index[vk]] = -dvi, 1.0
    mode, _ = lead_acid_control_mode(spec, state, program)
    jac[1, index[vk] if mode == "constant_voltage" else index[ik]] = (
        1.0 if mode == "constant_voltage" else 1 / spec.maximum_current_A
    )
    return sparse.csc_matrix(jac)


def lead_acid_fields(spec, state, *, current_A=None, program=None):
    if (current_A is None) == (program is None):
        raise ValueError("supply exactly one LOQS current or control program")
    if isinstance(state, CoupledState):
        validate_lead_acid_state(spec, state)
    if current_A is not None:
        if (
            isinstance(current_A, bool)
            or not isfinite(current_A)
            or abs(current_A) > spec.maximum_current_A
        ):
            raise ValueError("LOQS current exceeds the declared finite source")
        return _voltage(spec, state, current_A)[0]
    closure = lead_acid_algebraic(spec, program)
    if all(key in state.algebraic_values for key in BATTERY_ALGEBRAIC):
        f = _voltage(spec, state, state.algebraic_values[BATTERY_ALGEBRAIC[0]])[0]
        return {**f, "voltage_V": state.algebraic_values[BATTERY_ALGEBRAIC[1]]}

    def view(y):
        return CoupledStateView(
            state.amounts_mol,
            state.temperature_K,
            state.volume_L,
            state.time_s,
            coordinates=state.coordinates,
            algebraic_values=dict(zip(BATTERY_ALGEBRAIC, y, strict=True)),
        )

    system = ResidualSystem(
        "battery:LOQS-closure",
        closure.variables,
        closure.equations,
        lambda t, y, rate, p: closure.residual(view(y)),
        ResidualDerivatives(
            "analytic",
            state=lambda t, y, rate, p: _closure_jacobian(
                spec, view(y), program, BATTERY_ALGEBRAIC
            ),
            rate=lambda t, y, rate, p: sparse.csc_matrix((2, 2)),
        ),
    )
    guess = closure.initial_values(state)
    answer = solve_steady_residual(
        system, [guess[k] for k in BATTERY_ALGEBRAIC], policy=RootPolicy(tolerance=1e-12)
    )
    if not answer.root.success:
        raise ResidualNumericalError("LOQS electrical closure:" + answer.root.failure_reason)
    current, voltage = answer.root.value
    if abs(current) > spec.maximum_current_A * (1 + 1e-10):
        raise ResidualNumericalError("LOQS closure exceeds its finite source")
    return {
        **_voltage(spec, state, current)[0],
        "voltage_V": voltage,
        "closure_diagnostic": answer.root.diagnostics,
    }


@dataclass(frozen=True)
class LeadAcidReadout:
    voltage_V: float
    current_A: float
    acid_concentration_mol_m3: float
    water_concentration_mol_m3: float
    molality_mol_kg: float
    porosities: tuple[float, float, float]
    negative_ocp_V: float
    positive_ocp_V: float
    negative_overpotential_V: float
    positive_overpotential_V: float
    negative_exchange_current_A_m2: float
    positive_exchange_current_A_m2: float
    amounts_mol: dict[str, float]
    discharged_charge_C: float
    delivered_work_J: float


def lead_acid_readout(spec, state, current_A):
    f = lead_acid_fields(spec, state, current_A=current_A)
    return LeadAcidReadout(
        **{
            key: state.coordinates[key] if key in state.coordinates else f[key]
            for key in LeadAcidReadout.__dataclass_fields__
        }
    )


def lead_acid_profile(spec, state, *, current_A=None, program=None):
    """Return a JSON-safe read-only six-inventory LOQS profile.

    The profile is derived from the existing material state and voltage law.
    It exposes the closed pore/acid/water composition and Faradaic residuals
    without introducing a second inventory, volume, or integration path.
    """
    if current_A is None and program is None:
        current_A = 0.0
    fields = lead_acid_fields(spec, state, current_A=current_A, program=program)
    amounts = {key: float(state.amounts_mol[key]) for key in LOQS_KEYS}
    charge = float(state.coordinates[BATTERY_COORDINATES[0]])
    elements = {
        element: float(sum(count * amounts[key] for key, count in zip(
            LOQS_KEYS, counts, strict=True
        )))
        for element, counts in _ELEMENTS.items()
    }
    faradaic = {
        key: float(2 * F * (amounts[key] - initial) - nu * charge)
        for key, initial, nu in zip(
            LOQS_KEYS, spec.initial_amounts_mol, LOQS_STOICHIOMETRY, strict=True
        )
    }
    return {
        "model_family": LOQS_FAMILY,
        "inventory_keys": list(LOQS_KEYS),
        "amounts_mol": amounts,
        "coordinates": {key: float(value) for key, value in state.coordinates.items()},
        "volume_L": float(state.volume_L),
        "current_A": float(fields["current_A"]),
        "voltage_V": float(fields["voltage_V"]),
        "acid_concentration_mol_m3": float(fields["acid_concentration_mol_m3"]),
        "water_concentration_mol_m3": float(fields["water_concentration_mol_m3"]),
        "molality_mol_kg": float(fields["molality_mol_kg"]),
        "porosities": [float(value) for value in fields["porosities"]],
        "pore_volumes_m3": [float(value) for value in fields["pore_volumes_m3"]],
        "element_mol": elements,
        "faradaic_residual_C": faradaic,
        "negative_ocp_V": float(fields["negative_ocp_V"]),
        "positive_ocp_V": float(fields["positive_ocp_V"]),
        "negative_overpotential_V": float(fields["negative_overpotential_V"]),
        "positive_overpotential_V": float(fields["positive_overpotential_V"]),
        "negative_interfacial_current_A_m2": float(
            fields["negative_interfacial_current_A_m2"]
        ),
        "positive_interfacial_current_A_m2": float(
            fields["positive_interfacial_current_A_m2"]
        ),
    }


def lead_acid_control_current(spec, state, program):
    return lead_acid_fields(spec, state, program=program)["current_A"]


class _LeadAcidThermodynamics:
    inventory_keys = LOQS_KEYS

    def heat_capacity_j_k(self, state):
        return 1.0

    def internal_energy_j(self, state):
        return None

    def chemical_energy_rate_w(self, state, rates, contributions):
        return 0.0


def lead_acid_system(spec, program, *, thermal_device=None, control_events=()):
    if thermal_device is not None:
        raise ValueError("lead-acid LOQS is isothermal; external thermal coupling is unsupported")
    closure = lead_acid_algebraic(spec, program)

    def fields(state):
        return lead_acid_fields(spec, state, program=program)

    def evaluate(state):
        f = fields(state)
        current, voltage = f["current_A"], f["voltage_V"]
        return ProcessContribution(
            amount_rates_mol_s={
                key: nu * current / (2 * F)
                for key, nu in zip(LOQS_KEYS, LOQS_STOICHIOMETRY, strict=True)
            },
            coordinate_rates={
                "discharged_charge_C": current,
                "delivered_work_J": current * voltage,
            },
        )

    events = []
    for key, nu in zip(LOQS_KEYS[:4], LOQS_STOICHIOMETRY[:4], strict=True):
        events.append(
            CoupledEvent(
                f"lead-acid:{key}:depleted",
                lambda s, k=key, v=nu: s.amounts_mol[k] if v * fields(s)["current_A"] < 0 else 1.0,
            )
        )
    for name, minimum in (
        ("acid", spec.minimum_acid_concentration_mol_m3),
        ("water", spec.minimum_water_concentration_mol_m3),
    ):
        events.append(
            CoupledEvent(
                f"lead-acid:minimum-{name}",
                lambda s, n=name, m=minimum: (
                    fields(s)[n + "_concentration_mol_m3"] - m
                    if (fields(s)["current_A"] > 0 if n == "acid" else fields(s)["current_A"] < 0)
                    else 1.0
                ),
            )
        )
    for index, name in ((0, "negative"), (2, "positive")):
        # Pore rate sign depends on the explicitly declared molar volumes.
        pb, pbo2, sulfate, _, _ = spec.material.molar_volumes_m3_mol
        change = (pb if index == 0 else pbo2) - sulfate
        for bound, limit, sense in (
            ("minimum", spec.minimum_porosity, 1),
            ("maximum", spec.maximum_porosity, -1),
        ):
            events.append(
                CoupledEvent(
                    f"lead-acid:{name}:{bound}-porosity",
                    lambda s, i=index, bound_value=limit, d=sense, c=change: (
                        d * (fields(s)["porosities"][i] - bound_value)
                        if d * c * fields(s)["current_A"] < 0
                        else 1.0
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
        ik, vk = BATTERY_ALGEBRAIC
        jac = np.zeros((len(identities), len(identities)))
        for key, nu in zip(LOQS_KEYS, LOQS_STOICHIOMETRY, strict=True):
            jac[index[key], index[ik]] = nu / (2 * F)
        jac[index[BATTERY_COORDINATES[0]], index[ik]] = 1.0
        jac[index[BATTERY_COORDINATES[1]], index[ik]] = state.algebraic_values[vk]
        jac[index[BATTERY_COORDINATES[1]], index[vk]] = state.algebraic_values[ik]
        jac[[index[k] for k in BATTERY_ALGEBRAIC], :] = -_closure_jacobian(
            spec, state, program, identities
        ).toarray()
        return sparse.csc_matrix(jac)

    laws = {
        "element:" + element: dict(zip(LOQS_KEYS, counts, strict=True))
        for element, counts in _ELEMENTS.items()
    }
    for key, nu in zip(LOQS_KEYS, LOQS_STOICHIOMETRY, strict=True):
        laws["faradaic:" + key] = {key: 2 * F, BATTERY_COORDINATES[0]: -nu}
    return CoupledSystem(
        _LeadAcidThermodynamics(),
        (
            CoupledProcess(
                "lead-acid:two-electron",
                evaluate,
                validate_state=lambda s: validate_lead_acid_state(spec, s),
            ),
        ),
        isothermal=True,
        numerical_form="residual",
        algebraic=closure,
        residual_rhs_jacobian=jacobian,
        events=(*events, *control_events),
        inventory_scales_mol={
            key: max(n, spec.usable_charge_c / (2 * F))
            for key, n in zip(LOQS_KEYS, spec.initial_amounts_mol, strict=True)
        },
        coordinates=(
            CoupledCoordinate(BATTERY_COORDINATES[0], "C", spec.usable_charge_c),
            CoupledCoordinate(BATTERY_COORDINATES[1], "J", spec.usable_charge_c * 2),
        ),
        conservation_laws=laws,
    )


def advance_lead_acid(
    spec,
    state,
    program,
    duration_s,
    *,
    evaluation_times_s=None,
    thermal_device=None,
    control_events=(),
):
    result = lead_acid_system(
        spec, program, thermal_device=thermal_device, control_events=control_events
    ).advance(state, duration_s, evaluation_times_s=evaluation_times_s)
    return replace(
        result,
        solver_diagnostic={
            **result.solver_diagnostic,
            "battery_control_mode": lead_acid_control_mode(spec, result.final_state, program)[0],
            "energy_scope": (
                "isothermal LOQS; terminal electrical work only; no chemical/thermal closure"
            ),
        },
    )
