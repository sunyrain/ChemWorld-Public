"""Finite planar O + e <-> R with a nonlinear Randles boundary.

The common CoupledSystem integrates both Fick inventories and signed electrical
coordinates. Positive current is anodic (R -> O). The supported electrolyte is
isothermal; no assertion of complete cell energy or counterelectrode closure is
made. Coordinates are a transient view of caller-owned material and equipment.
"""

from dataclasses import asdict, dataclass
from functools import cached_property
from math import isfinite

import numpy as np

from chemworld.physchem.coupled_system import (
    CoupledCoordinate,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    ProcessContribution,
)
from chemworld.physchem.electrochemistry import (
    FARADAY_C_PER_MOL,
    butler_volmer_factors,
)
from chemworld.physchem.finite_volume import ConservativeDiffusion, DiffusionMesh
from chemworld.physchem.reactor_shared import ScalarControlProgram

F = FARADAY_C_PER_MOL
ELECTRICAL_COORDINATES = (
    "double_layer_charge_C",
    "applied_charge_C",
    "faradaic_charge_C",
    "electrical_work_J",
    "ohmic_loss_J",
)


@dataclass(frozen=True)
class TransientElectrodeSpec:
    area_m2: float = 1e-4
    thickness_m: float = 1e-3
    cells: int = 40
    mesh_exponent: float = 2.0
    diffusivity_O_m2_s: float = 1e-9
    diffusivity_R_m2_s: float = 1e-9
    standard_rate_m_s: float = 1e-4
    standard_potential_V: float = 0.0
    alpha_anodic: float = 0.5
    alpha_cathodic: float = 0.5
    capacitance_F_m2: float = 0.2
    series_resistance_ohm: float = 10.0
    potential_of_zero_charge_V: float = 0.0
    maximum_potential_V: float = 1.0
    temperature_K: float = 298.15
    oxidized_species: str = "O"
    reduced_species: str = "R"
    provenance: str = "synthetic-planar-supported-electrolyte-v1"

    def __post_init__(self):
        for name in (
            "area_m2",
            "thickness_m",
            "diffusivity_O_m2_s",
            "diffusivity_R_m2_s",
            "alpha_anodic",
            "alpha_cathodic",
            "capacitance_F_m2",
            "series_resistance_ohm",
            "maximum_potential_V",
            "temperature_K",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.mesh_exponent) or not 1 <= self.mesh_exponent <= 3:
            raise ValueError("mesh_exponent must be in [1, 3]")
        if not isfinite(self.standard_rate_m_s) or self.standard_rate_m_s < 0:
            raise ValueError("standard_rate_m_s must be finite and nonnegative")
        for value in (self.standard_potential_V, self.potential_of_zero_charge_V):
            if not isfinite(value) or abs(value) >= self.maximum_potential_V:
                raise ValueError("reference potential is outside the finite potential domain")
        if abs(self.alpha_anodic + self.alpha_cathodic - 1) > 1e-12:
            raise ValueError("one-electron elementary BV requires alpha_a + alpha_c = 1")
        if (
            not self.oxidized_species
            or not self.reduced_species
            or self.oxidized_species == self.reduced_species
            or not self.provenance
        ):
            raise ValueError("distinct species identities and parameter provenance are required")
        _ = self.mesh  # Validate the finite integer mesh through its actual owner.

    @cached_property
    def mesh(self):
        # Nested grids resolve the short-time interfacial layer without adding
        # another diffusion approximation or changing the physical domain.
        uniform = DiffusionMesh.planar(self.thickness_m, self.cells, area_m2=self.area_m2)
        faces = (
            self.thickness_m
            * (np.asarray(uniform.faces_m) / self.thickness_m) ** self.mesh_exponent
        )
        return DiffusionMesh("planar", tuple(faces), area_m2=self.area_m2)

    @property
    def capacitance_f(self):
        return self.capacitance_F_m2 * self.area_m2

    @property
    def volume_l(self):
        return self.area_m2 * self.thickness_m * 1000

    def keys(self, species):
        return tuple(f"{species}:cell:{i}" for i in range(self.cells))

    @property
    def inventory_keys(self):
        return (*self.keys(self.oxidized_species), *self.keys(self.reduced_species))

    def to_dict(self):
        return asdict(self)

    def initial_state(self, concentration_O=1.0, concentration_R=1.0, *, potential_V=0.0):
        if any(not isfinite(v) or v < 0 for v in (concentration_O, concentration_R)):
            raise ValueError("initial concentrations must be finite and nonnegative")
        if not isfinite(potential_V) or abs(potential_V) > self.maximum_potential_V:
            raise ValueError("initial potential is outside the electrode domain")
        amounts = {}
        for species, c in (
            (self.oxidized_species, concentration_O),
            (self.reduced_species, concentration_R),
        ):
            amounts.update(zip(self.keys(species), c * self.mesh.cell_volumes_m3, strict=True))
        coords = dict.fromkeys(ELECTRICAL_COORDINATES, 0.0)
        coords["double_layer_charge_C"] = self.capacitance_f * (
            potential_V - self.potential_of_zero_charge_V
        )
        return CoupledState(amounts, self.temperature_K, self.volume_l, coordinates=coords)


@dataclass(frozen=True)
class ElectrodeReadout:
    terminal_potential_V: float
    interfacial_potential_V: float
    applied_current_A: float
    faradaic_current_A: float
    capacitive_current_A: float
    surface_O_mol_m3: float
    surface_R_mol_m3: float


def electrode_readout(spec, state, terminal_potential_V):
    """Couple BV at the actual face to both half-cell diffusion resistances."""
    phi = (
        spec.potential_of_zero_charge_V
        + state.coordinates["double_layer_charge_C"] / spec.capacitance_f
    )
    anodic, cathodic = butler_volmer_factors(
        phi - spec.standard_potential_V,
        temperature_K=state.temperature_K,
        alpha_anodic=spec.alpha_anodic,
        alpha_cathodic=spec.alpha_cathodic,
    )
    ko, kr = spec.standard_rate_m_s * anodic, spec.standard_rate_m_s * cathodic
    volume = spec.mesh.cell_volumes_m3[0]
    co = state.amounts_mol[spec.keys(spec.oxidized_species)[0]] / volume
    cr = state.amounts_mol[spec.keys(spec.reduced_species)[0]] / volume
    distance = spec.mesh.centers_m[0]
    ro, rr = distance / spec.diffusivity_O_m2_s, distance / spec.diffusivity_R_m2_s
    # Solve the linear Robin face relation analytically. Using cell-center
    # concentrations directly in BV would lose the half-cell resistance.
    denominator = 1 + kr * ro + ko * rr
    reduction_flux = (kr * co - ko * cr) / denominator
    # Algebraically nonnegative face concentrations, avoiding cancellation.
    surface_o = (co * (1 + ko * rr) + ko * cr * ro) / denominator
    surface_r = (cr * (1 + kr * ro) + kr * co * rr) / denominator
    faradaic = -F * spec.area_m2 * reduction_flux
    applied = (terminal_potential_V - phi) / spec.series_resistance_ohm
    return ElectrodeReadout(
        terminal_potential_V, phi, applied, faradaic, applied - faradaic, surface_o, surface_r
    )


class _IsothermalElectrolyte:
    """Temperature is prescribed; full cell U is intentionally not supplied."""

    def __init__(self, spec):
        self.inventory_keys = spec.inventory_keys

    def heat_capacity_j_k(self, state):
        return 1.0  # Unused normalization of the prescribed dT/dt = 0.

    def internal_energy_j(self, state):
        return None

    def chemical_energy_rate_w(self, state, derivatives, contributions):
        return 0.0


def electrode_system(spec: TransientElectrodeSpec, program: ScalarControlProgram):
    if program.unit != "V" or any(abs(v) > spec.maximum_potential_V for _, v in program.setpoints):
        raise ValueError("electrode control must have in-domain potential setpoints in V")
    o_keys, r_keys = spec.keys(spec.oxidized_species), spec.keys(spec.reduced_species)
    processes = [
        ConservativeDiffusion(spec.mesh, diffusivity)
        .couple(keys, process_id=f"diffusion:{species}", owner_id="electrolyte")
        .process
        for species, keys, diffusivity in (
            (spec.oxidized_species, o_keys, spec.diffusivity_O_m2_s),
            (spec.reduced_species, r_keys, spec.diffusivity_R_m2_s),
        )
    ]

    def interface(state):
        read = electrode_readout(spec, state, program.value_at(state.time_s))
        rate = read.faradaic_current_A / F
        return ProcessContribution(
            amount_rates_mol_s={o_keys[0]: rate, r_keys[0]: -rate},
            reaction_progress_mol_s={"electrode:O+e=R": -rate},
            coordinate_rates={
                "double_layer_charge_C": read.capacitive_current_A,
                "applied_charge_C": read.applied_current_A,
                "faradaic_charge_C": read.faradaic_current_A,
                "electrical_work_J": read.terminal_potential_V * read.applied_current_A,
                "ohmic_loss_J": read.applied_current_A**2 * spec.series_resistance_ohm,
            },
        )

    def validate(state):
        if (
            abs(state.temperature_K - spec.temperature_K) > 1e-8
            or abs(state.volume_L - spec.volume_l) > 1e-12
        ):
            raise ValueError("electrode requires its declared isothermal finite liquid layer")

    processes.append(
        CoupledProcess(
            "electrode:charge-transfer",
            interface,
            validate_state=validate,
            inventory_dependencies={
                o_keys[0]: (o_keys[0], r_keys[0]),
                r_keys[0]: (o_keys[0], r_keys[0]),
            },
        )
    )
    charge_scale = max(spec.capacitance_f, F * spec.area_m2 * spec.thickness_m)
    coordinates = tuple(
        CoupledCoordinate(key, "J" if key.endswith("_J") else "C", charge_scale)
        for key in ELECTRICAL_COORDINATES
    )

    # Terminal setpoints are constrained; the interface still has independent
    # physical stopping events, useful after a material/electrode replacement.
    def phi(state):
        return (
            spec.potential_of_zero_charge_V
            + state.coordinates["double_layer_charge_C"] / spec.capacitance_f
        )

    return CoupledSystem(
        _IsothermalElectrolyte(spec),
        tuple(processes),
        isothermal=True,
        coordinates=coordinates,
        inventory_scales_mol=dict.fromkeys(
            spec.inventory_keys, spec.area_m2 * spec.thickness_m / spec.cells
        ),
        events=(
            CoupledEvent(
                "electrode:positive-potential-limit", lambda s: spec.maximum_potential_V - phi(s)
            ),
            CoupledEvent(
                "electrode:negative-potential-limit", lambda s: spec.maximum_potential_V + phi(s)
            ),
        ),
        conservation_laws={
            "O_plus_R_mol": dict.fromkeys(spec.inventory_keys, 1.0),
            "applied_faradaic_capacitive_C": {
                "applied_charge_C": 1,
                "faradaic_charge_C": -1,
                "double_layer_charge_C": -1,
            },
            "faradaic_inventory_C": {**dict.fromkeys(o_keys, F), "faradaic_charge_C": -1},
        },
    )


def advance_electrode(spec, state, program, duration_s, *, evaluation_times_s=None):
    """Independent physics entry, using exactly the runtime's shared system."""
    system = electrode_system(spec, program)
    return system.advance(state, duration_s, evaluation_times_s=evaluation_times_s)
