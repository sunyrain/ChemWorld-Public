"""Many-particle electrodes: independent size inventories, shared BV potentials.

The ideal uniform electrolyte is the declared MPM reduction. No private time
integrator, prescribed current fractions or averaged independent cells are used.
"""

from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np

from chemworld.physchem import battery_particle_electrodes as shared
from chemworld.physchem.battery_models import (
    BATTERY_ALGEBRAIC,
    BatteryReadout,
    BatterySPMSpec,
    F,
    particle_surface_fraction,
)
from chemworld.physchem.battery_ocp import (
    hysteresis_key,
)
from chemworld.physchem.battery_particles import ParticleSizeDistribution
from chemworld.physchem.coupled_system import (
    CoupledState,
)
from chemworld.physchem.finite_volume import ConservativeDiffusion

MPM_ID = "mpm-isothermal-v1"
PHI_E = "battery:mpm:electrolyte-potential_V"


@dataclass(frozen=True)
class BatteryMPMSpec(BatterySPMSpec):
    negative_distribution: ParticleSizeDistribution | None = None
    positive_distribution: ParticleSizeDistribution | None = None
    model_family: str = MPM_ID
    provenance: str = "synthetic-MPM-v1"

    def __post_init__(self) -> None:
        # The base validates enabled plating against actual bin areas. Decode
        # PSDs before that validation, including when restoring a JSON payload.
        for name in ("negative_distribution", "positive_distribution"):
            value = getattr(self, name)
            if isinstance(value, dict):
                value = ParticleSizeDistribution.from_dict(value)
                object.__setattr__(self, name, value)
            if value is not None and not isinstance(value, ParticleSizeDistribution):
                raise ValueError("MPM requires a declared particle size distribution")
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

    @property
    def model_id(self):
        return MPM_ID if self.plating is None else "mpm-plating-isothermal-v1"

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
        from chemworld.physchem.battery_plating import PLATING_SPECIES

        return {**{key: "Li" for _, _, _, keys, _ in self.particle_bins for key in keys},
                **({} if self.plating is None else PLATING_SPECIES)}

    @cached_property
    def counts(self):
        return tuple(
            sum(name == electrode for name, *_ in self.particle_bins)
            for electrode in ("negative", "positive")
        )

    @cached_property
    def current_coordinates(self):
        return shared.current_coordinates(self)

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
        from chemworld.physchem.battery_plating import STRIPPING_CURRENT

        return (*self.reaction_keys, *((STRIPPING_CURRENT,) if self.plating is not None else ()),
                PHI_E, *BATTERY_ALGEBRAIC)

    @cached_property
    def particle_side_reaction(self):
        from chemworld.physchem.battery_plating import _PlatingSideReaction

        return None if self.plating is None else _PlatingSideReaction(self)

    @cached_property
    def reaction_map(self):
        return shared.reaction_map(self)

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
        if self.plating is not None:
            amounts.update(self.plating.initial_amounts())
        return CoupledState(
            amounts,
            self.temperature_K,
            self.volume_l,
            coordinates=self.initial_coordinates(),
        )

    def to_dict(self):
        result = super().to_dict()
        for name in ("negative_distribution", "positive_distribution"):
            distribution = getattr(self, name)
            result[name] = None if distribution is None else distribution.to_dict()
        return result


    @property
    def electrolyte_potential_key(self):
        return PHI_E

    @property
    def closure_id(self):
        return "battery:mpm-closure"

    @property
    def intercalation_id(self):
        return "battery:mpm-intercalation"

    @property
    def population_equations(self):
        return tuple(f"battery:bv:{name}:size:{i}" for name, i, *_ in self.particle_bins)

    def population_ocp_derivatives(self, population, theta, current, state):
        name = population[0]
        u, slope, di, dh = self.ocp_derivatives(name, theta, current, state)
        return u, slope, di, {hysteresis_key(name): dh}

    def electrode_initial_potential(self, name, state, current):
        total = sum(state.amounts_mol[k] for n, _, _, keys, _ in self.particle_bins
                    if n == name for k in keys)
        return self.operational_ocp(name, total / getattr(self, name).capacity_mol, current, state)

    def validate_particle_owner(self, state):
        if self.plating is not None:
            from chemworld.physchem.battery_plating import validate_plating_state

            validate_plating_state(self, state)
        # Historical no-plating MPM checks remain in the shared kernel.

    def particle_events(self, fields):
        return shared.legacy_particle_events(self, fields)


def _fields(spec, state, values):
    return shared._fields(spec, state, values)


def mpm_algebraic(spec, program):
    return shared.particle_algebraic(spec, program)


def _closure_jacobian(spec, state, program, identities):
    return shared._closure_jacobian(spec, state, program, identities)


def mpm_fields(spec, state, *, current_A=None, program=None):
    if spec.plating is not None:
        from chemworld.physchem.battery_plating import plating_fields

        return plating_fields(spec, state, current_A=current_A, program=program)
    return shared.particle_fields(spec, state, current_A=current_A, program=program)


def mpm_system(spec, program, *, thermal_device=None, control_events=()):
    return shared.particle_system(spec, program, thermal_device=thermal_device,
                                  control_events=control_events)


def mpm_readout(spec, state, current_A):
    if spec.plating is not None:
        from chemworld.physchem.battery_plating import plating_readout

        return plating_readout(spec, state, current_A)
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

    if spec.plating is not None:
        raise ValueError("plating does not declare complete chemical energy or dissipation")
    if spec.has_directional_ocp:
        raise ValueError(
            "directional OCP does not declare complete hysteresis energy or dissipation")
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
