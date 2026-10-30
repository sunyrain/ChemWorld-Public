"""Isothermal, binary-electrolyte SPMe contributions; no private time integrator.

Equation reference: PyBaMM 25.4.2 SPMe, full electrolyte diffusion and composite /
integrated electrolyte conductivity (BSD-3-Clause). Independent constant-property
realisation using ChemWorld's conservative porous finite-volume operator.
"""

from dataclasses import asdict, dataclass
from functools import cached_property
from math import isfinite
from numbers import Integral

import numpy as np

from chemworld.physchem.coupled_system import CoupledProcess, ProcessContribution
from chemworld.physchem.electrochemistry import FARADAY_C_PER_MOL, R_J_PER_MOL_K
from chemworld.physchem.finite_volume import ConservativeDiffusion, DiffusionMesh

REGIONS = ("negative", "separator", "positive")
ELECTROLYTE_SPECIES = "battery:LiPF6"


@dataclass(frozen=True)
class BatteryElectrolyteSpec:
    """Explicit SPMe-v1 model and parameter domain; effective sigma is in S/m."""

    model_version: str = "spme-isothermal-v1"
    area_m2: float = 0.02
    lengths_m: tuple[float, float, float] = (1e-4, 2.5e-5, 1e-4)
    porosities: tuple[float, float, float] = (0.4, 0.5, 0.4)
    bruggeman: tuple[float, float, float] = (1.5, 1.5, 1.5)
    cells: tuple[int, int, int] = (20, 20, 20)
    diffusivity_m2_s: float = 2e-10
    conductivity_S_m: float = 1.0
    cation_transference: float = 0.4
    thermodynamic_factor: float = 1.0
    initial_concentration_mol_m3: float = 1000.0
    minimum_concentration_mol_m3: float = 1.0
    solid_conductivity_S_m: tuple[float, float] = (100.0, 100.0)
    potential_form: str = "composite"

    def __post_init__(self) -> None:
        if self.model_version != "spme-isothermal-v1":
            raise ValueError("unknown electrolyte model version")
        if self.potential_form not in {"composite", "integrated"}:
            raise ValueError("unknown electrolyte potential form")
        for name, count in (("lengths_m", 3), ("porosities", 3), ("bruggeman", 3),
                            ("cells", 3), ("solid_conductivity_S_m", 2)):
            values = tuple(getattr(self, name))
            if len(values) != count or any(not isfinite(x) or x <= 0 for x in values):
                raise ValueError(f"{name} requires {count} positive finite entries")
            object.__setattr__(self, name, values)
        if any(not isinstance(n, Integral) or isinstance(n, bool) or n < 2 for n in self.cells):
            raise ValueError("each electrolyte region requires at least two integer cells")
        if any(e >= 1 for e in self.porosities):
            raise ValueError("porosities must be strictly between zero and one")
        for name in ("area_m2", "diffusivity_m2_s", "conductivity_S_m", "thermodynamic_factor",
                     "initial_concentration_mol_m3", "minimum_concentration_mol_m3"):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 <= self.cation_transference < 1:
            raise ValueError("cation transference must be in [0, 1)")
        if self.minimum_concentration_mol_m3 >= self.initial_concentration_mol_m3:
            raise ValueError("initial electrolyte must be above the depletion cutoff")

    @cached_property
    def mesh(self):
        faces = [0.0]
        for length, count in zip(self.lengths_m, self.cells, strict=True):
            faces.extend(faces[-1] + np.linspace(0, length, count + 1)[1:])
        return DiffusionMesh("planar", tuple(faces), area_m2=self.area_m2)

    @cached_property
    def region_slices(self):
        cuts = np.r_[0, np.cumsum(self.cells)]
        return {name: slice(int(cuts[i]), int(cuts[i + 1])) for i, name in enumerate(REGIONS)}

    @cached_property
    def keys(self):
        return tuple(f"electrolyte:salt:{i}" for i in range(sum(self.cells)))

    @cached_property
    def diffusion(self):
        efficiencies = np.power(self.porosities, self.bruggeman)
        return ConservativeDiffusion(
            self.mesh, tuple(np.repeat(efficiencies * self.diffusivity_m2_s, self.cells)),
            storage_fractions=tuple(np.repeat(self.porosities, self.cells)),
        )

    def concentrations(self, state):
        return np.array([state.amounts_mol[k] for k in self.keys]) / (
            self.diffusion.storage_volumes_m3
        )

    def minimum_resolved_concentration(self, state):
        """Include collector boundaries, where the salt gradient is zero.

        The persistent concentrations are cell averages. A quadratic satisfying
        the Neumann boundary reconstructs c_boundary=(7*c0-c1)/6 on a uniform
        mesh, not the minimum of cell averages a half-cell inside the domain.
        No inventory is adjusted and the signed reconstructed value is retained.
        """
        c = self.concentrations(state)
        widths = np.diff(self.mesh.faces_m)
        boundaries = []
        for values, hs in ((c, widths), (c[::-1], widths[::-1])):
            h0, h1 = hs[:2]
            weight = h0**2 / ((h0 + h1) * (2 * h0 + h1))
            boundaries.append(values[0] + weight * (values[0] - values[1]))
        return float(min(*boundaries, np.min(c)))

    def initial_amounts(self):
        return dict(zip(self.keys, self.diffusion.storage_volumes_m3
                        * self.initial_concentration_mol_m3, strict=True))

    def to_dict(self):
        return {key: list(value) if isinstance(value, tuple) else value
                for key, value in asdict(self).items()}

    def region_average(self, values, region):
        selected = self.region_slices[region]
        return float(np.average(np.asarray(values)[selected],
                                weights=self.mesh.cell_volumes_m3[selected]))

    def kinetics_factor(self, state, region):
        # Trial extension is used only below the physical positive cutoff; it
        # changes no stored amounts. Accepted trajectories stop before this point.
        c = np.maximum(self.concentrations(state), self.minimum_concentration_mol_m3 / 2)
        return self.region_average(np.sqrt(c / self.initial_concentration_mol_m3), region)

    def potential(self, state, current_A):
        """Return centre potentials in a zero-mean-negative-electrode gauge.

        Constant transport makes the composite and integrated formulations
        identical. Integrate the piecewise-linear ionic current analytically,
        not through a second numerical Poisson solver.
        """
        c = np.maximum(self.concentrations(state), self.minimum_concentration_mol_m3 / 2)
        ln_c = np.log(c / self.initial_concentration_mol_m3)
        coefficient = (2 * R_J_PER_MOL_K * state.temperature_K / FARADAY_C_PER_MOL
                       * (1 - self.cation_transference) * self.thermodynamic_factor)
        lengths = self.lengths_m
        conductivities = self.conductivity_S_m * np.power(self.porosities, self.bruggeman)
        current_density = current_A / self.area_m2
        x = self.mesh.centers_m
        ionic_current = np.empty_like(x)
        resistance_integral = np.empty_like(x)
        offset = resistance = 0.0
        for i, name in enumerate(REGIONS):
            selected = self.region_slices[name]
            local = x[selected] - offset
            length, conductivity = lengths[i], conductivities[i]
            if i == 0:
                factor = local / length
                integral = local**2 / (2 * length * conductivity)
                full = length / (2 * conductivity)
            elif i == 1:
                factor = np.ones_like(local)
                integral = local / conductivity
                full = length / conductivity
            else:
                factor = 1 - local / length
                integral = (local - local**2 / (2 * length)) / conductivity
                full = length / (2 * conductivity)
            ionic_current[selected] = current_density * factor
            resistance_integral[selected] = current_density * (resistance + integral)
            offset += length
            resistance += full
        concentration_voltage = coefficient * (
            self.region_average(ln_c, "positive") - self.region_average(ln_c, "negative")
        )
        # Exact region means of the quadratic ohmic potential, independent of grid.
        electrolyte_ohmic = -current_density * (
            lengths[0] / (3 * conductivities[0]) + lengths[1] / conductivities[1]
            + lengths[2] / (3 * conductivities[2])
        )
        solid_ohmic = -current_density / 3 * (
            lengths[0] / self.solid_conductivity_S_m[0]
            + lengths[2] / self.solid_conductivity_S_m[1]
        )
        phi = coefficient * ln_c - resistance_integral
        # Set the continuum negative-electrode average to zero. Concentration
        # averages use the same finite-volume quadrature as the voltage law.
        phi -= coefficient * self.region_average(ln_c, "negative") - (
            current_density * lengths[0] / (6 * conductivities[0])
        )
        return {
            "potential_V": phi, "ionic_current_A_m2": ionic_current,
            "concentration_overpotential_V": concentration_voltage,
            "electrolyte_ohmic_V": electrolyte_ohmic, "solid_ohmic_V": solid_ohmic,
        }

    def processes(self, control_current):
        diffusion = self.diffusion.couple(
            self.keys, process_id="battery:electrolyte-diffusion", owner_id="battery",
        ).process
        source_weights = np.zeros(len(self.keys))
        source_weights[self.region_slices["negative"]] = 1 / self.cells[0]
        source_weights[self.region_slices["positive"]] = -1 / self.cells[2]

        def source(state):
            current = control_current(state)
            rates = source_weights * ((1 - self.cation_transference) * current
                                      / FARADAY_C_PER_MOL)
            return ProcessContribution(amount_rates_mol_s=dict(zip(self.keys, rates, strict=True)))

        return (diffusion, CoupledProcess("battery:electroneutral-salt-redistribution", source))
