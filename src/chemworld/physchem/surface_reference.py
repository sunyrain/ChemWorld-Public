"""Optional Cantera comparison for the finite synthetic surface mechanism.

This module is an evaluator-side unit conversion, never a world provider.
No upstream mechanism data or kinetic implementation is copied. Equations and
parameters come from ChemWorld's authored development specification; Cantera
independently evaluates mass action and integrates its reactor/surface system.
"""

import json
from math import exp

from chemworld.physchem.surface_rates import adsorbate_key
from chemworld.physchem.surface_specs import (
    DEFAULT_SURFACE_MECHANISM,
    GAS_SPECIES,
    surface_species_energies,
)


class CanteraSurfaceReference:
    def __init__(self, state, slot, spec=DEFAULT_SURFACE_MECHANISM):
        import cantera as ct

        self.ct, self.slot = ct, slot
        self.version = ct.__version__
        self.volume_L = state.volume_L
        energies = surface_species_energies()

        def species(name, composition, *, gas=False):
            item = ct.Species(name, composition)
            internal = energies.get(name)
            # Thermal properties are only needed to instantiate Cantera phases.
            # They do not define the independent finite-Cv heat-coupling check.
            cv = 0.0 if internal is None else internal.cv_J_mol_K
            cp = cv + ct.gas_constant / 1000 if gas else max(cv, 1.0)
            item.thermo = ct.ConstantCp(200, 1000, 100000, [350, 0, 0, cp * 1000])
            return item

        gas_species = [
            species(name, composition, gas=True)
            for name, composition in (
                ("CO", {"C": 1, "O": 1}),
                ("O2", {"O": 2}),
                ("CO2", {"C": 1, "O": 2}),
                ("N2", {"N": 2}),
            )
        ]
        # Pt is an inert reference site label, not a fitted platinum mechanism.
        surface_species = [
            species("vacant", {"Pt": 1}),
            species("CO_ads", {"Pt": 1, "C": 1, "O": 1}),
            species("O_ads", {"Pt": 1, "O": 1}),
        ]
        gamma = slot.values["site_density_mol_m2"] / 1000  # kmol/m2
        factors = (
            ct.gas_constant / spec.reference_pressure_Pa,
            1.0,
            ct.gas_constant / (spec.reference_pressure_Pa * gamma),
            1 / gamma,
            1 / gamma,
        )
        reactions = []
        equations = (
            "CO + vacant => CO_ads",
            "CO_ads => CO + vacant",
            "O2 + 2 vacant => 2 O_ads",
            "2 O_ads => O2 + 2 vacant",
            "CO_ads + O_ads => CO2 + 2 vacant",
        )
        for i, equation in enumerate(equations):
            energy = spec.activation_energies_J_mol[i] * 1000
            coefficient = (
                spec.rate_constants_s_inv[i]
                * exp(energy / (ct.gas_constant * spec.reference_temperature_K))
                * factors[i]
            )
            rate = ct.InterfaceArrheniusRate(coefficient, 1 if i in (0, 2) else 0, energy)
            if i == 4:
                rate.coverage_dependencies = {
                    "CO_ads": {
                        "a": 0.0,
                        "m": 0.0,
                        "E": spec.coverage_activation_energy_J_mol * 1000,
                    }
                }
            reactions.append(ct.Reaction(equation=equation, rate=rate))
        # Explicit phase adjacency is required by ReactorSurface. Construct the
        # authored mechanism in memory through Cantera's native YAML/JSON input.
        mechanism = {
            "phases": [
                {
                    "name": "gas",
                    "thermo": "ideal-gas",
                    "kinetics": "gas",
                    "species": list(GAS_SPECIES),
                    "reactions": "none",
                },
                {
                    "name": "surface",
                    "thermo": "ideal-surface",
                    "kinetics": "surface",
                    "species": [s.name for s in surface_species],
                    "adjacent-phases": ["gas"],
                    "site-density": gamma,
                    "reactions": "all",
                },
            ],
            "species": [dict(s.input_data) for s in (*gas_species, *surface_species)],
            "reactions": [dict(r.input_data) for r in reactions],
        }
        self.surface = ct.Interface(yaml=json.dumps(mechanism), name="surface")
        self.gas = self.surface.adjacent["gas"]
        volume = state.volume_L / 1000
        pressure = sum(state.amounts_mol[key] for key in GAS_SPECIES) * (
            ct.gas_constant / 1000 * state.temperature_K / volume
        )
        self.gas.TPX = (
            state.temperature_K,
            pressure,
            {key: state.amounts_mol[key] for key in GAS_SPECIES},
        )
        self.surface.TP = state.temperature_K, pressure
        sites = slot.values["sites_mol"]
        co = state.amounts_mol[adsorbate_key(slot, "CO_ads")] / sites
        oxygen = state.amounts_mol[adsorbate_key(slot, "O_ads")] / sites
        self.surface.coverages = {"vacant": 1 - co - oxygen, "CO_ads": co, "O_ads": oxygen}

    def per_site_rates_s_inv(self):
        return tuple(
            float(x / self.surface.site_density) for x in self.surface.net_rates_of_progress
        )

    def trajectory(self, times):
        ct = self.ct
        reactor = ct.IdealGasReactor(
            self.gas, energy="off", volume=self.volume_L / 1000, clone=False
        )
        surface = ct.ReactorSurface(
            self.surface, reactor, A=self.slot.values["area_m2"], clone=False
        )
        network = ct.ReactorNet([reactor])
        network.rtol, network.atol, network.max_steps = 1e-11, 1e-18, 100000
        rows = []
        for time in times:
            if time:
                network.advance(time)
            amounts = {
                key: float(
                    reactor.phase.concentrations[reactor.phase.species_index(key)]
                    * 1000
                    * reactor.volume
                )
                for key in GAS_SPECIES
            }
            amounts.update(
                {
                    adsorbate_key(self.slot, key): float(
                        surface.coverages[self.surface.species_index(key)]
                        * self.slot.values["sites_mol"]
                    )
                    for key in ("CO_ads", "O_ads")
                }
            )
            rows.append({"time_s": float(time), "amounts_mol": amounts})
        return rows
