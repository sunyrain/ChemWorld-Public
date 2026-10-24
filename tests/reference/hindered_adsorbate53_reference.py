"""Public ASE API and independent finite-gas scalar balance for fixed 53B units."""

from itertools import product
from math import exp, log

import numpy as np
from ase.thermochemistry import HinderedThermo
from scipy.integrate import solve_ivp

from chemworld.examples.native_hindered_adsorbate import PROFILE_NAMES, profile_parameters

# Published SI and CODATA-2014 conversion values. Preserve ASE's reference
# constants; never patch its modules or read its implementation.
NA = 6.02214076e23
R = 8.31446261815324
E2014 = 1.6021766208e-19
KB2014 = 1.38064852e-23
AMU2014 = 1.660539040e-27
EV_MOLAR = NA * E2014
TIMES = (0.0, 1.0, 2.0, 4.0, 8.0)
CAPACITY, GAS_INITIAL, ADS_INITIAL, SUPPORT = 1e-4, 0.002, 2e-5, 0.05
TOTAL = GAS_INITIAL + ADS_INITIAL


class ReferenceThermo:
    def __init__(self, profile="intermediate", binding=0):
        data = profile_parameters(profile)
        self.density = data["site_density_mol_m2"]
        retained = [v / EV_MOLAR for v in data["vibrational_energies_J_mol"]]
        # Declared placeholders for the replaced coordinates. ASE's documented
        # highest-3N-3 selection excludes these; native never sees these modes.
        full = [min(retained) * k for k in (0.001, 0.002, 0.003)] + retained
        self.model = HinderedThermo(
            vib_energies=np.array(full),
            trans_barrier_energy=data["translation_barrier_J_mol"] / EV_MOLAR,
            rot_barrier_energy=data["rotation_barrier_J_mol"] / EV_MOLAR,
            sitedensity=self.density * NA / 1e4,
            rotationalminima=data["rotational_minima"],
            potentialenergy=(data["electronic_energy_J_mol"] + 0.5 * binding) / EV_MOLAR,
            mass=data["molar_mass_kg_mol"] / (NA * AMU2014),
            inertia=data["moment_of_inertia_kg_m2"] / (AMU2014 * 1e-20),
            symmetrynumber=data["symmetry_number"],
        )

    def energy(self, temperature):
        return float(self.model.get_internal_energy(temperature, verbose=False) * EV_MOLAR)

    def values(self, temperature, *, with_cv=True):
        n0 = exp(1 / 3) * (1e5 / (KB2014 * temperature)) ** (2 / 3)
        concentration_entropy = NA * KB2014 * (1 - log(n0 / (NA * self.density)))
        entropy = (
            self.model.get_entropy(temperature, verbose=False) * EV_MOLAR - concentration_entropy
        )
        gibbs = (
            self.model.get_helmholtz_energy(temperature, verbose=False) * EV_MOLAR
            + temperature * concentration_entropy
        )
        return {
            "U": self.energy(temperature),
            "S": float(entropy),
            "G": float(gibbs),
            "Cv": (self.energy(temperature + 0.01) - self.energy(temperature - 0.01)) / 0.02
            if with_cv
            else None,
            "ZPE": float(self.model.get_zero_point_energy(verbose=False) * EV_MOLAR),
            "removed_concentration_entropy": concentration_entropy,
        }


def gas_values(temperature):
    cp = 3.5 * R
    h = cp * (temperature - 298.15)
    s = 40 + cp * log(temperature / 298.15)
    return {"U": h - R * temperature, "G": h - temperature * s, "Cv": cp - R}


def rate_values(reference, temperature, bound, *, with_cv=False):
    surface, gas = reference.values(temperature, with_cv=with_cv), gas_values(temperature)
    delta = surface["G"] - gas["G"]
    theta = bound / CAPACITY
    gas_activity = (TOTAL - bound) * R * temperature / 0.001 / 1e5
    forward = CAPACITY * 10000 * exp(-(30000 + 0.5 * delta) / (R * temperature))
    reverse = CAPACITY * 10000 * exp(-(30000 - 0.5 * delta) / (R * temperature))
    return {
        "delta": delta,
        "equilibrium": exp(-delta / (R * temperature)),
        "forward": forward * gas_activity * (1 - theta),
        "reverse": reverse * theta,
        "surface": surface,
        "gas": gas,
    }


def total_energy(reference, temperature, bound):
    return (
        (TOTAL - bound) * gas_values(temperature)["U"]
        + bound * reference.energy(temperature)
        + SUPPORT * (temperature - 298.15)
    )


def reference_history(temperature, isothermal):
    reference = ReferenceThermo()

    def derivative(_time, state):
        bound, current_t = state
        rates = rate_values(reference, current_t, bound, with_cv=not isothermal)
        rate = rates["forward"] - rates["reverse"]
        thermal_rate = 0.0
        if not isothermal:
            capacity = (
                SUPPORT + (TOTAL - bound) * rates["gas"]["Cv"] + bound * rates["surface"]["Cv"]
            )
            thermal_rate = -(rates["surface"]["U"] - rates["gas"]["U"]) * rate / capacity
        return [rate, thermal_rate]

    result = solve_ivp(
        derivative,
        (0, TIMES[-1]),
        (ADS_INITIAL, temperature),
        method="DOP853",
        t_eval=TIMES,
        rtol=1e-10,
        atol=(1e-14, 1e-9),
    )
    if not result.success or result.y.shape != (2, len(TIMES)):
        raise RuntimeError("independent finite-gas reference failed: " + result.message)
    initial_energy = total_energy(reference, temperature, ADS_INITIAL)
    rows = []
    for at, (bound, current_t) in zip(TIMES, result.y.T, strict=True):
        rows.append(
            {
                "time_s": at,
                "gas": float(TOTAL - bound),
                "adsorbed": float(bound),
                "temperature": float(current_t),
                "Q": total_energy(reference, current_t, bound) - initial_energy
                if isothermal
                else 0.0,
                "elements": [TOTAL, TOTAL],
            }
        )
    return rows, result.nfev


def cases():
    rows = []
    for name, temperature in product(PROFILE_NAMES, (298.15, 600.0, 1000.0)):
        rows.append(
            {
                "name": f"thermo-{name}-{temperature}",
                "kind": "thermo",
                "profile": name,
                "temperature": temperature,
                "expected_comparisons": 6,
            }
        )
    for temperature, binding in product((298.15, 600.0), (-4000.0, 4000.0)):
        rows.append(
            {
                "name": f"rates-{temperature}-{binding}",
                "kind": "rates",
                "temperature": temperature,
                "binding": binding,
                "expected_comparisons": 6,
            }
        )
    for temperature, isothermal in product((298.15, 600.0), (True, False)):
        rows.append(
            {
                "name": f"world-{temperature}-{'iso' if isothermal else 'adiabatic'}",
                "kind": "world",
                "temperature": temperature,
                "isothermal": isothermal,
                "expected_comparisons": 29,
            }
        )
    return rows
