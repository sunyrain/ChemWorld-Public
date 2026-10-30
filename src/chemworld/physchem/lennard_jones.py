"""Native Lennard-Jones estimates, collision integrals and dilute gas transport.

Equations: chemicals 1.5.2 public lennard_jones documentation; Chapman-Enskog
first approximation for a classical 12-6 potential. No reference runtime is
imported. Energy per molecule and epsilon/k are distinct input quantities.
"""
# ruff: noqa: E741 -- conventional collision-integral indices (l, s)

from dataclasses import dataclass, field
from math import cos, exp, isfinite, log, pi, sin, sqrt

from chemworld.physchem.lennard_jones_coefficients import (
    KIM_MONROE_COEFFICIENTS,
    NEUFELD_COEFFICIENTS,
)
from chemworld.physchem.property_reports import PropertyEvaluation

BOLTZMANN_J_K = 1.380649e-23
AVOGADRO_PER_MOL = 6.02214076e23
COLLISION_PAIRS = tuple(sorted(NEUFELD_COEFFICIENTS))
_REFERENCE = "https://chemicals.readthedocs.io/chemicals.lennard_jones.html"
EPSILON_METHODS = {
    "BSL_boiling": ("Tb",), "BSL_critical": ("Tc",), "BSL_melting": ("Tm",),
    "Flynn": ("Tc",), "Stiel_Thodos": ("Tc", "Zc"),
    "Tee_Gotoh_Steward_1": ("Tc",), "Tee_Gotoh_Steward_2": ("Tc", "omega"),
}
DIAMETER_METHODS = {
    "BSL_boiling": ("Vb",), "BSL_critical_1": ("Vc",), "BSL_critical_2": ("Tc", "Pc"),
    "BSL_melting": ("Vm",), "Flynn": ("Vc",), "Flynn_published": ("Vc",),
    "Silva_Liu_Macedo": ("Tc", "Pc"),
    "Stiel_Thodos": ("Vc", "Zc"), "Tee_Gotoh_Steward_1": ("Tc", "Pc"),
    "Tee_Gotoh_Steward_2": ("Tc", "Pc", "omega"),
}


def _positive(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or value <= 0):
        raise ValueError(f"{label} must be finite and positive")
    return value


def _parameters(methods, method, parameters):
    if method not in methods:
        raise ValueError(f"unknown Lennard-Jones estimate {method!r}")
    if set(parameters) != set(methods[method]):
        raise ValueError(f"{method} requires exactly {methods[method]}")
    for key, value in parameters.items():
        if key == "omega":
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not isfinite(value)):
                raise ValueError("omega must be finite")
        else:
            _positive(value, key)


def lennard_jones_energy(method, **parameters):
    """Estimate epsilon/k in K; input temperatures K and Zc/omega dimensionless."""
    _parameters(EPSILON_METHODS, method, parameters)
    p = parameters
    if method in {"BSL_boiling", "BSL_critical", "BSL_melting"}:
        key, coefficient = {"BSL_boiling": ("Tb", 1.15), "BSL_critical": ("Tc", .77),
                            "BSL_melting": ("Tm", 1.92)}[method]
        value = coefficient * p[key]
    elif method == "Flynn":
        value = 1.77 * p["Tc"] ** (5 / 6)
    elif method == "Stiel_Thodos":
        value = 65.3 * p["Tc"] * p["Zc"]**3.6
    elif method == "Tee_Gotoh_Steward_1":
        value = .7740 * p["Tc"]
    else:
        value = (.7915 + .1693 * p["omega"]) * p["Tc"]
    return _positive(value, "estimated epsilon/k")


def lennard_jones_diameter(method, **parameters):
    """Estimate sigma in metres; inputs K, Pa, m³/mol, dimensionless Zc/omega.

    Flynn selects chemicals 1.5.2 numerical compatibility, exponent 0.4.
    Flynn_published selects the documented/original exponent 5/12; the
    original 1962 abstract (doi:10.1002/aic.690080320) confirms slope 5/4
    against cube-root volume. The discrepancy is not a unit conversion.
    """
    _parameters(DIAMETER_METHODS, method, parameters)
    p = parameters
    volume_rules = {"BSL_boiling": ("Vb", 1.166), "BSL_critical_1": ("Vc", .841),
                    "BSL_melting": ("Vm", 1.222)}
    if method in volume_rules:
        key, coefficient = volume_rules[method]
        angstrom = coefficient * (p[key] * 1e6)**(1 / 3)
    elif method in {"Flynn", "Flynn_published"}:
        exponent = .4 if method == "Flynn" else 5 / 12
        angstrom = .561 * (p["Vc"] * 1e6)**exponent
    elif method == "Stiel_Thodos":
        angstrom = .1866 * (p["Vc"] * 1e6)**(1 / 3) * p["Zc"]**(-6 / 5)
    elif method == "Silva_Liu_Macedo":
        ratio = p["Tc"] / (p["Pc"] / 1e5)
        cubic = .17791 + 11.779 * ratio - .049029 * ratio * ratio
        _positive(cubic, "Silva-Liu-Macedo sigma cubed; outside correlation domain")
        angstrom = cubic**(1 / 3)
    else:
        coefficient = {"BSL_critical_2": 2.44, "Tee_Gotoh_Steward_1": 2.3647,
                       "Tee_Gotoh_Steward_2": 2.3551 - .0874 * p.get("omega", 0)}[method]
        angstrom = coefficient * (p["Tc"] / (p["Pc"] / 101325))**(1 / 3)
    return _positive(angstrom * 1e-10, "estimated diameter")


def lennard_jones_reduced_temperature(T, *, epsilon_k=None, epsilon_J=None):
    """T*=kT/epsilon. Exactly one of epsilon/k (K) and epsilon (J/molecule)."""
    _positive(T, "temperature")
    if (epsilon_k is None) == (epsilon_J is None):
        raise ValueError("provide exactly one of epsilon_k and epsilon_J")
    energy_K = (_positive(epsilon_J, "epsilon_J") / BOLTZMANN_J_K
                if epsilon_k is None else _positive(epsilon_k, "epsilon_k"))
    return _positive(T / energy_K, "reduced temperature")


def lennard_jones_collision_integral(T_star, *, method, l=1, s=1, extrapolate=False):
    """Complete 16-pair fits. Extrapolation requires explicit opt-in."""
    _positive(T_star, "reduced temperature")
    if type(l) is not int or type(s) is not int or (l, s) not in COLLISION_PAIRS:
        raise ValueError("unsupported collision integral order pair")
    if method not in {"Neufeld_Janzen_Aziz", "Kim_Monroe"}:
        raise ValueError("unknown collision integral method")
    if type(extrapolate) is not bool:
        raise ValueError("extrapolate must be explicitly true or false")
    maximum = 100 if method == "Neufeld_Janzen_Aziz" else 400
    if not extrapolate and not .3 < T_star < maximum:
        raise ValueError(f"collision fit requires 0.3 < T_star < {maximum}")
    if method == "Kim_Monroe":
        a, b, c = KIM_MONROE_COEFFICIENTS[l, s]
        logarithm = log(T_star)
        value = a + sum(bk / T_star**k + ck * logarithm**k
                        for k, (bk, ck) in enumerate(zip(b, c, strict=True), start=1))
    else:
        p = NEUFELD_COEFFICIENTS[l, s]
        value = p["A"] / T_star**p["B"]
        value += sum(p[a] * exp(-p[b] * T_star)
                     for a, b in (("C", "D"), ("E", "F"), ("G", "H")) if a in p)
        if "R" in p:
            value += p["R"] * T_star**p["B"] * sin(p["S"] * T_star**p["W"] - p["P"])
    return _positive(value, "collision integral")


def collision_value_slope(T_star, *, method, l=1, s=1, extrapolate=False):
    """The same collision value and its analytic reduced-temperature slope."""
    value = lennard_jones_collision_integral(T_star, method=method, l=l, s=s,
                                             extrapolate=extrapolate)
    t = T_star
    if method == "Kim_Monroe":
        _, b, c = KIM_MONROE_COEFFICIENTS[l, s]
        slope = sum(-k*bk/t**(k+1) + k*ck*log(t)**(k-1)/t
                    for k, (bk, ck) in enumerate(zip(b, c, strict=True), start=1))
    else:
        p = NEUFELD_COEFFICIENTS[l, s]
        slope = -p["A"]*p["B"]/t**(p["B"]+1)
        slope -= sum(p[a]*p[b]*exp(-p[b]*t)
                     for a, b in (("C", "D"), ("E", "F"), ("G", "H")) if a in p)
        if "R" in p:
            angle = p["S"]*t**p["W"]-p["P"]
            slope += p["R"] * (p["B"]*t**(p["B"]-1)*sin(angle)
                               + p["S"]*p["W"]*t**(p["B"]+p["W"]-1)*cos(angle))
    return value, slope


def kinetic_viscosity_value_slope(T, MW, sigma_m, epsilon_k, *, method, extrapolate=False):
    """Single value/slope kernel for reports, correlations and property packages."""
    mass = _positive(MW, "molecular weight") * 1e-3 / AVOGADRO_PER_MOL
    _positive(sigma_m, "diameter")
    reduced = lennard_jones_reduced_temperature(T, epsilon_k=epsilon_k)
    omega, slope = collision_value_slope(reduced, method=method, l=2, s=2,
                                         extrapolate=extrapolate)
    value = 5 / 16 * sqrt(mass * BOLTZMANN_J_K * T / pi) / sigma_m**2 / omega
    return value, value * (.5 / T - slope / epsilon_k / omega)


@dataclass(frozen=True)
class LennardJonesSpec:
    """Parameters for an existing component; configuration, never live state."""

    component_id: str
    epsilon_k_K: float
    sigma_m: float
    source_note: str
    sources: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _positive(self.epsilon_k_K, "epsilon/k")
        _positive(self.sigma_m, "diameter")
        if not isinstance(self.component_id, str) or not self.component_id.strip():
            raise ValueError("Lennard-Jones parameters require component identity")
        if not isinstance(self.source_note, str) or not self.source_note.strip():
            raise ValueError("Lennard-Jones parameters require an explicit source note")

    def to_dict(self):
        return {"component_id": self.component_id, "epsilon_k_K": self.epsilon_k_K,
                "sigma_m": self.sigma_m, "source_note": self.source_note, "sources": self.sources}

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)

    @classmethod
    def estimate(cls, component, *, energy_method, energy_parameters,
                 diameter_method, diameter_parameters):
        return cls(
            component.identifier, lennard_jones_energy(energy_method, **energy_parameters),
            lennard_jones_diameter(diameter_method, **diameter_parameters),
            f"Empirical Lennard-Jones estimates; {_REFERENCE}; uncertainty not established",
            {"epsilon_k": {"method": energy_method, "inputs": dict(energy_parameters)},
             "sigma": {"method": diameter_method, "inputs": dict(diameter_parameters)}},
        )

    def viscosity_correlation(self, component, *, collision_method, temperature_range_K):
        """Build the original property declaration with an explicit application range."""
        from chemworld.physchem.specs import PropertyCorrelation

        _molecular_mass(component, self)
        methods = {"Neufeld_Janzen_Aziz": "lj_viscosity_neufeld",
                   "Kim_Monroe": "lj_viscosity_kim_monroe"}
        if collision_method not in methods:
            raise ValueError("unknown collision integral method")
        for endpoint in temperature_range_K:
            kinetic_viscosity_value_slope(endpoint, component.molecular_weight_g_mol,
                                          self.sigma_m, self.epsilon_k_K,
                                          method=collision_method)
        return PropertyCorrelation(
            f"{self.component_id}:{methods[collision_method]}", methods[collision_method],
            {"MW": component.molecular_weight_g_mol, "sigma": self.sigma_m,
             "epsilon_k": self.epsilon_k_K}, {"temperature": "K"}, "Pa*s", "gas_viscosity",
            {"temperature": tuple(temperature_range_K)}, self.source_note,
            {"lennard_jones_parameters": self.to_dict(),
             "model_scope": "classical dilute gas; Chapman-Enskog first approximation"},
        )


def _molecular_mass(component, parameters):
    if component.identifier != parameters.component_id:
        raise ValueError("Lennard-Jones parameters do not match component identity")
    if component.charge:
        raise ValueError("neutral classical Lennard-Jones transport does not model ionic species")
    return _positive(component.molecular_weight_g_mol, "molecular weight") * 1e-3 / AVOGADRO_PER_MOL


def _report(property_id, value, unit, inputs, method, extrapolate):
    from chemworld.physchem.transport_properties import TransportPropertyReport

    return TransportPropertyReport(
        PropertyEvaluation(property_id, method, method, value, unit, inputs,
                           ("Classical dilute gas; first Chapman-Enskog approximation",)
                           + (("Collision integral extrapolation explicitly enabled",)
                              if extrapolate else ())),
        phase="gas", method_family="Chapman-Enskog Lennard-Jones",
        validity_status="estimated", relative_uncertainty=None,
        uncertainty_note="Parameter and model uncertainty not established",
        reference_reading=("Chapman and Cowling, The Mathematical Theory of Non-Uniform Gases",),
    )


def lennard_jones_viscosity_report(component, parameters, *, temperature_K,
                                  collision_method, extrapolate=False):
    """5 sqrt(m kT/pi)/(16 sigma² Omega(2,2)); SI dynamic viscosity."""
    _molecular_mass(component, parameters)
    reduced = lennard_jones_reduced_temperature(temperature_K, epsilon_k=parameters.epsilon_k_K)
    omega = lennard_jones_collision_integral(reduced, method=collision_method,
                                            l=2, s=2, extrapolate=extrapolate)
    value, _ = kinetic_viscosity_value_slope(
        temperature_K, component.molecular_weight_g_mol, parameters.sigma_m,
        parameters.epsilon_k_K, method=collision_method, extrapolate=extrapolate,
    )
    return _report("gas_viscosity", value, "Pa*s",
                   {"temperature": temperature_K, "T_star": reduced, "Omega_22": omega,
                    "sigma_m": parameters.sigma_m, "epsilon_k_K": parameters.epsilon_k_K,
                    "molecular_weight_g_mol": component.molecular_weight_g_mol},
                   "chapman_enskog_viscosity:" + collision_method, extrapolate)


def lennard_jones_diffusivity_report(component_a, parameters_a, component_b, parameters_b, *,
                                    temperature_K, pressure_Pa, collision_method,
                                    combining_rule, extrapolate=False):
    """Binary diffusion with explicit Lorentz-Berthelot sigma/energy mixing."""
    if combining_rule != "Lorentz_Berthelot":
        raise ValueError("explicit supported cross rule Lorentz_Berthelot is required")
    _positive(pressure_Pa, "pressure")
    mass_a, mass_b = _molecular_mass(component_a, parameters_a), _molecular_mass(component_b,
                                                                                 parameters_b)
    reduced_mass = mass_a * mass_b / (mass_a + mass_b)
    sigma = (parameters_a.sigma_m + parameters_b.sigma_m) / 2
    energy = sqrt(parameters_a.epsilon_k_K * parameters_b.epsilon_k_K)
    reduced = lennard_jones_reduced_temperature(temperature_K, epsilon_k=energy)
    omega = lennard_jones_collision_integral(reduced, method=collision_method,
                                            l=1, s=1, extrapolate=extrapolate)
    kT = BOLTZMANN_J_K * temperature_K
    value = 3 / 16 * kT / pressure_Pa * sqrt(2 * kT / (pi * reduced_mass)) / sigma**2 / omega
    return _report("binary_gas_diffusivity", value, "m^2/s",
                   {"temperature": temperature_K, "pressure": pressure_Pa,
                    "T_star": reduced, "Omega_11": omega, "cross_sigma_m": sigma,
                    "cross_epsilon_k_K": energy},
                   "chapman_enskog_diffusivity:" + collision_method, extrapolate)
