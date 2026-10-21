"""Complete-combustion balances and explicit calorific/fuel correlations.

Independent atom balances and public equations, not a kinetic mechanism.
Reaction enthalpies are negative for heat release. Reference numbers below
are the selected historical source convention, not an implicit phase database.
"""

from dataclasses import dataclass
from math import fsum, isclose, isfinite
from types import MappingProxyType

from chemworld.physchem.elements import formula_charge, molecular_weight, parse_formula
from chemworld.physchem.property_reports import PropertyEvaluation
from chemworld.physchem.viscosity_kernels import _positive

WATER_REFERENCE_VAPORIZATION_J_MOL = 44011.496
COMBUSTION_REFERENCE_ENTHALPIES = MappingProxyType(
    {
        "H2O": -285825.0,
        "CO2": -393474.0,
        "SO2": -296800.0,
        "Br2": 30880.0,
        "I2": 62417.0,
        "HCl": -92173.0,
        "HF": -272711.0,
        "P4O10": -3009940.0,
        "O2": 0.0,
        "N2": 0.0,
        "Ar": 0.0,
        "Ne": 0.0,
        "Kr": 0.0,
        "Xe": 0.0,
        "He": 0.0,
        "Ash": 0.0,
    }
)
COMBUSTION_PRODUCT_IDENTITIES = MappingProxyType(
    {
        "Br2": "7726-95-6",
        "CO2": "124-38-9",
        "H2O": "7732-18-5",
        "HCl": "7647-01-0",
        "HF": "7664-39-3",
        "I2": "7553-56-2",
        "N2": "7727-37-9",
        "O2": "7782-44-7",
        "Ar": "7440-37-1",
        "He": "7440-59-7",
        "Ne": "7440-01-9",
        "Kr": "7439-90-9",
        "Xe": "7440-63-3",
        "P4O10": "16752-60-6",
        "SO2": "7446-09-5",
    }
)
_SUPPORTED = frozenset(("C", "H", "N", "O", "S", "Br", "I", "Cl", "F", "P"))


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _composition(formula):
    if isinstance(formula, str) and formula_charge(formula):
        raise ValueError("complete-combustion formulas require a neutral species")
    atoms = parse_formula(formula) if isinstance(formula, str) else dict(formula)
    if not atoms:
        raise ValueError("combustion requires a nonempty composition")
    for key, value in atoms.items():
        _positive(value, key + " count", zero=True)
        if key != "Ash":
            molecular_weight({key: 1})  # The existing element/isotope contract owns identity.
    return atoms


def combustion_stoichiometry(atoms, *, missing_handling="elemental"):
    """Net products per declared fuel unit; negative O2/H2O are reactants.

    Ash has 1 g per pseudo-unit. Its original unresolved atoms remain caller
    inputs and are retained by CombustionData; ash is not a new chemical atom.
    Halogen-rich compounds can consume water. A finite consumer must supply it.
    """
    atoms = _composition(atoms)
    if missing_handling not in {"elemental", "ash"}:
        raise ValueError("select elemental or ash residual handling")
    h_water = (atoms.get("H", 0) - atoms.get("Cl", 0) - atoms.get("F", 0)) / 2
    products = {
        "CO2": atoms.get("C", 0),
        "H2O": h_water,
        "SO2": atoms.get("S", 0),
        "Br2": atoms.get("Br", 0) / 2,
        "I2": atoms.get("I", 0) / 2,
        "HCl": atoms.get("Cl", 0),
        "HF": atoms.get("F", 0),
        "P4O10": atoms.get("P", 0) / 4,
        "N2": atoms.get("N", 0) / 2,
    }
    products["O2"] = (
        atoms.get("O", 0)
        - 2 * products["CO2"]
        - h_water
        - 2 * products["SO2"]
        - 10 * products["P4O10"]
    ) / 2
    unresolved = {k: v for k, v in atoms.items() if k not in _SUPPORTED and k != "Ash"}
    if missing_handling == "elemental":
        products.update(unresolved)
        products["Ash"] = atoms.get("Ash", 0)
    else:
        products["Ash"] = atoms.get("Ash", 0) + (
            molecular_weight(unresolved) if any(unresolved.values()) else 0
        )
    return {k: v for k, v in products.items() if v != 0 or k == "O2"}


def combustion_products_mixture(
    atom_compositions,
    amounts,
    *,
    identities=None,
    reactivities=None,
    missing_handling="elemental",
    stoichiometries=None,
):
    """Extensive linear balance; amounts may be mol, mol/s or normalized fractions."""
    size = len(atom_compositions)
    if not size or len(amounts) != size:
        raise ValueError("composition and amount vectors must be nonempty and aligned")
    reactivities = (True,) * size if reactivities is None else tuple(reactivities)
    identities = (None,) * size if identities is None else tuple(identities)
    if len(reactivities) != size or len(identities) != size:
        raise ValueError("reactivity and identity vectors must align with every component")
    if any(type(v) is not bool for v in reactivities):
        raise ValueError("reactivities must be explicit booleans")
    if stoichiometries is not None and len(stoichiometries) != size:
        raise ValueError("precomputed stoichiometries must align with components")
    result = {}
    for i, (atoms, n, active, identity) in enumerate(
        zip(atom_compositions, amounts, reactivities, identities, strict=True)
    ):
        _positive(n, "amount", zero=True)
        # Existing product species are carried through with their physical formula.
        product = next((p for p, c in COMBUSTION_PRODUCT_IDENTITIES.items() if c == identity), None)
        if product is not None:
            if dict(_composition(atoms)) != parse_formula(product):
                raise ValueError("product identity and declared atoms disagree")
            values = {product: 1}
        elif not active:
            if not identity:
                raise ValueError("nonreactive components require a preserved identity")
            values = {identity: 1}
        else:
            values = combustion_stoichiometry(atoms, missing_handling=missing_handling)
            if stoichiometries is not None:
                supplied = stoichiometries[i]
                if set(supplied) != set(values) or any(
                    not isclose(_finite(supplied[k], k), v, rel_tol=1e-12, abs_tol=1e-12)
                    for k, v in values.items()
                ):
                    raise ValueError("precomputed stoichiometry violates declared atom balance")
        for k, v in values.items():
            result[k] = result.get(k, 0) + n * v
    return result


def combustion_enthalpy(stoichiometry, formation_enthalpy_J_mol, *, product_enthalpies=None):
    """Hess law in explicit source units; never assume unknown products have Hf=0."""
    _finite(formation_enthalpy_J_mol, "fuel formation enthalpy")
    refs = COMBUSTION_REFERENCE_ENTHALPIES if product_enthalpies is None else product_enthalpies
    missing = {k for k, v in stoichiometry.items() if v and k not in refs}
    if missing:
        raise ValueError(f"missing product formation enthalpies: {sorted(missing)}")
    return (
        fsum(
            _finite(n, k) * _finite(refs[k], k + " enthalpy") for k, n in stoichiometry.items() if n
        )
        - formation_enthalpy_J_mol
    )


def heating_value_convert(
    value_J_mol,
    net_water,
    *,
    target,
    vaporization_enthalpy_J_mol=WATER_REFERENCE_VAPORIZATION_J_MOL,
):
    _finite(value_J_mol, "heating value")
    _finite(net_water, "net water coefficient")
    _positive(vaporization_enthalpy_J_mol, "water vaporization enthalpy")
    if target not in {"HHV", "LHV"}:
        raise ValueError("target must be HHV or LHV")
    return value_J_mol + (1 if target == "LHV" else -1) * net_water * vaporization_enthalpy_J_mol


def heating_value_mass(mass_fractions, *, method):
    """Boie or modified Dulong, signed J/g from normalized mass fractions.

    Perry's percent-basis coefficients are multiplied by 100 for fractions.
    Dulong's <10 wt% oxygen domain is enforced, not silently extrapolated.
    """
    if set(mass_fractions) - {"C", "H", "N", "O", "S", "Ash"}:
        raise ValueError("mass estimate supports CHNOS and explicit ash only")
    for k, v in mass_fractions.items():
        _positive(v, k + " mass fraction", zero=True)
    if not isclose(fsum(mass_fractions.values()), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("mass fractions including ash must sum to one")
    c, h, n, o, s = (mass_fractions.get(k, 0) for k in ("C", "H", "N", "O", "S"))
    if method == "Boie":
        return -100 * (347.3 * c + 1151 * h + 29 * n + 42 * s - 108 * o)
    if method == "Dulong":
        if o >= 0.1:
            raise ValueError("modified Dulong requires less than 10 wt% oxygen")
        return -100 * (338 * c + 1428 * (h - o / 8) + 95 * s)
    raise ValueError("select Boie or Dulong")


@dataclass(frozen=True)
class CombustionData:
    stoichiometry: dict
    HHV: float
    Hf: float
    MW: float
    atoms: dict
    method: str
    missing_handling: str
    product_enthalpies: dict

    def __post_init__(self):
        for key in ("HHV", "Hf"):
            _finite(getattr(self, key), key)
        _positive(self.MW, "molecular weight")
        if not self.stoichiometry or not any(self.stoichiometry.values()):
            raise ValueError("combustion data require a nonempty net stoichiometry")
        hess = combustion_enthalpy(
            self.stoichiometry, self.Hf, product_enthalpies=self.product_enthalpies
        )
        if not isclose(hess, self.HHV, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("combustion enthalpy and formation data are inconsistent")
        for key in ("stoichiometry", "atoms", "product_enthalpies"):
            object.__setattr__(self, key, MappingProxyType(dict(getattr(self, key))))

    @property
    def LHV(self):  # noqa: N802 -- public thermochemical quantity
        return heating_value_convert(self.HHV, self.stoichiometry.get("H2O", 0), target="LHV")

    def heat(self, amount, *, basis="HHV"):
        if basis not in {"HHV", "LHV"}:
            raise ValueError("select HHV or LHV heat basis")
        n = _positive(amount.to("mol").value, "fuel amount", zero=True)
        return PropertyEvaluation(
            "combustion_heat",
            "explicit-fuel",
            self.method,
            n * getattr(self, basis),
            "J",
            {"fuel_mol": n, "molar_enthalpy": getattr(self, basis)},
            (
                "Complete conversion at the declared calorific reference; "
                "no flame-temperature model",
            ),
        )

    def to_dict(self):
        return {
            "model": "combustion-data-1",
            "stoichiometry": dict(self.stoichiometry),
            "HHV": self.HHV,
            "Hf": self.Hf,
            "MW": self.MW,
            "atoms": dict(self.atoms),
            "method": self.method,
            "missing_handling": self.missing_handling,
            "product_enthalpies": dict(self.product_enthalpies),
        }

    @classmethod
    def from_dict(cls, payload):
        if (
            set(payload)
            != {
                "model",
                "stoichiometry",
                "HHV",
                "Hf",
                "MW",
                "atoms",
                "method",
                "missing_handling",
                "product_enthalpies",
            }
            or payload["model"] != "combustion-data-1"
        ):
            raise ValueError("unknown combustion data schema")
        return combustion_data(
            formula=payload["atoms"] or None,
            stoichiometry=payload["stoichiometry"],
            Hf=payload["Hf"],
            HHV=payload["HHV"],
            MW=payload["MW"],
            method=payload["method"],
            missing_handling=payload["missing_handling"],
            product_enthalpies=payload["product_enthalpies"],
        )


def combustion_data(
    formula=None,
    *,
    stoichiometry=None,
    Hf=None,
    MW=None,
    method,
    missing_handling="elemental",
    LHV=None,
    HHV=None,
    product_enthalpies=None,
):
    atoms = _composition(formula) if formula is not None else {}
    if not atoms and stoichiometry is None:
        raise ValueError("formula or explicit stoichiometry is required")
    if atoms:
        calculated = combustion_stoichiometry(atoms, missing_handling=missing_handling)
        if stoichiometry is not None and dict(stoichiometry) != calculated:
            raise ValueError("formula and stoichiometry disagree")
        stoichiometry = calculated
        if MW is None:
            real_atoms = {k: v for k, v in atoms.items() if k != "Ash"}
            MW = (molecular_weight(real_atoms) if any(real_atoms.values()) else 0) + atoms.get(
                "Ash", 0
            )
    if MW is None:
        raise ValueError("explicit stoichiometry requires molecular weight")
    _positive(MW, "molecular weight")
    refs = dict(
        COMBUSTION_REFERENCE_ENTHALPIES if product_enthalpies is None else product_enthalpies
    )
    # Independent Hess datum; this also checks completeness for every used product.
    product_h = combustion_enthalpy(stoichiometry, 0, product_enthalpies=refs)
    supplied = []
    if Hf is not None:
        supplied.append(product_h - _finite(Hf, "Hf"))
    if HHV is not None:
        supplied.append(_finite(HHV, "HHV"))
    if LHV is not None:
        supplied.append(heating_value_convert(LHV, stoichiometry.get("H2O", 0), target="HHV"))
    if method in {"Dulong", "Boie"}:
        if not atoms:
            raise ValueError("mass-fraction estimate needs original atoms")
        masses = {
            k: (v if k == "Ash" or v == 0 else molecular_weight({k: v})) for k, v in atoms.items()
        }
        total = fsum(masses.values())
        supplied.append(
            heating_value_mass({k: v / total for k, v in masses.items()}, method=method) * MW
        )
    elif method == "Stoichiometry":
        if Hf is None:
            raise ValueError("Stoichiometry method requires formation enthalpy")
    elif method == "Specification":
        if HHV is None and LHV is None:
            raise ValueError("Specification method requires HHV or LHV")
    else:
        raise ValueError("unknown combustion data method")
    if any(not isclose(v, supplied[0], rel_tol=1e-10, abs_tol=1e-8) for v in supplied[1:]):
        raise ValueError("inconsistent calorific specifications")
    return CombustionData(
        dict(stoichiometry),
        supplied[0],
        product_h - supplied[0],
        MW,
        atoms,
        method,
        missing_handling,
        refs,
    )


def combustion_reference_heat(component, reference, product_references, amount):
    """Original component and phase-specific caloric references -> finite heat.

    No entropy or heat capacity is fabricated. The caller supplies all product
    phases/enthalpies at the same T/P as the fuel; these override historical
    default constants entirely. A changed water phase changes the result.
    """
    if component.charge:
        raise ValueError("complete combustion requires a neutral fuel component")
    coefficients = combustion_stoichiometry(component.composition)
    keys = {k for k, v in coefficients.items() if v}
    if set(product_references) != keys:
        raise ValueError("exactly the nonzero product/reactant references are required")
    if reference.phase != component.default_phase:
        raise ValueError("fuel component and caloric reference phases disagree")
    refs = (reference, *product_references.values())
    if any(r.energy_basis != "enthalpy" or r.phase is None for r in refs):
        raise ValueError("phase-specific enthalpy references are required")
    if any(
        (r.temperature_K, r.pressure_Pa) != (reference.temperature_K, reference.pressure_Pa)
        for r in refs
    ):
        raise ValueError("combustion references must share temperature and pressure")
    value = combustion_enthalpy(
        coefficients,
        reference.energy_J_mol,
        product_enthalpies={k: r.energy_J_mol for k, r in product_references.items()},
    )
    n = _positive(amount.to("mol").value, "fuel amount", zero=True)
    return PropertyEvaluation(
        "combustion_heat",
        component.identifier,
        "source_Hess_balance",
        n * value,
        "J",
        {
            "fuel_mol": n,
            "molar_enthalpy": value,
            "temperature": reference.temperature_K,
            "pressure": reference.pressure_Pa,
        },
        tuple(r.source_note for r in refs),
    )


def octane_index(research_octane, motor_octane, *, quantity="AKI"):
    _finite(research_octane, "RON")
    _finite(motor_octane, "MON")
    if quantity == "AKI":
        return (research_octane + motor_octane) / 2
    if quantity == "sensitivity":
        return research_octane - motor_octane
    raise ValueError("select AKI or sensitivity")


def ignition_delay_quality(delay_s, *, quantity):
    """Public Perez-Boehman and D6890-derived correlations; delay supplied in s.

    Perez fits ~3.6..67 ms; algebraic extrapolation is returned, never clipped
    to a nominal octane interval. Source consumers carry that applicability.
    """
    t = 1000 * _positive(delay_s, "ignition delay")
    if quantity == "RON":
        return 120.77 - 425.48 / t
    if quantity == "MON":
        return 109.93 - 374.73 / t
    if quantity == "DCN":
        # The selected reference-version convention assigns both exact endpoints
        # to the outer branch; the two empirical branches are not continuous.
        if 3.1 < t < 6.5:
            return 4.46 + 186.6 / t
        if t <= 1.512:
            raise ValueError("DCN fractional-power branch requires delay above 1.512 ms")
        return 83.99 * (t - 1.512) ** (-0.658) + 3.547
    raise ValueError("select RON, MON or DCN")
