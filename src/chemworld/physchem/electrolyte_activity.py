"""Explicit molality activity models from one excess Gibbs energy.

Debye-Huckel, Davies, SIT and Pitzer share analytic derivative arithmetic.
Parameters are caller-owned data with explicit units, sources and T/P domains;
there is no implicit zero interaction, reference-library import or solvent stock.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from itertools import combinations, combinations_with_replacement, product
from math import exp, factorial, isfinite, log, sqrt
from types import MappingProxyType
from typing import Any

from chemworld.physchem.caloric_math import (
    FirstDerivative,
    compensated_sum,
    derivative_depth,
    exponential,
    logarithm,
    scalar_value,
)

_SQRT_UNIT = "(kg/mol)^0.5"
_PAIR_UNIT = "kg/mol"
_TRIPLE_UNIT = "kg^2/mol^2"
_GLOBAL_UNITS = {
    "ideal": {},
    "DebyeHuckelLimiting": {"A": _SQRT_UNIT},
    "DebyeHuckel": {"A": _SQRT_UNIT, "B": _SQRT_UNIT},
    "Davies": {"A": _SQRT_UNIT, "b": _SQRT_UNIT},
    "SIT": {"A": _SQRT_UNIT, "B": _SQRT_UNIT},
    "Pitzer": {"Aphi": _SQRT_UNIT, "b": _SQRT_UNIT},
}
_INTERACTION_UNITS = {
    "epsilon": {"epsilon": _PAIR_UNIT},
    "binary": {
        "beta0": _PAIR_UNIT,
        "beta1": _PAIR_UNIT,
        "beta2": _PAIR_UNIT,
        "C0": _TRIPLE_UNIT,
        "C1": _TRIPLE_UNIT,
        "alpha1": _SQRT_UNIT,
        "alpha2": _SQRT_UNIT,
        "omega": _SQRT_UNIT,
    },
    "theta": {"theta": _PAIR_UNIT},
    "psi": {"psi": _TRIPLE_UNIT},
    "lambda": {"lambda": _PAIR_UNIT},
    "zeta": {"zeta": _TRIPLE_UNIT},
    "mu": {"mu": _TRIPLE_UNIT},
}


@dataclass(frozen=True)
class ElectrolyteParameter:
    """v=v0+a(T-Tr)+b(1/T-1/Tr)+c ln(T/Tr)+d(P-Pr).

    ``unit`` is the value unit. Coefficients carry respectively unit/K, unit*K,
    unit and unit/Pa. Explicit bounds also apply to constant parameters.
    """

    value: float
    unit: str
    source_id: str
    temperature_bounds_K: tuple[float, float]
    pressure_bounds_Pa: tuple[float, float]
    reference_temperature_K: float = 298.15
    reference_pressure_Pa: float = 100000
    per_K: float = 0
    times_K: float = 0
    per_log_T: float = 0
    per_Pa: float = 0

    def __post_init__(self) -> None:
        for name in ("temperature_bounds_K", "pressure_bounds_Pa"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            low, high = getattr(self, name)
            if not 0 < low <= high or not isfinite(low) or not isfinite(high):
                raise ValueError("electrolyte parameter requires finite positive T/P bounds")
        if (
            not self.unit
            or not self.source_id
            or any(
                isinstance(v, bool) or not isfinite(v)
                for v in (
                    self.value,
                    self.per_K,
                    self.times_K,
                    self.per_log_T,
                    self.per_Pa,
                    self.reference_temperature_K,
                    self.reference_pressure_Pa,
                )
            )
        ):
            raise ValueError("electrolyte parameter requires finite coefficients, unit and source")
        self.evaluate(self.reference_temperature_K, self.reference_pressure_Pa)

    def evaluate(self, temperature_K: Any, pressure_Pa: Any) -> Any:
        t, p, tr, pr = (
            temperature_K,
            pressure_Pa,
            self.reference_temperature_K,
            self.reference_pressure_Pa,
        )
        if any(
            not isfinite(scalar_value(v)) or not bounds[0] <= scalar_value(v) <= bounds[1]
            for v, bounds in ((t, self.temperature_bounds_K), (p, self.pressure_bounds_Pa))
        ):
            raise ValueError("electrolyte parameter outside declared temperature/pressure domain")
        return (
            self.value
            + self.per_K * (t - tr)
            + self.times_K * (1 / t - 1 / tr)
            + self.per_log_T * logarithm(t / tr)
            + self.per_Pa * (p - pr)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ElectrolyteInteraction:
    kind: str
    species: tuple[str, ...]
    parameters: Mapping[str, ElectrolyteParameter]

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", tuple(sorted(self.species)))
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        count = 3 if self.kind in ("psi", "zeta", "mu") else 2
        if self.kind not in _INTERACTION_UNITS or len(self.species) != count:
            raise ValueError("unknown electrolyte interaction kind or component arity")
        for name, parameter in self.parameters.items():
            if (
                name not in _INTERACTION_UNITS[self.kind]
                or parameter.unit != _INTERACTION_UNITS[self.kind][name]
            ):
                raise ValueError("electrolyte interaction has unknown parameter or wrong unit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "species": list(self.species),
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ElectrolyteInteraction:
        return cls(
            payload["kind"],
            tuple(payload["species"]),
            {k: ElectrolyteParameter(**v) for k, v in payload["parameters"].items()},
        )


def pitzer_binary_from_cphi(
    *, cation_charge: int, anion_charge: int, Cphi: ElectrolyteParameter
) -> ElectrolyteParameter:
    """Explicit convention conversion C0=Cphi/(2 sqrt(|zc za|)); no estimates."""
    if cation_charge <= 0 or anion_charge >= 0 or Cphi.unit != _TRIPLE_UNIT:
        raise ValueError("Cphi conversion requires opposite charges and kg^2/mol^2")
    values = Cphi.to_dict()
    factor = 2 * sqrt(abs(cation_charge * anion_charge))
    for name in ("value", "per_K", "times_K", "per_log_T", "per_Pa"):
        values[name] /= factor
    return ElectrolyteParameter(**values)


def _screening_integral(A: Any, B: Any, ionic: Any) -> Any:
    """G/RT/kg giving log10 gamma_i=-A zi虏 sqrt(I)/(1+B sqrt(I))."""
    if scalar_value(ionic) == 0:
        return ionic * 0
    root = ionic**0.5
    if abs(scalar_value(B * root)) < 0.1:
        return (
            -4
            * log(10)
            * A
            * compensated_sum((-B) ** k * ionic ** ((k + 3) / 2) / (k + 3) for k in range(16))
        )
    return -4 * log(10) * A * (ionic / (2 * B) - root / B**2 + logarithm(1 + B * root) / B**3)


def _radial_integral(x: Any, power: int) -> Any:
    """Integral_0^1 t^power exp(-x t) dt, stable at zero using its Taylor series."""
    if abs(scalar_value(x)) < 0.5:
        return compensated_sum((-x) ** k / (factorial(k) * (power + k + 1)) for k in range(20))
    return (
        factorial(power)
        / x ** (power + 1)
        * (1 - exponential(-x) * compensated_sum(x**k / factorial(k) for k in range(power + 1)))
    )


def pitzer_mixing_J(x: Any) -> Any:  # noqa: N802 - named scientific J function
    """Pitzer (1975) equation 47; mathematical approximation, not Harvie coefficients."""
    if scalar_value(x) <= 0:
        if scalar_value(x) == 0:
            if derivative_depth(x) > 1:
                raise ValueError("Pitzer J has no finite second Taylor derivative at zero")
            return x * 0
        raise ValueError("Pitzer J requires nonnegative argument")
    return x**1.7237 / (4 * x**0.7237 + 4.581 * exponential(-0.012 * x**0.528))


@dataclass(frozen=True)
class ElectrolyteActivityResult:
    model_id: str
    ionic_strength_mol_kg: float
    log_activity_coefficients: dict[str, float]
    log_activities: dict[str, float | None]
    log_solvent_activity: float
    osmotic_coefficient: float
    excess_gibbs_RT_mol_kg: float
    net_charge_mol_kg: float

    @property
    def solvent_activity(self) -> float:
        return exp(self.log_solvent_activity)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ElectrolyteActivityModel:
    model_id: str
    family: str
    charges: tuple[tuple[str, int], ...]
    source_id: str
    maximum_ionic_strength_mol_kg: float
    temperature_bounds_K: tuple[float, float]
    pressure_bounds_Pa: tuple[float, float]
    parameters: Mapping[str, ElectrolyteParameter] = field(default_factory=dict)
    interactions: tuple[ElectrolyteInteraction, ...] = ()
    unsymmetrical_mixing: str = "Pitzer1975"

    def __post_init__(self) -> None:
        object.__setattr__(self, "charges", tuple(tuple(row) for row in self.charges))
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "interactions", tuple(self.interactions))
        for name in ("temperature_bounds_K", "pressure_bounds_Pa"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            low, high = getattr(self, name)
            if not 0 < low <= high or not isfinite(low) or not isfinite(high):
                raise ValueError("activity model needs finite positive T/P bounds")
        if (
            not self.model_id
            or not self.source_id
            or self.family not in _GLOBAL_UNITS
            or len(dict(self.charges)) != len(self.charges)
            or any(not key or type(z) is not int for key, z in self.charges)
            or not isfinite(self.maximum_ionic_strength_mol_kg)
            or self.maximum_ionic_strength_mol_kg <= 0
            or self.unsymmetrical_mixing not in ("Pitzer1975", "none")
        ):
            raise ValueError("activity model requires identity, catalog family and finite domain")
        for name, parameter in self.parameters.items():
            if (
                name not in _GLOBAL_UNITS[self.family]
                or parameter.unit != _GLOBAL_UNITS[self.family][name]
            ):
                raise ValueError("activity model global parameter has unknown name or wrong unit")
        expected = set(self.required_interactions())
        keys = [(item.kind, item.species) for item in self.interactions]
        if len(set(keys)) != len(keys) or any(key not in expected for key in keys):
            raise ValueError("duplicate or incompatible electrolyte interaction declaration")

    def required_interactions(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        ids = tuple(key for key, _ in self.charges)
        if self.family == "SIT":
            return tuple(
                ("epsilon", tuple(sorted(pair))) for pair in combinations_with_replacement(ids, 2)
            )
        if self.family != "Pitzer":
            return ()
        cs = [key for key, z in self.charges if z > 0]
        anions = [key for key, z in self.charges if z < 0]
        ns = [key for key, z in self.charges if not z]
        result: list[tuple[str, tuple[str, ...]]] = [
            ("binary", (c, a)) for c, a in product(cs, anions)
        ]
        for group, opposite in ((cs, anions), (anions, cs)):
            for i, j in combinations(group, 2):
                result.append(("theta", (i, j)))
                result.extend(("psi", (i, j, k)) for k in opposite)
        result.extend(("lambda", pair) for pair in combinations_with_replacement(ns, 2))
        result.extend(("lambda", (n, ion)) for n, ion in product(ns, cs + anions))
        result.extend(("zeta", (n, c, a)) for n, c, a in product(ns, cs, anions))
        result.extend(("mu", (n, n, n)) for n in ns)
        return tuple((kind, tuple(sorted(pair))) for kind, pair in result)

    def missing_parameters(self) -> tuple[str, ...]:
        missing = [
            "global." + name for name in _GLOBAL_UNITS[self.family] if name not in self.parameters
        ]
        present = {(item.kind, item.species): item for item in self.interactions}
        for kind, ids in self.required_interactions():
            item = present.get((kind, ids))
            for name in _INTERACTION_UNITS[kind]:
                if item is None or name not in item.parameters:
                    missing.append(f"{kind}[{','.join(ids)}].{name}")
        return tuple(missing)

    def source_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    self.source_id,
                    *(p.source_id for p in self.parameters.values()),
                    *(p.source_id for item in self.interactions for p in item.parameters.values()),
                }
            )
        )

    def _validate(self, molalities: Mapping[str, Any], t: Any, p: Any) -> None:
        missing = self.missing_parameters()
        if missing:
            raise ValueError("missing electrolyte parameters: " + ", ".join(missing))
        if set(molalities) != set(dict(self.charges)):
            raise ValueError("activity composition must explicitly include every declared solute")
        if any(not isfinite(scalar_value(v)) or scalar_value(v) < 0 for v in molalities.values()):
            raise ValueError("molalities must be finite and nonnegative")
        for value, bounds in ((t, self.temperature_bounds_K), (p, self.pressure_bounds_Pa)):
            if (
                not isfinite(scalar_value(value))
                or not bounds[0] <= scalar_value(value) <= bounds[1]
            ):
                raise ValueError("activity state outside declared T/P domain")
        ionic = sum(molalities[key] * z * z / 2 for key, z in self.charges)
        if scalar_value(ionic) > self.maximum_ionic_strength_mol_kg:
            raise ValueError("ionic strength outside declared activity-model domain")

    def excess_gibbs_RT_mol_kg(  # noqa: N802 - explicit thermodynamic normalization
        self, molalities: Mapping[str, Any], temperature_K: Any, pressure_Pa: Any
    ) -> Any:
        """Gex/(RT kg solvent), with parameters expressed on the molality scale."""
        m, t, p = molalities, temperature_K, pressure_Pa
        self._validate(m, t, p)
        values = {k: v.evaluate(t, p) for k, v in self.parameters.items()}
        if any(scalar_value(v) < 0 for v in values.values()):
            raise ValueError("global screening coefficients must be nonnegative")
        charges = dict(self.charges)
        ionic = compensated_sum(m[key] * z * z / 2 for key, z in self.charges)
        if self.family == "ideal":
            return ionic * 0
        if (
            scalar_value(ionic) == 0
            and derivative_depth(ionic) > 1
            and scalar_value(values.get("A", values.get("Aphi", 0))) > 0
        ):
            raise ValueError(
                "Debye screening has no finite second derivative at zero ionic strength"
            )
        if self.family == "Pitzer":
            b = values["b"]
            if scalar_value(b) <= 0:
                raise ValueError("Pitzer b must be positive")
            energy = (
                -4 * values["Aphi"] * ionic * logarithm(1 + b * ionic**0.5) / b
                if scalar_value(ionic)
                else ionic * 0
            )
        else:
            B = (
                values["B"]
                if self.family in ("DebyeHuckel", "SIT")
                else (1 if self.family == "Davies" else 0)
            )
            energy = _screening_integral(values["A"], B, ionic)
            if self.family == "Davies":
                energy += log(10) * values["A"] * values["b"] * ionic**2
        zsum = compensated_sum(m[key] * abs(z) for key, z in self.charges)
        for item in self.interactions:
            v = {k: parameter.evaluate(t, p) for k, parameter in item.parameters.items()}
            ids, kind = item.species, item.kind
            population = m[ids[0]] * m[ids[1]]
            if kind == "epsilon":
                energy += log(10) * population * v["epsilon"] * (0.5 if ids[0] == ids[1] else 1)
            elif kind == "binary":
                if any(scalar_value(v[key]) <= 0 for key in ("alpha1", "alpha2", "omega")):
                    raise ValueError("Pitzer binary screening ranges must be positive")
                # At an ion-free face, binary terms and their first derivatives
                # vanish. Keep the finite limiting coefficients without sqrt jets.
                root = ionic**0.5 if scalar_value(ionic) else 0
                B = v["beta0"] + 2 * v["beta1"] * _radial_integral(v["alpha1"] * root, 1)
                B += 2 * v["beta2"] * _radial_integral(v["alpha2"] * root, 1)
                C = v["C0"] + 4 * v["C1"] * _radial_integral(v["omega"] * root, 3)
                energy += population * (2 * B + zsum * C)
            elif kind == "theta":
                theta = v["theta"]
                zi, zj = (charges[key] for key in ids)
                if (
                    self.unsymmetrical_mixing == "Pitzer1975"
                    and zi != zj
                    and scalar_value(ionic)
                    and scalar_value(values["Aphi"])
                ):
                    scale = 6 * values["Aphi"] * ionic**0.5
                    theta += (
                        zi
                        * zj
                        / (4 * ionic)
                        * (
                            pitzer_mixing_J(scale * zi * zj)
                            - (pitzer_mixing_J(scale * zi * zi) + pitzer_mixing_J(scale * zj * zj))
                            / 2
                        )
                    )
                energy += 2 * population * theta
            elif kind == "lambda":
                energy += 2 * population * v["lambda"]
            else:
                energy += population * m[ids[2]] * v[kind]
        return energy

    def evaluate(
        self,
        molalities_mol_kg: Mapping[str, float],
        *,
        temperature_K: float,
        pressure_Pa: float,
        solvent_molar_mass_kg_mol: float,
    ) -> ElectrolyteActivityResult:
        if not isfinite(solvent_molar_mass_kg_mol) or solvent_molar_mass_kg_mol <= 0:
            raise ValueError("positive actual solvent molar mass is required")
        m = dict(molalities_mol_kg)
        energy = self.excess_gibbs_RT_mol_kg(m, temperature_K, pressure_Pa)
        coefficients = {}
        for key in m:
            values: dict[str, FirstDerivative[float, float]] = {
                k: FirstDerivative(v, int(k == key)) for k, v in m.items()
            }
            out = self.excess_gibbs_RT_mol_kg(values, temperature_K, pressure_Pa)
            coefficients[key] = float(out.slope if isinstance(out, FirstDerivative) else 0)
        total = sum(m.values())
        solvent = total + sum(m[k] * coefficients[k] for k in m) - energy
        if any(not isfinite(v) for v in (energy, solvent, *coefficients.values())):
            raise ValueError("nonfinite electrolyte activity evaluation")
        return ElectrolyteActivityResult(
            self.model_id,
            sum(m[key] * z * z / 2 for key, z in self.charges),
            coefficients,
            {k: log(v) + coefficients[k] if v else None for k, v in m.items()},
            -solvent_molar_mass_kg_mol * solvent,
            solvent / total if total else 1,
            energy,
            sum(m[key] * z for key, z in self.charges),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "chemworld-electrolyte-activity-v1",
            "model_id": self.model_id,
            "family": self.family,
            "charges": [list(row) for row in self.charges],
            "source_id": self.source_id,
            "maximum_ionic_strength_mol_kg": self.maximum_ionic_strength_mol_kg,
            "temperature_bounds_K": list(self.temperature_bounds_K),
            "pressure_bounds_Pa": list(self.pressure_bounds_Pa),
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
            "interactions": [v.to_dict() for v in self.interactions],
            "unsymmetrical_mixing": self.unsymmetrical_mixing,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ElectrolyteActivityModel:
        data = dict(payload)
        if data.pop("model", None) != "chemworld-electrolyte-activity-v1":
            raise ValueError("unknown electrolyte activity encoding")
        data["parameters"] = {k: ElectrolyteParameter(**v) for k, v in data["parameters"].items()}
        data["interactions"] = tuple(
            ElectrolyteInteraction.from_dict(v) for v in data["interactions"]
        )
        return cls(**data)


def electrolyte_activity_catalog() -> dict[str, dict[str, Any]]:
    """Same declarations as runtime validation, including explicit interaction units."""
    return {
        family: {
            "parameters": dict(units),
            "interactions": (
                {"epsilon": dict(_INTERACTION_UNITS["epsilon"])}
                if family == "SIT"
                else {k: dict(v) for k, v in _INTERACTION_UNITS.items() if k != "epsilon"}
                if family == "Pitzer"
                else {}
            ),
            "activity_basis": "molality / (1 mol/kg solvent)",
            "parameters_required": "all coefficients including declared zeros",
        }
        for family, units in _GLOBAL_UNITS.items()
    }
