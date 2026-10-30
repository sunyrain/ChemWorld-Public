"""Native T/concentration correlations with explicit, consistent caloric datum.

This is a zero-compressibility engineering approximation, not a stable fluid
EOS. In particular no cv, acoustic speed, chemical equilibrium or measured
mixing enthalpy follows from these fits. No reference backend is imported.
"""

from dataclasses import asdict, dataclass
from math import isfinite

from chemworld.physchem.caloric_math import exponential, logarithm, scalar_value
from chemworld.physchem.second_order_pair import SecondOrderPair


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{label} must be a finite real number")
    return value


def _horner(coefficients, value):
    result = 0.0
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


@dataclass(frozen=True)
class IncompressibleCorrelation:
    family: str
    coefficients: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        coefficients = tuple(tuple(row) for row in self.coefficients)
        if not coefficients or not coefficients[0] or any(
            len(row) != len(coefficients[0]) for row in coefficients
        ):
            raise ValueError("incompressible coefficients must be a nonempty rectangular matrix")
        for row in coefficients:
            for value in row:
                _finite(value, "coefficient")
        if self.family not in {
            "polynomial", "exppolynomial", "exponential", "logexponential", "polyoffset"
        }:
            raise ValueError("unknown incompressible correlation family")
        if self.family in {"exponential", "logexponential"} and (
            len(coefficients) != 3 or len(coefficients[0]) != 1
        ):
            raise ValueError("exponential correlation requires three column coefficients")
        if self.family == "polyoffset" and (
            min(len(coefficients), len(coefficients[0])) != 1
            or max(len(coefficients), len(coefficients[0])) < 2
        ):
            raise ValueError("offset polynomial requires an offset and a nonempty vector")
        object.__setattr__(self, "coefficients", coefficients)

    def evaluate(self, y, x, *, y_center=0.0, x_center=0.0):
        for value in (y, x, y_center, x_center):
            _finite(scalar_value(value), "correlation coordinate")
        coefficients = self.coefficients
        if self.family in {"polynomial", "exppolynomial"}:
            result = _horner(tuple(_horner(row, x-x_center) for row in coefficients),
                             y-y_center)
            if self.family == "exppolynomial":
                result = exponential(result)
        elif self.family == "polyoffset":
            row = len(coefficients) == 1
            vector = coefficients[0] if row else tuple(c[0] for c in coefficients)
            result = _horner(vector[1:], (x if row else y)-vector[0])
        else:
            a, b, c = (row[0] for row in coefficients)
            if self.family == "exponential":
                denominator = y+b
                if scalar_value(denominator) == 0:
                    raise ValueError("singular exponential correlation")
                result = exponential(a/denominator-c)
            else:
                denominator = y+a
                if scalar_value(denominator) == 0:
                    raise ValueError("singular logarithmic exponential correlation")
                argument = 1/denominator + 1/denominator**2
                if scalar_value(argument) <= 0:
                    raise ValueError("logarithmic exponential correlation outside real domain")
                result = exponential(b*logarithm(argument)+c)
        if not isfinite(scalar_value(result)):
            raise ValueError("nonfinite incompressible correlation")
        return result

    def temperature_derivative(self, temperature, fraction, *, y_center, x_center):
        if self.family != "polynomial":
            raise ValueError("caloric density derivative requires the declared polynomial")
        coefficients = tuple(i*_horner(row, fraction-x_center)
                             for i, row in enumerate(self.coefficients) if i)
        return _horner(coefficients, temperature-y_center)

    def caloric_integrals(self, temperature, fraction, reference, *, y_center, x_center):
        """Definite integrals of cp and cp/T; algebraic even outside the fit domain.

        The latter permission is only for the declared reference datum. State
        evaluation checks the temperature interval before calling this method.
        """
        if self.family != "polynomial":
            raise ValueError("caloric integration requires a polynomial heat capacity")
        y, r = temperature-y_center, reference-y_center
        coefficients = tuple(_horner(row, fraction-x_center) for row in self.coefficients)

        def difference(power):
            # Factor the difference of powers to retain small temperature changes.
            return (temperature-reference)*sum(y**j*r**(power-1-j) for j in range(power))

        h, s = 0.0, 0.0
        integral = logarithm(temperature/reference)
        for i, coefficient in enumerate(coefficients):
            h += coefficient*difference(i+1)/(i+1)
            if i:
                integral = difference(i)/i-y_center*integral
            s += coefficient*integral
        return h, s

    def to_dict(self):
        return {"family": self.family, "coefficients": [list(row) for row in self.coefficients]}

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


@dataclass(frozen=True)
class IncompressibleReference:
    temperature_K: float = 293.15
    pressure_Pa: float = 101325.0
    enthalpy_J_kg: float = 0.0
    entropy_J_kg_K: float = 0.0

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            _finite(value, key)
        if self.temperature_K <= 0 or self.pressure_Pa <= 0:
            raise ValueError("incompressible reference requires positive T/P")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class IncompressibleState:
    fluid_id: str
    temperature_K: float
    pressure_Pa: float
    fraction: float
    fraction_basis: str
    mass_density_kg_m3: float
    specific_volume_m3_kg: float
    source_cp_J_kg_K: float
    cp_J_kg_K: float
    enthalpy_J_kg: float
    entropy_J_kg_K: float
    internal_energy_J_kg: float
    gibbs_energy_J_kg: float
    drho_dT_kg_m3_K: float
    drho_dx_kg_m3: float
    dh_dx_J_kg: float
    ds_dx_J_kg_K: float
    dh_dp_J_kg_Pa: float
    ds_dp_J_kg_K_Pa: float
    thermal_expansion_K_inv: float
    reference_temperature_outside_fit: bool
    freezing_boundary_available: bool
    saturation_boundary_available_at_temperature: bool
    caloric_scope: str = "same_composition_reference; mixing_heat_not_measured"
    stability_scope: str = "incompressible_correlation; no_cv_sound_or_stability_claim"

    def to_dict(self):
        return asdict(self)


_CORRELATIONS = (
    "density", "specific_heat", "viscosity", "conductivity", "saturation_pressure",
    "T_freeze", "mass2input", "volume2input", "mole2input",
)


@dataclass(frozen=True)
class IncompressibleFluid:
    fluid_id: str
    description: str
    citation: str
    source: str
    fraction_basis: str
    temperature_bounds_K: tuple[float, float]
    fraction_bounds: tuple[float, float]
    temperature_center_K: float
    fraction_center: float
    minimum_saturation_temperature_K: float
    correlations: tuple[tuple[str, IncompressibleCorrelation | None], ...]
    reference: IncompressibleReference = IncompressibleReference()

    def __post_init__(self) -> None:
        if not self.fluid_id or not self.source or self.fraction_basis not in {
            "pure", "mass", "volume", "mole"
        }:
            raise ValueError("incompressible fluid requires an identity, source and fraction basis")
        for field in ("temperature_bounds_K", "fraction_bounds"):
            values = tuple(getattr(self, field))
            if len(values) != 2:
                raise ValueError("incompressible bounds must have two endpoints")
            for value in values:
                _finite(value, field)
            object.__setattr__(self, field, values)
        low, high = self.temperature_bounds_K
        xmin, xmax = self.fraction_bounds
        if not 0 < low < high or not 0 <= xmin <= xmax <= 1:
            raise ValueError("invalid incompressible temperature/fraction interval")
        for value in (self.temperature_center_K, self.fraction_center,
                      self.minimum_saturation_temperature_K):
            _finite(value, "correlation center/bound")
        pairs = tuple(tuple(pair) for pair in self.correlations)
        mapping = dict(pairs)
        if len(pairs) != len(mapping) or set(mapping) != set(_CORRELATIONS):
            raise ValueError("incompressible correlations must declare each known field once")
        if any(c is not None and not isinstance(c, IncompressibleCorrelation)
               for c in mapping.values()):
            raise TypeError("incompressible correlations require validated declarations")
        if any(mapping[key] is None or mapping[key].family != "polynomial"
               for key in ("density", "specific_heat")):
            raise ValueError("incompressible calorics requires polynomial density and cp")
        if self.fraction_basis == "pure" and any(
            len(mapping[key].coefficients[0]) != 1 for key in ("density", "specific_heat")
        ):
            raise ValueError("pure fluid density/cp cannot depend on concentration")
        if not isinstance(self.reference, IncompressibleReference):
            raise TypeError("incompressible reference must be explicitly validated")
        object.__setattr__(self, "correlations", pairs)

    def correlation(self, name):
        if name not in _CORRELATIONS:
            raise ValueError("unknown incompressible property")
        result = dict(self.correlations)[name]
        if result is None:
            raise ValueError(f"{self.fluid_id}: {name} is unavailable in the source card")
        return result

    def concentration(self, fraction, *, basis):
        _finite(fraction, "concentration")
        if basis != self.fraction_basis:
            raise ValueError("concentration conversion lacks a qualified definition")
        if self.fraction_basis == "pure":
            if fraction != 1:
                raise ValueError("pure incompressible correlation requires fraction=1")
        elif not self.fraction_bounds[0] <= fraction <= self.fraction_bounds[1]:
            raise ValueError("concentration outside source interval")
        return fraction

    def _value(self, name, T, x):
        return self.correlation(name).evaluate(
            T, x, y_center=self.temperature_center_K, x_center=self.fraction_center)

    def freezing_temperature(self, pressure_Pa, fraction):
        _finite(pressure_Pa, "pressure")
        if pressure_Pa <= 0:
            raise ValueError("positive pressure required")
        self.concentration(fraction, basis=self.fraction_basis)
        fit = self.correlation("T_freeze")
        if fit.family in {"exponential", "logexponential"}:
            return fit.evaluate(fraction, 0.0)
        return fit.evaluate(pressure_Pa, fraction, y_center=0.0, x_center=self.fraction_center)

    def saturation_pressure(self, temperature_K, fraction):
        self._check_coordinates(temperature_K, 1.0, fraction)
        if temperature_K <= self.minimum_saturation_temperature_K:
            raise ValueError("saturation pressure unavailable at/below source TminPsat")
        value = self._value("saturation_pressure", temperature_K, fraction)
        if value <= 0:
            raise ValueError("nonpositive saturation-pressure fit")
        return value

    def _check_coordinates(self, T, p, x):
        _finite(T, "temperature")
        _finite(p, "pressure")
        self.concentration(x, basis=self.fraction_basis)
        if p <= 0 or not self.temperature_bounds_K[0] <= T <= self.temperature_bounds_K[1]:
            raise ValueError("temperature/pressure outside incompressible coordinate domain")

    def validate_state(self, T, p, x):
        self._check_coordinates(T, p, x)
        fits = dict(self.correlations)
        if fits["T_freeze"] is not None and self.freezing_temperature(p, x) > T:
            raise ValueError("temperature below source freezing boundary")
        if (fits["saturation_pressure"] is not None
                and self.minimum_saturation_temperature_K < T
                and p < self.saturation_pressure(T, x)):
            raise ValueError("pressure below source saturation boundary")

    def property_value(self, name, temperature_K, pressure_Pa, fraction):
        """A source correlation; missing or nonpositive transport is not fabricated."""
        if name not in {"density", "specific_heat", "viscosity", "conductivity"}:
            raise ValueError("use explicit methods for phase boundaries and caloric state")
        self.validate_state(temperature_K, pressure_Pa, fraction)
        value = self._value(name, temperature_K, fraction)
        if value <= 0:
            raise ValueError(f"nonpositive source {name} fit")
        return value

    def _volume(self, T, x):
        rho = self._value("density", T, x)
        if scalar_value(rho) <= 0:
            raise ValueError("nonpositive density at state or declared reference")
        rho_T = self.correlation("density").temperature_derivative(
            T, x, y_center=self.temperature_center_K, x_center=self.fraction_center)
        return 1/rho, -rho_T/rho**2

    def _caloric(self, T, p, x):
        ref = self.reference
        h, s = self.correlation("specific_heat").caloric_integrals(
            T, x, ref.temperature_K,
            y_center=self.temperature_center_K, x_center=self.fraction_center)
        v, vT = self._volume(T, x)
        vr, vrT = self._volume(ref.temperature_K, x)
        return (ref.enthalpy_J_kg+h+p*(v-T*vT)
                - ref.pressure_Pa*(vr-ref.temperature_K*vrT),
                ref.entropy_J_kg_K+s-p*vT+ref.pressure_Pa*vrT)

    def state(self, temperature_K, pressure_Pa, fraction):
        self.validate_state(temperature_K, pressure_Pa, fraction)
        T, p, x = temperature_K, pressure_Pa, fraction
        tjet = SecondOrderPair(T, dx=1.0)
        xjet = SecondOrderPair(x, dy=0.0 if self.fraction_basis == "pure" else 1.0)
        rho = self._value("density", tjet, xjet)
        source_cp = self.property_value("specific_heat", T, p, x)
        v, vT = self._volume(tjet, xjet)
        h, s = self._caloric(tjet, p, xjet)
        result = IncompressibleState(
            self.fluid_id, T, p, x, self.fraction_basis, rho.value, v.value,
            source_cp, h.dx, h.value, s.value, h.value-p*v.value, h.value-T*s.value,
            rho.dx, rho.dy, h.dy, s.dy, v.value-T*vT.value, -vT.value, vT.value/v.value,
            not self.temperature_bounds_K[0] <= self.reference.temperature_K
            <= self.temperature_bounds_K[1], dict(self.correlations)["T_freeze"] is not None,
            dict(self.correlations)["saturation_pressure"] is not None
            and self.minimum_saturation_temperature_K < T)
        if any(not isfinite(value) for value in asdict(result).values()
               if isinstance(value, (float, int))):
            raise ValueError("nonfinite incompressible caloric response")
        return result

    def temperature_from_enthalpy(self, pressure_Pa, enthalpy_J_kg, fraction, *, bounds_K):
        """Explicit single branch/interval PH closure using the shared residual solver."""
        from chemworld.physchem.enthalpy import caloric_temperature

        _finite(enthalpy_J_kg, "enthalpy")

        def state(T):
            return self.state(T, pressure_Pa, fraction)

        return caloric_temperature(
            lambda T: state(T).enthalpy_J_kg, enthalpy_J_kg, bounds_K,
            energy_basis="enthalpy", algorithm="residual",
            heat_capacity_at_temperature=lambda T: state(T).cp_J_kg_K,
            xtol=1e-9, rtol=1e-12)

    def to_dict(self):
        return {"model": "incompressible-fluid-v1", "fluid_id": self.fluid_id,
                "description": self.description, "citation": self.citation, "source": self.source,
                "fraction_basis": self.fraction_basis,
                "temperature_bounds_K": list(self.temperature_bounds_K),
                "fraction_bounds": list(self.fraction_bounds),
                "temperature_center_K": self.temperature_center_K,
                "fraction_center": self.fraction_center,
                "minimum_saturation_temperature_K": self.minimum_saturation_temperature_K,
                "correlations": {k: None if v is None else v.to_dict()
                                 for k, v in self.correlations},
                "reference": self.reference.to_dict()}

    @classmethod
    def from_dict(cls, payload):
        data = dict(payload)
        if data.pop("model") != "incompressible-fluid-v1":
            raise ValueError("unknown incompressible fluid schema")
        data["correlations"] = tuple((k, None if v is None
                                      else IncompressibleCorrelation.from_dict(v))
                                     for k, v in data["correlations"].items())
        data["reference"] = IncompressibleReference(**data.get("reference", {}))
        return cls(**data)
