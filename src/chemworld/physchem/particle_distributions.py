"""Diameter distributions with explicit weighting bases and finite moments.

Distribution tools own parameters, never a process particle inventory. Number,
length, area and volume correspond to orders 0, 1, 2 and 3 for similar shapes
and uniform density. Mass weighting is volume weighting only at that density.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from math import exp, inf, isfinite, log, pi, sqrt
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.special import gammainc, gammaincc, gammaln, ndtr

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.battery_particles import ParticleSizeDistribution
from chemworld.physchem.flow_meters import _engineering_root, _positive


def _diameter(value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError("diameter must be finite and nonnegative")


def _normal_interval(lower: float, upper: float) -> float:
    return float(ndtr(-lower) - ndtr(-upper) if lower > 0 else ndtr(upper) - ndtr(lower))


class ContinuousDiameterDistribution:
    """Common normalization, basis conversion, quantiles and moment means.

    Subclasses integrate dimensionless x=d/scale before restoring dimensions.
    Every converted basis is normalized over the actual truncated support.
    Divergent zero-endpoint moments are rejected instead of tiny-cutoff fitting.
    """

    order: float
    scale_m: float
    lower_m: float
    upper_m: float

    def _scaled_density(self, x: float) -> float:
        raise NotImplementedError

    def _scaled_moment(self, power: float, lower: float, upper: float) -> float:
        value, error = quad(
            lambda x: x**power * self._scaled_density(x),
            lower,
            upper,
            epsabs=1e-11,
            epsrel=1e-10,
            limit=200,
        )
        if not isfinite(value) or value < 0 or error > max(1e-10, abs(value) * 1e-7):
            raise ValueError("distribution moment is divergent or quadrature did not converge")
        return value

    def _moment(
        self, power: float, lower: float | None = None, upper: float | None = None
    ) -> float:
        a = self.lower_m if lower is None else max(self.lower_m, lower)
        b = self.upper_m if upper is None else min(self.upper_m, upper)
        if b <= a:
            return 0.0
        return self._scaled_moment(power, a / self.scale_m, b / self.scale_m)

    def pdf(self, diameter_m: float, *, order: float | None = None) -> float:
        _diameter(diameter_m)
        if diameter_m < self.lower_m or diameter_m > self.upper_m:
            return 0.0
        power = (self.order if order is None else order) - self.order
        denominator = self._moment(power)
        if diameter_m == 0:
            # Endpoint density can be singular; return the right limit only
            # where finite. CDF and finite moments remain well defined.
            return self._zero_density(power) / (self.scale_m * denominator)
        x = diameter_m / self.scale_m
        return x**power * self._scaled_density(x) / (self.scale_m * denominator)

    def _zero_density(self, power: float) -> float:
        return 0.0

    def cdf(self, diameter_m: float, *, order: float | None = None) -> float:
        _diameter(diameter_m)
        if diameter_m <= self.lower_m:
            return 0.0
        if diameter_m >= self.upper_m:
            return 1.0
        power = (self.order if order is None else order) - self.order
        return min(1.0, max(0.0, self._moment(power, upper=diameter_m) / self._moment(power)))

    def interval_fraction(
        self, lower_m: float, upper_m: float, *, order: float | None = None
    ) -> float:
        if lower_m > upper_m:
            raise ValueError("diameter interval must increase")
        _diameter(lower_m)
        _diameter(upper_m)
        power = (self.order if order is None else order) - self.order
        return self._moment(power, lower_m, upper_m) / self._moment(power)

    def quantile(self, fraction: float, *, order: float | None = None) -> float:
        if not isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("CDF fraction must lie in [0,1]")
        if fraction == 0:
            return self.lower_m
        if fraction == 1:
            return self.upper_m
        upper = self.upper_m / self.scale_m
        if upper == inf:
            upper = 1.0
            while self.cdf(upper * self.scale_m, order=order) < fraction:
                upper *= 2
                if upper > 1e100:
                    raise ValueError("could not bracket distribution quantile")
        lower = self.lower_m / self.scale_m
        # Tighten the monotone CDF bracket before the shared local solver;
        # a broad bracket midpoint can lie in a numerically flat tail.
        for _ in range(12):
            middle = (lower + upper) / 2
            if self.cdf(middle * self.scale_m, order=order) < fraction:
                lower = middle
            else:
                upper = middle
        root = _engineering_root(
            lambda x: self.cdf(x * self.scale_m, order=order) - fraction,
            (lower + upper) / 2,
            lower,
            upper,
            name="scaled_particle_quantile",
            unit="dimensionless",
            scale=1,
        )
        return root * self.scale_m

    def moment(self, power: float, *, order: float | None = None) -> float:
        shift = (self.order if order is None else order) - self.order
        return self.scale_m**power * self._moment(power + shift) / self._moment(shift)

    def basis_integral(self, power: float, lower_m: float, upper_m: float) -> float:
        """Definite integral of d**power times the original normalized density."""
        if lower_m > upper_m:
            raise ValueError("moment interval must increase")
        _diameter(lower_m)
        _diameter(upper_m)
        return self.scale_m**power * self._moment(power, lower_m, upper_m) / self._moment(0)

    def mean_size(self, p: float, q: float) -> float:
        """D[p,q] from number moments; p=q uses a log moment, not differencing."""
        denominator = self._moment(q - self.order)
        if p != q:
            return self.scale_m * (self._moment(p - self.order) / denominator) ** (1 / (p - q))
        a, b = self.lower_m / self.scale_m, self.upper_m / self.scale_m
        numerator, error = quad(
            lambda x: log(x) * x ** (p - self.order) * self._scaled_density(x),
            a,
            b,
            epsabs=1e-10,
            epsrel=1e-10,
            limit=200,
        )
        if not isfinite(numerator) or error > max(1e-9, abs(numerator) * 1e-7):
            raise ValueError("log moment did not converge")
        return self.scale_m * exp(numerator / denominator)

    def mean_size_iso(self, k: float, r: float) -> float:
        return self.mean_size(k + r, r)

    @property
    def volume_specific_surface_area_m_inv(self) -> float:
        return 6 / self.mean_size(3, 2)

    def sample_cdf(
        self, diameters_m: Sequence[float], *, order: float | None = None
    ) -> tuple[float, ...]:
        return tuple(self.cdf(d, order=order) for d in diameters_m)

    def sample_fractions(
        self, upper_edges_m: Sequence[float], *, order: float | None = None
    ) -> tuple[float, ...]:
        if any(a >= b for a, b in pairwise(upper_edges_m)):
            raise ValueError("upper edges must increase")
        cdf = (0.0, *self.sample_cdf(upper_edges_m, order=order))
        return tuple(b - a for a, b in pairwise(cdf))

    def sample_curve(
        self, diameters_m: Sequence[float], *, order: float | None = None
    ) -> tuple[dict[str, float], ...]:
        """Plot-ready data; plotting/rendering belongs to the caller."""
        return tuple(
            {
                "diameter_m": d,
                "pdf_per_m": self.pdf(d, order=order),
                "cdf": self.cdf(d, order=order),
            }
            for d in diameters_m
        )

    def to_particle_quadrature(
        self, *, lower_m: float, upper_m: float, nodes: int, source: InputSource
    ) -> ParticleSizeDistribution:
        """Explicit finite Gauss-Legendre approximation in the existing radius PSD.

        Bounds are in diameter units. Output uses volume-weighted radius nodes;
        no mutation of the battery model or process inventory occurs.
        """
        if type(nodes) is not int or not 1 <= nodes <= 64 or not 0 < lower_m < upper_m:
            raise ValueError("finite positive diameter bounds and 1..64 nodes required")
        if lower_m < self.lower_m or upper_m > self.upper_m:
            raise ValueError("quadrature must remain within the distribution support")
        x, weights = np.polynomial.legendre.leggauss(nodes)
        diameters = lower_m + (x + 1) * (upper_m - lower_m) / 2
        weights *= np.array([self.pdf(float(d), order=3) for d in diameters])
        selected = weights > 0
        return ParticleSizeDistribution.from_weights(
            diameters[selected] / 2, weights[selected], source=source, basis="volume"
        )


@dataclass(frozen=True)
class AnalyticDiameterDistribution(ContinuousDiameterDistribution):
    """Lognormal, Gates-Gaudin-Schuhman, or Rosin-Rammler distributions.

    scale is lognormal median, GGS maximum, or RR characteristic diameter;
    RR k = scale**(-shape), avoiding a dimensionally ambiguous k parameter.
    """

    family: str
    scale_m: float
    shape: float
    order: float = 3.0
    lower_m: float = 0.0
    upper_m: float = inf

    def __post_init__(self) -> None:
        if self.family not in {"lognormal", "GGS", "RR"}:
            raise ValueError("unknown diameter distribution family")
        _positive(self.scale_m, "distribution diameter scale")
        _positive(self.shape, "distribution shape")
        _diameter(self.lower_m)
        if not isfinite(self.order) or not self.upper_m > self.lower_m:
            raise ValueError("finite order and increasing support required")
        if self.family == "GGS":
            object.__setattr__(self, "upper_m", min(self.upper_m, self.scale_m))
            if self.lower_m >= self.upper_m:
                raise ValueError("GGS support is empty")

    @classmethod
    def rosin_rammler_from_k(
        cls, k: float, shape: float, **kwargs: Any
    ) -> AnalyticDiameterDistribution:
        _positive(k, "Rosin-Rammler k")
        _positive(shape, "Rosin-Rammler shape")
        return cls("RR", exp(-log(k) / shape), shape, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "upper_m": None if self.upper_m == inf else self.upper_m}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AnalyticDiameterDistribution:
        return cls(**{**value, "upper_m": inf if value["upper_m"] is None else value["upper_m"]})

    def _scaled_density(self, x: float) -> float:
        if x <= 0:
            return self._zero_density(0)
        if self.family == "lognormal":
            return exp(-0.5 * (log(x) / self.shape) ** 2) / (x * self.shape * sqrt(2 * pi))
        if self.family == "GGS":
            return self.shape * x ** (self.shape - 1) if x <= 1 else 0.0
        return self.shape * x ** (self.shape - 1) * exp(-(x**self.shape))

    def _zero_density(self, power: float) -> float:
        if self.family == "lognormal":
            return 0.0
        exponent = self.shape + power - 1
        return 0.0 if exponent > 0 else self.shape if exponent == 0 else inf

    def _scaled_moment(self, power: float, lower: float, upper: float) -> float:
        if self.family == "lognormal":
            a = -inf if lower == 0 else (log(lower) - power * self.shape**2) / self.shape
            b = inf if upper == inf else (log(upper) - power * self.shape**2) / self.shape
            return exp(0.5 * (power * self.shape) ** 2) * _normal_interval(a, b)
        exponent = self.shape + power
        if lower == 0 and exponent <= 0:
            raise ValueError("requested diameter moment diverges at zero")
        if self.family == "GGS":
            if exponent == 0:
                return self.shape * log(upper / lower)
            return self.shape / exponent * (upper**exponent - lower**exponent)
        a = 1 + power / self.shape
        if a <= 0:
            return super()._scaled_moment(power, lower, upper)
        lo, hi = lower**self.shape, upper**self.shape
        probability = (
            gammaincc(a, lo) - gammaincc(a, hi) if lo > a else gammainc(a, hi) - gammainc(a, lo)
        )
        return exp(float(gammaln(a))) * float(probability)

    def mean_size(self, p: float, q: float) -> float:
        if self.family == "lognormal" and self.lower_m == 0 and self.upper_m == inf:
            return self.scale_m * exp((p + q - 2 * self.order) * self.shape**2 / 2)
        return super().mean_size(p, q)


class CustomDiameterDistribution(ContinuousDiameterDistribution):
    """Caller density with explicit support and basis; callable is not serialized."""

    def __init__(
        self,
        pdf_per_m: Callable[[float], float],
        *,
        scale_m: float,
        lower_m: float,
        upper_m: float,
        order: float = 3,
    ) -> None:
        _positive(scale_m, "diameter scale")
        _diameter(lower_m)
        if not isfinite(upper_m) or upper_m <= lower_m or not isfinite(order):
            raise ValueError("custom distribution requires finite increasing support and order")
        self.scale_m, self.lower_m, self.upper_m, self.order = scale_m, lower_m, upper_m, order
        self.density = pdf_per_m

    def _scaled_density(self, x: float) -> float:
        result = float(self.density(x * self.scale_m)) * self.scale_m
        if not isfinite(result) or result < 0:
            raise ValueError("custom density must be finite and nonnegative")
        return result


class InterpolatedDiameterDistribution(ContinuousDiameterDistribution):
    """Monotone PCHIP of normalized CDF knots, bounded by declared endpoints."""

    def __init__(
        self,
        diameters_m: Sequence[float],
        fractions: Sequence[float],
        *,
        order: float = 3,
        cumulative: bool = False,
    ) -> None:
        ds, fs = np.asarray(diameters_m, dtype=float), np.asarray(fractions, dtype=float)
        if (
            ds.ndim != 1
            or fs.ndim != 1
            or len(ds) < 2
            or not np.all(np.isfinite(ds))
            or np.any(ds < 0)
            or np.any(np.diff(ds) <= 0)
            or not np.all(np.isfinite(fs))
            or np.any(fs < 0)
            or not isfinite(order)
        ):
            raise ValueError("ordered nonnegative diameter knots and fractions required")
        if cumulative:
            if len(fs) != len(ds) or fs[0] != 0 or np.any(np.diff(fs) < 0):
                raise ValueError("CDF knots require equal lengths, initial zero and monotonicity")
            cdf = fs
        else:
            if len(ds) == len(fs) + 1:
                fs = np.r_[0, fs]
            elif len(ds) == len(fs) and ds[0] > 0:
                ds, fs = np.r_[0, ds], np.r_[0, fs]
            elif len(ds) != len(fs) or fs[0] != 0:
                raise ValueError("fraction count must describe bins or positive diameter nodes")
            cdf = np.cumsum(fs)
        if cdf[-1] <= 0:
            raise ValueError("distribution must contain positive total weight")
        self.diameters_m = tuple(float(d) for d in ds)
        self.cumulative = tuple(float(f) for f in cdf / cdf[-1])
        self.scale_m, self.lower_m, self.upper_m, self.order = (
            float(ds[-1]),
            float(ds[0]),
            float(ds[-1]),
            order,
        )
        self.spline = PchipInterpolator(ds / self.scale_m, self.cumulative, extrapolate=False)
        self.derivative = self.spline.derivative()

    def _scaled_density(self, x: float) -> float:
        return max(0.0, float(self.derivative(x)))

    def _zero_density(self, power: float) -> float:
        density = float(self.derivative(0))
        return density if power == 0 else 0.0 if power > 0 or density == 0 else inf

    def _scaled_moment(self, power: float, lower: float, upper: float) -> float:
        if power == 0:
            return float(self.spline(upper) - self.spline(lower))
        if lower == 0:
            coefficients = self.derivative.c[:, 0]
            leading = next(
                (
                    len(coefficients) - 1 - i
                    for i in reversed(range(len(coefficients)))
                    if coefficients[i] != 0
                ),
                None,
            )
            if leading is not None and power + leading <= -1:
                raise ValueError(
                    "interpolated converted basis diverges at zero; use positive support"
                )
        boundaries = [
            lower,
            *(d / self.scale_m for d in self.diameters_m if lower < d / self.scale_m < upper),
            upper,
        ]
        return sum(
            super(InterpolatedDiameterDistribution, self)._scaled_moment(power, a, b)
            for a, b in pairwise(boundaries)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "diameters_m": list(self.diameters_m),
            "fractions": list(self.cumulative),
            "order": self.order,
            "cumulative": True,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InterpolatedDiameterDistribution:
        return cls(**value)


class BinnedDiameterDistribution:
    """Measured diameter nodes or uniform-diameter bin-power convention.

    The source's finite-bin convention converts fractions through bin-average
    powers, then computes number moments. It is not an exact continuous density
    reconstructed from volume bins. CDF/PDF interpolation is separately exposed.
    """

    def __init__(
        self,
        diameters_m: Sequence[float],
        fractions: Sequence[float],
        *,
        order: float = 3,
        cumulative: bool = False,
    ) -> None:
        ds, fs = tuple(diameters_m), tuple(fractions)
        if (
            not fs
            or len(ds) not in (len(fs), len(fs) + 1)
            or any(not isfinite(d) or d <= 0 for d in ds)
            or any(a >= b for a, b in pairwise(ds))
            or any(not isfinite(f) or f < 0 for f in fs)
            or not isfinite(order)
        ):
            raise ValueError("positive increasing diameters and nonnegative fractions required")
        if cumulative:
            if any(a > b for a, b in pairwise(fs)):
                raise ValueError("cumulative fractions must be monotone")
            fs = tuple(b - a for a, b in zip((0, *fs), fs, strict=False))
        if sum(fs) <= 0:
            raise ValueError("distribution must have positive total weight")
        self.diameters_m, self.input_fractions, self.order = ds, fs, order
        self.is_binned = len(ds) == len(fs) + 1
        weights = [f * self.bin_power(i, 3 - order) for i, f in enumerate(fs)]
        self.volume_fractions = tuple(w / sum(weights) for w in weights)

    def bin_power(self, index: int, power: float) -> float:
        lower = self.diameters_m[index]
        if not self.is_binned:
            return lower**power
        upper = self.diameters_m[index + 1]
        if power == -1:
            return log(upper / lower) / (upper - lower)
        return (upper ** (power + 1) - lower ** (power + 1)) / ((upper - lower) * (power + 1))

    def fractions(self, order: float) -> tuple[float, ...]:
        weights = [f / self.bin_power(i, 3 - order) for i, f in enumerate(self.volume_fractions)]
        return tuple(w / sum(weights) for w in weights)

    def mean_size(self, p: float, q: float) -> float:
        numbers = self.fractions(0)
        denominator = sum(self.bin_power(i, q) * f for i, f in enumerate(numbers))
        if p == q:
            numerator = sum(
                log(self.bin_power(i, 1)) * self.bin_power(i, p) * f for i, f in enumerate(numbers)
            )
            return exp(numerator / denominator)
        return (sum(self.bin_power(i, p) * f for i, f in enumerate(numbers)) / denominator) ** (
            1 / (p - q)
        )

    def mean_size_iso(self, k: float, r: float) -> float:
        return self.mean_size(k + r, r)

    @property
    def volume_specific_surface_area_m_inv(self) -> float:
        return sum(f * 6 / self.bin_power(i, 1) for i, f in enumerate(self.volume_fractions))

    def interpolated(self) -> InterpolatedDiameterDistribution:
        return InterpolatedDiameterDistribution(self.diameters_m, self.volume_fractions, order=3)

    def to_particle_quadrature(self, *, source: InputSource) -> ParticleSizeDistribution:
        selected = [
            (self.bin_power(i, 1) / 2, f) for i, f in enumerate(self.volume_fractions) if f > 0
        ]
        return ParticleSizeDistribution.from_weights(
            [item[0] for item in selected],
            [item[1] for item in selected],
            basis="volume",
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "diameters_m": list(self.diameters_m),
            "fractions": list(self.input_fractions),
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BinnedDiameterDistribution:
        return cls(**value)


def diameter_spacing(
    *,
    lower_m: float | None = None,
    upper_m: float | None = None,
    points: int = 20,
    method: str = "logarithmic",
) -> tuple[float, ...]:
    if type(points) is not int or points < 2:
        raise ValueError("at least two diameter points required")
    if method.lower().startswith("r"):
        series = float(method[1:])
        _positive(series, "Renard series index")
        if (lower_m is None) == (upper_m is None):
            raise ValueError("Renard spacing requires exactly one endpoint")
        anchor = lower_m if lower_m is not None else upper_m
        if anchor is None:
            raise AssertionError("invariant violated: anchor is not None")
        _positive(anchor, "diameter endpoint")
        powers = np.arange(points) if lower_m is not None else np.arange(1 - points, 1)
        return tuple(float(anchor * 10 ** (i / series)) for i in powers)
    if lower_m is None or upper_m is None or not 0 <= lower_m < upper_m:
        raise ValueError("increasing nonnegative diameter endpoints required")
    if method == "linear":
        return tuple(float(d) for d in np.linspace(lower_m, upper_m, points))
    if method == "logarithmic" and lower_m > 0:
        return tuple(float(d) for d in np.geomspace(lower_m, upper_m, points))
    raise ValueError("unknown spacing method or nonpositive logarithmic endpoint")
