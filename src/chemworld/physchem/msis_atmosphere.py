"""NRLMSISE-00 atmosphere from explicitly sourced empirical parameters.

Picone et al. (2002), JGR 107(A12), 1468. Source model equations and
coefficient schema were consulted; native implementation composes recurrence
harmonics, SciPy cubic inverse-temperature profiles and species transitions.
Public scope is the all-effects-on GTD7 model, including seven-index AP history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, exp, isfinite, log, sin
from typing import TYPE_CHECKING, Any

from chemworld.physchem.msis_harmonics import MSISHarmonics
from chemworld.physchem.msis_profiles import (
    BatesProfile,
    InverseTemperatureProfile,
    chemistry_factor,
    mixed_density,
)
from chemworld.physchem.transport import FluidState

if TYPE_CHECKING:
    from chemworld.data.scientific_inputs import ScientificTable

MSIS_PARAMETER_SHAPES = {
    "pt": (1, 150),
    "pd": (9, 150),
    "ps": (1, 150),
    "pdl": (2, 25),
    "ptm": (1, 10),
    "pdm": (8, 10),
    "ptl": (4, 100),
    "pma": (10, 100),
    "pavgm": (1, 10),
}
_SPECIES = ("He", "O", "N2", "O2", "Ar", "H", "N", "O_anomalous")
_MASSES = (4.002602, 15.9994, 28.0134, 31.9988, 39.948, 1.00794, 14.0067)


class MSISParameters:
    """3200 model coefficients as index/value rows grouped by named array.

    Coefficients retain their original model-unit numerical convention; table
    cells are dimensionless encoded parameters, not interchangeable SI values.
    Row indices are zero-based row-major offsets in MSIS_PARAMETER_SHAPES.
    """

    def __init__(self, table: ScientificTable) -> None:
        if table.kind != "parameters":
            raise ValueError("MSIS requires an explicit scientific parameter table")
        self.table = table
        groups: dict[str, dict[int, float]] = {}
        for row in table.rows:
            index, value = (
                row.values["index"].to("dimensionless"),
                row.values["value"].to("dimensionless"),
            )
            if index.quality != "valid" or value.quality != "valid":
                raise ValueError("valid MSIS parameter cells required")
            numeric_index, numeric_value = float(index.value), float(value.value)
            if (
                not isfinite(numeric_index)
                or not numeric_index.is_integer()
                or not isfinite(numeric_value)
            ):
                raise ValueError("finite MSIS coefficient and integer index required")
            group = groups.setdefault(row.group_id, {})
            if int(numeric_index) in group:
                raise ValueError("duplicate MSIS parameter index")
            group[int(numeric_index)] = numeric_value
        if set(groups) != set(MSIS_PARAMETER_SHAPES):
            raise ValueError("all nine MSIS parameter arrays required")
        self._arrays = {}
        for name, (rows, columns) in MSIS_PARAMETER_SHAPES.items():
            values = groups[name]
            if set(values) != set(range(rows * columns)):
                raise ValueError(f"incomplete MSIS array: {name}")
            self._arrays[name] = tuple(
                tuple(values[r * columns + c] for c in range(columns)) for r in range(rows)
            )

    def vector(self, name: str, row: int = 0) -> tuple[float, ...]:
        return self._arrays[name][row]

    def to_dict(self) -> dict[str, Any]:
        return self.table.to_dict()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MSISParameters:
        from chemworld.data.scientific_inputs import ScientificTable

        return cls(ScientificTable.from_dict(payload))


@dataclass(frozen=True)
class MSISAtmosphereState:
    elevation_m: float
    temperature_K: float
    exospheric_temperature_K: float
    number_densities_m3: tuple[float, ...]
    density_kg_m3: float
    composition_density_kg_m3: float
    pressure_Pa: float
    mole_fractions: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "species": _SPECIES}

    def fluid_state(
        self,
        viscosity_Pa_s: float,
        heat_capacity_J_kg_K: float,
        thermal_conductivity_W_m_K: float,
    ) -> FluidState:
        """Caller supplies transport/caloric properties absent from this empirical model."""
        return FluidState(
            self.density_kg_m3, viscosity_Pa_s, heat_capacity_J_kg_K, thermal_conductivity_W_m_K
        )


class MSISAtmosphere:
    def __init__(self, parameters: MSISParameters) -> None:
        self.parameters = parameters

    def evaluate(
        self,
        elevation_m: float,
        *,
        latitude_degrees: float = 0,
        longitude_degrees: float = 0,
        day_of_year: int = 0,
        seconds_utc: float = 0,
        solar_flux: float = 150,
        mean_solar_flux: float = 150,
        daily_ap: float = 4,
        ap_history: tuple[float, ...] | None = None,
    ) -> MSISAtmosphereState:
        """0..1000km, SI outputs; quiet AP=4 is explicit (source wrapper defaults0).

        F10.7 uses the model's solar flux unit (10^-22 W/m²/Hz) at actual
        Earth-Sun distance. Day0 is the source-compatible preceding calendar
        boundary. AP history is daily,current3h,3h,6h,9h,12..33h,36..57h.
        No time-varying weather, transport coefficients or dynamic inventory.
        """
        values = (
            elevation_m,
            latitude_degrees,
            longitude_degrees,
            seconds_utc,
            solar_flux,
            mean_solar_flux,
            daily_ap,
        )
        if not all(isfinite(v) for v in values):
            raise ValueError("finite MSIS inputs required")
        if not (
            0 <= elevation_m <= 1e6
            and -90 <= latitude_degrees <= 90
            and -180 <= longitude_degrees <= 180
            and 0 <= seconds_utc < 86400
        ):
            raise ValueError("MSIS altitude/geography/time outside declared domain")
        if not isinstance(day_of_year, int) or not 0 <= day_of_year <= 366:
            raise ValueError("integer model day in [0,366] required")
        if min(solar_flux, mean_solar_flux) <= 0 or daily_ap < 0:
            raise ValueError("positive solar flux and nonnegative AP required")
        if ap_history is not None:
            if len(ap_history) != 7 or any(not isfinite(v) or v < 0 for v in ap_history):
                raise ValueError("seven finite nonnegative AP indices required")
            ap_history = tuple(ap_history)
        harmonic = MSISHarmonics(
            latitude_degrees,
            longitude_degrees,
            day_of_year,
            seconds_utc,
            solar_flux,
            mean_solar_flux,
            daily_ap,
            ap_history,
        )
        arrays = self.parameters
        pt, ptm, ps = (arrays.vector(name) for name in ("pt", "ptm", "ps"))
        pdl = [arrays.vector("pdl", i) for i in range(2)]
        pdm = [arrays.vector("pdm", i) for i in range(8)]
        pd = [arrays.vector("pd", i) for i in range(9)]
        ptl = [arrays.vector("ptl", i) for i in range(4)]
        pma = [arrays.vector("pma", i) for i in range(10)]
        pavg = arrays.vector("pavgm")
        evaluate = harmonic.evaluate
        latitude_cosine = cos(2 * 0.0174533 * latitude_degrees)
        gravity = 980.616 * (1 - 0.0026373 * latitude_cosine)
        radius = 2 * gravity / (3.085462e-6 + 2.27e-9 * latitude_cosine) * 1e-5
        height = max(elevation_m / 1000, 72.5)
        top = pdl[1][15]
        exosphere = ptm[0] * pt[0] * (1 + evaluate(pt) if height > top else 1)
        gradient = ptm[3] * ps[0] * (1 + evaluate(ps) if height > 72.5 else 1)
        base_temperature = ptm[1] * pd[3][0] * (1 + evaluate(pd[3]))
        slope = gradient / (exosphere - base_temperature)
        temperatures = [1.0] + [
            ptm[index] * ptl[i][0] / (1 - evaluate(ptl[i], lower=True) if height < 300 else 1)
            for i, index in enumerate((6, 2, 7, 4))
        ]
        bottom_gradient = ptm[8] * pma[8][0] * temperatures[4] ** 2 / (ptm[4] * ptl[3][0]) ** 2
        bottom_gradient *= 1 + evaluate(pma[8], lower=True) if height < 300 else 1
        profile = BatesProfile(
            exosphere,
            base_temperature,
            ptm[5],
            slope,
            (top, 110, 100, 90, 72.5),
            tuple(temperatures),
            bottom_gradient,
            radius,
            gravity,
        )
        mean_mass = pdm[2][4]
        width = pdm[2][3] * pdl[1][5]
        base_n2 = pdm[2][0] * pd[2][0] * exp(evaluate(pd[2]))
        turbopause = (
            pdm[2][2]
            * pdl[1][24]
            * (
                1
                + pdl[0][24]
                * sin(0.0174533 * latitude_degrees)
                * cos(0.0172142 * (day_of_year - pt[13]))
            )
        )
        mixed_n2_base = profile.density(turbopause, base_n2, 28 - mean_mass, -1)
        mixed_n2 = profile.density(height, mixed_n2_base, mean_mass)
        densities = [0.0] * 8
        densities[2] = profile.density(height, base_n2, 28)
        if height <= 160:
            densities[2] = mixed_density(densities[2], mixed_n2, width, mean_mass, 28)
        # species output, parameter row, harmonic row, mass, thermal exponent,
        # upper mixing altitude, ground-mixing correction pdl-row/index.
        species = (
            (0, 0, 0, 4, -0.38, 200, 1, 0),
            (1, 1, 1, 16, 0, 300, 1, 2),
            (3, 3, 4, 32, 0, 250, 1, 6),
            (4, 4, 5, 40, 0.17, 240, 1, 8),
            (5, 5, 6, 1, -0.38, 320, 1, 10),
            (6, 6, 7, 14, 0, 450, 0, 0),
        )
        for output, parameter, variation, mass, alpha, limit, row, index in species:
            pars = pdm[parameter]
            initial = pars[0] * pd[variation][0] * exp(evaluate(pd[variation]))
            density = profile.density(height, initial, mass, alpha)
            if height <= limit and not (output == 0 and height == limit):
                mixing_base = profile.density(pars[2], initial, mass - mean_mass, alpha - 1)
                mixing = profile.density(height, mixing_base, mean_mass)
                density = mixed_density(density, mixing, width, mean_mass, mass)
                if output == 1:
                    ratio = pars[1] * pdl[1][16] * (1 + pdl[0][23] * (mean_solar_flux - 150))
                    density *= chemistry_factor(
                        height, ratio, pars[4] * pdl[1][2], pars[5] * pdl[1][3], pars[5] * pdl[1][4]
                    )
                else:
                    factor = (
                        abs(pdl[1][17]) if output == 5 else (abs(pdl[0][2]) if output == 6 else 1)
                    )
                    ratio = log(mixed_n2_base * pars[1] * factor / mixing_base)
                    density *= chemistry_factor(
                        height, ratio, pars[4] * pdl[row][index], pars[5] * pdl[row][index + 1]
                    )
                if output in (1, 5, 6):
                    correction_row, correction_index = {1: (1, 12), 5: (1, 18), 6: (0, 3)}[output]
                    factors = pdl[correction_row]
                    density *= chemistry_factor(
                        height,
                        pars[3] * factors[correction_index + 2],
                        pars[6] * factors[correction_index],
                        pars[7] * factors[correction_index + 1],
                    )
            if output == 3:
                ratio = pars[3] * pdl[1][23] * (1 + pdl[0][23] * (mean_solar_flux - 150))
                density *= chemistry_factor(
                    height, ratio, pars[6] * pdl[1][21], pars[7] * pdl[1][22], pars[7] * pdl[0][22]
                )
            densities[output] = density
        hot_temperature = pdm[7][9] * pdl[0][6]
        hot_profile = BatesProfile(
            hot_temperature,
            hot_temperature,
            ptm[5],
            slope,
            (top, 110, 100, 90, 72.5),
            tuple(temperatures),
            bottom_gradient,
            radius,
            gravity,
        )
        initial_hot = pdm[7][0] * pd[8][0] * exp(evaluate(pd[8]))
        hot_density = hot_profile.density(height, initial_hot, 16)
        center, hot_scale = pdm[7][4], pdm[7][5]
        scale_height = 831.4 * hot_temperature / (gravity / (1 + center / radius) ** 2 * 16)
        densities[7] = hot_density * exp(
            -hot_scale / scale_height * (exp(-(height - center) / hot_scale) - 1)
        )
        temperature = profile.temperature(height)
        if elevation_m < 72500:
            actual_height = elevation_m / 1000
            meso_temperatures = (
                profile.temperature(72.5),
                *(pma[i][0] * pavg[i] / (1 - evaluate(pma[i], lower=True)) for i in range(3)),
            )
            meso_gradient = pavg[8] * pma[9][0] * (1 + evaluate(pma[9], lower=True))
            meso_gradient *= (meso_temperatures[-1] / (pma[2][0] * pavg[2])) ** 2
            meso = InverseTemperatureProfile(
                (72.5, 55, 45, 32.5),
                meso_temperatures,
                (bottom_gradient, meso_gradient),
                radius,
                gravity,
            )
            target_height = max(actual_height, 32.5)
            fully_mixed = meso.density(target_height, mixed_n2, mean_mass)
            temperature = meso.temperature(target_height)
            if actual_height < 32.5:
                low_temperatures = (
                    meso_temperatures[-1],
                    *(
                        pma[i][0] * pavg[i] / (1 - evaluate(pma[i], lower=True))
                        for i in range(3, 7)
                    ),
                )
                low_gradient = pma[7][0] * pavg[7] * (1 + evaluate(pma[7], lower=True))
                low_gradient *= (low_temperatures[-1] / (pma[6][0] * pavg[6])) ** 2
                low = InverseTemperatureProfile(
                    (32.5, 20, 15, 10, 0),
                    low_temperatures,
                    (meso_gradient, low_gradient),
                    radius,
                    gravity,
                )
                fully_mixed = low.density(actual_height, fully_mixed, mean_mass)
                temperature = low.temperature(actual_height)
            fraction = max(0.0, (actual_height - 62.5) / 10)
            n2 = fully_mixed * (1 + (densities[2] / mixed_n2 - 1) * fraction)
            for output, parameter in ((0, 0), (3, 3), (4, 4)):
                ratio = pdm[parameter][1]
                densities[output] = (
                    n2 * ratio * (1 + (densities[output] / (densities[2] * ratio) - 1) * fraction)
                )
            densities[2] = n2
            for output in (1, 5, 6, 7):
                densities[output] = 0.0
        numbers = tuple(value * 1e6 for value in densities)
        total = sum(numbers[:7])
        model_density = 1.66e-27 * sum(
            n * m for n, m in zip(numbers, (4, 16, 28, 32, 40, 1, 14), strict=False)
        )
        composition_density = sum(n * m for n, m in zip(numbers, _MASSES, strict=False)) / (
            1000 * 6.02214076e23
        )
        pressure = total * 1.380649e-23 * temperature
        if not all(isfinite(v) and v >= 0 for v in (*numbers, pressure, temperature)):
            raise ValueError("MSIS empirical coefficients produced a nonphysical state")
        return MSISAtmosphereState(
            elevation_m,
            temperature,
            exosphere,
            numbers,
            model_density,
            composition_density,
            pressure,
            tuple(n / total for n in numbers[:7]),
        )
