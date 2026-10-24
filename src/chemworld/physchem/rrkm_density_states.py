"""Sourced harmonic state counting on an explicitly declared finite energy grid.

Counts are dimensionless, and density is count / (J/mol). Zero-point energy is
handled exactly once at authoring. Quantization is reported, never implicit.
This is a vibrational finite-grid model, without rotors, tunneling or QSD.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import exp, expm1, floor, fsum, isclose, log
from typing import Any

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.energy_grain_specs import R, finite, identity

PLANCK_J_S = 6.62607015e-34
AVOGADRO_PER_MOL = 6.02214076e23
LIGHT_M_S = 299792458.0


def _source(source: DataSourceProvenance) -> None:
    if not source.version.strip() or not source.license.strip():
        raise ValueError("RRKM source must identify its version and license")


def _integer(value: int, label: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")


@dataclass(frozen=True)
class RRKMEnergyGrid:
    grain_width_J_mol: float
    maximum_grain_index: int
    quantization: str = "exact"
    maximum_quantization_error_J_mol: float = 0.0

    def __post_init__(self) -> None:
        finite(self.grain_width_J_mol, "grain width", positive=True)
        _integer(self.maximum_grain_index, "maximum grain index")
        finite(self.maximum_quantization_error_J_mol, "quantization error bound")
        if self.quantization not in {"exact", "nearest"} or (
            self.maximum_quantization_error_J_mol < 0
            or (self.quantization == "exact" and self.maximum_quantization_error_J_mol != 0)
        ):
            raise ValueError("declare exact grid or nearest-half-up with an explicit error bound")

    def quantize(
        self, energy_J_mol: float, *, label: str, positive: bool = False
    ) -> dict[str, Any]:
        finite(energy_J_mol, label, positive=positive)
        if energy_J_mol < 0:
            raise ValueError("negative excitation or RRKM barrier is outside this model")
        index = floor(energy_J_mol / self.grain_width_J_mol + 0.5)
        quantized = index * self.grain_width_J_mol
        error = quantized - energy_J_mol
        arithmetic_match = isclose(quantized, energy_J_mol, rel_tol=1e-12, abs_tol=1e-10)
        if (self.quantization == "exact" and not arithmetic_match) or (
            self.quantization == "nearest"
            and abs(error) > self.maximum_quantization_error_J_mol
            and not arithmetic_match
        ):
            raise ValueError(f"{label} is outside its declared grid quantization contract")
        if positive and index == 0:
            raise ValueError("a positive harmonic mode cannot quantize to zero")
        return {
            "label": label,
            "original_J_mol": energy_J_mol,
            "index": index,
            "quantized_J_mol": quantized,
            "error_J_mol": error,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RRKMEnergyGrid:
        return cls(**dict(value))


@dataclass(frozen=True)
class RRKMHarmonicState:
    state_id: str
    vibrational_quanta_J_mol: tuple[float, ...]
    reference_energy_J_mol: float
    energy_basis: str
    electronic_degeneracy: int
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        identity(self.state_id)
        object.__setattr__(self, "vibrational_quanta_J_mol", tuple(self.vibrational_quanta_J_mol))
        finite(self.reference_energy_J_mol, "harmonic reference energy")
        for value in self.vibrational_quanta_J_mol:
            finite(value, "stable harmonic quantum", positive=True)
        _integer(self.electronic_degeneracy, "electronic degeneracy")
        if self.energy_basis not in {"electronic", "zero-point-corrected"}:
            raise ValueError("declare electronic or already zero-point-corrected energy")
        _source(self.source)

    @property
    def mode_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{self.state_id}:vibration:{i}" for i in range(len(self.vibrational_quanta_J_mol))
        )

    @property
    def ground_energy_J_mol(self) -> float:  # noqa: N802
        return self.reference_energy_J_mol + (
            0.5 * fsum(self.vibrational_quanta_J_mol) if self.energy_basis == "electronic" else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["vibrational_quanta_J_mol"] = list(self.vibrational_quanta_J_mol)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RRKMHarmonicState:
        return cls(**{**value, "source": DataSourceProvenance.from_dict(value["source"])})


def harmonic_state_counts(state: RRKMHarmonicState, grid: RRKMEnergyGrid) -> dict[str, Any]:
    """Beyer-Swinehart recurrence; integer coefficients before float conversion."""
    quantization = [
        grid.quantize(q, label=mode, positive=True)
        for mode, q in zip(state.mode_ids, state.vibrational_quanta_J_mol, strict=True)
    ]
    counts = [0] * (grid.maximum_grain_index + 1)
    counts[0] = state.electronic_degeneracy
    for quantum in quantization:
        step = quantum["index"]
        for index in range(step, len(counts)):
            counts[index] += counts[index - step]
    total = 0
    sums = []
    for count in counts:
        total += count
        sums.append(total)
    return {
        "state_id": state.state_id,
        "ground_energy_J_mol": state.ground_energy_J_mol,
        "counts": tuple(counts),
        "sum_of_states": tuple(sums),
        "density_states_per_J_mol": tuple(c / grid.grain_width_J_mol for c in counts),
        "quantization": tuple(quantization),
        "zero_point_added_J_mol": state.ground_energy_J_mol - state.reference_energy_J_mol,
    }


def harmonic_log_partition(
    state: RRKMHarmonicState, temperature_K: float, grid: RRKMEnergyGrid | None = None
) -> float:
    """Excitation-only infinite harmonic partition; absolute ground is separate."""
    finite(temperature_K, "temperature", positive=True)
    quanta = (
        state.vibrational_quanta_J_mol
        if grid is None
        else tuple(
            grid.quantize(q, label=mode, positive=True)["quantized_J_mol"]
            for mode, q in zip(state.mode_ids, state.vibrational_quanta_J_mol, strict=True)
        )
    )
    return log(state.electronic_degeneracy) - fsum(
        log(-expm1(-q / (R * temperature_K))) for q in quanta
    )


def finite_harmonic_partition(
    state: RRKMHarmonicState, grid: RRKMEnergyGrid, temperature_K: float
) -> dict[str, Any]:
    finite(temperature_K, "temperature", positive=True)
    counts = harmonic_state_counts(state, grid)["counts"]
    logs = [
        log(c) - j * grid.grain_width_J_mol / (R * temperature_K) for j, c in enumerate(counts) if c
    ]
    offset = max(logs)
    log_q = offset + log(fsum(exp(v - offset) for v in logs))
    return {
        "log_partition": log_q,
        "retained_partition_fraction": exp(
            log_q - harmonic_log_partition(state, temperature_K, grid)
        ),
        "fractions": tuple(
            exp(log(c) - j * grid.grain_width_J_mol / (R * temperature_K) - log_q) if c else 0.0
            for j, c in enumerate(counts)
        ),
    }
