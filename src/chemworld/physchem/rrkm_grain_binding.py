"""Generate RRKM channels into the original finite-grain gas world.

One harmonic well, competing irreversible same-formula products, no additional
population owner. Canonical diagnostics are finite-grid averages, not k(T,P)
reduction or a substitute for instantaneous bath-dependent original kinetics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from math import exp, expm1, fsum, log
from typing import Any

from chemworld.physchem.energy_grain_binding import compile_energy_grain_network
from chemworld.physchem.energy_grain_specs import (
    EnergyGrainBathGas,
    EnergyGrainMasterEquationSpec,
    EnergyGrainState,
    EnergyGrainWellSpec,
    EnergyTransferBath,
    InternalModeBaseline,
    MicrocanonicalGrainChannel,
    R,
    finite,
    identity,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.rrkm_density_states import (
    AVOGADRO_PER_MOL,
    PLANCK_J_S,
    RRKMEnergyGrid,
    RRKMHarmonicState,
    finite_harmonic_partition,
    harmonic_log_partition,
    harmonic_state_counts,
)


@dataclass(frozen=True)
class RRKMProductChannel:
    channel_id: str
    transition_state: RRKMHarmonicState
    imaginary_frequency_cm_inverse: float
    path_degeneracy: int
    product: EnergyGrainState

    def __post_init__(self) -> None:
        identity(self.channel_id)
        finite(self.imaginary_frequency_cm_inverse, "TS imaginary frequency")
        if self.imaginary_frequency_cm_inverse >= 0:
            raise ValueError(
                "declare exactly one negative TS frequency separately from stable modes"
            )
        if (
            isinstance(self.path_degeneracy, bool)
            or not isinstance(self.path_degeneracy, int)
            or self.path_degeneracy < 1
        ):
            raise ValueError("RRKM path degeneracy must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "transition_state": self.transition_state.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RRKMProductChannel:
        return cls(
            **{
                **value,
                "transition_state": RRKMHarmonicState.from_dict(value["transition_state"]),
                "product": EnergyGrainState(**value["product"]),
            }
        )


@dataclass(frozen=True)
class RRKMGrainSpec:
    network_id: str
    molecular_formula: str
    baseline: InternalModeBaseline
    well: RRKMHarmonicState
    grid: RRKMEnergyGrid
    baths: tuple[EnergyTransferBath, ...]
    bath_gases: tuple[EnergyGrainBathGas, ...]
    channels: tuple[RRKMProductChannel, ...]

    def __post_init__(self) -> None:
        identity(self.network_id)
        for field in ("baths", "bath_gases", "channels"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if not self.well.vibrational_quanta_J_mol or set(
            self.baseline.excluded_internal_modes
        ) != set(self.well.mode_ids):
            raise ValueError("baseline must explicitly exclude exactly the well's harmonic modes")
        if not self.channels or len({c.channel_id for c in self.channels}) != len(self.channels):
            raise ValueError("declare unique nonempty RRKM product channels")
        if len({c.product.species_id for c in self.channels}) != len(self.channels):
            raise ValueError("this slice requires one distinct absorbing product per channel")
        for channel in self.channels:
            if (
                len(channel.transition_state.vibrational_quanta_J_mol)
                != len(self.well.vibrational_quanta_J_mol) - 1
            ):
                raise ValueError(
                    "TS stable modes must exclude exactly one well reaction coordinate"
                )
            self.grid.quantize(
                channel.transition_state.ground_energy_J_mol - self.well.ground_energy_J_mol,
                label=channel.channel_id + ":barrier",
            )

    def to_dict(self) -> dict[str, Any]:
        baseline = asdict(self.baseline)
        baseline["excluded_internal_modes"] = list(self.baseline.excluded_internal_modes)
        return {
            "model": "harmonic-rrkm-grain-v1",
            "network_id": self.network_id,
            "molecular_formula": self.molecular_formula,
            "baseline": baseline,
            "well": self.well.to_dict(),
            "grid": self.grid.to_dict(),
            "baths": [asdict(b) for b in self.baths],
            "bath_gases": [asdict(b) for b in self.bath_gases],
            "channels": [c.to_dict() for c in self.channels],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RRKMGrainSpec:
        if value.get("model") != "harmonic-rrkm-grain-v1":
            raise ValueError("unknown RRKM declaration")
        return cls(
            str(value["network_id"]),
            str(value["molecular_formula"]),
            InternalModeBaseline(**value["baseline"]),
            RRKMHarmonicState.from_dict(value["well"]),
            RRKMEnergyGrid.from_dict(value["grid"]),
            tuple(EnergyTransferBath(**b) for b in value["baths"]),
            tuple(EnergyGrainBathGas(**b) for b in value["bath_gases"]),
            tuple(RRKMProductChannel.from_dict(c) for c in value["channels"]),
        )


def rrkm_microcanonical_report(spec: RRKMGrainSpec) -> dict[str, Any]:
    well = harmonic_state_counts(spec.well, spec.grid)
    channels = {}
    for channel in spec.channels:
        ts = harmonic_state_counts(channel.transition_state, spec.grid)
        barrier = spec.grid.quantize(
            ts["ground_energy_J_mol"] - well["ground_energy_J_mol"],
            label=channel.channel_id + ":barrier",
        )
        rates = []
        for j, count in enumerate(well["counts"]):
            n = ts["sum_of_states"][j - barrier["index"]] if j >= barrier["index"] else 0
            rate = (
                channel.path_degeneracy
                * spec.grid.grain_width_J_mol
                / (PLANCK_J_S * AVOGADRO_PER_MOL)
                * (n / count)
                if count
                else None
            )
            if rate is not None:
                finite(rate, "RRKM rate")
            rates.append(rate)
        channels[channel.channel_id] = {
            "transition_state": ts,
            "barrier": barrier,
            "rate_s": tuple(rates),
            "product_id": channel.product.species_id,
            "path_degeneracy": channel.path_degeneracy,
            "excluded_imaginary_frequency_cm_inverse": channel.imaginary_frequency_cm_inverse,
        }
    return {
        "well": well,
        "channels": channels,
        "grid": spec.grid.to_dict(),
        "rate_convention": "d * DeltaE * N_TS / (h * N_A * g_well); zero for closed channels",
    }


def compile_rrkm_grains(spec: RRKMGrainSpec) -> EnergyGrainMasterEquationSpec:
    report = rrkm_microcanonical_report(spec)
    grains = tuple(
        EnergyGrainState(
            f"{spec.well.state_id}:grain:{j}",
            spec.well.ground_energy_J_mol + j * spec.grid.grain_width_J_mol,
            float(count),
        )
        for j, count in enumerate(report["well"]["counts"])
        if count
    )
    channels = []
    for channel in spec.channels:
        for j, rate in enumerate(report["channels"][channel.channel_id]["rate_s"]):
            if rate is not None and rate > 0:
                channels.append(
                    MicrocanonicalGrainChannel(
                        f"{channel.channel_id}:grain:{j}",
                        f"{spec.well.state_id}:grain:{j}",
                        channel.product.species_id,
                        rate,
                    )
                )
    return EnergyGrainMasterEquationSpec(
        spec.network_id,
        spec.molecular_formula,
        spec.baseline,
        (EnergyGrainWellSpec(spec.well.state_id, grains, spec.baths),),
        spec.bath_gases,
        tuple(channels),
        tuple(c.product for c in spec.channels),
        source_note="sourced harmonic RRKM; explicit finite grid; no rotors/tunneling/reduction",
    )


def rrkm_grain_network(spec: RRKMGrainSpec) -> ReactionNetworkSpec:
    network = compile_energy_grain_network(compile_rrkm_grains(spec))
    return replace(network, metadata={**network.metadata, "rrkm_declaration": spec.to_dict()})




def rrkm_canonical_report(spec: RRKMGrainSpec, temperature_K: float) -> dict[str, Any]:
    """Finite-grid high-collision-limit average, retaining cutoff and grid bias."""
    if not spec.baseline.min_temperature_K <= temperature_K <= spec.baseline.max_temperature_K:
        raise ValueError("temperature outside the RRKM baseline domain")
    micro = rrkm_microcanonical_report(spec)
    partition = finite_harmonic_partition(spec.well, spec.grid, temperature_K)
    width = spec.grid.grain_width_J_mol
    factor = width / (R * temperature_K) / (-expm1(-width / (R * temperature_K)))
    # With zero-DOS bins, cancelling rho in a canonical sum is not valid there.
    # A one-grid-quantum mode guarantees positive counts at every energy index.
    dense = any(q["index"] == 1 for q in micro["well"]["quantization"])
    rows = {}
    for channel in spec.channels:
        mean = fsum(
            f * rate
            for f, rate in zip(
                partition["fractions"], micro["channels"][channel.channel_id]["rate_s"], strict=True
            )
            if rate is not None
        )
        log_ratio = harmonic_log_partition(
            channel.transition_state, temperature_K, spec.grid
        ) - harmonic_log_partition(spec.well, temperature_K, spec.grid)
        barrier = micro["channels"][channel.channel_id]["barrier"]["quantized_J_mol"]
        log_tst = (
            log(channel.path_degeneracy * R * temperature_K / (PLANCK_J_S * AVOGADRO_PER_MOL))
            + log_ratio
            - barrier / (R * temperature_K)
        )
        tst = exp(log_tst)
        rows[channel.channel_id] = {
            "finite_grid_average_s": mean,
            "infinite_grid_average_s": tst * factor if dense else None,
            "continuous_tst_quantized_modes_s": tst,
            "finite_to_infinite_ratio": mean / (tst * factor) if dense and tst > 0 else None,
        }
    return {
        "temperature_K": temperature_K,
        "well_partition": partition,
        "channels": rows,
        "dense_grid_prefactor_ratio_to_continuous_tst": factor,
        "infinite_grid_formula_applicable": dense,
        "empty_bin_scope": "sparse harmonic states need smoothing/rotors for a continuum limit",
        "interpretation": "finite-grid canonical average, not pressure-dependent reduced kinetics",
    }
