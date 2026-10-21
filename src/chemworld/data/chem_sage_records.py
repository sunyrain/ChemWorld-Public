"""Immutable ChemSage declarations; none of these records owns physical state."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChemSageHeader:
    title: str
    solution_sizes: tuple[int, ...]
    stoichiometric_phases: int
    components: tuple[str, ...]
    masses_g_mol: tuple[float, ...]
    gibbs_indices: tuple[int, ...]
    excess_indices: tuple[int, ...]


@dataclass(frozen=True)
class ChemSageInterval:
    upper_K: float
    coefficients: tuple[float, ...]
    extra_powers: tuple[tuple[float, float], ...] = ()
    pressure_volume_terms: tuple[float, ...] = ()
    transition_enthalpy_J_mol: float = 0.0


@dataclass(frozen=True)
class ChemSageEndmember:
    name: str
    status: str
    option: int
    stoichiometry: tuple[float, ...]
    intervals: tuple[ChemSageInterval, ...]
    enthalpy_298_J_mol: float | None = None
    entropy_298_J_mol_K: float | None = None
    curie_temperature_K: float | None = None
    magnetic_moment: float | None = None
    stoichiometric_factor: float | None = None
    chemical_group: int | None = None
    pair_stoichiometry: tuple[float, ...] = ()
    zeta: float | None = None
    real_gas_parameters: tuple[float, ...] = ()
    aqueous_charge: float | None = None


@dataclass(frozen=True)
class ChemSageInteraction:
    kind: str
    indices: tuple[int, ...]
    coefficients: tuple[float, ...]
    order: int = 0
    exponents: tuple[int, ...] = ()
    mixing_type: int = 0
    mixing_code: str = ""
    metadata: tuple[float, ...] = ()
    additional_constituent: int = 0
    additional_anion_constituent: int = 0


@dataclass(frozen=True)
class ChemSageMQMQA:
    cations: tuple[str, ...]
    anions: tuple[str, ...]
    cation_charges: tuple[float, ...]
    anion_charge_magnitudes: tuple[float, ...]
    cation_groups: tuple[int, ...]
    anion_groups: tuple[int, ...]
    pair_indices: tuple[tuple[int, int], ...]
    coordinations: tuple[tuple[tuple[int, ...], tuple[float, ...]], ...]


@dataclass(frozen=True)
class ChemSagePhase:
    name: str
    phase_type: str
    status: str
    gas_slot: bool
    declared_size: int
    endmembers: tuple[ChemSageEndmember, ...]
    site_ratios: tuple[float, ...] = ()
    constituents: tuple[tuple[str, ...], ...] = ()
    endmember_indices: tuple[tuple[int, ...], ...] = ()
    interactions: tuple[ChemSageInteraction, ...] = ()
    overrides: tuple[tuple[str, ...], ...] = ()
    magnetic_afm_factor: float | None = None
    magnetic_structure_factor: float | None = None
    mqmqa: ChemSageMQMQA | None = None


@dataclass(frozen=True)
class ChemSageInput:
    header: ChemSageHeader
    phases: tuple[ChemSagePhase, ...]
    trailer: str
