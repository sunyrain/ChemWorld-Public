"""Composition of existing SAFT/potential kernels and explicit site association.

This module orchestrates contributions; site equilibria, derivatives, states
and roots remain the existing shared implementations. The teqp GenericSAFT
factory has four nonpolar families and optional canonical/Dufal association.
Its top-level polar field is not an implemented contribution in version 0.23.2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from chemworld.physchem.caloric_math import (
    compensated_sum,
    exponential_minus_one,
    logarithm,
    scalar_value,
)
from chemworld.physchem.dufal_association import DufalAssociation
from chemworld.physchem.molecular_potential_extensions import TwoCenterLJModel
from chemworld.physchem.molecular_potentials import SoftSAFTModel
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL, PCSAFTModel
from chemworld.physchem.saft_association import (
    SAFTAssociation,
    association_helmholtz,
    association_site_fractions,
)
from chemworld.physchem.saftvrmie import SAFTVRMieModel


@dataclass(frozen=True)
class CanonicalSAFTAssociation:
    """Independent CR1 site data and an explicit CS/KG contact reference."""

    component_ids: tuple[str, ...]
    covolumes_m3_mol: tuple[float, ...]
    sites: SAFTAssociation
    source: str
    radial_distribution: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_ids", tuple(self.component_ids))
        object.__setattr__(self, "covolumes_m3_mol", tuple(self.covolumes_m3_mol))
        if (
            not self.component_ids
            or len(set(self.component_ids)) != len(self.component_ids)
            or not all(self.component_ids)
            or not self.source
            or len(self.covolumes_m3_mol) != len(self.component_ids)
            or any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in self.covolumes_m3_mol)
            or any(s.component_id not in self.component_ids for s in self.sites.sites)
            or self.radial_distribution not in ("Carnahan-Starling", "Kontogeorgis")
        ):
            raise ValueError(
                "canonical association needs ordered components, b>0 and explicit CS/KG"
            )

    def contacts(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> list[list[Any]]:
        _validate_association_state(self.component_ids, temperature, density, fractions)
        eta = (
            density
            * compensated_sum(x * b for x, b in zip(fractions, self.covolumes_m3_mol, strict=True))
            / 4
        )
        limit = 1.0 if self.radial_distribution == "Carnahan-Starling" else 1 / 1.9
        if not 0 <= scalar_value(eta) < limit:
            raise ValueError("canonical contact outside positive packing domain")
        contact = (
            (1 - eta / 2) / (1 - eta) ** 3
            if self.radial_distribution == "Carnahan-Starling"
            else 1 / (1 - 1.9 * eta)
        )
        return [[contact] * len(fractions) for _ in fractions]

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        return association_helmholtz(
            self.sites,
            temperature,
            density * AVOGADRO_PER_MOL,
            self.component_ids,
            fractions,
            self.contacts(temperature, density, fractions),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CanonicalSAFTAssociation:
        data = dict(payload)
        data["sites"] = SAFTAssociation.from_dict(data["sites"])
        return cls(**data)


def _validate_association_state(
    ids: Sequence[str], temperature: Any, density: Any, fractions: Sequence[Any]
) -> None:
    values = tuple(scalar_value(x) for x in fractions)
    if (
        not isfinite(scalar_value(temperature))
        or scalar_value(temperature) <= 0
        or not isfinite(scalar_value(density))
        or scalar_value(density) < 0
        or len(values) != len(ids)
        or any(not isfinite(x) or x < 0 for x in values)
        or abs(sum(values) - 1) > 1e-10
    ):
        raise ValueError(
            "association requires positive T, nonnegative rho and normalized fractions"
        )


@dataclass(frozen=True)
class MaskedSAFTAssociation:
    """Source-directed site connectivity on the existing site-equilibrium solver.

    Symmetric molecular bonding strengths remain in the underlying parameter
    card. An explicit directed connectivity matrix can reproduce the source's
    interaction_partners option, including asymmetric declarations. Such a
    declaration is an algebraic source convention, not reciprocal pair physics.
    """

    kernel: CanonicalSAFTAssociation | DufalAssociation
    interaction_mask: tuple[tuple[bool, ...], ...]

    def __post_init__(self) -> None:
        mask = tuple(tuple(row) for row in self.interaction_mask)
        size = len(self.kernel.sites.sites)
        if (
            len(mask) != size
            or any(len(row) != size for row in mask)
            or any(type(v) is not bool for row in mask for v in row)
        ):
            raise ValueError("site interaction mask must be an explicit square boolean matrix")
        object.__setattr__(self, "interaction_mask", mask)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return self.kernel.component_ids

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        _validate_association_state(self.component_ids, temperature, density, fractions)
        spec = self.kernel.sites
        if spec.geometric_pairs:
            raise ValueError("directed site masks require explicit pair strengths")
        owners = [self.component_ids.index(site.component_id) for site in spec.sites]
        contacts = self.kernel.contacts(temperature, density, fractions)
        populations = [
            fractions[owner] * site.count for owner, site in zip(owners, spec.sites, strict=True)
        ]
        coupling = [
            [
                density
                * AVOGADRO_PER_MOL
                * spec.bonding_volume_m3[i][j]
                * exponential_minus_one(spec.energy_K[i][j] / temperature)
                * contacts[left][right]
                if self.interaction_mask[i][j]
                else 0.0
                for j, right in enumerate(owners)
            ]
            for i, left in enumerate(owners)
        ]
        values = association_site_fractions(populations, coupling)
        return compensated_sum(
            n * (logarithm(value) - value / 2 + 0.5)
            for n, value in zip(populations, values, strict=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel": _encode_association(self.kernel),
            "interaction_mask": self.interaction_mask,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MaskedSAFTAssociation:
        kernel = _decode_association(payload["kernel"])
        if not isinstance(kernel, (CanonicalSAFTAssociation, DufalAssociation)):
            raise ValueError("nested site masks are not a source model")
        return cls(kernel, payload["interaction_mask"])


_NONPOLAR: dict[str, Any] = {
    "PCSAFT": PCSAFTModel,
    "SAFTVRMie": SAFTVRMieModel,
    "softSAFT": SoftSAFTModel,
    "2CLJ": TwoCenterLJModel,
}
_ASSOCIATION: dict[str, Any] = {
    "canonical": CanonicalSAFTAssociation,
    "Dufal": DufalAssociation,
    "masked": MaskedSAFTAssociation,
}


def _encode_association(kernel: Any) -> dict[str, Any] | None:
    if kernel is None:
        return None
    return {
        "kind": next(k for k, cls in _ASSOCIATION.items() if isinstance(kernel, cls)),
        "model": kernel.to_dict(),
    }


def _decode_association(payload: Mapping[str, Any] | None) -> Any:
    if payload is None:
        return None
    if payload["kind"] not in _ASSOCIATION:
        raise ValueError("unknown generic association encoding")
    return _ASSOCIATION[payload["kind"]].from_dict(payload["model"])


@dataclass(frozen=True)
class GenericSAFTModel:
    """One native nonpolar model plus independently declared association.

    Existing PC-SAFT/Mie polar declarations remain part of their native model;
    the source importer never invents an ignored top-level polar contribution.
    """

    nonpolar: PCSAFTModel | SAFTVRMieModel | SoftSAFTModel | TwoCenterLJModel
    source: str
    association: CanonicalSAFTAssociation | DufalAssociation | MaskedSAFTAssociation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.nonpolar, tuple(_NONPOLAR.values())) or not self.source:
            raise ValueError("GenericSAFT needs a supported native base and provenance")
        if self.association is not None:
            if not isinstance(self.association, tuple(_ASSOCIATION.values())):
                raise ValueError("unknown GenericSAFT association kernel")
            if self.association.component_ids != self.component_ids:
                raise ValueError("GenericSAFT association must match component ordering")
            if getattr(self.nonpolar, "association", None) is not None:
                raise ValueError(
                    "association cannot be present in both base and outer contribution"
                )

    @property
    def component_ids(self) -> tuple[str, ...]:
        return self.nonpolar.component_ids

    @property
    def gas_constant_J_mol_K(self) -> float:  # noqa: N802 - public physical unit
        return self.nonpolar.gas_constant_J_mol_K

    @property
    def temperature_range_K(self) -> tuple[float, float]:  # noqa: N802 - public physical unit
        if isinstance(self.nonpolar, TwoCenterLJModel):
            return self.nonpolar.temperature_bounds_K
        return self.nonpolar.temperature_range_K

    def contributions(
        self, temperature: Any, density: Any, fractions: Sequence[Any]
    ) -> dict[str, Any]:
        result = {"nonpolar": self.nonpolar.alphar(temperature, density, fractions)}
        if self.association is not None:
            result["association"] = self.association.alphar(temperature, density, fractions)
        return result

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        return compensated_sum(self.contributions(temperature, density, fractions).values())

    def to_dict(self) -> dict[str, Any]:
        kind = next(k for k, cls in _NONPOLAR.items() if isinstance(self.nonpolar, cls))
        return {
            "model": "chemworld-generic-saft-v1",
            "source": self.source,
            "nonpolar": {"kind": kind, "model": self.nonpolar.to_dict()},
            "association": _encode_association(self.association),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GenericSAFTModel:
        if payload.get("model") != "chemworld-generic-saft-v1":
            raise ValueError("unknown GenericSAFT encoding")
        base = payload["nonpolar"]
        if base["kind"] not in _NONPOLAR:
            raise ValueError("unknown GenericSAFT base encoding")
        return cls(
            _NONPOLAR[base["kind"]].from_dict(base["model"]),
            payload["source"],
            _decode_association(payload.get("association")),
        )
