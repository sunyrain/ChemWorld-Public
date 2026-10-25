"""Explicit association and activity departures on the same multifluid potential.

The association is interspecies only: pure reference EOS already include their
self association. The activity departure follows Jaeger et al. (2018),
doi:10.1016/j.fluid.2018.04.015. These are new parameterized combinations,
not inherited qualification of an original GERG or other mixture fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log1p
from typing import Any

from chemworld.physchem.activity_kernels import reduced_gibbs
from chemworld.physchem.caloric_math import compensated_sum, logarithm, scalar_value
from chemworld.physchem.dufal_association import DufalAssociation
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.generic_saft import CanonicalSAFTAssociation, MaskedSAFTAssociation
from chemworld.physchem.molecular_eos import MolecularEOSState, molecular_eos_state
from chemworld.physchem.molecular_mixture import MolecularCaloricModel
from chemworld.physchem.multifluid import MultifluidModel
from chemworld.physchem.saft_association import SAFTAssociation


def _strict(data: Mapping[str, Any], fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or set(data) != fields:
        raise ValueError(f"{label} requires exactly {sorted(fields)}")
    return dict(data)


class _MultifluidCoupling:
    base: MultifluidModel
    source: str
    combination_policy: str

    def _validate_combination(self) -> None:
        if not isinstance(self.base, MultifluidModel):
            raise ValueError("coupling needs an explicit MultifluidModel base")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("coupling parameterization source is required")
        if self.combination_policy != "explicit_joint_parameterization":
            raise ValueError(
                "explicit_joint_parameterization must acknowledge the new combination; "
                "base binary departures and the added coupling need joint qualification"
            )

    @property
    def component_ids(self) -> tuple[str, ...]:
        return self.base.component_ids

    @property
    def gas_constant_J_mol_K(self) -> float:  # noqa: N802
        return self.base.gas_constant_J_mol_K

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        raise NotImplementedError

    def state(
        self,
        temperature_K: float,
        molar_density_mol_m3: float,
        mole_fractions: Sequence[float],
        *,
        require_stable: bool = True,
    ) -> MolecularEOSState:
        return molecular_eos_state(
            self,
            temperature_K,
            molar_density_mol_m3,
            mole_fractions,
            require_stable=require_stable,
        )

    def validate_pressure(self, pressure_Pa: float) -> None:
        """The final coupled pressure, including at singular critical states."""
        self.base.validate_pressure(pressure_Pa)

    def validate_state(self, state: MolecularEOSState) -> None:
        self.validate_pressure(state.pressure_Pa)

    def caloric_model(self) -> MolecularCaloricModel:
        return MolecularCaloricModel(self, self.base.caloric_model().ideal_references)


@dataclass(frozen=True)
class MultifluidAssociationModel(_MultifluidCoupling):
    """Multifluid + reciprocal cross association using the existing site solver.

    Every same-component site-pair mask entry must be false, including different
    sites on molecules of the same species. Pure T/rho derivatives are preserved; insertion
    chemical potentials of absent species can still contain association.
    """

    base: MultifluidModel
    association: MaskedSAFTAssociation
    source: str
    combination_policy: str

    def __post_init__(self) -> None:
        self._validate_combination()
        if not isinstance(self.association, MaskedSAFTAssociation):
            raise ValueError("multifluid association requires explicit site masks")
        if self.association.component_ids != self.component_ids:
            raise ValueError("association component ordering differs from multifluid")
        kernel, mask = self.association.kernel, self.association.interaction_mask
        if kernel.sites.geometric_pairs:
            raise ValueError("masked multifluid association requires explicit pair strengths")
        for i, left in enumerate(kernel.sites.sites):
            for j, right in enumerate(kernel.sites.sites):
                if mask[i][j] != mask[j][i]:
                    raise ValueError("multifluid association requires reciprocal site masks")
                if left.component_id == right.component_id and mask[i][j]:
                    raise ValueError("self association must be masked to preserve pure EOS")

    def association_departure(
        self,
        temperature: Any,
        density: Any,
        fractions: Sequence[Any],
    ) -> Any:
        return self.association.alphar(temperature, density, fractions)

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        return self.base.alphar(temperature, density, fractions) + self.association_departure(
            temperature,
            density,
            fractions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "chemworld_multifluid_association_v1",
            "base": self.base.to_dict(),
            "association": self.association.to_dict(),
            "source": self.source,
            "combination_policy": self.combination_policy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MultifluidAssociationModel:
        data = _strict(payload, set(cls.__dataclass_fields__) | {"model"}, "association coupling")
        if data.pop("model") != "chemworld_multifluid_association_v1":
            raise ValueError("unknown multifluid association encoding")
        assoc = _strict(data["association"], {"kernel", "interaction_mask"}, "masked association")
        kernel = _strict(assoc["kernel"], {"kind", "model"}, "association kernel")
        kinds: dict[str, Any] = {"canonical": CanonicalSAFTAssociation, "Dufal": DufalAssociation}
        if kernel["kind"] not in kinds:
            raise ValueError("unsupported multifluid association kernel")
        _strict(kernel["model"], set(kinds[kernel["kind"]].__dataclass_fields__), "kernel model")
        _strict(kernel["model"]["sites"], set(SAFTAssociation.__dataclass_fields__), "site model")
        data["association"] = MaskedSAFTAssociation.from_dict(assoc)
        data["base"] = MultifluidModel.from_dict(data["base"])
        return cls(**data)


@dataclass(frozen=True)
class MultifluidActivityModel(_MultifluidCoupling):
    """Jaeger activity departure with explicit residual/combinatorial convention.

    ``wilson_volume_residual`` subtracts sum(x_i ln(b_i/b_mix)) from the existing
    Wilson total excess potential, as in teqp 0.23.2. ``declared_residual``
    explicitly identifies the supplied equation as the residual excess term.
    No total-excess to residual conversion is guessed for other models.
    Reference densities are mathematical integration reference points, not
    stable fluid states. Physical source limits apply to the final state.
    """

    base: MultifluidModel
    activity: ActivityModelSpec
    covolumes_m3_mol: tuple[float, ...]
    u: float
    activity_convention: str
    source: str
    combination_policy: str

    def __post_init__(self) -> None:
        self._validate_combination()
        object.__setattr__(self, "covolumes_m3_mol", tuple(self.covolumes_m3_mol))
        if self.activity.component_ids != self.component_ids:
            raise ValueError("activity component ordering differs from multifluid")
        if len(self.covolumes_m3_mol) != len(self.component_ids) or any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (*self.covolumes_m3_mol, self.u)
        ):
            raise ValueError("activity coupling needs explicit positive b_i and u")
        if self.activity_convention not in {"wilson_volume_residual", "declared_residual"}:
            raise ValueError("explicit residual activity convention is required")
        if self.activity_convention == "wilson_volume_residual" and self.activity.model != "wilson":
            raise ValueError("Wilson volume residual requires a Wilson activity model")

    def residual_activity(self, temperature: Any, fractions: Sequence[Any]) -> Any:
        xs = self.base._fractions(fractions)
        value = reduced_gibbs(self.activity, xs, temperature)
        if self.activity_convention == "wilson_volume_residual":
            bm = compensated_sum(x * b for x, b in zip(xs, self.covolumes_m3_mol, strict=True))
            value -= compensated_sum(
                x * logarithm(b / bm) for x, b in zip(xs, self.covolumes_m3_mol, strict=True)
            )
        return value

    def activity_departure(
        self,
        temperature: Any,
        density: Any,
        fractions: Sequence[Any],
    ) -> Any:
        xs = self.base._fractions(fractions)
        if not isfinite(scalar_value(density)) or scalar_value(density) < 0:
            raise ValueError("activity departure requires nonnegative finite density")
        # Reuse the base T-domain guard, including when evaluating the vacuum limit.
        self.base._inputs(temperature, 1.0, xs)
        tr, rhor = self.base.reducing(xs)
        bm = compensated_sum(x * b for x, b in zip(xs, self.covolumes_m3_mol, strict=True))
        delta_ref, tau = 1 / (self.u * bm * rhor), tr / temperature
        pure_correction = compensated_sum(
            x
            * (
                fluid.alpha("residual", delta_ref, tau)
                - fluid.alpha(
                    "residual",
                    1 / (self.u * b * fluid.reducing_molar_density_mol_m3),
                    fluid.reducing_temperature_K / temperature,
                )
            )
            for x, b, fluid in zip(xs, self.covolumes_m3_mol, self.base.fluids, strict=True)
        )
        return (
            logarithm(1 + density * bm)
            / log1p(1 / self.u)
            * (self.residual_activity(temperature, xs) - pure_correction)
        )

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        return self.base.alphar(temperature, density, fractions) + self.activity_departure(
            temperature,
            density,
            fractions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "chemworld_multifluid_activity_v1",
            "base": self.base.to_dict(),
            "activity": self.activity.to_dict(),
            "covolumes_m3_mol": self.covolumes_m3_mol,
            "u": self.u,
            "activity_convention": self.activity_convention,
            "source": self.source,
            "combination_policy": self.combination_policy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MultifluidActivityModel:
        data = _strict(payload, set(cls.__dataclass_fields__) | {"model"}, "activity coupling")
        if data.pop("model") != "chemworld_multifluid_activity_v1":
            raise ValueError("unknown multifluid activity encoding")
        activity = dict(data["activity"])
        activity["component_ids"] = tuple(activity["component_ids"])
        data["activity"] = ActivityModelSpec(**activity)
        data["base"] = MultifluidModel.from_dict(data["base"])
        return cls(**data)
