"""Bind an explicit ideal reference to molecular residuals for existing consumers.

The original Helmholtz phase, flash, stream and finite-inventory machinery
continues to own closure and execution. This module adds no phase solver,
inventory or energy zero. Mixtures are not projected onto a pure-fluid EOS.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from importlib import import_module
from typing import Any, cast

from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_eos import ResidualHelmholtzModel

# Only delivered native model codecs are selectable; payloads cannot name an
# arbitrary Python import. Existing model declarations keep their own schema.
_MODELS = {
    "MultifluidModel": "multifluid",
    "GenericSAFTModel": "generic_saft",
    "FundamentalMeasureBulkModel": "fundamental_measure",
    "PCSAFTModel": "pcsaft",
    "CPAModel": "cpa",
    "SAFTVRMieModel": "saftvrmie",
    "PurePotentialModel": "molecular_potentials",
    "SoftSAFTModel": "molecular_potentials",
    "TwoCenterLJModel": "molecular_potential_extensions",
    "PohlMieModel": "molecular_potential_extensions",
    "ChaparroMieModel": "molecular_potential_extensions",
    "SAFTVRQMieModel": "quantum_molecular",
    "QuantumCubicModel": "quantum_molecular_cubic",
    "LJSplineBHModel": "lj_spline",
    "LJSplineWCAModel": "lj_spline",
    "LJSplineUVModel": "lj_spline",
    "LeeKeslerPlockerModel": "corresponding_states",
    "HuberElyModel": "corresponding_states",
    "ExtendedCorrespondingStatesModel": "corresponding_states",
    "MBWRReference": "corresponding_states",
}


def molecular_model_to_dict(model: ResidualHelmholtzModel) -> dict[str, Any]:
    name = type(model).__name__
    if name not in _MODELS or type(model).__module__ != f"chemworld.physchem.{_MODELS[name]}":
        raise ValueError("unsupported native molecular model codec")
    return {"kind": name, "parameters": cast(Any, model).to_dict()}


def molecular_model_from_dict(payload: Mapping[str, Any]) -> ResidualHelmholtzModel:
    name = payload.get("kind")
    if not isinstance(name, str) or name not in _MODELS:
        raise ValueError("unknown native molecular model codec")
    module = import_module(f"chemworld.physchem.{_MODELS[name]}")
    return cast(ResidualHelmholtzModel, getattr(module, name).from_dict(payload["parameters"]))


@dataclass(frozen=True)
class MolecularHelmholtzFluid(HelmholtzFluid):
    """Pure molecular residual plus caller-declared ideal-gas caloric reference.

    Reducing coordinates are coordinates, not an inferred critical point.
    Phase callers must still supply explicit critical/ancillary declarations.
    Both this reference's domain and the residual model's domain apply.
    """

    residual_model: ResidualHelmholtzModel = field(kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.residual_model.component_ids != (self.fluid_id,):
            raise ValueError("pure molecular consumer requires exactly the declared fluid identity")
        if self.residual_model.gas_constant_J_mol_K != self.gas_constant_J_mol_K:
            raise ValueError("ideal and residual gas constants must agree")
        if self.residual_terms or self.pseudo_pure or self.ideal_prefactor != 1:
            raise ValueError("molecular consumer requires an unambiguous pure ideal reference")

    @classmethod
    def from_reference(cls, residual_model: ResidualHelmholtzModel,
                       ideal_reference: HelmholtzFluid) -> MolecularHelmholtzFluid:
        if isinstance(ideal_reference, MolecularHelmholtzFluid):
            raise ValueError("supply an ideal reference, not an already bound molecular fluid")
        return cls(**{f.name: getattr(ideal_reference, f.name) for f in fields(HelmholtzFluid)},
                   residual_model=residual_model)

    def alpha(self, kind: str, delta: Any, tau: Any) -> Any:
        if kind not in {"ideal", "residual", "total"}:
            raise ValueError("Helmholtz contribution must be ideal, residual or total")
        ideal = super().alpha("ideal", delta, tau) if kind != "residual" else 0.0
        if kind == "ideal":
            return ideal
        return ideal + self.residual_model.alphar(
            self.reducing_temperature_K/tau, delta*self.reducing_molar_density_mol_m3, (1.,))

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["model"] = "chemworld_molecular_helmholtz_v1"
        result["residual_model"] = molecular_model_to_dict(self.residual_model)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularHelmholtzFluid:
        values = dict(data)
        if values.pop("model", None) != "chemworld_molecular_helmholtz_v1":
            raise ValueError("unknown molecular Helmholtz consumer encoding")
        values["residual_model"] = molecular_model_from_dict(values["residual_model"])
        for key in ("ideal_terms", "residual_terms"):
            values[key] = tuple(HelmholtzTerm.from_dict(t) for t in values[key])
        return cls(**values)
