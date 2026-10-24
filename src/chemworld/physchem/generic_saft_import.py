"""Explicit teqp 0.23.2 GenericSAFT JSON translation, without source-runtime imports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any

from chemworld.physchem.dufal_association import DufalAssociation
from chemworld.physchem.eos import R_J_PER_MOL_K
from chemworld.physchem.generic_saft import (
    CanonicalSAFTAssociation,
    GenericSAFTModel,
    MaskedSAFTAssociation,
)
from chemworld.physchem.molecular_potential_extensions import TwoCenterLJModel
from chemworld.physchem.molecular_potentials import SoftSAFTModel
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL, PCSAFTModel, SAFTSegment
from chemworld.physchem.saft_association import AssociationSite, SAFTAssociation
from chemworld.physchem.saft_polar import SAFTMultipole, SAFTPolar
from chemworld.physchem.saftvrmie import MieSegment, SAFTVRMieModel


@dataclass(frozen=True)
class GenericSAFTImportResult:
    model: GenericSAFTModel
    notes: tuple[str, ...]


def _matrix(raw: Any, size: int) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(row) for row in raw) if raw else ((0.0,) * size,) * size


def _coefficients(
    spec: Mapping[str, Any], library: Mapping[str, Mapping[str, Any]] | None
) -> list[Mapping[str, Any]]:
    if "names" in spec:
        if library is None:
            raise ValueError("named source fluids require an explicit parameter_library")
        try:
            return [library[name] for name in spec["names"]]
        except KeyError as exc:
            raise ValueError(f"missing explicitly supplied source fluid {exc.args[0]}") from exc
    if "coeffs" not in spec:
        raise ValueError("source PC-SAFT/Mie input needs names or coeffs")
    return list(spec["coeffs"])


def _base(
    payload: Mapping[str, Any],
    ids: tuple[str, ...],
    temperature_range: tuple[float, float],
    source: str,
    library: Mapping[str, Mapping[str, Any]] | None,
    scales: tuple[float, float] | None,
    notes: list[str],
) -> Any:
    kind, spec = payload["kind"], payload["model"]
    if kind in ("PCSAFT", "PC-SAFT", "SAFTVRMie", "SAFT-VR-Mie"):
        rows = _coefficients(spec, library)
        if len(rows) != len(ids):
            raise ValueError("source coefficient count does not match explicit component_ids")
        interaction = _matrix(spec.get("kmat"), len(ids))
        if kind in ("PCSAFT", "PC-SAFT"):
            components = tuple(
                SAFTSegment(
                    ids[i],
                    row["m"],
                    row["sigma_Angstrom"] * 1e-10,
                    row["epsilon_over_k"],
                    row.get("BibTeXKey", source),
                )
                for i, row in enumerate(rows)
            )
            sets = {
                None: "Gross-Sadowski-2001",
                "Liang-IECR-2012": "Liang-2012",
                "Liang-IECR-2014": "Liang-2014",
            }
            if spec.get("ab") not in sets:
                raise ValueError("unknown public PC-SAFT coefficient selection")
            moments = tuple(
                SAFTMultipole(
                    ids[i],
                    row.get("(mu^*)^2", 0.0) if "nmu" in row else 0.0,
                    row.get("(Q^*)^2", 0.0) if "nQ" in row else 0.0,
                    source,
                    row.get("nmu", 0.0),
                    row.get("nQ", 0.0),
                )
                for i, row in enumerate(rows)
            )
            polar = (
                SAFTPolar(moments, source, "teqp-0.23.2", ("DD", "QQ"))
                if any(
                    p.reduced_dipole_squared * p.dipole_count
                    or p.reduced_quadrupole_squared * p.quadrupole_count
                    for p in moments
                )
                else None
            )
            return PCSAFTModel(
                components,
                interaction,
                temperature_range,
                source,
                coefficient_set=sets[spec.get("ab")],
                polar=polar,
            )
        mie_components = tuple(
            MieSegment(
                ids[i],
                row["m"],
                row["sigma_m"] if "sigma_m" in row else row["sigma_Angstrom"] * 1e-10,
                row["epsilon_over_k"],
                row.get("BibTeXKey", source),
                row["lambda_r"],
                row["lambda_a"],
            )
            for i, row in enumerate(rows)
        )
        if (
            "SAFTVRMie_flags" in spec
            or "polar_model" in spec
            or any("(mu^*)^2" in row or "(Q^*)^2" in row for row in rows)
        ):
            notes.append(
                "GenericSAFT uses the Mie nonpolar factory: polar fields and energy flags "
                "are ignored by that source branch; Lafitte mixing is retained."
            )
        return SAFTVRMieModel(mie_components, interaction, temperature_range, source)
    if kind in ("softSAFT", "Johnson+Johnson"):
        components = tuple(
            SAFTSegment(cid, m, sigma, eps, source)
            for cid, m, sigma, eps in zip(
                ids, spec["m"], spec["sigma / m"], spec["epsilon/kB / K"], strict=True
            )
        )
        return SoftSAFTModel(components, temperature_range, source)
    if kind in ("2CLJ", "2CLJF"):
        if len(ids) != 1 or scales is None or len(scales) != 2:
            raise ValueError(
                "source two-center branch requires one component and explicit SI scales"
            )
        notes.append(
            "Source two-center alpha accepts reduced T/rho; explicit diameter and energy "
            "scales map the native SI state. Association retains its own SI parameters."
        )
        return TwoCenterLJModel(
            ids[0],
            scales[0],
            scales[1],
            temperature_range,
            source,
            author={"2CLJF_Mecke": "Mecke", "2CLJF_Lisal": "Lisal"}.get(
                spec["author"], spec["author"]
            ),
            elongation=spec["L^*"],
        )
    raise ValueError(f"unknown public GenericSAFT nonpolar kind {kind}")


def _association(
    payload: Mapping[str, Any],
    ids: tuple[str, ...],
    source: str,
    canonical_radial_distribution: str | None,
    notes: list[str],
) -> MaskedSAFTAssociation | None:
    if payload["kind"] not in ("canonical", "Dufal"):
        raise ValueError("source GenericSAFT association kind must be canonical or Dufal")
    spec, n = payload["model"], len(ids)
    molecules = spec["molecule_sites"]
    if len(molecules) != n:
        raise ValueError("association site list does not match component ordering")
    if any(not isinstance(label, str) or not label for sites in molecules for label in sites):
        raise ValueError("association site labels must be nonempty strings")
    sites = tuple(
        AssociationSite(cid, label, count)
        for cid, labels in zip(ids, molecules, strict=True)
        for label, count in sorted(Counter(labels).items())
    )
    if not sites:
        notes.append("Empty association site lists contribute exactly zero.")
        return None
    opts = spec.get("options", {})
    types = {site.site_id for site in sites}
    partners = opts.get("interaction_partners")
    if partners is not None and partners:
        if set(partners) != types or any(set(values) - types for values in partners.values()):
            raise ValueError("interaction_partners must define every declared site type")
    else:
        # Absent option defaults to unlike labels; an explicitly empty source
        # map reaches make_D's all-partners branch and therefore includes self.
        partners = {
            label: list(types if "interaction_partners" in opts else types - {label})
            for label in types
        }
    self_mask = tuple(opts.get("self_association_mask", ())) or (True,) * n
    if len(self_mask) != n or any(type(v) is not bool for v in self_mask):
        raise ValueError("self_association_mask must have one boolean per component")
    owners = [ids.index(site.component_id) for site in sites]
    mask = tuple(
        tuple(
            right.site_id in partners[left.site_id]
            and (owners[i] != owners[j] or self_mask[owners[i]])
            for j, right in enumerate(sites)
        )
        for i, left in enumerate(sites)
    )
    if any(
        key in opts for key in ("alpha", "rtol", "atol", "max_iters", "allow_explicit_fractions")
    ):
        notes.append(
            "Source iteration controls are provenance only; native association uses the "
            "common root solver and analytic implicit derivatives."
        )
    if "site_order" in opts:
        notes.append(
            "Source 0.23.2 JSON does not parse site_order; native site order is canonical."
        )

    def vector(name: str) -> tuple[float, ...]:
        values = tuple(spec[name])
        if len(values) != n:
            raise ValueError(f"association vector {name} does not match components")
        if any(
            isinstance(v, bool) or not isfinite(v) or v < 0 or (name == "lambda_r" and v <= 3)
            for v in values
        ):
            raise ValueError(f"association vector {name} has invalid physical parameters")
        return values

    rule = spec.get("Delta_rule")
    if rule == "CR1":
        if canonical_radial_distribution is None:
            raise ValueError(
                "canonical JSON needs an explicit native radial distribution: source "
                "0.23.2 does not initialize it from JSON"
            )
        b, beta, eps = vector("b / m^3/mol"), vector("beta"), vector("epsilon / J/mol")
        energy = tuple(
            tuple((eps[i] + eps[j]) / (2 * R_J_PER_MOL_K) for j in owners) for i in owners
        )
        volume = tuple(
            tuple((b[i] + b[j]) / 2 * sqrt(beta[i] * beta[j]) / AVOGADRO_PER_MOL for j in owners)
            for i in owners
        )
        card = SAFTAssociation(sites, energy, volume, source)
        kernel: CanonicalSAFTAssociation | DufalAssociation = CanonicalSAFTAssociation(
            ids, b, card, source, canonical_radial_distribution
        )
        notes.append(
            "Canonical radial distribution is explicit native data; source JSON leaves "
            "its radial_dist enum uninitialized and does not reliably select CS/KG."
        )
    elif rule == "Dufal":
        sigma, lr = vector("sigma / m"), vector("lambda_r")
        eps = (
            tuple(v / R_J_PER_MOL_K for v in vector("epsilon / J/mol"))
            if "epsilon / J/mol" in spec
            else vector("epsilon/kB / K")
        )
        hb, khb = vector("epsilon_HB / J/mol"), vector("K_HB / m^3")
        kij = _matrix(spec["kmat"], n)
        energy = tuple(tuple(sqrt(hb[i] * hb[j]) / R_J_PER_MOL_K for j in owners) for i in owners)
        volume = tuple(
            tuple(((khb[i] ** (1 / 3) + khb[j] ** (1 / 3)) / 2) ** 3 for j in owners)
            for i in owners
        )
        pair_energy = tuple(
            tuple(
                (1 - kij[i][j])
                * sqrt(eps[i] * eps[j])
                * (sigma[i] * sigma[j]) ** 1.5
                / ((sigma[i] + sigma[j]) / 2) ** 3
                for j in range(n)
            )
            for i in range(n)
        )
        pair_lr = tuple(
            tuple(3 + sqrt((lr[i] - 3) * (lr[j] - 3)) for j in range(n)) for i in range(n)
        )
        kernel = DufalAssociation(
            ids,
            (1.0,) * n,
            sigma,
            pair_energy,
            pair_lr,
            SAFTAssociation(sites, energy, volume, source),
            source,
            density_basis="molecular",
        )
    else:
        raise ValueError("public association Delta_rule must explicitly be CR1 or Dufal")
    return MaskedSAFTAssociation(kernel, mask)


def generic_saft_from_teqp(
    payload: Mapping[str, Any],
    *,
    component_ids: Sequence[str],
    temperature_range_K: tuple[float, float],
    source: str,
    parameter_library: Mapping[str, Mapping[str, Any]] | None = None,
    canonical_radial_distribution: str | None = None,
    two_center_scales: tuple[float, float] | None = None,
) -> GenericSAFTImportResult:
    """Translate source combinations with explicit identities, SI domain and provenance.

    ``two_center_scales`` is (diameter_m, well_depth_K). Named-fluid lookup
    consumes a caller-supplied parameter library, never an upstream runtime or
    a hidden material database. Returned notes expose source-ignored fields.
    """
    if payload.get("kind") != "genericSAFT":
        raise ValueError("expected teqp genericSAFT factory envelope")
    ids, notes = tuple(component_ids), []
    if not ids or len(set(ids)) != len(ids) or not all(ids):
        raise ValueError("explicit unique component identities required")
    spec = payload["model"]
    if "polar" in spec:
        notes.append(
            "Top-level GenericSAFT polar is not implemented in teqp 0.23.2 and "
            "contributes no energy; configure native base polar data explicitly."
        )
    nonpolar = _base(
        spec["nonpolar"],
        ids,
        temperature_range_K,
        source,
        parameter_library,
        two_center_scales,
        notes,
    )
    association = (
        _association(spec["association"], ids, source, canonical_radial_distribution, notes)
        if "association" in spec
        else None
    )
    return GenericSAFTImportResult(GenericSAFTModel(nonpolar, source, association), tuple(notes))
