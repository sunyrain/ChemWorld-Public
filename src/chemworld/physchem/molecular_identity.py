"""One bounded graph identity rule shared by static and growing mechanisms.

Resonance changes electron/bond placement, never nuclei, isotopes or spin.
Mappings relate complete labelled nuclei; an unfinished closure is not evidence
of inequivalence. This reuses the existing finite 1,3-shift enumerator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from chemworld.physchem.molecular_graph import (
    MolecularGraph,
    enumerate_allylic_resonance,
    molecular_isomorphisms,
)


def molecular_identity_mapping(
    left: MolecularGraph,
    right: MolecularGraph,
    *,
    allylic_resonance: bool = False,
    maximum_forms: int = 64,
) -> dict[str, str] | None:
    """Map a reachable left form to right, or establish complete inequivalence.

    The exact default uses the original VF2 iteration without normalization. A
    resonance match preserves each nucleus' label across all shifts before the
    final isomorphism. It does not match merely equal formulas or heavy skeletons.
    """
    if type(allylic_resonance) is not bool:
        raise ValueError("allylic resonance policy must be explicit boolean")
    variants: tuple[MolecularGraph, ...] = (left,)
    if allylic_resonance:
        resonance = enumerate_allylic_resonance(left, maximum_forms=maximum_forms)
        if not resonance.complete:
            raise ValueError("resonance_identity_search_truncated")
        variants = resonance.forms
    for variant in variants:
        mapping = next(molecular_isomorphisms(variant, right), None)
        if mapping is not None:
            return mapping
    return None


def _relabel_graph(graph: MolecularGraph, mapping: Mapping[str, str]) -> MolecularGraph:
    """Change only nuclear coordinates, retaining the actual electronic form."""
    if set(mapping) != {a.atom_id for a in graph.atoms} or len(set(mapping.values())) != len(
        mapping
    ):
        raise ValueError("graph relabelling requires a complete nuclear bijection")
    return replace(
        graph,
        atoms=tuple(replace(a, atom_id=mapping[a.atom_id]) for a in graph.atoms),
        bonds=tuple(
            replace(b, first=mapping[b.first], second=mapping[b.second]) for b in graph.bonds
        ),
    )
