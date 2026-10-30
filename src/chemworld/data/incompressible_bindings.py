"""Checked convenience identities; arbitrary catalog names are not chemicals.

Other materials require an explicit IncompressibleMaterialModel declaration,
including the evidence for its liquid identity and concentration convention.
"""

from chemworld.data.incompressible import incompressible_fluid
from chemworld.physchem.elements import parse_formula
from chemworld.physchem.incompressible_properties import IncompressibleMaterialModel
from chemworld.physchem.specs import ComponentSpec

_IDENTITIES = {
    "MEG": ("C2H6O2", "107-21-1", "https://webbook.nist.gov/cgi/cbook.cgi?ID=C107211"),
    "MEG2": ("C2H6O2", "107-21-1", "https://webbook.nist.gov/cgi/cbook.cgi?ID=C107211"),
    "MPG": ("C3H8O2", "57-55-6", "https://webbook.nist.gov/cgi/cbook.cgi?ID=57-55-6"),
    "MPG2": ("C3H8O2", "57-55-6", "https://webbook.nist.gov/cgi/cbook.cgi?ID=57-55-6"),
    "MCA": ("CaCl2", "10043-52-4", "https://pubchem.ncbi.nlm.nih.gov/compound/10043-52-4"),
    "MCA2": ("CaCl2", "10043-52-4", "https://pubchem.ncbi.nlm.nih.gov/compound/10043-52-4"),
}


def bind_incompressible_liquid(
    name: str, *, water: ComponentSpec, solute: ComponentSpec | None = None,
) -> IncompressibleMaterialModel:
    """Bind supplied original component IDs to an identified aqueous/pure fit.

    Salt n denotes neutral anhydrous formula units, not a dissociation model.
    The generic propylene-glycol CAS does not qualify an enantiopure sample.
    """
    if (
        water.composition != {"H": 2, "O": 1}
        or water.cas_number != "7732-18-5"
        or water.charge != 0
    ):
        raise ValueError("incompressible aqueous binding requires identified neutral H2O")
    fluid = incompressible_fluid(name)
    if name == "Water":
        if solute is not None:
            raise ValueError("pure Water binding cannot include a solute")
        return IncompressibleMaterialModel(
            fluid,
            (water,),
            None,
            fluid.source + "; H2O CAS 7732-18-5; engineering fit of pure water",
        )
    if name not in _IDENTITIES:
        raise ValueError("catalog identity is not a qualified aqueous material binding")
    formula, cas, source = _IDENTITIES[name]
    if (
        solute is None
        or solute.composition != parse_formula(formula)
        or solute.cas_number != cas
        or solute.charge != 0
        or water.identifier == solute.identifier
    ):
        raise ValueError("solute identity differs from the named source solution")
    if fluid.fraction_basis != "mass":
        raise ValueError("qualified aqueous binding requires the source mass-fraction convention")
    return IncompressibleMaterialModel(
        fluid,
        (water, solute),
        solute.identifier,
        fluid.source + "; constituent identity: " + source + "; water CAS 7732-18-5; "
        "bulk composition only; no measured dilution heat or ionic/speciation qualification",
    )
