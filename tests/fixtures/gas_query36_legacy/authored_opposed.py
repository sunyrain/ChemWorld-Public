"""Exact authored input factory from the original opposed compatibility tests.

No material/source declaration or saved source state is used. This fixture is
not a runtime fallback or an independent scientific implementation.
"""

from math import fsum

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.gas_opposed_flow import GasOpposedFlowSpec, opposed_flow_initial_profile
from chemworld.physchem.gas_polynomial_transport import GasTransportPolynomialSpec
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.specs import ComponentSpec


def authored(*, cells=4, faces=None, reacting=False, spreads=2.0, flux_model="mixture-averaged"):
    species = (
        SpeciesSpec("A", "H2", phase="gas"),
        SpeciesSpec("B", "H2" if reacting else "O2", phase="gas"),
    )
    reactions = (
        (
            ReactionSpec.from_equation(
                reaction_id="authored-isomerization",
                equation="A -> B",
                rate_law=RateLawSpec("authored-k", "mass_action", {"k": 0.1}),
            ),
        )
        if reacting
        else ()
    )
    network = ReactionNetworkSpec(
        "authored-opposed",
        species,
        reactions,
        metadata={
            "gas_thermochemistry": {
                s.species_id: {
                    "thermo": {
                        "model": "constant-cp",
                        "T0": 600.0,
                        "h0": f"{0 if s.species_id == 'A' else -1000} J/mol",
                        "s0": "30 J/mol/K",
                        "cp0": "30 J/mol/K",
                        "T-min": 200.0,
                        "T-max": 2000.0,
                    },
                    "composition": s.composition,
                    "reference_pressure_Pa": 101325,
                    "units": {},
                }
                for s in species
            },
        },
    )
    ids = network.species_ids
    masses = {
        s.species_id: ComponentSpec(s.species_id, s.formula).molecular_weight_g_mol / 1000
        for s in species
    }
    transport = GasTransportPolynomialSpec(
        ids,
        {s.species_id: s.to_dict() for s in species},
        network.metadata["gas_thermochemistry"],
        masses,
        dict.fromkeys(ids, (0.002, 0, 0, 0, 0)),
        dict.fromkeys(ids, (0.001, 0, 0, 0, 0)),
        {s: dict.fromkeys(ids, (0.0001, 0, 0, 0, 0)) for s in ids},
        DataSourceProvenance(
            "authored-opposed", 0, "Authored functional coefficients", "v1", "CC0-1.0", "0" * 64
        ),
        "authored-synthetic",
        (300.0, 1200.0),
        200000.0,
        "off",
    )
    mesh = (
        DiffusionMesh.planar(0.02, cells, area_m2=0.023)
        if faces is None
        else DiffusionMesh("planar", tuple(faces), area_m2=0.023)
    )
    x = {"A": 0.8, "B": 0.2}
    rho = 1e5 / (R * 600) * fsum(x[s] * masses[s] for s in ids)
    # Analytic constant V>0, both true inlets and an interior G sign reversal.
    mdot = rho * 2 * mesh.faces_m[-1]
    spec = GasOpposedFlowSpec(
        mesh,
        transport,
        1e5,
        600,
        600,
        x,
        x,
        mdot,
        mdot,
        "B",
        (300, 1200),
        spreads,
        spreads,
        flux_model,
    )
    initial = opposed_flow_initial_profile(
        spec,
        network,
        temperatures_K=[600.0] * mesh.cells,
        mole_fractions=[x] * mesh.cells,
        spread_rates_per_s=[2.0] * mesh.cells,
        radial_pressure_gradient_Pa_m2=-rho * 4,
    )
    return spec, network, initial
