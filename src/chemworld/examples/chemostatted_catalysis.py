"""Two independently declared synthetic surface mechanisms with real aqueous H+."""

from typing import Any

from chemworld.physchem.catalysis_descriptors import FreeEnergyBEP
from chemworld.physchem.catalysis_network import CatalyticMechanismSpec, CatalyticReactionSpec
from chemworld.physchem.catalysis_reservoirs import (
    AqueousStandardThermo,
    CatalyticReservoirBoundary,
    ChemostatSolute,
)
from chemworld.physchem.catalysis_thermochemistry import CatalyticSpeciesThermo
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.heat_transfer_units import CondensedBackgroundSpec, ThermalDeviceSpec
from chemworld.physchem.reaction_network import SpeciesSpec
from chemworld.world.chemostatted_catalysis_world import chemostatted_catalysis_world
from chemworld.world.composition import WorldCompositionSpec

SOURCE = DataSourceProvenance(
    "authored-54C",
    0,
    "Synthetic constant-Cp half-cell; H+ reference is explicitly half of H2 standard thermo",
    "1",
    "CC0-1.0",
)
FRAME = "authored-SHE-paired-hydrogen-reference-54C"


def example_mechanism(kind: str = "her") -> CatalyticMechanismSpec:
    def species(
        name: str, formula: str, phase: str, h: float, s: float, cp: float, charge: int = 0
    ) -> CatalyticSpeciesThermo:
        native = SpeciesSpec(name, formula, phase=phase, charge=charge)
        if phase == "liquid":
            return CatalyticSpeciesThermo(
                native,
                FRAME,
                SOURCE,
                aqueous_standard=AqueousStandardThermo(h, s, cp, 298.15, 280, 340, SOURCE),
            )
        declaration: dict[str, Any] = {
            "thermo": {
                "model": "constant-cp",
                "T0": 298.15,
                "h0": h,
                "s0": s,
                "cp0": cp,
                "T-min": 280,
                "T-max": 340,
            },
            "composition": native.composition,
            "units": {"quantity": "mol", "energy": "J"},
        }
        if phase == "gas":
            return CatalyticSpeciesThermo(native, FRAME, SOURCE, gas_declaration=declaration)
        return CatalyticSpeciesThermo(native, FRAME, SOURCE, surface_declaration=declaration)

    common = (
        species("hydrogen", "H2", "gas", 0, 100, 30),
        species("proton", "H", "liquid", 0, 50, 15, 1),
        species("chloride", "Cl", "liquid", -10000, 20, 10, -1),
    )

    def reaction(
        name: str,
        left: dict[str, int],
        right: dict[str, int],
        frequency: float = 1000,
        external: dict[str, int] | None = None,
        electron: int = 0,
    ) -> CatalyticReactionSpec:
        return CatalyticReactionSpec(
            name,
            left,
            right,
            "metal",
            frequency,
            FreeEnergyBEP(30000, 0.5, -50000, 50000, SOURCE),
            SOURCE,
            reservoir_reactants=external or {},
            electron_stoichiometry=electron,
        )

    all_species: tuple[CatalyticSpeciesThermo, ...]
    if kind in {"her", "volmer"}:
        all_species = (*common, species("H_ads", "H", "surface", -10000, 0, 10))
        reactions = [
            reaction(
                "electron-proton-capture",
                {"vacancy:metal": 1},
                {"H_ads": 1},
                external={"proton": 1},
                electron=-1,
            )
        ]
        if kind == "her":
            reactions.append(
                reaction(
                    "hydrogen-recombination", {"H_ads": 2}, {"hydrogen": 1, "vacancy:metal": 2}
                )
            )
        adsorbates = {"H_ads": "metal"}
    elif kind == "hydroxylation":
        all_species = (
            *common,
            species("oxygen_bound", "O", "surface", -25000, 0, 10),
            species("hydroxyl_bound", "OH", "surface", -35000, 0, 20),
        )
        adsorbates = {"oxygen_bound": "metal", "hydroxyl_bound": "metal"}
        reactions = [
            reaction(
                "surface-hydroxylation",
                {"oxygen_bound": 1},
                {"hydroxyl_bound": 1},
                external={"proton": 1},
                electron=-1,
            )
        ]
    else:
        raise ValueError("choose her, volmer or hydroxylation")
    return CatalyticMechanismSpec(
        "authored-" + kind,
        all_species,
        ("metal",),
        adsorbates,
        tuple(reactions),
        tuple(tuple(0 for _ in adsorbates) for _ in adsorbates),
        280,
        340,
        SOURCE,
        reservoir_species=("proton", "chloride"),
    )


def example_boundary(*, pH: float = 2.0, temperature: float = 298.15) -> CatalyticReservoirBoundary:
    return CatalyticReservoirBoundary(
        {key: ChemostatSolute(10 ** (-pH), 1.0, 1.0, SOURCE) for key in ("proton", "chloride")},
        0.01,
        0.01,
        CondensedBackgroundSpec(4180, 298.15),
        ThermalDeviceSpec(
            setpoint_K=temperature,
            minimum_setpoint_K=280,
            maximum_setpoint_K=340,
            maximum_power_W=10,
            maximum_cooling_power_W=10,
            mode="isothermal",
        ),
        -0.4,
        0.1,
        2e5,
        1.0,
        SOURCE,
        (
            "H+ standard H/S/Cp equals half H2 in an authored SHE paired reference; "
            "not measured single-ion calorimetry"
        ),
        proton_species_id="proton",
    )


def example_world(
    kind: str = "her", *, pH: float = 2.0, potential: float = -0.2
) -> WorldCompositionSpec:
    spec, boundary = example_mechanism(kind), example_boundary(pH=pH)
    ads = (
        {"H_ads": 1e-7}
        if kind != "hydroxylation"
        else {"oxygen_bound": 9e-7, "hydroxyl_bound": 1e-7}
    )
    return chemostatted_catalysis_world(
        spec,
        boundary,
        {"hydrogen": 1e-6},
        capacities_mol={"metal": 1e-6},
        initial_adsorbates_mol=ads,
        potential_V=potential,
        support_heat_capacity_J_K=0.1,
    )


def main() -> None:
    import json

    from chemworld.runtime import ChemWorldRuntime

    runtime = ChemWorldRuntime(composition=example_world())
    runtime.reset()
    for action in (
        {"operation": "configure_electrical_control", "potential_V": -0.2},
        {"operation": "wait", "duration_s": 20},
        {"operation": "measure", "instrument": "electrical_meter"},
    ):
        result = runtime.execute(action)
        if result.info["transaction_status"] != "committed":
            raise RuntimeError(result.info)
    print(json.dumps(result.info, indent=2))


if __name__ == "__main__":
    main()
