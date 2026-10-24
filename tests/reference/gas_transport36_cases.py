"""Declared transport fixtures; optional public source data are read only by reference runs."""

from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec

IDS = ("N2", "O2", "AR")
TEMPERATURES = (300.0, 700.0, 1200.0)
PRESSURES = (50000.0, 100000.0, 200000.0)
COMPOSITIONS = ((0.7, 0.2, 0.1), (0.2, 0.5, 0.3), (0.4, 0.6, 0.0))
COLLISION_MODELS = ("Neufeld_Janzen_Aziz", "Kim_Monroe")


def network_from_species(species):
    return ReactionNetworkSpec(
        "declared-dilute-gas-mixture",
        tuple(
            SpeciesSpec(
                s["name"],
                "".join(k + (str(int(v)) if v != 1 else "") for k, v in s["composition"].items()),
                phase="gas",
                metadata={"source_declaration": s},
            )
            for s in species
        ),
        (),
        metadata={
            "gas_thermochemistry": {
                s["name"]: {
                    "thermo": s["thermo"],
                    "composition": s["composition"],
                    "reference_pressure_Pa": 101325,
                    "units": {},
                }
                for s in species
            }
        },
    )


def synthetic_species():
    """Declared constant-Cp fixtures, independent of optional source installation."""
    return [
        {
            "name": name,
            "composition": composition,
            "transport": {
                "model": "gas",
                "geometry": geometry,
                "diameter": diameter,
                "well-depth": depth,
            },
            "thermo": {
                "model": "constant-cp",
                "T0": 298.15,
                "h0": f"{h} J/mol",
                "s0": "30 J/mol/K",
                "cp0": f"{cp} J/mol/K",
                "T-min": 200,
                "T-max": 3000,
            },
        }
        for name, composition, geometry, diameter, depth, cp, h in (
            ("N2", {"N": 2}, "linear", 3.621, 97.53, 29.1, 0),
            ("O2", {"O": 2}, "linear", 3.458, 107.4, 29.4, -1000),
            ("AR", {"Ar": 1}, "atom", 3.33, 136.5, 20.8, 2000),
        )
    ]


def public_species():
    from pathlib import Path

    import cantera as ct
    import yaml

    path = next(
        Path(p) / "gri30.yaml"
        for p in ct.get_data_directories()
        if (Path(p) / "gri30.yaml").exists()
    )
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))["species"]
    return [next(s for s in entries if s["name"] == key) for key in IDS]
