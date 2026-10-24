"""Fixed 16-cell development coverage for 38D."""

from chemworld.examples.rrkm_gas_network import example_inputs
from chemworld.physchem.energy_grain_specs import (
    EnergyGrainBathGas,
    EnergyGrainMasterEquationSpec,
    EnergyGrainState,
    EnergyGrainWellSpec,
    EnergyTransferBath,
    InternalModeBaseline,
    MicrocanonicalGrainChannel,
    R,
)
from chemworld.physchem.rrkm_grain_binding import compile_rrkm_grains


def explicit(branch):
    return EnergyGrainMasterEquationSpec(
        "qsd38-branch" if branch else "qsd38-single",
        "H2",
        InternalModeBaseline(30, 0, 40, ("authored-internal-mode",)),
        (
            EnergyGrainWellSpec(
                "A",
                (EnergyGrainState("A0", 0), EnergyGrainState("A1", 4000, 2)),
                (
                    EnergyTransferBath("He", 3000, 4000, 300, 0.5, 0.1),
                    EnergyTransferBath("Ar", 900, 6000, 300, 0.3, 0.2),
                ),
            ),
        ),
        (
            EnergyGrainBathGas("He", "He", 2.5 * R, 0, 20),
            EnergyGrainBathGas("Ar", "Ar", 2.5 * R, 0, 20),
        ),
        (
            MicrocanonicalGrainChannel("P-high", "A1", "P", 5),
            *(
                (
                    MicrocanonicalGrainChannel("Q-low", "A0", "Q", 0.2),
                    MicrocanonicalGrainChannel("Q-high", "A1", "Q", 0.7),
                )
                if branch
                else ()
            ),
        ),
        (EnergyGrainState("P", -3000), *((EnergyGrainState("Q", -1000),) if branch else ())),
    )


def cases():
    result = []
    for kind in ("single", "branch", "rrkm"):
        model = (
            compile_rrkm_grains(example_inputs()[0])
            if kind == "rrkm"
            else explicit(kind == "branch")
        )
        for temperature in (400.0, 900.0):
            for pressure in (1000.0, 100000.0) if kind == "rrkm" else (1000.0, 10000.0, 100000.0):
                fraction = 0.35 if kind == "branch" else 0.8
                baths = {
                    "He": fraction * pressure / (1000 * R * temperature),
                    "Ar": (1 - fraction) * pressure / (1000 * R * temperature),
                }
                result.append(
                    (f"{kind}-T{temperature:g}-Pb{pressure:g}", model, temperature, baths)
                )
    return result
