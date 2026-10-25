"""Fixed-width CHEMKIN subset fixtures and source-boundary checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemworld.physchem.chemkin_import import (
    chemkin_import_capabilities,
    convert_chemkin,
    import_chemkin,
)


def _nasa_record(name: str, composition: str) -> str:
    """Build one fixed-width NASA7 record with distinct low/high coefficients."""

    header = [" "] * 80
    header[:18] = name.ljust(18)
    header[24:44] = composition.ljust(20)
    # CHEMKIN stores Tlow, Thigh, Tmid in columns 46--75 (not ascending).
    for start, value in ((45, 200.0), (55, 3000.0), (65, 1000.0)):
        header[start : start + 10] = f"{value:10.3f}"
    header[79] = "1"
    low = [3.5, 1.0e-4, -2.0e-8, 3.0e-12, -4.0e-16, -1200.0, 2.0]
    high = [3.0, 2.0e-4, -3.0e-8, 4.0e-12, -5.0e-16, -1100.0, 2.5]

    def line(values: list[float], number: str) -> str:
        return "".join(f"{value:15.8E}" for value in values).ljust(79) + number

    return "\n".join(
        (
            "".join(header),
            line(high[:5], "2"),
            line(high[5:] + low[:3], "3"),
            line(low[3:], "4"),
        )
    )


def _thermo(*species: tuple[str, str]) -> str:
    return "\n".join(_nasa_record(name, composition) for name, composition in species)


def _mechanism(reactions: str, species: tuple[str, ...] = ("H2", "O2", "H2O")) -> str:
    return "\n".join(
        (
            "ELEMENTS",
            "H O",
            "END",
            "SPECIES",
            " ".join(species),
            "END",
            "REACTIONS CAL/MOLE MOLES",
            reactions,
            "END",
        )
    )


def _single_file_mechanism(
    reactions: str,
    species: tuple[str, ...] = ("H2", "O2", "H2O"),
) -> str:
    thermo = _thermo(
        ("H2", "H 2"),
        ("O2", "O 2"),
        ("H2O", "H 2 O 1"),
    )
    return "\n".join(
        (
            "ELEMENTS",
            "H O",
            "END",
            "SPECIES",
            " ".join(species),
            "END",
            "THERMO ALL",
            "200.000 1000.000 5000.000",
            thermo,
            "END",
            "REACTIONS CAL/MOLE MOLES",
            reactions,
            "END",
        )
    )


def test_fixed_width_nasa7_elementary_reaction_roundtrips_native_network():
    network = convert_chemkin(
        _mechanism("2 H2 + O2 = 2 H2O 1.0E+13 0.0 5.0E+4"),
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        source="synthetic.ckin",
    )

    assert len(network.species) == 3
    assert len(network.reactions) == 1
    assert network.reactions[0].reversible
    assert network.reactions[0].rate_law.equation_id == "arrhenius"
    assert network.metadata["import_format"] == "chemkin-gas-mechanism-v1"
    assert network.imported_gas_thermochemistry["H2O"].evaluate(700).cp_J_mol_K > 0


def test_cantera_exported_elem_header_is_accepted():
    mechanism = _mechanism("2 H2 + O2 = 2 H2O 1.0E+13 0.0 5.0E+4").replace(
        "ELEMENTS", "ELEM", 1
    )
    network = convert_chemkin(
        mechanism,
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
    )
    assert network.metadata["import_format"] == "chemkin-gas-mechanism-v1"


def test_third_body_and_troe_falloff_are_converted_with_efficiencies():
    species = ("A", "B", "C")
    thermo = _thermo(("A", "H 1"), ("B", "H 1"), ("C", "H 2"))
    mechanism = _mechanism(
        "\n".join(
            (
                "A + B + M = C + M 2.0E+12 0.0 1.0E+3",
                "A + B (+M) = C (+M) 3.0E+10 0.0 1.0E+3",
                "LOW / 2.0E+15 0.0 2.0E+3 /",
                "TROE / 0.5 100.0 1000.0 3000.0 /",
                "A/2.0/ B/0.5/",
                "DUPLICATE",
            )
        ),
        species,
    )

    network = convert_chemkin(mechanism, thermo)
    assert [reaction.rate_law.equation_id for reaction in network.reactions] == [
        "third_body_arrhenius",
        "troe_falloff",
    ]
    assert network.reactions[0].rate_law.parameters["default_efficiency"] == 1
    assert network.reactions[1].metadata["source_declaration"]["duplicate"] is True
    assert network.reactions[1].rate_law.parameters["third_body_efficiencies"] == {
        "A": 2.0,
        "B": 0.5,
    }
    assert network.metadata["duplicate_reaction_count"] == 1


def test_duplicate_reactions_remain_distinct_native_entries():
    mechanism = _mechanism(
        "\n".join(
            (
                "2 H2 + O2 => 2 H2O 1.0E+3 0.0 0.0",
                "2 H2 + O2 => 2 H2O 2.0E+3 0.0 0.0",
                "DUPLICATE",
            )
        )
    )
    network = convert_chemkin(
        mechanism,
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
    )
    assert len(network.reactions) == 2
    assert network.reaction_ids == ("reaction-001", "reaction-002")
    assert network.metadata["duplicate_reaction_count"] == 1


def test_embedded_thermo_single_file_is_supported():
    network = convert_chemkin(
        _single_file_mechanism("2 H2 + O2 = 2 H2O 1.0E+13 0.0 5.0E+4"),
    )
    assert network.metadata["thermo_source"] == "embedded THERMO section"
    assert len(network.species) == 3


def test_import_embedded_thermo_requires_no_companion_file(tmp_path: Path):
    mechanism_path = tmp_path / "embedded.ckin"
    mechanism_path.write_text(
        _single_file_mechanism("2 H2 + O2 = 2 H2O 1.0E+4 0.0 0.0"),
        encoding="utf-8",
    )
    network = import_chemkin(mechanism_path)
    assert network.metadata["source_mechanism"] == "embedded.ckin; embedded THERMO"


def test_missing_companion_or_embedded_thermo_is_explicitly_rejected():
    with pytest.raises(ValueError, match="companion NASA7 thermo text or embedded THERMO"):
        convert_chemkin(_mechanism("2 H2 + O2 = 2 H2O 1.0E+3 0.0 0.0"))


def test_unsupported_continuation_rejects_whole_import():
    mechanism = _mechanism("A => B 1.0E+3 0.0 0.0\nPLOG / 1.0 2.0 0.0 /")
    with pytest.raises(ValueError, match="PLOG"):
        convert_chemkin(
            mechanism,
            _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        )


def test_import_reads_local_mechanism_and_thermo_files(tmp_path: Path):
    mechanism_path = tmp_path / "minimal.ckin"
    thermo_path = tmp_path / "minimal.thermo"
    mechanism_path.write_text(
        _mechanism("2 H2 + O2 = 2 H2O 1.0D+4 0.0 0.0"), encoding="utf-8"
    )
    thermo_path.write_text(
        _thermo(("H2", "H 2"), ("O2", "O 2"), ("H2O", "H 2 O 1")),
        encoding="utf-8",
    )
    assert import_chemkin(mechanism_path, thermo_path).metadata["import_format"] == (
        "chemkin-gas-mechanism-v1"
    )


def test_capabilities_are_json_ready_and_keep_explicit_boundaries():
    capabilities = chemkin_import_capabilities()
    assert "NASA7" in capabilities["thermo_models"]
    assert "REV/PLOG/CHEB/FORD/RORD continuations" in capabilities["unsupported"]
    assert capabilities["whole_mechanism_policy"] == "all species and reactions converted or reject"
