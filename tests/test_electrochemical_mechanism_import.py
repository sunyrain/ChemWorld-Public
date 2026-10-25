"""Coverage for the explicit Cantera electrochemical mechanism extension."""

from __future__ import annotations

import json

import pytest

from chemworld.physchem.electrochemical_mechanism_import import (
    ELECTROCHEMICAL_MECHANISM_SCHEMA_VERSION,
    ImportedElectrochemicalMechanism,
    convert_electrochemical_mechanism,
    electrochemical_mechanism_import_capabilities,
)
from chemworld.physchem.mechanism_capabilities import cantera_mechanism_capabilities


def _payload() -> dict[str, object]:
    return {
        "schema_version": ELECTROCHEMICAL_MECHANISM_SCHEMA_VERSION,
        "description": "two electron reduction",
        "cantera-version": "3.2",
        "phases": [
            {
                "name": "electrolyte",
                "thermo": "ideal-electrolyte",
                "kinetics": "electrochemical",
                "species": ["A", "P", "e-"],
            }
        ],
        "species": [
            {"name": "A", "charge": 0},
            {"name": "P", "charge": -1},
            {"name": "e-", "charge": -1},
        ],
        "reactions": [
            {
                "id": "A-to-P",
                "electrons": 2,
                "standard-potential": "700 mV",
                "reaction-quotient": {"P": 1, "A": -1},
                "exchange-current-density": "10 mA/cm^2",
                "electrode-area": "1 cm^2",
                "faradaic-efficiency": 0.95,
                "product-selectivity": 0.90,
            }
        ],
        "resistance": {
            "electrolyte-conductivity": "10 mS/cm",
            "electrode-gap": "1 mm",
            "electrode-area": "1 cm^2",
            "contact-resistance": "2 ohm",
            "voltage-window": "2 V",
        },
    }


def test_electrochemical_import_normalizes_units_and_replays() -> None:
    bundle = convert_electrochemical_mechanism(_payload())

    reaction = bundle.reactions[0]
    assert reaction.standard_potential_V == pytest.approx(0.7)
    assert reaction.exchange_current_density_A_m2 == pytest.approx(100.0)
    assert reaction.electrode_area_m2 == pytest.approx(1.0e-4)
    assert bundle.resistance is not None
    assert bundle.resistance.electrolyte_conductivity_S_m == pytest.approx(1.0)
    assert bundle.resistance.electrode_gap_m == pytest.approx(1.0e-3)

    replay = ImportedElectrochemicalMechanism.from_dict(bundle.to_dict())
    assert replay.to_dict() == bundle.to_dict()
    assert len(json.dumps(bundle.to_dict(), allow_nan=False)) > 0


def test_import_rejects_unknown_quotient_species_and_duplicate_reactions() -> None:
    payload = _payload()
    payload["reactions"] = [
        {
            "id": "bad",
            "electrons": 1,
            "standard-potential": "1 V",
            "reaction-quotient": {"missing": 1},
            "exchange-current-density": "1 A/m^2",
            "electrode-area": "1 m^2",
        }
    ]
    with pytest.raises(ValueError, match="unknown reaction-quotient species"):
        convert_electrochemical_mechanism(payload)

    payload = _payload()
    payload["reactions"] = [payload["reactions"][0], dict(payload["reactions"][0])]
    with pytest.raises(ValueError, match="reaction ids must be unique"):
        convert_electrochemical_mechanism(payload)


def test_capability_inventory_keeps_boundaries_and_qualification_split() -> None:
    electrochemical = electrochemical_mechanism_import_capabilities()
    inventory = cantera_mechanism_capabilities()

    assert "Nernst" in electrochemical["rate_models"]
    assert "CHEMKIN input" in electrochemical["unsupported"]
    assert inventory["schema_version"] == "chemworld-cantera-capability-inventory-v1"
    assert "reactor_trajectories" in inventory["qualification_boundaries"]
    assert "CHEMKIN input" in inventory["unsupported"]

