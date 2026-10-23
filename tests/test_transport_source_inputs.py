"""Source assembly conflicts, real pressure consumption and runtime continuation."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.transport_devices import FiniteTransportSpec, LiquidTransportProperties
from chemworld.physchem.transport_properties import ideal_gas_transport_state
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.transport_world import transport_world_spec
from reference.transport_process_fixtures import synthetic_card


def test_combine_preserves_sources_and_rejects_ambiguous_overwrites():
    card = synthetic_card(ComponentSpec("ethanol", "C2H6O"))
    a = PropertyInputCard.combine((card, card), reference_note="Explicit identical inputs")
    assert a.package == card.package and a.sources == card.sources
    assert PropertyInputCard.from_json(json.dumps(a.to_dict())).to_dict() == a.to_dict()
    other = synthetic_card(card.package.component, mu=0.002)
    with pytest.raises(ValueError, match="correlation"):
        PropertyInputCard.combine((card, other), reference_note="Reject conflict")
    with pytest.raises(ValueError, match="component"):
        PropertyInputCard.combine(
            (card, synthetic_card(ComponentSpec("other", "H2O"))),
            reference_note="Reject identity conflict",
        )
    with pytest.raises(ValueError, match="caloric"):
        PropertyInputCard.combine(
            (card, replace(card, reference=replace(card.reference, energy_J_mol=1))),
            reference_note="Reject datum conflict",
        )
    other_source = replace(
        card.sources[0],
        provenance=replace(
            card.sources[0].provenance,
            citation="Conflicting description for same identifier",
        ),
    )
    with pytest.raises(ValueError, match="source"):
        PropertyInputCard.combine(
            (card, replace(card, sources=(other_source,))), reference_note="Reject source conflict"
        )
    curve = replace(card.package.correlations[0], correlation_id="alternative-cp")
    alternative = replace(
        card,
        package=replace(card.package, correlations=(curve,)),
        uncertainties={
            curve.correlation_id: replace(
                next(iter(card.uncertainties.values())), field_id=curve.correlation_id
            )
        },
    )
    with pytest.raises(ValueError, match="overlapping"):
        PropertyInputCard.combine((card, alternative), reference_note="Explicit method needed")


def test_gas_closure_uses_declared_phase_pressure_and_rejects_missing_inputs():
    card = synthetic_card(ComponentSpec("nitrogen", "N2", default_phase="gas"), cp=30)
    args = ((card.package,), {"nitrogen": 0.01}, 400)
    references = {"nitrogen": card.reference}
    low = ideal_gas_transport_state(*args, 1e5, references)
    high = ideal_gas_transport_state(*args, 2e5, references)
    assert high[0].density_kg_m3 == pytest.approx(2 * low[0].density_kg_m3)
    assert low[2] == pytest.approx(2 * high[2])
    for quantities, amounts, t, refs in (
        ((), {"nitrogen": 0.01}, 400, references),
        (card.package.correlations, {"nitrogen": 0}, 400, references),
        (card.package.correlations, {"nitrogen": 0.01}, 500, references),
        (card.package.correlations, {"nitrogen": 0.01}, 400, {}),
        (
            card.package.correlations,
            {"nitrogen": 0.01},
            400,
            {"nitrogen": CaloricReference(phase="liquid")},
        ),
    ):
        with pytest.raises(ValueError):
            ideal_gas_transport_state(
                (replace(card.package, correlations=quantities),), amounts, t, 1e5, refs
            )


def liquid(pressure):
    component = ComponentSpec("ethanol", "C2H6O")
    card = synthetic_card(component, pressure=pressure, variable=True, vapor_pressure=True)
    parameters = {
        "ethanol": {
            "critical_temperature_K": 572.2,
            "critical_pressure_Pa": 3.47e6,
            "acentric_factor": 0.236,
        }
    }
    return LiquidTransportProperties(
        (card.package,), (280, 330), pressure, {"ethanol": card.reference}, parameters
    )


def test_pressure_correction_reaches_real_pump_and_survives_snapshot():
    states = []
    for pressure in (1e5, 2e5):
        props = liquid(pressure)
        assert LiquidTransportProperties.from_dict(json.loads(json.dumps(props.to_dict()))) == props
        runtime = ChemWorldRuntime(
            composition=transport_world_spec(
                FiniteTransportSpec(props),
                feed_temperature_K=300,
                receiver_temperature_K=300,
            )
        )
        restored = None
        try:
            runtime.reset(seed=24)
            assert (
                runtime.execute({"operation": "set_flow_rate", "flow_rate_mL_min": 20}).info[
                    "transaction_status"
                ]
                == "committed"
            )
            assert (
                runtime.execute({"operation": "run_flow", "duration_s": 1}).info[
                    "transaction_status"
                ]
                == "committed"
            )
            restored = ChemWorldRuntime.from_snapshot(
                json.loads(json.dumps(runtime.snapshot().to_dict(), allow_nan=False))
            )
            for current in (runtime, restored):
                assert (
                    current.execute({"operation": "run_flow", "duration_s": 1}).info[
                        "transaction_status"
                    ]
                    == "committed"
                )
            assert runtime._state.to_dict() == restored._state.to_dict()
            states.append(runtime._state.volume_L)
        finally:
            runtime.close()
            if restored is not None:
                restored.close()
    assert states[0] > states[1]


def test_missing_pressure_baseline_and_mismatched_composition_reject():
    props = liquid(1e5)
    incomplete = replace(
        props.packages[0],
        correlations=tuple(
            c for c in props.packages[0].correlations if c.property_id != "vapor_pressure"
        ),
    )
    with pytest.raises(ValueError, match="vapor_pressure"):
        replace(props, packages=(incomplete,)).evaluate({"ethanol": 1}, 300)
    for fractions in ({"other": 1}, {"ethanol": 0.8}, {"ethanol": float("nan")}):
        with pytest.raises(ValueError):
            transport_world_spec(FiniteTransportSpec(props), feed_mole_fractions=fractions)
    with pytest.raises(ValueError, match="Lucas"):
        replace(props, viscosity_pressure_parameters={"other": {}})
