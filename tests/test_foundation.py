from __future__ import annotations

from chemworld.foundation import Observation, Quantity, convert_value
from chemworld.runtime import make_chemworld_constitution
from chemworld.world.instruments import chemworld_instruments
from chemworld.world.state_factory import initial_chemworld_state


def test_unit_conversion_table() -> None:
    assert convert_value(1.0, "h", "s") == 3600.0
    assert convert_value(25.0, "degC", "K") == 298.15
    assert Quantity(1000.0, "mL").to("L").value == 1.0
    assert convert_value(1, "MPa", "kPa") == 1000
    assert convert_value(1, "kPa", "Pa") == 1000
    assert convert_value(1, "MPa/K^2", "kPa/K^2") == 1000


def test_constitution_initial_state_passes() -> None:
    constitution = make_chemworld_constitution()
    report = constitution.check_state(initial_chemworld_state())
    assert report.passed


def test_constitution_rejects_negative_amount() -> None:
    constitution = make_chemworld_constitution()
    state = initial_chemworld_state().replace(species_amounts={"A": -1.0})
    report = constitution.check_state(state)
    assert not report.passed
    assert any(check.name.startswith("nonnegative") for check in report.failures())


def test_constitution_rejects_nonfinite_nested_state_values() -> None:
    constitution = make_chemworld_constitution()
    state = initial_chemworld_state().replace(
        metadata={"nested_diagnostic": {"residual": float("nan")}}
    )

    report = constitution.check_state(state)

    assert "state_numeric_values_finite" in {
        check.name for check in report.failures()
    }


def test_constitution_rejects_unregistered_species_and_conservation_is_fail_closed() -> None:
    constitution = make_chemworld_constitution()
    before = initial_chemworld_state()
    after = before.replace(
        species_amounts={**before.species_amounts, "UNKNOWN_SPECIES": 999.0}
    )

    state_report = constitution.check_state(after)
    material_report = constitution.check_material_conservation(before, after)

    assert "species_registry_membership" in {
        check.name for check in state_report.failures()
    }
    assert material_report.passed is False
    assert "unregistered species" in material_report.message


def test_final_assay_requires_termination() -> None:
    constitution = make_chemworld_constitution()
    state = initial_chemworld_state().replace(volume_L=0.02)
    preconditions = constitution.check_preconditions(
        "measure",
        state,
        {"instrument": "final_assay"},
    )
    assert not preconditions["measure_final_requires_terminated"]
    assert chemworld_instruments()["final_assay"].requires_terminated


def test_observation_constitution_rejects_nonfinite_and_private_payloads() -> None:
    constitution = make_chemworld_constitution()
    observation = Observation(
        values={"score": float("inf")},
        units={"score": "dimensionless"},
        observed_mask={"score": True},
        raw_signal={"species_amounts": {"A": 1.0}},
        processed_estimate={"score": float("nan")},
        uncertainty={"score_std": -0.1},
        cost=float("inf"),
    )

    report = constitution.check_observation(observation)
    failures = {check.name for check in report.failures()}

    assert "observation_non_omniscient" in failures
    assert "observation_values_finite_and_bounded" in failures
    assert "observation_processed_estimate_finite_and_bounded" in failures
    assert "observation_uncertainty_finite_nonnegative" in failures
    assert "observation_accounting_finite_nonnegative" in failures


def test_observation_constitution_rejects_inconsistent_missingness() -> None:
    constitution = make_chemworld_constitution()
    observation = Observation(
        values={"score": None},
        units={"score": "dimensionless"},
        observed_mask={"score": True},
    )

    report = constitution.check_observation(observation)

    # Legacy constructor inputs are converted to native channels; missingness
    # is checked there before the optional Gym projection is reconstructed.
    assert "observation_native_channels" in {
        check.name for check in report.failures()
    }
    assert any(error.endswith(":missing_quality") for error in observation.errors())
