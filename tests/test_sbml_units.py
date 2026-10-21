"""Unit import preserves SBML powers/scales and rejects implicit conventions."""

import pytest

from chemworld.data.sbml_units import SBMLUnitTerm, base_unit, unit_definition
from chemworld.foundation.dimensions import dimension_definition


def test_unit_multiplies_scale_before_exponent_and_uses_common_dimensions():
    source = SBMLUnitTerm("metre", 3, -2, 2)
    # (2 cm)^3 is 8 mL, not 2e-6 m^3.
    signature = source.signature()
    assert signature.dimension == dimension_definition("volume").vector
    assert signature.volume_l(1) == pytest.approx(.008)
    assert source.scale == -2 and source.multiplier == 2 and source.exponent == 3


def test_millimolar_inverse_minute_and_source_item_policy_are_not_guessed():
    concentration = unit_definition((SBMLUnitTerm("mole", 1, -3, 1),
                                     SBMLUnitTerm("litre", -1, 0, 1)))
    assert concentration.dimension == dimension_definition("concentration").vector
    assert concentration.si_factor == pytest.approx(1.)  # 1 mmol/L = 1 mol/m^3
    assert SBMLUnitTerm("second", 1, 0, 60).signature().time_s(2) == 120
    assert SBMLUnitTerm("mole", 1, -3, 1).signature().amount_mol(3) == .003
    with pytest.raises(ValueError, match="explicit items_per_mole"):
        base_unit("item")
    assert base_unit("item", items_per_mole=6.02214076e23).amount_mol(
        6.02214076e23) == pytest.approx(1, abs=3e-16)
    with pytest.raises(ValueError, match="dimension"):
        concentration.amount_mol(1)


@pytest.mark.parametrize("fields", [
    ("celsius", 1, 0, 1), ("litre", float("nan"), 0, 1), ("litre", 1, True, 1),
    ("litre", 1, 0, -1), ("litre", 1, 0, 0), ("mole", 1, 0, float("inf")),
])
def test_invalid_or_unsupported_source_unit_is_not_dimensionless(fields):
    with pytest.raises(ValueError):
        SBMLUnitTerm(*fields)


def test_unit_overflow_underflow_and_nonfinite_native_amount_reject():
    for scale in (1000, -1000):
        with pytest.raises(ValueError):
            SBMLUnitTerm("mole", 1, scale, 1).signature()
    with pytest.raises(ValueError):
        base_unit("mole").amount_mol(float("inf"))


@pytest.mark.parametrize("invalid", [True, None, "2", 1j])
def test_unit_normalization_and_count_policy_do_not_coerce_non_real_inputs(invalid):
    with pytest.raises(ValueError):
        base_unit("mole").amount_mol(invalid)
    with pytest.raises(ValueError):
        base_unit("item", items_per_mole=invalid)
