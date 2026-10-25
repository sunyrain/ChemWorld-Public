"""One molecular transfer cannot disappear behind a macroscopic mol tolerance."""

from copy import deepcopy

import pytest

from chemworld.foundation.discrete_inventory import (
    AVOGADRO,
    CROSS_PARTITION_TRANSFER_SCHEMA,
    validate_cross_partition_transition,
)


def transition():
    before = {"D": 10 / AVOGADRO, "A": 3.25 / AVOGADRO, "B": 2.5 / AVOGADRO}
    # Two release and one capture: one net molecule passes into the continuous pool.
    after = {"D": 9 / AVOGADRO, "A": 3.75 / AVOGADRO, "B": 3 / AVOGADRO}
    receipt = {
        "schema": CROSS_PARTITION_TRANSFER_SCHEMA,
        "channel_stoichiometry": {
            "release": {"D": -1, "A": 1},
            "capture": {"D": 1, "A": -1},
        },
        "firings": {"release": 0, "capture": 0},
    }
    final = deepcopy(receipt)
    final["firings"] = {"release": 2, "capture": 1}
    return before, after, receipt, final


def validate(before, after, receipt, final):
    return validate_cross_partition_transition(
        before, after, ("D",), {s: {"H": 1} for s in before}, receipt, final
    )


def test_internal_transfer_and_fractional_continuous_conversion_share_one_atom_balance():
    args = transition()
    saved = deepcopy(args)
    assert abs(validate(*args)["H"]) < 1e-12
    assert args == saved


@pytest.mark.parametrize("species", ("D", "A", "B"))
def test_loss_of_one_molecule_is_rejected_in_either_partition(species):
    before, after, receipt, final = transition()
    after[species] -= 1 / AVOGADRO
    with pytest.raises(ValueError, match="atom transfer does not close"):
        validate(before, after, receipt, final)


def test_large_unresolvable_continuous_stock_is_rejected_even_with_zero_transfer():
    before, _, receipt, _ = transition()
    before["A"] = 1e-6
    with pytest.raises(ValueError, match="quantum is not numerically resolved"):
        validate(before, before.copy(), receipt, deepcopy(receipt))


@pytest.mark.parametrize("bad", (True, 1.5, -1))
def test_receipts_require_nonnegative_integer_firings(bad):
    before, after, receipt, final = transition()
    final["firings"]["release"] = bad
    with pytest.raises(ValueError, match="invalid stoichiometry or count"):
        validate(before, after, receipt, final)


def test_transfer_cannot_rewrite_stoichiometry_or_reverse_cumulative_counts():
    before, after, receipt, final = transition()
    final["channel_stoichiometry"]["release"] = {"D": -1, "B": 1}
    with pytest.raises(ValueError, match="changed its declaration"):
        validate(before, after, receipt, final)
    before, after, receipt, final = transition()
    receipt["firings"]["release"] = 3
    with pytest.raises(ValueError, match="lost events"):
        validate(before, after, receipt, final)


def test_transfer_stoichiometry_must_conserve_atoms_and_cross_partitions():
    before, after, receipt, final = transition()
    receipt["channel_stoichiometry"]["release"] = {"D": -1, "A": 2}
    with pytest.raises(ValueError, match="does not conserve atoms"):
        validate(before, after, receipt, final)
    receipt["channel_stoichiometry"]["release"] = {"A": -1, "B": 1}
    with pytest.raises(ValueError, match="invalid stoichiometry or count"):
        validate(before, after, receipt, final)
