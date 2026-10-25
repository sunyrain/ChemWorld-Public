"""Reference adapter alignment uses declared axes, including partial SDK flattening."""

import numpy as np
import pytest

from reference.sedml_experiment_references import normalized_array


def test_partial_repeat_collapse_preserves_outer_inner_time_order():
    logical = np.arange(42).reshape(2, 3, 7)
    values, mapping = normalized_array(logical.reshape(6, 7).T, (2, 3, 7))
    assert values == logical.ravel().tolist()
    assert mapping["collapsed_groups"] == [(2, 3), (7,)]
    assert mapping["permutation"] == [1, 0]


def test_separate_axes_and_fully_flat_sdk_order():
    logical = np.arange(42).reshape(2, 3, 7)
    values, _ = normalized_array(logical.transpose(2, 1, 0), (2, 3, 7))
    assert values == logical.ravel().tolist()
    values, mapping = normalized_array(logical.ravel(), (2, 3, 7))
    assert values == logical.ravel().tolist()
    assert mapping["mapping"] == "SDK flat order"


@pytest.mark.parametrize("array,shape", [
    (np.zeros((6, 6)), (2, 3, 6)),  # two equal-length physical axes cannot be ordered by lengths
    (np.zeros((7, 5)), (2, 3, 7)),  # incomplete SDK rows are never selected by their values
    (np.full((7, 6), np.nan), (2, 3, 7)),
])
def test_ambiguous_incomplete_or_nonfinite_reference_is_rejected(array, shape):
    with pytest.raises(ValueError):
        normalized_array(array, shape)
