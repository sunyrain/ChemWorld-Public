"""Physical declaration, same-table energy derivative and binary ownership checks."""

from dataclasses import replace

import pytest

from chemworld.physchem.aqueous_transport import BinaryAqueousProperties
from chemworld.physchem.distributed_ro import DistributedROSpec
from reference.ro47_fixtures import feed, spec


def test_declaration_roundtrip_and_exact_ion_basis():
    model = spec()
    assert DistributedROSpec.from_dict(model.to_dict()).to_dict() == model.to_dict()
    n = feed(model)
    n["Cl-"] *= 1.01
    with pytest.raises(ValueError, match="ion inventory"):
        model.properties.validate_amounts(n)
    with pytest.raises(ValueError, match="electroneutral"):
        replace(model.properties, salt_stoichiometry=(("Na+", 2), ("Cl-", 1)))
    with pytest.raises(ValueError, match="source"):
        replace(model.properties, viscosity=replace(model.properties.viscosity, source_id="absent"))


@pytest.mark.parametrize("family", ["ideal", "Davies"])
def test_same_table_enthalpy_direction_includes_nonideal_mixing(family):
    model = spec(family)
    prop, n = model.properties, feed(model)
    direction = feed(model, 80, 0.01)
    exact = prop.enthalpy_rate(n, direction, 310, 1e5)
    eps = 1e-3
    plus = {k: n[k] + eps * direction[k] for k in n}
    minus = {k: n[k] - eps * direction[k] for k in n}
    finite = (prop.enthalpy(plus, 310, 1e5) - prop.enthalpy(minus, 310, 1e5)) / (2 * eps)
    assert exact == pytest.approx(finite, rel=1e-7, abs=1e-6)
    empty = dict.fromkeys(n, 0.0)
    assert prop.enthalpy(empty, 310, 1e5) == 0
    assert prop.enthalpy_rate(empty, direction, 310, 1e5) == prop.enthalpy(direction, 310, 1e5)


def test_property_and_membrane_domains_are_explicit():
    model = spec()
    with pytest.raises(ValueError, match="domain"):
        model.properties.viscosity.evaluate(350, 1e5, 1)
    with pytest.raises(ValueError, match="source"):
        replace(model, water_permeability=replace(model.water_permeability, source_id="absent"))
    restored = BinaryAqueousProperties.from_dict(model.properties.to_dict())
    assert restored.water_id == "H2O"
