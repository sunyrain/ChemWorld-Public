"""JSON key ordering must not change original material-conservation arithmetic."""

import json

from chemworld.foundation.constitution import PhysicalConstitution
from chemworld.foundation.state import WorldState


def test_json_key_order_preserves_three_vessel_material_total_exactly():
    def vessel(name, amount):
        return WorldState(
            species_amounts={"A": amount}, volume_L=1.0, temperature_K=300.0,
            pressure_Pa=1e5, phase="liquid", vessel_id=name,
        )

    original = vessel("receiver", .01).replace(
        inactive_vessels={"source": vessel("source", .1), "liquid": vessel("liquid", .2)}
    )
    restored = WorldState.from_dict(json.loads(json.dumps(original.to_dict(), sort_keys=True)))
    assert list(original.inactive_vessels) != list(restored.inactive_vessels)
    assert original.to_dict() == restored.to_dict()
    assert PhysicalConstitution.material_inventory(original) == (
        PhysicalConstitution.material_inventory(restored)
    )
