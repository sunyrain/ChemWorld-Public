"""Fixed baths never silently acquire movable heater queue semantics."""

import pytest

from chemworld.runtime.scheduling import validate_lab
from chemworld.world.composition import WorldCompositionSpec
from reference.chemostatted_catalysis54_shared import shared_world


@pytest.mark.parametrize(
    "defect, message",
    (
        ("listed_heater", "physical heater identities"),
        ("finite_power", "enabled local isotherm"),
        ("disabled", "enabled local isotherm"),
        ("unknown_role", "unsupported laboratory thermal device role"),
    ),
)
def test_fixed_bath_declaration_rejects_queue_or_thermal_mode_aliases(defect, message):
    payload = shared_world().to_dict()
    settings = payload["physics"]["initial_states"]["reactor"]["equipment"]["heater"]["settings"]
    if defect == "listed_heater":
        payload["physics"]["lab_clock"]["heater_ids"] = ["heater", "observer:heater"]
    elif defect == "finite_power":
        settings["thermal_device"]["mode"] = "finite_power"
    elif defect == "disabled":
        settings["thermal_device"]["enabled"] = False
    else:
        settings["lab_role"] = "unknown"
    with pytest.raises(ValueError, match=message):
        validate_lab(WorldCompositionSpec.from_dict(payload))
