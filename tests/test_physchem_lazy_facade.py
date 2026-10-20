from __future__ import annotations

import json
import subprocess
import sys


def test_physchem_facade_loads_public_models_on_first_access() -> None:
    program = """
import json
import sys
import chemworld.physchem as physchem

deferred = {
    "chemworld.physchem.crystallization_units",
    "chemworld.physchem.electrochemical_scenarios",
    "chemworld.physchem.eos",
    "chemworld.physchem.reference_validation",
    "chemworld.physchem.finite_volume",
}
before = sorted(deferred.intersection(sys.modules))
module_name = physchem.CoolingCrystallizationResult.__module__
after = sorted(deferred.intersection(sys.modules))
diffusion_module = physchem.ConservativeDiffusion.__module__
print(json.dumps({
    "before": before,
    "after": after,
    "module_name": module_name,
    "diffusion_module": diffusion_module,
    "after_diffusion": sorted(deferred.intersection(sys.modules)),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["before"] == []
    assert result["after"] == ["chemworld.physchem.crystallization_units"]
    assert result["module_name"] == "chemworld.physchem.crystallization_units"
    assert result["diffusion_module"] == "chemworld.physchem.finite_volume"
    assert result["after_diffusion"] == [
        "chemworld.physchem.crystallization_units", "chemworld.physchem.finite_volume",
    ]
