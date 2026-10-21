"""The separately specified 52 right-branch checks preserve the strict block."""

import importlib.util
import sys
from pathlib import Path

PHASES = {
    "cuo.tdb": ("FCC_A1", "GAS"),
    "Al-Cu-Y.tdb": (
        "AL2Y", "AL2Y3", "AL42CU68Y10", "AL45CU65Y10", "AL7CU2Y3", "AL8CU4Y",
        "ALCU11Y3", "ALCU17Y2", "ALCU5Y", "ALCUY", "ALY", "ALY2", "BCC_A2",
        "CU4Y", "CU6Y", "CU7Y2", "CUY", "FCC_A1", "HCP_A3", "H_AL3Y", "H_CU2Y",
        "LIQUID", "L_AL3Y", "L_CU2Y",
    ),
}


def main():
    spec = importlib.util.spec_from_file_location(
        "calphad_boundary_driver", Path(__file__).with_name("run_calphad_block.py"),
    )
    block = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(block)
    block.DATABASES = tuple(PHASES)
    block.TEMPERATURES = (1000.0,)
    block.PHASE_FILTER = frozenset((name, phase) for name, phases in PHASES.items()
                                    for phase in phases)
    block.REFERENCE_SCRIPT = Path(__file__).resolve()
    ordinary = block.native_second_values

    def checked_values(model, sites, temperature):
        for operation in (model.derivatives, model.temperature_response):
            try:
                operation(sites, temperature, block.PRESSURE)
            except ValueError as exc:
                if "one-sided" not in str(exc):
                    raise
            else:
                raise AssertionError("a TDB segment boundary was accepted without a side")
        values, diagnostics = ordinary(model, sites, temperature)
        return values, (*diagnostics, "default branch rejection independently checked")

    block.native_second_values = checked_values
    for flag in ("--second-order", "--allow-one-sided"):
        if flag not in sys.argv:
            sys.argv.append(flag)
    return block.main()


if __name__ == "__main__":
    raise SystemExit(main())
