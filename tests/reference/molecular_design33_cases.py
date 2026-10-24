"""Native declarations for the frozen independent 33C case matrix."""

from chemworld.examples.molecular_process_design import two_stage_design
from reference.molecular_design33_reference import CASES


def specification(index, start=0.2):
    x, tin, pout, eta1, eta2, capacity, price, tmax, upper = CASES[index]
    return two_stage_design(
        x_a=x,
        temperature_K=tin,
        final_pressure_Pa=pout,
        efficiencies=(eta1, eta2),
        cooling_capacity_W=capacity,
        cooling_price_per_J=price,
        maximum_pressure_Pa=upper,
        stage1_maximum_temperature_K=tmax,
        start_coordinate=start,
    )
