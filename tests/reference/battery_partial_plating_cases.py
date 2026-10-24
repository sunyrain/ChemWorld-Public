"""Prospective constant-deactivation SPM inputs: 12 units, 62 routes."""

from copy import deepcopy
from dataclasses import replace

from battery_plating_cases import (  # Frozen physical input and tracing conventions only.
    ATOL,
    GRIDS,
    PROFILE_RADII,
    RESPONSES,
    ROOT_TOL,
    RTOL,
    F,
    R,
)
from battery_plating_cases import (
    case_data as prior_case,
)
from battery_plating_cases import (
    native_spec as reversible_spec,
)

__all__ = ["ATOL", "GRIDS", "PROFILE_RADII", "RESPONSES", "ROOT_TOL", "RTOL", "F", "R"]
NAMES = (
    "pure-deactivation",
    "gamma-zero-reversible-limit",
    "zero-metal-charge",
    "seeded-internal-rest",
    "stripping-and-deactivation",
    "CC-rest-reverse",
    "kinetics-geometry-temperature",
    "interior-CV",
    "signed-limited-CV",
    "surface-stop",
    "voltage-stop",
    "refusals",
)
REFUSALS = (
    "negative-gamma",
    "missing-gamma",
    "gamma-unit",
    "version",
    "MPM",
    "negative-plated",
    "negative-dead",
    "SEI",
    "thermal",
    "EIS",
    "SOH",
    "chemical-energy",
    "negative-k",
    "nonpositive-volume",
)


def case_data(index):
    if index not in range(1, 13):
        raise ValueError("partial block fixes exactly P01-P12")
    data = deepcopy(prior_case({6: 5, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12}.get(index, 2)))
    data.update(index=index, name=NAMES[index - 1], mode="partially reversible", gamma_s_inv=0.002)
    if index == 1:
        data.update(
            k_m_s=0.0, gamma_s_inv=0.005, initial_plated_mol=0.001, waveform=((600.0, 0.0, "A"),)
        )
    elif index == 2:
        data.update(
            gamma_s_inv=0.0,
            initial_plated_mol=0.0005,
            waveform=((150.0, -0.6, "A"), (100.0, 0.0, "A"), (150.0, 0.4, "A")),
        )
    elif index in (4, 5):
        data.update(initial_plated_mol=0.002, waveform=((300.0, 0.0 if index == 4 else 0.4, "A"),))
    elif index == 7:
        data["gamma_s_inv"] = 0.01
    elif index == 12:
        data["expected"] = "refusals"
    return data


def native_spec(data, cells, *, old_reversible=False):
    from chemworld.data.scientific_inputs import InputSource, InputValue, QuantityColumn
    from chemworld.physchem.battery_plating import LithiumPlatingSpec
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    original = reversible_spec({**data, "mode": "reversible"}, cells)
    if old_reversible:
        return original
    table = original.plating.parameters
    row = table.rows[0]
    declaration = InputSource(
        DataSourceProvenance(
            "partial-plating-synthetic",
            0,
            "Explicit synthetic constant metal decay and reversible kinetics; no SEI dependence",
            "fixed-partial-plating-block-1",
            "authored development qualification",
        ),
        "synthetic",
    )
    table = replace(
        table,
        dataset_id="partial-plating-synthetic",
        sources=(declaration,),
        columns=(
            *table.columns,
            QuantityColumn(
                "gamma_s_inv",
                "1/s",
                "parameter",
                (-1.0, 1e12),
                "Explicit constant metal deactivation; no SEI-thickness dependence",
            ),
        ),
        rows=(
            replace(
                row,
                source_id=declaration.provenance.source_id,
                values={
                    **row.values,
                    "gamma_s_inv": InputValue(
                        data["gamma_s_inv"],
                        "1/s",
                        None,
                        "Authored constant; no measured uncertainty",
                    ),
                },
            ),
        ),
    )
    return replace(
        original,
        plating=LithiumPlatingSpec(
            table,
            row.row_id,
            "partially reversible",
            data["initial_plated_mol"],
            data["initial_dead_mol"],
        ),
        provenance="Declared constant-decay SPM qualification",
    )
