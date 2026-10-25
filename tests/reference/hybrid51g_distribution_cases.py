"""All original controlled/cross-partition models; lazy native authoring only."""

from dataclasses import replace


def units():
    from cross_partition51_cases import CASES as CROSS_CASES
    from stochastic_hybrid51_cases import CASES as FEEDBACK_CASES

    result = []
    for kind, rows in (("feedback", FEEDBACK_CASES), ("cross", CROSS_CASES)):
        for index, parameters in enumerate(rows):
            frozen = kind == "cross" and index == 5
            rare = kind == "cross" and index == 2
            result.append(
                {
                    "id": f"S{len(result) + 1:02d}",
                    "index": len(result),
                    "kind": kind,
                    "case_index": index,
                    "parameters": parameters,
                    "duration_s": 3.0,
                    "output_steps": 24,
                    "frozen": frozen,
                    "rare": rare,
                    "native_replicates": 1 if frozen else 4096 if rare else 1024,
                    "reference_replicates": 1 if frozen else 16384 if rare else 8192,
                    "copasi_replicates": 1 if frozen else 4096 if rare else 1024,
                }
            )
    assert len(result) == 10
    assert sum(u["native_replicates"] for u in result) == 12289
    assert sum(u["reference_replicates"] for u in result) == 81921
    return result


def author(unit, method="RK45"):
    from hybrid51g_adapter_cases import author as original

    network, spec, amounts = original(unit)
    return network, replace(spec, ode_method=method), amounts


def jobs(unit):
    from hybrid51g_adapter_cases import jobs as original

    common = original({**unit, "seeds": [17]})[0]
    return [
        {
            **common,
            "id": unit["id"] + "-" + str(index),
            "seed": 17 if unit["frozen"] else 512800000 + 100000 * unit["index"] + index,
            "output_steps": unit["output_steps"],
        }
        for index in range(unit["copasi_replicates"])
    ]
