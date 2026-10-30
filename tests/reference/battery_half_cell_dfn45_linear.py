"""Explicit direct-linear-solver policies and receipts from actual IDAKLU setup.

This module neither builds nor solves a model.  In particular, the receipt reads
the serialized analytic functions that the pinned solver passes to its C++
binding; it does not create a replacement Jacobian or claim access to an
unexposed C++ callback evaluation API.
"""

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version


def validate_linear_policy(policy):
    if policy is None:
        return None
    if (
        type(policy) is not dict
        or set(policy) != {"backend", "linear_solution_scaling"}
        or type(policy["backend"]) is not str
        or policy["backend"] not in {"klu", "dense"}
        or type(policy["linear_solution_scaling"]) is not bool
    ):
        raise ValueError(
            "linear policy requires exactly backend and boolean linear_solution_scaling"
        )
    return dict(policy)


def linear_options(policy):
    policy = validate_linear_policy(policy)
    if policy is None:
        return {}
    dense = policy["backend"] == "dense"
    return {
        "jacobian": "dense" if dense else "sparse",
        "linear_solver": "SUNLinSol_Dense" if dense else "SUNLinSol_KLU",
        "linear_solution_scaling": policy["linear_solution_scaling"],
    }


def _integer_array_receipt(values):
    import numpy as np

    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError("actual IDAKLU CSC metadata must be one-dimensional integers")
    canonical = np.ascontiguousarray(raw, dtype="<i8")
    return {
        "length": int(canonical.size),
        "sha256_le_int64": sha256(memoryview(canonical).cast("B")).hexdigest(),
        "head": canonical[:4].tolist(),
        "tail": canonical[-4:].tolist(),
    }


def _function_receipt(serialized, function):
    def ports(count, sparsity):
        return [
            {"shape": list(sparsity(i).shape), "nnz": int(sparsity(i).nnz())} for i in range(count)
        ]

    return {
        "serialized_sha256": sha256(serialized.encode("utf-8")).hexdigest(),
        "serialized_bytes": len(serialized.encode("utf-8")),
        "inputs": ports(function.n_in(), function.sparsity_in),
        "outputs": ports(function.n_out(), function.sparsity_out),
    }


def linear_kernel_contract(actual_solver):
    """Return compact evidence from an already compiled, actual pinned solver.

    Dense CSC slot order is checked in bounded chunks; neither the full dense
    values nor full CSC arrays are emitted.  No setup, root or integration is
    performed here.  Serialized hashes identify the actual functions and are
    provenance, not a substitute for entrywise/Jv equivalence tests.
    """
    import casadi
    import numpy as np
    import pybamm

    if (
        pybamm.__version__ != "25.4.2"
        or casadi.__version__ != "3.6.7"
        or version("pybammsolvers") != "0.3.3"
    ):
        raise RuntimeError("linear kernel receipt requires pinned PyBaMM/CasADi/pybammsolvers")
    if not isinstance(actual_solver, pybamm.IDAKLUSolver):
        raise ValueError("linear kernel receipt requires actual IDAKLUSolver")
    setup = getattr(actual_solver, "_setup", None)
    if not isinstance(setup, Mapping) or not setup:
        raise ValueError("linear kernel receipt requires completed actual IDAKLU setup")
    options = actual_solver._options
    backend = {
        ("sparse", "SUNLinSol_KLU"): "klu",
        ("dense", "SUNLinSol_Dense"): "dense",
    }.get((options.get("jacobian"), options.get("linear_solver")))
    if backend is None or type(options.get("linear_solution_scaling")) is not bool:
        raise ValueError("linear kernel receipt requires a declared direct solver policy")

    def actual_function(name):
        serialized = setup.get(name + "_pkl")
        if type(serialized) is not str or not serialized or setup.get(name) is None:
            raise ValueError("missing actual IDAKLU serialized function: " + name)
        try:
            function = casadi.Function.deserialize(serialized)
        except RuntimeError as exc:
            raise ValueError("invalid actual IDAKLU serialized function: " + name) from exc
        return function, _function_receipt(serialized, function)

    jac, jac_receipt = actual_function("jac_times_cjmass")
    n, inputs = setup.get("number_of_states"), setup.get("inputs")
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("invalid actual IDAKLU state count")
    n = int(n)
    if (
        not isinstance(inputs, (int, np.integer))
        or inputs < 0
        or jac.n_in() != 4
        or jac.n_out() != 1
        or jac.size_out(0) != (n, n)
        or jac.size_in(0) != (1, 1)
        or jac.size_in(1) != (n, 1)
        or jac.size_in(2) != (int(inputs), 1)
        or jac.size_in(3) != (1, 1)
    ):
        raise ValueError("actual IDAKLU Jacobian signature disagrees with setup")
    sparsity = jac.sparsity_out(0)
    colptrs = np.asarray(setup.get("jac_times_cjmass_colptrs"))
    rows = np.asarray(setup.get("jac_times_cjmass_rowvals"))
    col_receipt, row_receipt = _integer_array_receipt(colptrs), _integer_array_receipt(rows)
    nnz = setup.get("jac_times_cjmass_nnz")
    if (
        not isinstance(nnz, (int, np.integer))
        or int(nnz) != sparsity.nnz()
        or colptrs.size != n + 1
        or rows.size != nnz
    ):
        raise ValueError("actual IDAKLU Jacobian CSC dimensions disagree")
    dense = backend == "dense"
    if dense:
        if (
            not sparsity.is_dense()
            or nnz != n * n
            or not np.array_equal(colptrs, np.arange(n + 1, dtype=np.int64) * n)
        ):
            raise ValueError("actual dense Jacobian lacks complete column-major slots")
        for start in range(0, int(nnz), 65536):
            stop = min(start + 65536, int(nnz))
            if not np.array_equal(rows[start:stop], np.arange(start, stop) % n):
                raise ValueError("actual dense Jacobian row slots are not column-major")
    elif not np.array_equal(colptrs, sparsity.colind()) or not np.array_equal(rows, sparsity.row()):
        raise ValueError("actual sparse Jacobian CSC metadata disagrees with serialized kernel")

    kernels = {}
    expected_inputs = {
        "rhs_algebraic": [(1, 1), (n, 1), (int(inputs), 1)],
        "jac_rhs_algebraic_action": [(1, 1), (n, 1), (int(inputs), 1), (n, 1)],
        "mass_action": [(n, 1)],
        "rootfn": [(1, 1), (n, 1), (int(inputs), 1)],
    }
    events = setup.get("num_of_events")
    if not isinstance(events, (int, np.integer)) or events < 0:
        raise ValueError("invalid actual IDAKLU event count")
    for name, signatures in expected_inputs.items():
        function, receipt = actual_function(name)
        output_size = int(events) if name == "rootfn" else n
        if (
            function.n_in() != len(signatures)
            or any(function.size_in(i) != shape for i, shape in enumerate(signatures))
            or function.n_out() != 1
            or function.size_out(0) != (output_size, 1)
        ):
            raise ValueError("actual IDAKLU function signature disagrees: " + name)
        kernels[name] = receipt
    ids = np.asarray(setup.get("ids"), dtype="<f8")
    if ids.shape != (n,) or not np.isin(ids, [0.0, 1.0]).all():
        raise ValueError("invalid actual IDAKLU differential/algebraic IDs")
    return {
        "schema": "half-cell-dfn45-actual-idaklu-linear-kernel-v1",
        "backend": backend,
        "actual_options": deepcopy(options),
        "number_of_states": n,
        "number_of_inputs": int(inputs),
        "number_of_events": int(events),
        "jacobian": {
            **jac_receipt,
            "shape": [n, n],
            "nnz": int(nnz),
            "dense_full": bool(sparsity.is_dense()),
            "column_major_dense_slots": dense,
            "column_pointers": col_receipt,
            "row_indices": row_receipt,
        },
        "kernels": kernels,
        "ids": {
            "length": n,
            "differential": int(np.count_nonzero(ids == 1)),
            "algebraic": int(np.count_nonzero(ids == 0)),
            "sha256_le_float64": sha256(
                memoryview(np.ascontiguousarray(ids)).cast("B")
            ).hexdigest(),
        },
        "cpp_callback_evaluation": "unavailable: pinned C++ Function binding exposes no call API; "
        "serialized functions are the actual generate_function inputs, not a callback evaluation",
    }
