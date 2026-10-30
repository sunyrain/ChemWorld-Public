"""Observe actual public IDAKLU buffers without changing solver acceptance.

Install once on a fresh solver for one history. Only that instance is wrapped;
the original pinned method signatures and original calls are retained. Copies
are observation data, never replacement states. Public saved t0/steps are not
direct C++ post-IC storage or failed internal states. No public negative object
is not evidence of a particular C++ failure cause.
"""

from copy import deepcopy
from numbers import Real
from types import MethodType


def _error(exc):
    try:
        message = str(exc)
    except Exception:
        message = "<exception string unavailable>"
    return {"type": type(exc).__module__ + "." + type(exc).__qualname__, "message": message}


def _array(value):
    import numpy as np

    array = np.asarray(value)
    return {"shape": tuple(array.shape), "dtype": array.dtype.str, "data": array.copy(order="K")}


def _seconds(value):
    # Pinned TimerTime.value is the numeric second count. Its __str__ is a
    # human-readable duration and is not a lossless numeric representation.
    raw = value if isinstance(value, Real) else value.value
    if isinstance(raw, bool) or not isinstance(raw, Real):
        raise TypeError("integration time requires numeric seconds or a numeric .value")
    return float(raw)


def install_failure_observer(solver, records):
    """Return observation health and append independent snapshots to ``records``.

    Arrays retain their shape/dtype and nonfinite values; the archive consumer
    must preserve those data. Copy/sink failures are separate observer errors.
    They never suppress the original method, return value or exception. Call IDs
    join actual snapshots and do not substitute for recorded solver data.
    """
    if getattr(solver, "_chemworld_failure_observer_status", None) is not None:
        raise ValueError("install once on a fresh per-history solver")
    if not callable(getattr(records, "append", None)):
        raise TypeError("records must provide append")
    original_integrate = solver._integrate
    original_post_process = solver._post_process_solution
    status = {
        "schema": "half-cell-dfn45-public-idaklu-failure-observer-v1",
        "observer_errors": [],
        "next_call_id": 0,
        "cpp_internal_ic": "unavailable",
        "failed_internal_state": "unavailable",
    }
    frames = []

    def observer_error(where, exc, call_id):
        status["observer_errors"].append({"where": where, "call_id": call_id, **_error(exc)})

    def append(record):
        try:
            records.append(record)
        except Exception as exc:
            observer_error("record-append", exc, record.get("call_id"))

    def capture(record, key, getter, call_id, copier=deepcopy):
        try:
            record[key] = copier(getter())
        except Exception as exc:
            record.setdefault("copy_errors", {})[key] = _error(exc)
            observer_error("copy:" + key, exc, call_id)

    def observe_post_process(self, sol, model, integration_time, inputs_dict):
        frame = frames[-1] if frames else None
        call_id = None if frame is None else frame["call_id"]
        try:
            if frame is not None:
                frame["post_process_entries"] += 1
            flag = sol.flag
            if flag < 0:
                if frame is not None:
                    frame["negative_public_returns"] += 1
                record = {
                    "event": "public-cpp-negative-flag-return",
                    "call_id": call_id,
                    "scope": "public returned buffers; not failed internal state or direct C++ IC",
                    "fields": {},
                }
                # pybind exposes C++ int as Python int, with no NumPy shape/dtype.
                capture(
                    record["fields"],
                    "flag",
                    lambda: {
                        "data": flag,
                        "python_type": type(flag).__name__,
                        "shape": None,
                        "dtype": None,
                    },
                    call_id,
                )
                for name in ("t", "y", "yp", "yS", "ypS", "y_term"):
                    capture(
                        record["fields"],
                        name,
                        lambda name=name: getattr(sol, name),
                        call_id,
                        _array,
                    )
                capture(record, "number_of_states", lambda: model.len_rhs_and_alg, call_id)
                capture(record, "output_variables", lambda: self.output_variables, call_id)
                capture(record, "inputs_dict", lambda: inputs_dict, call_id)
                capture(
                    record,
                    "integration_time_python_type",
                    lambda: (
                        type(integration_time).__module__
                        + "."
                        + type(integration_time).__qualname__
                    ),
                    call_id,
                )
                capture(record, "integration_time_s", lambda: integration_time, call_id, _seconds)
                append(record)
        except Exception as exc:
            observer_error("post-process-observer", exc, call_id)
        return original_post_process(sol, model, integration_time, inputs_dict)

    def observe_integrate(self, model, t_eval, inputs_list=None, t_interp=None):
        call_id = status["next_call_id"]
        status["next_call_id"] += 1
        frame = {"call_id": call_id, "post_process_entries": 0, "negative_public_returns": 0}
        frames.append(frame)
        try:
            record = {
                "event": "python-integrate-entry",
                "call_id": call_id,
                "ic_scope": "Python values before original _integrate packs C++ inputs; "
                "not C++ post-IC",
            }
            capture(record, "actual_t_eval_argument", lambda: t_eval, call_id, _array)
            capture(
                record,
                "actual_t_interp_argument",
                lambda: t_interp,
                call_id,
                lambda value: None if value is None else _array(value),
            )
            capture(record, "python_pre_cpp_y0full", lambda: model.y0full, call_id, _array)
            capture(record, "python_pre_cpp_ydot0full", lambda: model.ydot0full, call_id, _array)
            capture(record, "inputs_list_argument", lambda: inputs_list, call_id)
            append(record)
        except Exception as exc:
            observer_error("integrate-entry-observer", exc, call_id)
        try:
            result = original_integrate(model, t_eval, inputs_list=inputs_list, t_interp=t_interp)
        except BaseException as exc:
            try:
                append(
                    {
                        "event": "original-integrate-exception",
                        **frame,
                        "public_return_classification": "negative-flag-object-observed"
                        if frame["negative_public_returns"]
                        else "no-public-negative-object",
                        "classification_limit": "no inference of C++ cause or "
                        "unexposed internal state",
                        "exception": _error(exc),
                    }
                )
            except Exception as observation_exc:
                observer_error("integrate-exception-observer", observation_exc, call_id)
            raise
        else:
            append({"event": "original-integrate-returned", **frame})
            return result
        finally:
            frames.pop()

    solver._integrate = MethodType(observe_integrate, solver)
    solver._post_process_solution = MethodType(observe_post_process, solver)
    solver._chemworld_failure_observer_status = status
    return status
