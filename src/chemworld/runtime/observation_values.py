"""Public scalar helpers; native readings retain their physical channels."""

from chemworld.foundation.observations import Observation
from chemworld.world.operations import PUBLIC_OBSERVATION_KEYS


def value_or_default(values: dict[str, float | None], key: str, default: float = 0.0) -> float:
    value = values.get(key)
    return default if value is None else float(value)


def empty_reading() -> Observation:
    return Observation(values={
        key: 0.0 if key in {"cost", "safety_risk", "score"} else None
        for key in PUBLIC_OBSERVATION_KEYS
    })
