"""Finite scheduling state; physical time remains solely in the world ledger."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SharedAdvanceSystem:
    """Assembly references only; stocks remain in the existing world state."""

    system_id: str
    strategy_id: str
    vessels: tuple[str, ...]
    connections: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("system_id", "strategy_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError("shared systems require explicit identities")
        for name in ("vessels", "connections"):
            values = tuple(getattr(self, name))
            if (any(not isinstance(v, str) or not v for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError("shared system references must be distinct nonempty identities")
            object.__setattr__(self, name, tuple(sorted(values)))
        if self.system_id not in self.vessels:
            raise ValueError("shared system anchor must own one of its vessels")


@dataclass(frozen=True)
class LabAnalyzerSpec:
    device_id: str
    instrument: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value
               for value in (self.device_id, self.instrument)):
            raise ValueError("laboratory analyzers require explicit device and instrument IDs")


@dataclass(frozen=True, init=False)
class LabClockSpec:
    heater_ids: tuple[str, ...]
    analyzers: tuple[LabAnalyzerSpec, ...]
    capacity: int = 64
    vessel_capacity: int = 2

    def __init__(self, heater_id=None, analyzer=None, capacity=64, vessel_capacity=2,
                 *, heater_ids=None, analyzers=None):
        if heater_id is not None and heater_ids:
            raise ValueError("declare heater identities once")
        if analyzer is not None and analyzers:
            raise ValueError("declare analyzer identities once")
        if heater_id is not None or heater_ids is None:
            heater_ids = () if heater_id is None else (heater_id,)
        if analyzer is not None or analyzers is None:
            from chemworld.foundation.state_helpers import instrument_equipment_id

            analyzers = () if analyzer is None else (
                LabAnalyzerSpec(instrument_equipment_id(analyzer), analyzer),
            )
        elif isinstance(analyzers, Mapping):
            analyzers = tuple(LabAnalyzerSpec(key, value) for key, value in analyzers.items())
        analyzers = tuple(
            value if isinstance(value, LabAnalyzerSpec) else LabAnalyzerSpec(**value)
            for value in analyzers
        )
        if isinstance(heater_ids, str | bytes):
            raise ValueError("heater identities must be an explicit finite collection")
        heater_ids = tuple(heater_ids)
        if (any(not isinstance(value, str) or not value for value in heater_ids)
                or len(set(heater_ids)) != len(heater_ids) or len(heater_ids) > 64):
            raise ValueError("declare at most 64 distinct nonempty heater identities")
        identities = tuple(value.device_id for value in analyzers)
        if (len(set(identities)) != len(identities) or set(identities) & set(heater_ids)
                or len(identities) > 64):
            raise ValueError("declare at most 64 distinct analyzers without device aliases")
        object.__setattr__(self, "heater_ids", tuple(sorted(heater_ids)))
        object.__setattr__(self, "analyzers", tuple(sorted(analyzers, key=lambda a: a.device_id)))
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "vessel_capacity", vessel_capacity)
        self.__post_init__()

    def __post_init__(self):
        if type(self.capacity) is not int or not 1 <= self.capacity <= 256:
            raise ValueError("lab queue capacity must be an integer in [1, 256]")
        if type(self.vessel_capacity) is not int or not 1 <= self.vessel_capacity <= 64:
            raise ValueError("lab vessel capacity must be an integer in [1, 64]")

    @property
    def heater_id(self):
        return self.heater_ids[0] if len(self.heater_ids) == 1 else None

    @property
    def analyzer(self):
        return self.analyzers[0].instrument if len(self.analyzers) == 1 else None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class LabClockState:
    next_sequence: int = 0
    queued: tuple[dict, ...] = ()
    running: tuple[dict, ...] = ()
    completed: tuple[dict, ...] = ()
    detector_rng: dict = field(default_factory=dict)
    detector_noise: dict = field(default_factory=dict)
    # Trusted sampling receipts, appended only by executed acquisitions.
    exchanges: tuple[dict, ...] = ()
    stopped: bool = False

    def __post_init__(self):
        for key in ("queued", "running", "completed", "exchanges"):
            object.__setattr__(self, key, tuple(deepcopy(getattr(self, key))))
        object.__setattr__(self, "detector_rng", deepcopy(self.detector_rng))
        object.__setattr__(self, "detector_noise", deepcopy(self.detector_noise))

    def to_dict(self, *, include_hidden=True):
        if include_hidden:
            return asdict(self)
        return {
            "mode": "shared",
            "stopped": self.stopped,
            "queued": [self.public_job(job) for job in self.queued],
            "running": [self.public_job(job) for job in self.running],
            "completed": deepcopy(list(self.completed)),
        }

    @staticmethod
    def public_job(job):
        private = {"observation", "measurement_status", "readout_error"}
        return {k: deepcopy(v) for k, v in job.items() if k not in private}
