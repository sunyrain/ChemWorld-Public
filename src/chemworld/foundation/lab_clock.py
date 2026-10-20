"""Finite scheduling state; physical time remains solely in the world ledger."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class LabClockSpec:
    heater_id: str | None = None
    analyzer: str | None = None
    capacity: int = 64

    def __post_init__(self):
        if type(self.capacity) is not int or not 1 <= self.capacity <= 256:
            raise ValueError("lab queue capacity must be an integer in [1, 256]")
        for name in (self.heater_id, self.analyzer):
            if name is not None and (not isinstance(name, str) or not name):
                raise ValueError("lab equipment identities must be nonempty strings")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class LabClockState:
    next_sequence: int = 0
    queued: tuple[dict, ...] = ()
    running: tuple[dict, ...] = ()
    completed: tuple[dict, ...] = ()
    detector_rng: dict = field(default_factory=dict)
    # Trusted sampling receipts, appended only by executed acquisitions.
    exchanges: tuple[dict, ...] = ()
    stopped: bool = False

    def __post_init__(self):
        for key in ("queued", "running", "completed", "exchanges"):
            object.__setattr__(self, key, tuple(deepcopy(getattr(self, key))))
        object.__setattr__(self, "detector_rng", deepcopy(self.detector_rng))

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
