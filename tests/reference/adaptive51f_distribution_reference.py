"""Independent direct sampling and closed linear flow; deliberately no native imports."""

import math

import numpy as np
from scipy.linalg import expm


class ReferenceKernel:
    def __init__(self, unit):
        self.unit = unit
        self.keys = tuple(unit["initial"])
        self.reactions = unit["reactions"]
        self.changes = np.array(
            [[r["changes"].get(k, 0) for r in self.reactions] for k in self.keys], dtype=float
        )
        self.participants = [
            set(r["orders"]) | {k for k, n in r["changes"].items() if n} for r in self.reactions
        ]
        self.matrices = {}
        self.full_flows = {}

    def classify(self, values, previous=None):
        numbers = dict(zip(self.keys, values, strict=True))
        if previous is None:
            middle = (self.unit["lower"] + self.unit["upper"]) / 2
            return {k for k, v in numbers.items() if v < middle}
        low = set(previous)
        for k, v in numbers.items():
            if v < self.unit["lower"]:
                low.add(k)
            elif v > self.unit["upper"]:
                low.discard(k)
        return low

    def rates(self, values, active):
        numbers = dict(zip(self.keys, values, strict=True))
        rates = []
        for enabled, r in zip(active, self.reactions, strict=True):
            if not enabled or any(numbers[k] < -n for k, n in r["changes"].items() if n < 0):
                rates.append(0.0)
                continue
            factors = [numbers[k] - i for k, n in r["orders"].items() for i in range(n)]
            rate = r["coefficient"] * math.prod(factors) if all(v >= 0 for v in factors) else 0.0
            if not math.isfinite(rate) or rate < 0:
                raise ArithmeticError("reference propensity is invalid")
            rates.append(rate)
        return np.array(rates)

    def flow(self, values, active, dt):
        if all(active) or dt == 0:
            return values.copy()
        mask = tuple(active)
        if mask not in self.matrices:
            matrix = np.zeros((len(self.keys), len(self.keys)))
            for j, r in enumerate(self.reactions):
                if active[j]:
                    continue
                if sum(r["orders"].values()) != 1:
                    raise ValueError("independent flow only supports declared linear ODE subnets")
                key = next(iter(r["orders"]))
                matrix[:, self.keys.index(key)] += self.changes[:, j] * r["coefficient"]
            self.matrices[mask] = matrix
        matrix = self.matrices[mask]
        if self.unit["method"] == "RK4":
            # Closed matrix stability polynomial, independent of native RK stages.
            answer, term = values.copy(), values.copy()
            for order in range(1, 5):
                term = (matrix @ term) * (dt / order)
                answer += term
            return answer
        if dt == self.unit["step_s"]:
            if mask not in self.full_flows:
                self.full_flows[mask] = expm(matrix * dt)
            return self.full_flows[mask] @ values
        return expm(matrix * dt) @ values

    def simulate(self, seed):
        rng = np.random.Generator(np.random.Philox(seed))
        values = np.array(list(self.unit["initial"].values()), dtype=float)
        low = self.classify(values)
        now, grid, events, segments, flows, draws = 0.0, 0, 0, 0, 0, 0
        changes = []
        target = self.unit["duration_s"]
        while now < target:
            active = [bool(p & low) for p in self.participants]
            rates = self.rates(values, active)
            total = float(sum(rates))
            if total:
                wait = float(rng.exponential(1 / total))
                draws += 1
            else:
                wait = math.inf
            event_at = now + wait
            grid_at = (grid + 1) * self.unit["step_s"]
            end = min(event_at, grid_at, target)
            if end <= now:
                raise ArithmeticError("reference cannot resolve the next clock")
            if segments >= 1000000:
                raise RuntimeError("reference segment budget exhausted")
            if not all(active):
                values = self.flow(values, active, end - now)
                flows += 1
            now = end
            segments += 1
            if event_at == now:
                if events >= 100000:
                    raise RuntimeError("reference event budget exhausted")
                draw = float(rng.random()) * total
                draws += 1
                channel = int(np.searchsorted(np.cumsum(rates), draw, side="right"))
                values = values + self.changes[:, channel]
                events += 1
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ArithmeticError("reference candidate is negative/nonfinite; no clipping")
            if now == grid_at:
                grid += 1
                after = self.classify(values, low)
                if after != low:
                    changes.append({"time_s": now, "before": sorted(low), "after": sorted(after)})
                low = after
        return {
            "populations": dict(zip(self.keys, map(float, values), strict=True)),
            "time_s": now,
            "events": events,
            "partition_changes": changes,
            "work": {"segments": segments, "flow_calls": flows, "random_draws": draws},
        }
