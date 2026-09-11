"""What the policy can actually do, measured -- the table both the compiler and the model read.

Two kinds of entry, keyed like the compiler's calibration keys:

* ``move:<dir>:<speed>`` / ``turn:<dir>:<speed>``: how far the robot really goes. Each attempt
  records the commanded duration and the measured length along the commanded axis (metres for a
  move, radians for a turn). :meth:`fit` turns those into ``length = a * t + b``.
* ``flip:<kind>`` / ``stance:<kind>``: attempts, successes and falls. A stance also records what
  fraction of its hold the robot actually spent up.

The table is the ground truth for three consumers: the compiler (distance -> duration), the dataset
generator (which skills to sample, which to teach the model to decline) and the reply writer (what
to promise). It is written by ``scripts/llm/validate_programs.py --calibrate`` and updated by every
validation run after that.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class MotionEntry:
    """A move or turn skill: the sufficient statistics of (duration, measured length) samples.

    Sums rather than the samples themselves, so the table stays a few kilobytes however many
    validation runs feed it, and two tables merge by adding fields.
    """

    commanded_rate: float
    """Nominal speed the compiler commanded, m/s or rad/s."""
    n: int = 0
    sum_t: float = 0.0
    sum_l: float = 0.0
    sum_tt: float = 0.0
    sum_tl: float = 0.0
    attempts: int = 0
    falls: int = 0

    def add(self, duration: float, length: float) -> None:
        self.n += 1
        self.sum_t += duration
        self.sum_l += length
        self.sum_tt += duration * duration
        self.sum_tl += duration * length

    def fit(self) -> tuple[float, float]:
        """Least-squares ``length = a * t + b``; one distinct duration gives a through-origin rate."""
        if self.n == 0:
            return self.commanded_rate, 0.0
        mean_t = self.sum_t / self.n
        mean_l = self.sum_l / self.n
        var_t = self.sum_tt / self.n - mean_t * mean_t
        if var_t < 1e-9:
            return (mean_l / mean_t if mean_t > 0 else self.commanded_rate), 0.0
        cov = self.sum_tl / self.n - mean_t * mean_l
        a = cov / var_t
        return a, mean_l - a * mean_t

    @property
    def measured_rate(self) -> float:
        return self.fit()[0]

    def to_dict(self) -> dict[str, Any]:
        a, b = self.fit()
        return {
            "type": "motion",
            "commanded_rate": self.commanded_rate,
            "fit_a": round(a, 4),
            "fit_b": round(b, 4),
            "ratio": round(a / self.commanded_rate, 3) if self.commanded_rate else None,
            "attempts": self.attempts,
            "falls": self.falls,
            "fall_rate": round(self.falls / self.attempts, 3) if self.attempts else None,
            "n": self.n,
            "sum_t": self.sum_t,
            "sum_l": self.sum_l,
            "sum_tt": self.sum_tt,
            "sum_tl": self.sum_tl,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotionEntry:
        entry = cls(commanded_rate=data["commanded_rate"], attempts=data.get("attempts", 0), falls=data.get("falls", 0))
        if "samples" in data:  # the format before sufficient statistics
            for t, length in data["samples"]:
                entry.add(t, length)
        else:
            for name in ("n", "sum_t", "sum_l", "sum_tt", "sum_tl"):
                setattr(entry, name, data.get(name, 0))
        return entry


@dataclass
class EventEntry:
    """A flip or stance skill: did it work."""

    attempts: int = 0
    successes: int = 0
    falls: int = 0
    hold_fractions: list[float] = field(default_factory=list)
    """Stances only: share of the commanded hold actually spent in the stance."""
    recover_times: list[float | None] = field(default_factory=list)
    """Stances only: seconds from release until the trunk was level again; ``None`` if it never was."""

    @property
    def rate(self) -> float | None:
        return self.successes / self.attempts if self.attempts else None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": "event",
            "attempts": self.attempts,
            "successes": self.successes,
            "falls": self.falls,
            "rate": round(self.rate, 3) if self.rate is not None else None,
            "fall_rate": round(self.falls / self.attempts, 3) if self.attempts else None,
        }
        if self.hold_fractions:
            out["hold_fraction_mean"] = round(sum(self.hold_fractions) / len(self.hold_fractions), 3)
        if self.recover_times:
            recovered = [t for t in self.recover_times if t is not None]
            out["recover_rate"] = round(len(recovered) / len(self.recover_times), 3)
            out["recover_s_mean"] = round(sum(recovered) / len(recovered), 3) if recovered else None
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventEntry:
        entry = cls(attempts=data.get("attempts", 0), successes=data.get("successes", 0), falls=data.get("falls", 0))
        if "hold_fraction_mean" in data and entry.attempts:
            # The list is not stored; keep the mean as one weighted sample so merges stay sane.
            entry.hold_fractions = [data["hold_fraction_mean"]] * entry.attempts
        return entry


@dataclass
class CapabilityTable:
    entries: dict[str, MotionEntry | EventEntry] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- recording -----------------------------------------------------------------------------

    def record_motion(self, key: str, commanded_rate: float, duration: float, length: float, fell: bool) -> None:
        entry = self.entries.get(key)
        if not isinstance(entry, MotionEntry):
            entry = MotionEntry(commanded_rate=commanded_rate)
            self.entries[key] = entry
        entry.attempts += 1
        if fell:
            entry.falls += 1
        else:
            entry.add(duration, length)

    def record_event(
        self, key: str, success: bool, fell: bool, hold_fraction: float | None = None, recover_s: float | None = ...
    ) -> None:
        entry = self.entries.get(key)
        if not isinstance(entry, EventEntry):
            entry = EventEntry()
            self.entries[key] = entry
        entry.attempts += 1
        entry.successes += int(success)
        entry.falls += int(fell)
        if hold_fraction is not None:
            entry.hold_fractions.append(hold_fraction)
        if recover_s is not ...:
            entry.recover_times.append(recover_s)

    # -- reading -------------------------------------------------------------------------------

    def calibration(self) -> dict[str, tuple[float, float]]:
        """The ``key -> (a, b)`` mapping the compiler consumes."""
        return {key: entry.fit() for key, entry in self.entries.items() if isinstance(entry, MotionEntry)}

    def rate(self, key: str) -> float | None:
        entry = self.entries.get(key)
        return entry.rate if isinstance(entry, EventEntry) else None

    def unreliable(self, threshold: float = 0.8) -> list[str]:
        """Event skills whose measured success rate is below ``threshold`` (with at least one attempt)."""
        return sorted(
            key for key, entry in self.entries.items()
            if isinstance(entry, EventEntry) and entry.rate is not None and entry.rate < threshold
        )

    def summary(self) -> str:
        lines = []
        for key in sorted(self.entries):
            entry = self.entries[key]
            if isinstance(entry, MotionEntry):
                a, b = entry.fit()
                lines.append(
                    f"{key:<30} rate {a:6.3f} (cmd {entry.commanded_rate:.2f}, x{a / entry.commanded_rate:.2f})"
                    f"  offset {b:+.3f}  n={entry.n}  falls={entry.falls}/{entry.attempts}"
                )
            else:
                rate = "  n/a" if entry.rate is None else f"{entry.rate:5.1%}"
                hold = ""
                if entry.hold_fractions:
                    hold = f"  hold {sum(entry.hold_fractions) / len(entry.hold_fractions):.2f}"
                if entry.recover_times:
                    recovered = [t for t in entry.recover_times if t is not None]
                    mean = f"{sum(recovered) / len(recovered):.2f}s" if recovered else "never"
                    hold += f"  recover {len(recovered)}/{len(entry.recover_times)} mean {mean}"
                lines.append(f"{key:<30} success {rate}  ({entry.successes}/{entry.attempts})  falls={entry.falls}{hold}")
        return "\n".join(lines)

    # -- persistence ---------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": {**self.meta, "written": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            "skills": {key: entry.to_dict() for key, entry in sorted(self.entries.items())},
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> CapabilityTable:
        data = json.loads(Path(path).read_text())
        table = cls(meta=data.get("meta", {}))
        for key, entry in data.get("skills", {}).items():
            if entry.get("type") == "motion":
                table.entries[key] = MotionEntry.from_dict(entry)
            else:
                table.entries[key] = EventEntry.from_dict(entry)
        return table

    @classmethod
    def load_or_empty(cls, path: str | Path | None) -> CapabilityTable:
        if path is not None and Path(path).exists():
            return cls.load(path)
        return cls()
