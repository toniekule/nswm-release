"""Typed commitments and request identities. All ranges are closed intervals."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from enum import Enum
import hashlib
import json
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Verdict(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class FrozenDict(dict):
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def _deny(self, *args, **kwargs):
        raise TypeError("mapping is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _deny
    __ior__ = _deny


def freeze(value):
    if isinstance(value, dict):
        return FrozenDict((k, freeze(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class Interval:
    lo: str
    hi: str

    def __post_init__(self):
        object.__setattr__(self, "lo", str(self.lo))
        object.__setattr__(self, "hi", str(self.hi))
        if not self.lower.is_finite() or not self.upper.is_finite() or self.lower > self.upper:
            raise ValueError("interval endpoints must be finite and ordered")

    @property
    def lower(self) -> Decimal:
        return Decimal(self.lo)

    @property
    def upper(self) -> Decimal:
        return Decimal(self.hi)

    @classmethod
    def point(cls, value: Any) -> Interval:
        return cls(str(value), str(value))

    def contains(self, other: Interval) -> bool:
        return self.lower <= other.lower and other.upper <= self.upper

    def __add__(self, other: Interval) -> Interval:
        with localcontext() as ctx:
            ctx.prec = 50
            ctx.rounding = ROUND_FLOOR
            lo = self.lower + other.lower
            ctx.rounding = ROUND_CEILING
            hi = self.upper + other.upper
        return Interval(str(lo), str(hi))

    def __sub__(self, other: Interval) -> Interval:
        with localcontext() as ctx:
            ctx.prec = 50
            ctx.rounding = ROUND_FLOOR
            lo = self.lower - other.upper
            ctx.rounding = ROUND_CEILING
            hi = self.upper - other.lower
        return Interval(str(lo), str(hi))

    def __mul__(self, other: Interval) -> Interval:
        with localcontext() as ctx:
            ctx.prec = 50
            ctx.rounding = ROUND_FLOOR
            lo = min(a * b for a in (self.lower, self.upper) for b in (other.lower, other.upper))
            ctx.rounding = ROUND_CEILING
            hi = max(a * b for a in (self.lower, self.upper) for b in (other.lower, other.upper))
        return Interval(str(lo), str(hi))


@dataclass(frozen=True)
class Commitment:
    slots: dict[str, Interval]

    def __post_init__(self):
        if not self.slots or not all(isinstance(k, str) and k and isinstance(v, Interval)
                                     for k, v in self.slots.items()):
            raise ValueError("commitment requires named interval slots")
        object.__setattr__(self, "slots", freeze(self.slots))

    def edited(self, edits: tuple[Edit, ...]) -> Commitment:
        slots = dict(self.slots)
        seen = set()
        for edit in edits:
            if edit.slot not in slots or edit.slot in seen:
                raise ValueError("edits require existing, distinct slot IDs")
            slots[edit.slot] = edit.value
            seen.add(edit.slot)
        return Commitment(slots)


@dataclass(frozen=True)
class Edit:
    slot: str
    value: Interval

    def __post_init__(self):
        if not isinstance(self.slot, str) or not self.slot or not isinstance(self.value, Interval):
            raise ValueError("edit requires a slot ID and interval")


@dataclass(frozen=True)
class Query:
    history_id: str
    actions: tuple[tuple[float, ...], ...]
    premises: dict[str, Any]
    horizon: int

    def __post_init__(self):
        if not self.history_id or self.horizon <= 0 or not self.actions:
            raise ValueError("query needs history, actions, and positive horizon")
        object.__setattr__(self, "actions", freeze(self.actions))
        object.__setattr__(self, "premises", freeze(self.premises))
        canonical(asdict(self))

    @property
    def binding(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class Request:
    request_id: str
    query: Query
    commitment: Commitment
    acceptance: Commitment
    parent_id: str = ""

    def __post_init__(self):
        if not self.request_id or not scope_contains(self.commitment, self.acceptance):
            raise ValueError("request acceptance must imply its commitment")

    @property
    def binding(self) -> str:
        return fingerprint(asdict(self))


def scope_contains(scope: Commitment, target: Commitment) -> bool:
    """Every assertion in scope must hold for every member of target.

    Target may constrain additional dimensions. A missing target dimension means
    unconstrained, never zero or a default value. Finite conjunctions only.
    """
    return all(k in target.slots and v.contains(target.slots[k]) for k, v in scope.slots.items())


@dataclass(frozen=True)
class Certificate:
    judgment: Verdict = Verdict.UNKNOWN
    premises: dict[str, Any] | None = None
    constraint: str | None = None
    entities: tuple[str, ...] | None = None
    time: tuple[float, float] | None = None
    witness: dict[str, Any] | None = None
    scope: Commitment | None = None
    repair: tuple[Edit, ...] | None = None

    def target(self) -> dict[str, Any]:
        # The insertion order is the paper's autoregressive field order.
        return {"judgment": self.judgment.value, "premises": self.premises,
                "constraint": self.constraint, "entities": self.entities,
                "time": self.time, "witness": self.witness,
                "scope": asdict(self.scope) if self.scope else None,
                "repair": [asdict(e) for e in self.repair] if self.repair is not None else None}


def commitment_from_dict(value: dict[str, Any]) -> Commitment:
    return Commitment({k: Interval(**v) for k, v in value["slots"].items()})
