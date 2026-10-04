"""Independent interval checks and scoped request cancellation.

Predicted certificate content changes ordering only. Measurements, bounds,
premises, scope and query binding come from an independently built inventory.
The primitives below prove scalar necessary conditions, not arbitrary 3D physics.
"""
from dataclasses import dataclass
from typing import Callable

from .schema import Certificate, Commitment, Interval, Query, Request, Verdict, scope_contains


@dataclass(frozen=True)
class Bound:
    residual: Interval | None
    unit: str
    premises_valid: bool
    covered_scope: Commitment | None
    query_binding: str
    reason: str = ""


@dataclass(frozen=True)
class Check:
    check_id: str
    constraint: str
    entities: tuple[str, ...]
    time: tuple[float, float]
    evaluate: Callable[[], Bound]


@dataclass(frozen=True)
class Verification:
    status: str  # V: veto, S: insufficient scope, N: no conflict, X: unresolved
    checked: tuple[str, ...]
    cancelled: bool
    bound: Bound | None = None


def verify(request: Request, inventory: tuple[Check, ...], certificate: Certificate | None,
           budget: int = 16) -> Verification:
    if budget < 0 or len({c.check_id for c in inventory}) != len(inventory):
        raise ValueError("nonnegative budget and unique inventory IDs required")
    ordered = list(inventory)
    if certificate:
        def priority(check):
            return (check.constraint != certificate.constraint,
                    not bool(set(check.entities) & set(certificate.entities or ())),
                    certificate.time is None or check.time[1] < certificate.time[0]
                    or check.time[0] > certificate.time[1])
        ordered.sort(key=priority)  # stable, preserving deterministic fallback order
    checked = []
    unresolved = False
    partial = None
    for check in ordered[:budget]:
        checked.append(check.check_id)
        bound = check.evaluate()
        if (not bound.premises_valid or bound.query_binding != request.query.binding
                or bound.residual is None or bound.covered_scope is None):
            unresolved = True
            continue
        if bound.residual.lower > 0:
            if scope_contains(bound.covered_scope, request.acceptance):
                return Verification("V", tuple(checked), True, bound)
            partial = bound
        elif bound.residual.upper > 0:
            unresolved = True
    if unresolved or len(checked) < len(ordered):
        return Verification("X", tuple(checked), False, partial)
    return Verification("S" if partial else "N", tuple(checked), False, partial)


def scalar_inventory(query: Query, commitment: Commitment,
                     measurements: dict[str, Interval], rules: list[dict]) -> tuple[Check, ...]:
    """Each rule states a necessary `slot <= measured_limit` condition.

    The caller must establish the limit's physical premises and obtain it without
    candidate future observations. Unknown evidence yields unknown checks.
    """
    inventory = []
    for i, rule in enumerate(rules):
        slot, limit_key = rule["slot"], rule["limit"]
        def evaluate(slot=slot, limit_key=limit_key, rule=dict(rule)):
            value, limit = commitment.slots.get(slot), measurements.get(limit_key)
            residual = value - limit if value is not None and limit is not None else None
            scope = Commitment({slot: value}) if value is not None else None
            return Bound(residual, rule.get("unit", "normalized"),
                         rule.get("premises_valid", False), scope, query.binding,
                         "independent scalar necessary condition")
        inventory.append(Check(str(rule.get("id", i)), rule["constraint"],
                               tuple(rule.get("entities", ())),
                               tuple(rule.get("time", (0.0, float(query.horizon)))), evaluate))
    return tuple(inventory)


def scalar_oracle(query: Query, commitment: Commitment,
                  measurements: dict[str, Interval], rules: list[dict]) -> Verdict:
    """Analytic oracle for the closed scalar fixture domain only.

    `valid` is returned only when this caller-declared rule set is COMPLETE for
    its fixture. It does not establish task-valid robotic continuation.
    """
    if not query.premises.get("complete_scalar_domain", False):
        return Verdict.UNKNOWN
    uncertain = False
    for check in scalar_inventory(query, commitment, measurements, rules):
        b = check.evaluate()
        if b.residual is None or not b.premises_valid:
            uncertain = True
        elif b.residual.lower > 0:
            return Verdict.INVALID
        elif b.residual.upper > 0:
            uncertain = True
    return Verdict.UNKNOWN if uncertain else Verdict.VALID
