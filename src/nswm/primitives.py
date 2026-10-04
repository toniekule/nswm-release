"""Conservative physical necessary-condition primitives."""
from decimal import Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR

from .schema import Commitment, Interval
from .verify import Bound


def unresolved(query, unit, reason):
    return Bound(None, unit, False, None, query.binding, reason)


def separation(left, right):
    a, b = left - right, right - left
    return max(a.lower, b.lower)


def sphere_penetration(query, commitment, a, b, radius_a, radius_b,
                       tolerance=Interval.point("0.001"), premises_valid=False):
    slots = [f"{entity}.{axis}" for entity in (a, b) for axis in "xyz"]
    if not premises_valid or any(slot not in commitment.slots for slot in slots):
        return unresolved(query, "m", "shape or position coverage missing")
    differences = [commitment.slots[f"{a}.{axis}"] - commitment.slots[f"{b}.{axis}"] for axis in "xyz"]
    squared = Interval.point(0)
    for d in differences:
        minimum = 0 if d.lower <= 0 <= d.upper else min(d.lower.copy_abs(), d.upper.copy_abs())
        maximum = max(d.lower.copy_abs(), d.upper.copy_abs())
        square_lo = Interval.point(minimum) * Interval.point(minimum)
        square_hi = Interval.point(maximum) * Interval.point(maximum)
        squared = squared + Interval(str(square_lo.lower), str(square_hi.upper))
    with localcontext() as ctx:
        ctx.prec = 50
        ctx.rounding = ROUND_FLOOR
        distance_lower = max(Decimal(0), squared.lower.sqrt().next_minus())
        ctx.rounding = ROUND_CEILING
        distance_upper = squared.upper.sqrt().next_plus()
    residual = radius_a + radius_b - Interval(str(distance_lower), str(distance_upper)) - tolerance
    scope = Commitment({s: commitment.slots[s] for s in slots})
    return Bound(residual, "m", True, scope, query.binding, "inscribed-sphere overlap bound")


def minimum_separation(query, commitment, slot, radius_a, radius_b,
                       tolerance=Interval.point("0.001"), premises_valid=False):
    if not premises_valid or slot not in commitment.slots:
        return unresolved(query, "m", "minimum pair-distance coverage missing")
    distance = commitment.slots[slot]
    if distance.lower < 0:
        return unresolved(query, "m", "distance bound cannot be negative")
    return Bound(radius_a + radius_b - distance - tolerance, "m", True,
                 Commitment({slot: distance}), query.binding, "sampled sphere-separation lower bound")


def conservation(query, commitment, slot, initial, exchange,
                 unit="kg*m/s", premises_valid=False):
    if not premises_valid or slot not in commitment.slots or initial is None or exchange is None:
        return unresolved(query, unit, "external exchange is unbounded")
    permitted = initial + exchange
    gap = separation(commitment.slots[slot], permitted)
    return Bound(Interval.point(gap), unit, True, Commitment({slot: commitment.slots[slot]}),
                 query.binding, "required final quantity is disjoint from permitted exchanges")


def reachable_box(query, commitment, outer_box, constraint="resource_reachability", premises_valid=False):
    if not premises_valid or not outer_box:
        return unresolved(query, "m", "reachable outer bound missing")
    gaps = [(separation(commitment.slots[k], v), k) for k, v in outer_box.items() if k in commitment.slots]
    if not gaps:
        return unresolved(query, "m", "no covered commitment coordinate")
    gap, slot = max(gaps)
    return Bound(Interval.point(gap), "m", True, Commitment({slot: commitment.slots[slot]}),
                 query.binding, constraint)


def persistence(query, commitment, identity_slot, identity_count=Interval.point(1), premises_valid=False):
    if not premises_valid or identity_slot not in commitment.slots:
        return unresolved(query, "count", "persistent identity premise missing")
    gap = separation(commitment.slots[identity_slot], identity_count)
    return Bound(Interval.point(gap), "count", True, Commitment({identity_slot: commitment.slots[identity_slot]}),
                 query.binding, "persistent identity count constraint")


def precedence(query, commitment, edges, premises_valid=False):
    events = sorted({name for before, after, delay in edges for name in (before, after)})
    if not premises_valid or not events or any(k not in commitment.slots for k in events):
        return unresolved(query, "s", "event coverage missing")
    if any(delay.lower < 0 for _, _, delay in edges):
        return unresolved(query, "s", "predecessor delays must be nonnegative")
    lower = {k: commitment.slots[k].lower for k in events}
    gap = Decimal(0)
    for iteration in range(len(events)):
        changed = False
        for before, after, delay in edges:
            proposed = (Interval.point(lower[before]) + Interval.point(delay.lower)).lower
            if proposed > lower[after]:
                lower[after] = proposed
                changed = True
            gap = max(gap, (Interval.point(lower[after]) - Interval.point(commitment.slots[after].upper)).lower)
        if gap > 0:
            break
        if not changed:
            break
        if iteration == len(events) - 1:
            # Strictly positive cycles eventually violate any finite time box.
            for before, after, delay in edges:
                candidate = (Interval.point(lower[before]) + Interval.point(delay.lower)).lower
                if candidate > lower[after]:
                    gap = max(gap, (Interval.point(candidate) - Interval.point(lower[after])).lower)
    scope = Commitment({k: commitment.slots[k] for k in events})
    return Bound(Interval.point(gap), "s", True, scope, query.binding, "precedence feasibility")
