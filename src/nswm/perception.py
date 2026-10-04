"""Observed state boxes and split-conformal joint calibration."""
from dataclasses import dataclass
from decimal import Decimal
import math

from .schema import Interval, Query


def joint_error_radius(history_errors, alpha=0.01):
    if not 0 < alpha < 1 or not history_errors:
        raise ValueError("nonempty independent histories and 0 < alpha < 1 required")
    maxima = []
    for errors in history_errors:
        if not errors or not all(math.isfinite(e) and e >= 0 for e in errors):
            raise ValueError("errors must be finite normalized magnitudes")
        maxima.append(max(errors))
    rank = math.ceil((len(maxima) + 1) * (1 - alpha))
    return None if rank > len(maxima) else sorted(maxima)[rank - 1]


@dataclass(frozen=True)
class ObservedPose:
    entity: str
    timestamp: float
    position: tuple[Interval, Interval, Interval] | None
    source: str
    query_binding: str


def pose_box(entity, timestamp, position, radius, query, source="rgbd"):
    if timestamp > 0 or len(position) != 3 or source not in {"rgbd", "calibrated_pose", "oracle"}:
        raise ValueError("metric pose requires a permitted source and past timestamp")
    if radius is None:
        return ObservedPose(entity, timestamp, None, source, query.binding)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("invalid calibration radius")
    r = Interval(str(-Decimal(str(radius))), str(radius))
    intervals = tuple(Interval.point(p) + r for p in position)
    return ObservedPose(entity, timestamp, intervals, source, query.binding)


def propagate_position(pose, velocity, acceleration_bound, seconds, query):
    if pose.query_binding != query.binding or seconds < 0 or acceleration_bound < 0:
        raise ValueError("pose binding or motion bounds invalid")
    if pose.position is None or velocity is None:
        return None
    t = Interval.point(seconds)
    a = Interval.point(acceleration_bound) * t * t * Interval.point("0.5")
    result = []
    for p, v in zip(pose.position, velocity):
        displacement = v * t + Interval(str(-a.upper), str(a.upper))
        result.append(p + displacement)
    return tuple(result)
