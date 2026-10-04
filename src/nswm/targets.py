"""Ordered targets, unknowns, and information-matched supervision formats."""
import json
import math


FIELDS = ("judgment", "premises", "constraint", "entities", "time", "witness", "scope", "repair")


def serialize_target(target, supervision="certificate", removed=()):
    if any(field not in FIELDS or field == "judgment" for field in removed):
        raise ValueError("invalid removed field")
    cleaned = {field: target.get(field) for field in FIELDS}
    for field in removed:
        cleaned[field] = None
    if supervision == "binary":
        cleaned = {"judgment": cleaned["judgment"]}
    elif supervision == "scalar":
        cleaned = {"residual": target.get("scalar_residual")}
        if cleaned["residual"] is None:
            raise ValueError("scalar supervision requires a signed residual target")
    elif supervision == "free_form":
        return "\n".join(f"{key}: {json.dumps(value)}" for key, value in cleaned.items())
    elif supervision != "certificate":
        raise ValueError("unknown supervision format")
    return json.dumps(cleaned, separators=(",", ":"), ensure_ascii=True)


def parse_certificate(text):
    from .schema import Certificate, Edit, Interval, Verdict, commitment_from_dict
    try:
        data = json.loads(text)
        entities, time = data.get("entities"), data.get("time")
        if entities is not None and (not isinstance(entities, list) or any(not isinstance(e, str) for e in entities)):
            raise ValueError("invalid entity IDs")
        if time is not None and (not isinstance(time, list) or len(time) != 2
                or any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in time) or time[0] > time[1]):
            raise ValueError("invalid certificate time")
        if data.get("constraint") is not None and not isinstance(data["constraint"], str):
            raise ValueError("invalid constraint")
        if data.get("repair") is not None and (not isinstance(data["repair"], list) or len(data["repair"]) > 6):
            raise ValueError("invalid repair")
        json.dumps(data, allow_nan=False)
        return Certificate(Verdict(data["judgment"]), data.get("premises"), data.get("constraint"),
                           tuple(data["entities"]) if data.get("entities") is not None else None,
                           tuple(data["time"]) if data.get("time") is not None else None,
                           data.get("witness"), commitment_from_dict(data["scope"]) if data.get("scope") is not None else None,
                           tuple(Edit(e["slot"], Interval(**e["value"])) for e in data["repair"])
                           if data.get("repair") is not None else None)
    except (KeyError, TypeError, ValueError, AttributeError):
        return Certificate()
