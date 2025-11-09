
try:
    import pint
except Exception:
    pint = None
def convert(value: float, from_unit: str, to_unit: str) -> float:
    if pint is None:
        if from_unit == to_unit: return float(value)
        raise RuntimeError("pint not installed; cannot convert units.")
    u = pint.UnitRegistry()
    q = value * u(from_unit)
    return float(q.to(to_unit).magnitude)
