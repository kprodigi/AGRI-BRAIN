
import re
from typing import List, Tuple

try:
    import pint
except Exception:
    pint = None

_ALLOWED_UNITS = [
    "K","C","F","Pa","kPa","bar","atm","m3","L","kg","g","mg","mol","kmol","s","min","h",
    "W","kW","MW","J","kJ","MJ","m","cm","mm","um","N","kN","m/s","m2","m3/s","%","ppm"
]

def extract_number_units(text: str) -> List[Tuple[float, str]]:
    pat = r"(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([A-Za-z/%]+(?:/[A-Za-z]+)?)"
    out = []
    for m in re.finditer(pat, text):
        try:
            val = float(m.group(1))
            u = m.group(2)
            out.append((val, u))
        except Exception:
            pass
    return out

def units_consistent(answer: str) -> bool:
    pairs = extract_number_units(answer)
    if not pairs:
        return True
    if pint is not None:
        ureg = pint.UnitRegistry()
        for _, u in pairs:
            try:
                _ = 1 * ureg(u)
            except Exception:
                return False
        return True
    else:
        for _, u in pairs:
            base = re.split(r"[/\s]", u)[0]
            if base not in _ALLOWED_UNITS:
                return False
        return True
