
import logging
import re
from typing import List, Tuple

_log = logging.getLogger(__name__)

try:
    import pint
except Exception:
    pint = None

_ALLOWED_UNITS = [
    "K","C","F","Pa","kPa","bar","atm","m3","L","kg","g","mg","mol","kmol","s","min","h",
    "W","kW","MW","J","kJ","MJ","m","cm","mm","um","N","kN","m/s","m2","m3/s","%","ppm"
]

# Documentary text frequently mixes numbers with English unit words
# ("8 degrees Celsius", "12 hours", "5 percent"). pint does not
# recognise the bare English forms, so a strict pint check would reject
# every regulatory / SOP passage in the KB and trip the guard
# aggregation in context_builder. The guard is meant to catch numeric
# formulae with bogus unit symbols (xyzUnit etc.), not to police
# documentary prose, so we whitelist the common English unit phrases
# that appear in the §3.7 corpus. Reviewers verifying this list should
# scan ``agribrain/backend/pirag/knowledge_base/*.txt`` for the actual
# vocabulary used.
_DOCUMENTARY_UNIT_WORDS = {
    "degree", "degrees", "celsius", "fahrenheit", "kelvin",
    "percent", "percentage",
    "hour", "hours", "minute", "minutes", "second", "seconds",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "metre", "metres", "meter", "meters", "kilometre", "kilometres",
    "kilometer", "kilometers", "mile", "miles", "foot", "feet", "inch", "inches",
    "kilogram", "kilograms", "gram", "grams", "tonne", "tonnes", "ton", "tons",
    "litre", "litres", "liter", "liters", "gallon", "gallons",
    "watt", "watts", "kilowatt", "kilowatts",
    "joule", "joules",
    "pound", "pounds", "lb", "lbs",
    "celsius.", "fahrenheit.",  # period-terminated sentence ends
}

def extract_number_units(text: str) -> List[Tuple[float, str]]:
    pat = r"(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([A-Za-z/%]+(?:/[A-Za-z]+)?)"
    out = []
    for m in re.finditer(pat, text):
        try:
            val = float(m.group(1))
            u = m.group(2)
            out.append((val, u))
        except Exception as _exc:
            _log.debug("unit parse skipped for %r: %s", m.group(0), _exc)
    return out

def _is_documentary_word(unit: str) -> bool:
    """True when the unit token matches a common English documentary
    unit word (degrees, percent, hours, etc.). Used to keep the guard
    permissive on regulatory / SOP prose without weakening rejection of
    bogus unit symbols like ``xyzUnit``."""
    base = re.split(r"[/\s]", unit)[0].rstrip(".,;:!?").lower()
    return base in _DOCUMENTARY_UNIT_WORDS


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
                # Documentary text frequently uses English unit words
                # ("degrees", "percent", "hours") that pint does not
                # ship as bare aliases. Treat the pair as benign when
                # the token is on the documentary whitelist; fail only
                # on truly unrecognised symbols.
                if not _is_documentary_word(u):
                    return False
        return True
    else:
        for _, u in pairs:
            base = re.split(r"[/\s]", u)[0]
            if base not in _ALLOWED_UNITS and not _is_documentary_word(u):
                return False
        return True
