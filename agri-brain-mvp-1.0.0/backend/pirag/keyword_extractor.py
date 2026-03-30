"""Extract actionable keywords, thresholds, and regulatory terms from piRAG passages.

Three types of keywords:
1. Regulatory references: "FSMA Section 204", "EU Directive 2008/98/EC"
2. Actionable thresholds: "5°C limit", "2-hour window", "rho < 0.30"
3. Required actions: "must be stored below 5°C", "corrective action within 1 hour"
"""
from __future__ import annotations

import re
from typing import Dict, List


def extract_keywords(passage: str) -> List[str]:
    """Extract actionable keywords from a single piRAG passage.

    Returns a list of 3-8 keyword phrases, ordered by specificity
    (thresholds first, then actions, then regulatory references).
    """
    if not passage:
        return []

    keywords: List[str] = []

    # 1. Temperature thresholds
    for m in re.finditer(
        r'(\d+(?:\.\d+)?)\s*(?:degrees?\s*(?:Celsius|C|Fahrenheit|F)|°C|°F)',
        passage, re.IGNORECASE,
    ):
        phrase = _surrounding_phrase(passage, m.start(), m.end())
        if phrase and phrase not in keywords:
            keywords.append(phrase)

    # 2. Time thresholds
    for m in re.finditer(
        r'(?:within\s+)?(\d+)\s*[-\s]?(?:hours?|minutes?|days?)\b',
        passage, re.IGNORECASE,
    ):
        phrase = _surrounding_phrase(passage, m.start(), m.end())
        if phrase and phrase not in keywords:
            keywords.append(phrase)

    # 3. Rho / spoilage thresholds
    for m in re.finditer(
        r'(?:rho|spoilage[\s_]risk)\s*[<>=]+\s*\d+(?:\.\d+)?',
        passage, re.IGNORECASE,
    ):
        if m.group() not in keywords:
            keywords.append(m.group())

    # 4. Percentage thresholds
    for m in re.finditer(r'\d+(?:\.\d+)?\s*(?:percent|%)', passage, re.IGNORECASE):
        phrase = _surrounding_phrase(passage, m.start(), m.end())
        if phrase and phrase not in keywords:
            keywords.append(phrase)

    # 5. Regulatory references
    for pattern in [
        r'FSMA\s+(?:Section\s+)?\d+[\w]*',
        r'FDA\s+(?:Food\s+Safety\s+)?(?:Modernization\s+Act|FSMA)',
        r'EU\s+(?:Waste\s+)?(?:Framework\s+)?Directive\s+[\d/]+\w*',
        r'UNEP[/]SETAC',
        r'GHG\s+Protocol',
        r'EPA\s+(?:40\s+CFR|GHG)',
        r'NIST\s+Cybersecurity\s+Framework',
        r'ISO\s+\d+',
        r'AAFCO\s+\w+',
        r'DOT\s+Hours\s+of\s+Service',
        r'ILO\s+\w+\s*\w*',
        r'EIP[-\s]?\d+',
        r'FEMA\s+\w+',
    ]:
        for m in re.findall(pattern, passage, re.IGNORECASE):
            cleaned = m.strip()
            if cleaned and cleaned not in keywords:
                keywords.append(cleaned)

    # 6. Required actions ("must", "shall", "requires")
    for m in re.finditer(
        r'(?:must|shall|should|requires?)\s+(?:be\s+)?(\w+(?:\s+\w+){1,5}?)(?=\.|,|;|$)',
        passage, re.IGNORECASE,
    ):
        phrase = m.group(0).rstrip('.,;').strip()
        if len(phrase) > 10 and phrase not in keywords:
            keywords.append(phrase)

    # Deduplicate and limit
    seen: set = set()
    unique: List[str] = []
    for kw in keywords:
        normalized = kw.lower().strip()
        if normalized not in seen and len(normalized) > 3:
            seen.add(normalized)
            unique.append(kw)

    return unique[:8]


def extract_keywords_by_type(passage: str) -> Dict[str, List[str]]:
    """Extract keywords categorized by type.

    Returns:
        {
            "thresholds": ["5°C", "2 hours", ...],
            "regulations": ["FSMA Section 204", ...],
            "required_actions": ["must be stored below 5°C", ...],
        }
    """
    result: Dict[str, List[str]] = {
        "thresholds": [],
        "regulations": [],
        "required_actions": [],
    }

    if not passage:
        return result

    # Thresholds
    for m in re.findall(
        r'(\d+(?:\.\d+)?)\s*(?:degrees?\s*(?:Celsius|C)|°C)',
        passage, re.IGNORECASE,
    ):
        result["thresholds"].append(f"{m}°C")

    for val, unit in re.findall(
        r'(?:within\s+)?(\d+)\s+(hours?|minutes?|days?)',
        passage, re.IGNORECASE,
    ):
        result["thresholds"].append(f"{val} {unit}")

    for m in re.findall(
        r'(?:rho|spoilage[\s_]risk)\s*([<>=]+\s*\d+(?:\.\d+)?)',
        passage, re.IGNORECASE,
    ):
        result["thresholds"].append(f"rho {m}")

    # Regulations
    for pattern in [
        r'FSMA\s+(?:Section\s+)?\d+[\w]*',
        r'EU\s+(?:Waste\s+)?(?:Framework\s+)?Directive\s+[\d/]+\w*',
        r'FDA\s+\w+(?:\s+\w+){1,3}\s+Rule',
        r'UNEP/SETAC',
        r'EPA\s+(?:40\s+CFR|GHG)\s*\w*',
        r'ISO\s+\d+',
        r'NIST\s+\w+\s+Framework',
    ]:
        for m in re.findall(pattern, passage, re.IGNORECASE):
            result["regulations"].append(m.strip())

    # Required actions
    for pattern in [
        r'(?:must|shall)\s+(?:be\s+)?(\w+(?:\s+\w+){1,4}?)(?=\.|,|;)',
        r'(?:within|before)\s+(\d+\s+\w+)(?:\s+of|\s+after)',
        r'(?:requires?|mandatory)\s+(\w+(?:\s+\w+){1,4}?)(?=\.|,|;)',
    ]:
        for m in re.findall(pattern, passage, re.IGNORECASE):
            if len(m) > 5:
                result["required_actions"].append(m.strip())

    # Deduplicate each category
    for key in result:
        result[key] = list(dict.fromkeys(result[key]))[:5]

    return result


def _surrounding_phrase(text: str, start: int, end: int, radius: int = 40) -> str:
    """Extract surrounding phrase context for a match."""
    # Find sentence boundaries
    sent_start = text.rfind('.', 0, start)
    sent_start = sent_start + 1 if sent_start >= 0 else 0
    sent_end = text.find('.', end)
    sent_end = sent_end if sent_end >= 0 else len(text)

    phrase = text[sent_start:sent_end].strip()
    if len(phrase) > 80:
        # Truncate to match + radius
        p_start = max(sent_start, start - radius)
        p_end = min(sent_end, end + radius)
        phrase = text[p_start:p_end].strip()

    return phrase if len(phrase) > 5 else ""
