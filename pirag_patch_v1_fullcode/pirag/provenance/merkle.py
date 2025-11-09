
import hashlib
from typing import List
def _h(x: bytes) -> bytes:
    return hashlib.sha256(x).digest()
def merkle_root(hashes_hex: List[str]) -> str:
    if not hashes_hex:
        return ""
    nodes = [bytes.fromhex(h) for h in hashes_hex]
    while len(nodes) > 1:
        nxt = []
        it = iter(nodes)
        for a in it:
            try:
                b = next(it)
            except StopIteration:
                b = a
            nxt.append(_h(a + b))
        nodes = nxt
    return nodes[0].hex()
