"""Safe arithmetic evaluator for the calculator MCP tool.

Bounded AST walker: rejects expressions whose AST has more than
``MAX_NODES`` nodes, exponents larger than ``MAX_EXPONENT``, or
modulus operands larger than ``MAX_MODULUS``. The previous version
allowed unbounded ``**`` and ``%`` which let an MCP caller DoS the
server thread with ``pow(10, 10**10)``-style payloads. The bounds
below preserve every realistic calculator use (engineering
arithmetic) while killing the obvious abuse vectors.
"""
import ast
import operator as op


OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Mod: op.mod,
}

MAX_NODES = 64
MAX_EXPONENT = 1_000
MAX_MODULUS = 1e12


def _eval(node, depth: int = 0):
    if depth > MAX_NODES:
        raise ValueError("Expression too complex (depth > MAX_NODES)")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Num):  # legacy fallback
        return node.n
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        if isinstance(node.op, ast.Pow):
            exponent = _eval(node.right, depth + 1)
            if abs(exponent) > MAX_EXPONENT:
                raise ValueError("Exponent exceeds calculator bound")
            base = _eval(node.left, depth + 1)
            return OPS[ast.Pow](base, exponent)
        if isinstance(node.op, ast.Mod):
            right = _eval(node.right, depth + 1)
            if abs(right) > MAX_MODULUS:
                raise ValueError("Modulus exceeds calculator bound")
            left = _eval(node.left, depth + 1)
            return OPS[ast.Mod](left, right)
        return OPS[type(node.op)](_eval(node.left, depth + 1), _eval(node.right, depth + 1))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand, depth + 1))
    raise ValueError("Unsupported expression")


def _count_nodes(tree) -> int:
    return sum(1 for _ in ast.walk(tree))


def calculate(expr: str) -> float:
    parsed = ast.parse(expr, mode="eval")
    if _count_nodes(parsed) > MAX_NODES:
        raise ValueError("Expression too complex (node count > MAX_NODES)")
    return float(_eval(parsed.body))
