
import ast, operator as op
OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
       ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos, ast.Mod: op.mod}
def _eval(node):
    if isinstance(node, ast.Num): return node.n
    if isinstance(node, ast.BinOp) and type(node.op) in OPS: return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS: return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")
def calculate(expr: str) -> float:
    parsed = ast.parse(expr, mode="eval")
    return float(_eval(parsed.body))
