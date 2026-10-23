"""
calculator.py
-------------
Safe mathematical expression evaluator for the Sovereign AI Workbench.

Security guarantees:
  - Strictly uses Python's AST in mode='eval' — NEVER calls eval() or exec().
  - Allows ONLY basic arithmetic (+, -, *, /, //, %), unary signs, parentheses,
    and numeric comparisons (==, !=, <, <=, >, >=).
  - Explicitly rejects function calls, attribute access, names/variables, imports,
    indexing, comprehensions, and statements.
  - Enforces strict length limits (<= 200 chars), AST node limits (<= 50 nodes),
    magnitude limits (|result| <= 1e15), and division by zero protection.

Returns a structured dict:
  {
    "expression": str,
    "result": float | int | bool | None,
    "formatted": str,
    "success": bool,
    "error": str | None
  }
"""

import ast
import math
from typing import Any, Dict, Union

MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 50
MAX_MAGNITUDE = 1e15

# Supported AST binary operators mapped to python functions
BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}

# Supported AST unary operators
UNARY_OPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}

# Supported AST comparison operators
COMPARE_OPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}


class CalculatorError(ValueError):
    """Raised when an expression violates calculation safety rules."""
    pass


def _eval_node(node: ast.AST) -> Union[int, float, bool]:
    """Recursively evaluate an AST node strictly against the allowlist."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        # Python 3.8+ represents numeric literals as ast.Constant
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if abs(node.value) > MAX_MAGNITUDE:
                raise CalculatorError(f"Number {node.value} exceeds maximum allowed magnitude ({MAX_MAGNITUDE})")
            return node.value
        raise CalculatorError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in UNARY_OPS:
            raise CalculatorError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_node(node.operand)
        res = UNARY_OPS[op_type](operand)
        if abs(res) > MAX_MAGNITUDE:
            raise CalculatorError("Result exceeds maximum allowed magnitude")
        return res

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in BIN_OPS:
            raise CalculatorError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)

        # Division by zero guards
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("Division by zero")

        res = BIN_OPS[op_type](left, right)
        if math.isnan(res) or math.isinf(res):
            raise CalculatorError("Calculation resulted in NaN or Infinity")
        if abs(res) > MAX_MAGNITUDE:
            raise CalculatorError("Result exceeds maximum allowed magnitude")
        return res

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left)
        # Handle chained comparisons e.g. 1 < 2 < 3
        current = left
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in COMPARE_OPS:
                raise CalculatorError(f"Unsupported comparison operator: {op_type.__name__}")
            right = _eval_node(comparator)
            if not COMPARE_OPS[op_type](current, right):
                return False
            current = right
        return True

    # Any other AST node (Call, Name, Attribute, Import, etc.) is strictly forbidden
    raise CalculatorError(
        f"Forbidden expression element: {type(node).__name__}. "
        "Only basic arithmetic and numeric literals are permitted."
    )


def safe_calculate(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate a mathematical expression string.

    Returns:
      {
        "expression": expression,
        "result": float | int | bool | None,
        "formatted": str,
        "success": bool,
        "error": str | None
      }
    """
    if not isinstance(expression, str):
        return {
            "expression": str(expression),
            "result": None,
            "formatted": "Error",
            "success": False,
            "error": "Expression must be a string",
        }

    expr_clean = expression.strip()

    if not expr_clean:
        return {
            "expression": expression,
            "result": None,
            "formatted": "Error",
            "success": False,
            "error": "Empty expression",
        }

    # Length guard
    if len(expr_clean) > MAX_EXPRESSION_LENGTH:
        return {
            "expression": expression[:50] + "...",
            "result": None,
            "formatted": "Error",
            "success": False,
            "error": f"Expression length ({len(expr_clean)}) exceeds maximum allowed ({MAX_EXPRESSION_LENGTH})",
        }

    try:
        tree = ast.parse(expr_clean, mode="eval")
    except SyntaxError as e:
        return {
            "expression": expr_clean,
            "result": None,
            "formatted": "Error",
            "success": False,
            "error": f"Syntax error in expression: {e.msg}",
        }

    # Node count guard to prevent deeply nested or complex trees
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > MAX_AST_NODES:
        return {
            "expression": expr_clean,
            "result": None,
            "formatted": "Error",
            "success": False,
            "error": f"Expression complexity ({node_count} nodes) exceeds limit ({MAX_AST_NODES})",
        }

    try:
        val = _eval_node(tree)

        # Format result for human display (up to 4 decimal places if float)
        if isinstance(val, float):
            # If float is an integer value like 4.0, format nicely
            if val.is_integer():
                formatted = str(int(val))
            else:
                formatted = f"{val:.4f}".rstrip("0").rstrip(".")
        else:
            formatted = str(val)

        return {
            "expression": expr_clean,
            "result": val,
            "formatted": formatted,
            "success": True,
            "error": None,
        }
    except ZeroDivisionError as e:
        return {
            "expression": expr_clean,
            "result": None,
            "formatted": "Error",
            "success": False,
            "error": "Division by zero",
        }
    except CalculatorError as e:
        return {
            "expression": expr_clean,
            "result": None,
            "formatted": "Error",
            "success": False,
            "error": str(e),
        }
    except Exception as e:
        return {
            "expression": expr_clean,
            "result": None,
            "formatted": "Error",
            "success": False,
            "error": f"Calculation error: {type(e).__name__}: {e}",
        }
