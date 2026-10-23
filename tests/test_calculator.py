"""
tests/test_calculator.py
------------------------
Unit tests for the safe calculation tool (calculator.py).

Tests:
  - Basic arithmetic (+, -, *, /, //, %)
  - Operator precedence and parentheses
  - Decimal calculations (e.g. 0.26 / 18.60 * 100)
  - Division by zero
  - Numeric comparisons (==, !=, <, <=, >, >=)
  - Rejection of function calls, attribute access, variables, and imports
  - Rejection of malformed syntax
  - Limits on expression length and numeric magnitude
"""

import pytest
from calculator import safe_calculate, MAX_EXPRESSION_LENGTH


class TestBasicArithmetic:
    def test_addition(self):
        res = safe_calculate("12 + 34")
        assert res["success"] is True
        assert res["result"] == 46
        assert res["error"] is None

    def test_subtraction(self):
        res = safe_calculate("100 - 35.5")
        assert res["success"] is True
        assert res["result"] == 64.5

    def test_multiplication(self):
        res = safe_calculate("7 * 8")
        assert res["success"] is True
        assert res["result"] == 56

    def test_division(self):
        res = safe_calculate("10 / 4")
        assert res["success"] is True
        assert res["result"] == 2.5

    def test_floor_division_and_modulo(self):
        res1 = safe_calculate("17 // 5")
        assert res1["success"] is True
        assert res1["result"] == 3

        res2 = safe_calculate("17 % 5")
        assert res2["success"] is True
        assert res2["result"] == 2

    def test_unary_operators(self):
        res = safe_calculate("-5 + (+10)")
        assert res["success"] is True
        assert res["result"] == 5


class TestPrecedenceAndDecimals:
    def test_operator_precedence(self):
        res = safe_calculate("2 + 3 * 4")
        assert res["success"] is True
        assert res["result"] == 14  # Not 20

    def test_parentheses_override(self):
        res = safe_calculate("(2 + 3) * 4")
        assert res["success"] is True
        assert res["result"] == 20

    def test_nested_parentheses(self):
        res = safe_calculate("((10 + 20) * 2) / (3 + 1)")
        assert res["success"] is True
        assert res["result"] == 15.0

    def test_sample_inventory_calculation(self):
        # The specific Clause 6.2 demonstration: 0.26 / 18.60 * 100
        res = safe_calculate("0.26 / 18.60 * 100")
        assert res["success"] is True
        assert round(res["result"], 4) == 1.3978
        assert "1.3978" in res["formatted"] or "1.4" in res["formatted"]


class TestComparisons:
    def test_greater_than(self):
        res = safe_calculate("1.3978 > 1.0")
        assert res["success"] is True
        assert res["result"] is True

    def test_less_than_or_equal(self):
        res1 = safe_calculate("1.4 <= 1.0")
        assert res1["success"] is True
        assert res1["result"] is False

        res2 = safe_calculate("1.0 <= 1.0")
        assert res2["success"] is True
        assert res2["result"] is True

    def test_chained_comparison(self):
        res = safe_calculate("0.5 < 1.4 < 2.0")
        assert res["success"] is True
        assert res["result"] is True


class TestDivisionByZero:
    def test_float_division_by_zero(self):
        res = safe_calculate("10 / 0")
        assert res["success"] is False
        assert "Division by zero" in res["error"]
        assert res["result"] is None

    def test_floor_division_by_zero(self):
        res = safe_calculate("10 // (5 - 5)")
        assert res["success"] is False
        assert "Division by zero" in res["error"]

    def test_modulo_by_zero(self):
        res = safe_calculate("10 % 0")
        assert res["success"] is False
        assert "Division by zero" in res["error"]


class TestSecurityAndRejections:
    def test_reject_function_call(self):
        res = safe_calculate("abs(-5)")
        assert res["success"] is False
        assert "Forbidden" in res["error"] or "Call" in res["error"]

    def test_reject_eval_and_exec(self):
        res1 = safe_calculate("eval('1 + 1')")
        assert res1["success"] is False
        assert res1["result"] is None

        res2 = safe_calculate("exec('x = 1')")
        assert res2["success"] is False

    def test_reject_import(self):
        res = safe_calculate("__import__('os').system('ls')")
        assert res["success"] is False
        assert res["result"] is None

    def test_reject_attribute_access(self):
        res = safe_calculate("(1).__class__")
        assert res["success"] is False
        assert "Forbidden" in res["error"] or "Attribute" in res["error"]

    def test_reject_variables_and_names(self):
        res = safe_calculate("x + y")
        assert res["success"] is False
        assert "Forbidden" in res["error"] or "Name" in res["error"]

    def test_reject_power_operator(self):
        # We deliberately disallow ** to prevent DoS via 99999**99999
        res = safe_calculate("2 ** 10")
        assert res["success"] is False
        assert "Unsupported binary operator" in res["error"]

    def test_reject_bitwise_and_shift(self):
        res1 = safe_calculate("5 & 3")
        assert res1["success"] is False

        res2 = safe_calculate("1 << 10")
        assert res2["success"] is False


class TestLimitsAndMalformedInputs:
    def test_empty_expression(self):
        res = safe_calculate("   ")
        assert res["success"] is False
        assert "Empty" in res["error"]

    def test_non_string_input(self):
        res = safe_calculate(None)
        assert res["success"] is False
        assert "string" in res["error"].lower()

    def test_syntax_error(self):
        res = safe_calculate("2 + +")
        assert res["success"] is False
        assert "Syntax error" in res["error"]

    def test_unbalanced_parentheses(self):
        res = safe_calculate("(2 + 3 * 4")
        assert res["success"] is False
        assert "Syntax error" in res["error"]

    def test_oversized_expression_length(self):
        long_expr = "1 + " * 100 + "1"  # > 300 chars
        assert len(long_expr) > MAX_EXPRESSION_LENGTH
        res = safe_calculate(long_expr)
        assert res["success"] is False
        assert "length" in res["error"].lower()

    def test_magnitude_limit(self):
        res = safe_calculate("1000000000000000000000 + 1")  # > 1e15
        assert res["success"] is False
        assert "magnitude" in res["error"].lower()
