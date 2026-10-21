"""Finite SBML MathML syntax on common scalar/chain arithmetic, without eval.

Symbols are case sensitive. Function calls use lexical lambda arguments, not
the caller's global values. Piecewise and Boolean operators evaluate lazily.
"""

from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass, field
from functools import reduce
from itertools import pairwise
from numbers import Real

from chemworld.data.xml_inputs import bounded_xml
from chemworld.physchem.analytic_functions import (
    SMOOTH_FUNCTIONS,
    absolute,
    real_power,
    smooth_function,
)
from chemworld.physchem.caloric_math import FirstDerivative, logarithm, scalar_value

MATHML_NS = "http://www.w3.org/1998/Math/MathML"
SBML_NS = "http://www.sbml.org/sbml/level3/version2/core"
SBML_AVOGADRO = 6.02214179e23  # L3V2 Core 3.4.6: deliberately not the newer SI value.
_SID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_REAL_LITERAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_COMPARISONS = {"eq": operator.eq, "neq": operator.ne, "lt": operator.lt,
                "leq": operator.le, "gt": operator.gt, "geq": operator.ge}
_UNARY = SMOOTH_FUNCTIONS | {"abs", "floor", "ceiling", "factorial", "not"}
_BINARY = {"divide", "power", "neq", "implies"}
_NARY = {"plus", "times", "min", "max", "and", "or", "xor", "eq", "lt", "leq", "gt", "geq"}
_IDENTITIES = {"plus", "times", "and", "or", "xor"}


@dataclass(frozen=True)
class MathNode:
    operation: str
    arguments: tuple[MathNode, ...] = ()
    value: float | str | None = None
    bindings: tuple[str, ...] = ()
    unit: str | None = None


def _tag(element):
    if not isinstance(element.tag, str) or not element.tag.startswith("{"+MATHML_NS+"}"):
        raise ValueError("MathML node requires the MathML namespace")
    return element.tag.split("}", 1)[1]


def _identifier(text):
    if not isinstance(text, str) or not _SID.fullmatch(text.strip()):
        raise ValueError("MathML identifier must be an SBML SId")
    return text.strip()


def _structural_text(element):
    if (element.text or "").strip() or any((c.tail or "").strip() for c in element):
        raise ValueError("unexpected text in MathML structural element")


def _parse(element):
    tag = _tag(element)
    allowed = {"id", "class", "style"}
    if tag == "cn":
        allowed |= {"type", "{"+SBML_NS+"}units"}
    if tag in {"ci", "csymbol", "semantics"}:
        allowed.add("definitionURL")
    if tag == "csymbol":
        allowed.add("encoding")
    if set(element.attrib) - allowed:
        raise ValueError("unknown MathML attributes on "+tag)
    children = list(element)
    if tag not in {"cn", "ci", "csymbol"} and (element.text or "").strip():
        raise ValueError("unexpected text outside a MathML token")
    if tag != "cn" and any((c.tail or "").strip() for c in children):
        raise ValueError("unexpected trailing text inside MathML")
    if tag == "math":
        if len(children) != 1:
            raise ValueError("MathML math must contain exactly one expression")
        return _parse(children[0])
    if tag == "semantics":
        if not children or any(_tag(c) not in {"annotation", "annotation-xml"}
                               for c in children[1:]):
            raise ValueError("invalid MathML semantics wrapper")
        return _parse(children[0])
    if tag == "cn":
        kind = element.get("type", "real")
        raw = (element.text or "").strip()
        try:
            if kind in {"real", "integer"} and not children:
                if not _REAL_LITERAL.fullmatch(raw):
                    raise ValueError("invalid real number")
                if kind == "integer" and not re.fullmatch(r"[+-]?[0-9]+", raw):
                    raise ValueError("invalid integer")
                number = float(raw)
            elif kind in {"rational", "e-notation"} and len(children) == 1:
                if (_tag(children[0]) != "sep" or list(children[0]) or children[0].attrib
                        or (children[0].text or "").strip()):
                    raise ValueError("invalid numeric separator")
                other = (children[0].tail or "").strip()
                if not re.fullmatch(r"[+-]?[0-9]+", other):
                    raise ValueError("numeric exponent/denominator must be integer")
                if kind == "rational":
                    if not re.fullmatch(r"[+-]?[0-9]+", raw):
                        raise ValueError("rational numerator must be integer")
                    number = float(raw)/float(other)
                else:
                    if not _REAL_LITERAL.fullmatch(raw) or "e" in raw.lower():
                        raise ValueError("invalid e-notation mantissa")
                    number = float(raw+"e"+other)
            else:
                raise ValueError("unsupported numeric layout")
            if not math.isfinite(number):
                raise ValueError("nonfinite cn")
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            raise ValueError("invalid finite MathML numeric literal") from exc
        unit = element.get("{"+SBML_NS+"}units")
        if unit is not None:
            unit = _identifier(unit)
        return MathNode("number", value=number, unit=unit)
    if tag == "ci" and not children:
        return MathNode("symbol", value=_identifier(element.text))
    if tag == "csymbol" and not children:
        if element.get("encoding", "text") != "text":
            raise ValueError("SBML csymbol encoding must be text")
        url = element.get("definitionURL")
        if url == "http://www.sbml.org/sbml/symbols/time":
            return MathNode("symbol", value="$time")
        if url == "http://www.sbml.org/sbml/symbols/avogadro":
            return MathNode("number", value=SBML_AVOGADRO)
        raise ValueError("unsupported SBML csymbol; delay/rateOf require their declared semantics")
    constants = {"true": True, "false": False, "pi": math.pi, "exponentiale": math.e,
                 "infinity": math.inf, "notanumber": math.nan}
    if tag in constants and not children and not (element.text or "").strip():
        return MathNode("constant", value=constants[tag])
    if tag == "lambda":
        names = []
        for child in children[:-1]:
            _structural_text(child)
            if (_tag(child) != "bvar" or child.attrib or len(child) != 1
                    or _tag(child[0]) != "ci"):
                raise ValueError("lambda requires bvar identifiers followed by one body")
            names.append(_parse(child[0]).value)
        if not children or len(set(names)) != len(names) or _tag(children[-1]) == "bvar":
            raise ValueError("invalid or repeated lambda binding")
        return MathNode("lambda", (_parse(children[-1]),), bindings=tuple(names))
    if tag == "piecewise":
        args = []
        for index, child in enumerate(children):
            _structural_text(child)
            kind = _tag(child)
            if child.attrib:
                raise ValueError("unknown piecewise attributes")
            if kind == "piece" and len(child) == 2:
                args.append(MathNode("piece", tuple(_parse(c) for c in child)))
            elif kind == "otherwise" and len(child) == 1 and index == len(children)-1:
                args.append(MathNode("otherwise", (_parse(child[0]),)))
            else:
                raise ValueError("invalid piecewise branch or misplaced otherwise")
        if not args:
            raise ValueError("piecewise requires at least one branch")
        return MathNode(tag, tuple(args))
    if tag == "apply" and children:
        head = children[0]
        name = _tag(head)
        if name == "ci":
            if list(head) or head.attrib:
                raise ValueError("function call requires a plain identifier")
            return MathNode("call", tuple(_parse(c) for c in children[1:]),
                            value=_identifier(head.text))
        if list(head) or head.attrib or (head.text or "").strip():
            raise ValueError("MathML operator must be empty")
        arguments = children[1:]
        if name in {"root", "log"}:
            qualifier = "degree" if name == "root" else "logbase"
            if arguments and _tag(arguments[0]) == qualifier:
                q = arguments[0]
                _structural_text(q)
                if len(q) != 1 or q.attrib:
                    raise ValueError("invalid "+qualifier)
                first, arguments = _parse(q[0]), arguments[1:]
            else:
                first = MathNode("number", value=2. if name == "root" else 10.)
            if len(arguments) != 1:
                raise ValueError(name+" requires one operand and optional qualifier")
            return MathNode(name, (first, _parse(arguments[0])))
        n = len(arguments)
        if not ((name in _UNARY and n == 1) or (name in _BINARY and n == 2)
                or (name in _NARY and n >= 2) or name in _IDENTITIES
                or (name in {"min", "max"} and n == 1) or (name == "minus" and n in {1, 2})):
            raise ValueError("unsupported MathML operator or arity: "+name)
        return MathNode(name, tuple(_parse(c) for c in arguments))
    raise ValueError("unsupported or malformed MathML element: "+tag)


def _walk(root):
    yield root
    for child in root.arguments:
        yield from _walk(child)


def _finite(value):
    if not isinstance(value, (Real, FirstDerivative)) or not math.isfinite(scalar_value(value)):
        raise ValueError("MathML requires a finite real or Boolean result")
    return value


@dataclass(frozen=True)
class MathExpression:
    xml: str
    _root: MathNode = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        root = bounded_xml(self.xml, max_bytes=262144, max_nodes=4096, max_depth=64)
        if _tag(root) != "math":
            raise ValueError("SBML MathML expression requires a math container")
        tree = _parse(root)
        if any(n.operation == "lambda" for n in _walk(tree) if n is not tree):
            raise ValueError("lambda is only permitted at the function-definition root")
        if tree.operation == "lambda" and self._symbols(tree)-set(tree.bindings):
            raise ValueError("SBML function may reference only its lexical arguments")
        object.__setattr__(self, "_root", tree)

    @staticmethod
    def _symbols(tree):
        return {n.value for n in _walk(tree) if n.operation == "symbol"}

    @property
    def symbols(self):
        return tuple(sorted(self._symbols(self._root)-set(self._root.bindings)))

    @property
    def function_names(self):
        return tuple(sorted({n.value for n in _walk(self._root) if n.operation == "call"}))

    @property
    def function_calls(self):
        return tuple((n.value, len(n.arguments)) for n in _walk(self._root)
                     if n.operation == "call")

    @property
    def function_arguments(self):
        return self._root.bindings if self._root.operation == "lambda" else None

    @property
    def number_units(self):
        return tuple(sorted({n.unit for n in _walk(self._root) if n.unit is not None}))

    def evaluate(self, values, *, functions=None, subtree=None, comparison_values=None):
        """Evaluate the parsed tree, optionally at a coordinated root boundary.

        Subtrees must belong to this already bounded declaration. Explicit
        comparison values let a shared event coordinator supply right limits
        without perturbing physical time, state, or assignment expressions.
        The ordinary expression path has no overrides.
        """
        functions = {} if functions is None else functions
        overrides = {} if comparison_values is None else dict(comparison_values)
        if subtree is not None or overrides:
            nodes = set(_walk(self._root))
            if ((subtree is not None and subtree not in nodes)
                    or any(n not in nodes or n.operation not in _COMPARISONS
                           or type(v) is not bool for n, v in overrides.items())):
                raise ValueError("MathML root views need declared subtrees and Boolean comparisons")
        remaining = 65536

        def visit(node, scope, calls=()):
            nonlocal remaining

            remaining -= 1
            if remaining < 0:
                raise ValueError("MathML exceeds the expression evaluation budget")
            op, args = node.operation, node.arguments
            if overrides and node in overrides:
                return overrides[node]
            if op in {"number", "constant"}:
                return _finite(node.value)
            if op == "symbol":
                if node.value not in scope:
                    raise ValueError("undefined MathML symbol "+node.value)
                return _finite(scope[node.value])
            if op == "piecewise":
                for branch in args:
                    if branch.operation == "otherwise" or bool(scalar_value(
                            visit(branch.arguments[1], scope, calls))):
                        return visit(branch.arguments[0], scope, calls)
                raise ValueError("piecewise has no defined active branch")
            if op in {"and", "or", "implies"}:
                def truth(arg):
                    return bool(scalar_value(visit(arg, scope, calls)))

                if op == "and":
                    return all(truth(arg) for arg in args)
                if op == "or":
                    return any(truth(arg) for arg in args)
                return not truth(args[0]) or truth(args[1])
            numbers = [visit(a, scope, calls) for a in args]
            if op == "call":
                name = node.value
                if name in calls or len(calls) >= 32:
                    raise ValueError("recursive or excessively nested MathML function call")
                if name not in functions or not isinstance(functions[name], MathExpression):
                    raise ValueError("undefined MathML function "+name)
                body = functions[name]._root
                if body.operation != "lambda" or len(body.bindings) != len(numbers):
                    raise ValueError("function requires lambda with matching argument count")
                return visit(body.arguments[0], dict(zip(body.bindings, numbers, strict=True)),
                             (*calls, name))
            if op in _COMPARISONS:
                primals = list(map(scalar_value, numbers))
                return all(_COMPARISONS[op](a, b) for a, b in pairwise(primals))
            if op == "not":
                return not bool(scalar_value(numbers[0]))
            if op == "xor":
                return sum(bool(scalar_value(n)) for n in numbers) % 2 == 1
            if op == "plus":
                return sum(numbers)
            if op == "times":
                return reduce(operator.mul, numbers, 1.)
            if op == "minus":
                return -numbers[0] if len(numbers) == 1 else numbers[0]-numbers[1]
            if op == "divide":
                return numbers[0]/numbers[1]
            if op == "power":
                return real_power(*numbers)
            if op == "root":
                degree, radicand = numbers
                if (not isinstance(degree, FirstDerivative) and scalar_value(radicand) < 0
                        and degree == int(degree) and int(degree) % 2):
                    return -real_power(-radicand, 1/degree)
                return real_power(radicand, 1/degree)
            if op == "log":
                return logarithm(numbers[1])/logarithm(numbers[0])
            if op in SMOOTH_FUNCTIONS:
                return smooth_function(op, numbers[0])
            if op == "abs":
                return absolute(numbers[0])
            if op in {"min", "max"}:
                select = min if op == "min" else max
                selected = select(numbers, key=scalar_value)
                if (any(isinstance(n, FirstDerivative) for n in numbers)
                        and sum(scalar_value(n) == scalar_value(selected) for n in numbers) > 1):
                    raise ValueError("MathML min/max derivative is undefined at a tie")
                return selected
            if op in {"floor", "ceiling", "factorial"}:
                x = scalar_value(numbers[0])
                if op == "factorial":
                    if isinstance(numbers[0], FirstDerivative) or x != int(x) or not 0 <= x <= 170:
                        raise ValueError("factorial requires a nonnegative integer <=170, no jet")
                    return float(math.factorial(int(x)))
                if isinstance(numbers[0], FirstDerivative) and x == int(x):
                    raise ValueError("floor/ceiling derivative undefined at integer boundary")
                return float(math.floor(x) if op == "floor" else math.ceil(x))
            raise ValueError("MathML lambda cannot be evaluated outside a function call")

        try:
            return _finite(visit(self._root if subtree is None else subtree, values))
        except (OverflowError, ZeroDivisionError, TypeError) as exc:
            raise ValueError("MathML operation outside its finite real domain") from exc
