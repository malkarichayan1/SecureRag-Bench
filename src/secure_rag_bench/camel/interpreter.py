"""CaMeL AST interpreter with provenance tracking and security enforcement."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable

from secure_rag_bench.camel.provenance import Capability, Provenance, Source, TrackedValue
from secure_rag_bench.camel.quarantined_llm import QuarantinedLLM, wrap_untrusted
from secure_rag_bench.security.policy import PolicyDecision, check_policy


class InterpreterError(Exception):
    """Raised when plan code violates security constraints or fails execution."""


class SecurityViolation(InterpreterError):
    """Raised when generated code violates security rules."""


class PolicyHalt(InterpreterError):
    """Raised when policy check requires user confirmation."""

    def __init__(self, message: str, decision: PolicyDecision) -> None:
        super().__init__(message)
        self.decision = decision


@dataclass
class ToolContext:
    """Runtime context for tool execution during interpretation."""

    user_query: str
    task_description: str = ""
    strict_mode: bool = True
    user_confirmed: bool = False


class CaMeLInterpreter:
    """Recursively interprets privileged-model Python code with provenance tracking."""

    FORBIDDEN_NAMES = frozenset({"eval", "exec", "__import__", "compile", "open", "globals", "locals"})
    FORBIDDEN_ATTRS = frozenset({"append", "extend", "insert", "pop", "remove", "clear"})

    def __init__(
        self,
        tools: dict[str, Callable[..., Any]] | None = None,
        quarantined_llm: QuarantinedLLM | None = None,
        enforce_policy: bool = True,
        enforce_provenance: bool = True,
    ) -> None:
        self.tools = tools or {}
        self.quarantined_llm = quarantined_llm
        self.enforce_policy = enforce_policy
        self.enforce_provenance = enforce_provenance
        self._env: dict[str, TrackedValue[Any]] = {}
        self._context = ToolContext(user_query="")

    def validate_plan(self, code: str) -> None:
        """Validate plan code against security constraints before execution."""
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise SecurityViolation(f"Invalid Python syntax: {exc}") from exc

        allowed_nodes = (
            ast.Module, ast.Assign, ast.Expr, ast.If, ast.Return,
            ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp, ast.Compare,
            ast.BoolOp, ast.List, ast.Dict, ast.Subscript, ast.Call,
            ast.JoinedStr, ast.FormattedValue, ast.IfExp, ast.keyword,
            ast.Load, ast.Store, ast.Add, ast.Sub, ast.Mult, ast.Div,
            ast.Mod, ast.Not, ast.USub, ast.Eq, ast.NotEq, ast.Lt, ast.Gt,
            ast.LtE, ast.GtE, ast.And, ast.Or,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise SecurityViolation(
                    f"Unsupported syntax in restricted plan: {type(node).__name__}"
                )
            if isinstance(node, ast.Call):
                self._validate_call(node)
            if isinstance(node, ast.Assign):
                if any(not isinstance(target, ast.Name) for target in node.targets):
                    raise SecurityViolation("Only name assignment targets are supported")

    def _validate_call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise SecurityViolation("Only direct calls to allowlisted tools are supported")
        if node.func.id in self.FORBIDDEN_NAMES:
            raise SecurityViolation(f"Forbidden call: {node.func.id}()")
        if node.func.id != "quarantine_parse" and node.func.id not in self.tools:
            raise SecurityViolation(f"Unknown or disallowed call: {node.func.id}()")
        if any(keyword.arg is None for keyword in node.keywords):
            raise SecurityViolation("Keyword unpacking is forbidden")

    def execute(
        self,
        code: str,
        *,
        user_query: str,
        task_description: str = "",
        strict_mode: bool = True,
        user_confirmed: bool = False,
    ) -> TrackedValue[Any]:
        """Validate and execute privileged-model plan code."""
        self.validate_plan(code)
        self._env = {
            "user_input": TrackedValue.from_user(user_query),
            "user_query": TrackedValue.from_user(user_query),
        }
        self._context = ToolContext(
            user_query=user_query,
            task_description=task_description or user_query,
            strict_mode=strict_mode,
            user_confirmed=user_confirmed,
        )

        tree = ast.parse(code)
        result: TrackedValue[Any] | None = None
        for stmt in tree.body:
            result = self._exec_stmt(stmt)
        if result is None:
            answer = self._env.get("answer")
            if answer is not None:
                return answer
            raise InterpreterError("Plan produced no result")
        return result

    def _exec_stmt(self, node: ast.stmt) -> TrackedValue[Any] | None:
        if isinstance(node, ast.Assign):
            return self._exec_assign(node)
        if isinstance(node, ast.Expr):
            value = self._eval_expr(node.value)
            return value
        if isinstance(node, ast.If):
            return self._exec_if(node)
        if isinstance(node, ast.Return):
            return self._eval_expr(node.value) if node.value else None
        raise InterpreterError(f"Unsupported statement: {type(node).__name__}")

    def _exec_assign(self, node: ast.Assign) -> TrackedValue[Any] | None:
        value = self._eval_expr(node.value)
        for target in node.targets:
            self._assign_target(target, value)
        return value

    def _assign_target(self, target: ast.expr, value: TrackedValue[Any]) -> None:
        if isinstance(target, ast.Name):
            self._env[target.id] = value
            return
        raise InterpreterError(f"Unsupported assignment target: {type(target).__name__}")

    def _exec_if(self, node: ast.If) -> TrackedValue[Any] | None:
        test = self._eval_expr(node.test)
        if test.value:
            result = None
            for stmt in node.body:
                result = self._exec_stmt(stmt)
            return result
        result = None
        for stmt in node.orelse:
            result = self._exec_stmt(stmt)
        return result

    def _eval_expr(self, node: ast.expr) -> TrackedValue[Any]:
        if isinstance(node, ast.Constant):
            return TrackedValue.from_user(node.value)
        if isinstance(node, ast.Name):
            if node.id not in self._env:
                raise InterpreterError(f"Undefined variable: {node.id}")
            return self._env[node.id]
        if isinstance(node, ast.BinOp):
            return self._eval_binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._eval_unaryop(node)
        if isinstance(node, ast.Compare):
            return self._eval_compare(node)
        if isinstance(node, ast.BoolOp):
            return self._eval_boolop(node)
        if isinstance(node, ast.List):
            return self._eval_list(node)
        if isinstance(node, ast.Dict):
            return self._eval_dict(node)
        if isinstance(node, ast.Subscript):
            return self._eval_subscript(node)
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        if isinstance(node, ast.JoinedStr):
            return self._eval_joined_str(node)
        if isinstance(node, ast.IfExp):
            test = self._eval_expr(node.test)
            branch = node.body if test.value else node.orelse
            return self._eval_expr(branch)
        raise InterpreterError(f"Unsupported expression: {type(node).__name__}")

    def _eval_binop(self, node: ast.BinOp) -> TrackedValue[Any]:
        left = self._eval_expr(node.left)
        right = self._eval_expr(node.right)
        ops: dict[type, Callable[[Any, Any], Any]] = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.Mod: lambda a, b: a % b,
        }
        op_type = type(node.op)
        if op_type not in ops:
            raise InterpreterError(f"Unsupported operator: {op_type.__name__}")
        result = ops[op_type](left.value, right.value)
        return TrackedValue(result, left.provenance.merge(right.provenance))

    def _eval_unaryop(self, node: ast.UnaryOp) -> TrackedValue[Any]:
        operand = self._eval_expr(node.operand)
        if isinstance(node.op, ast.Not):
            return TrackedValue(not operand.value, operand.provenance)
        if isinstance(node.op, ast.USub):
            return TrackedValue(-operand.value, operand.provenance)
        raise InterpreterError(f"Unsupported unary op: {type(node.op).__name__}")

    def _eval_compare(self, node: ast.Compare) -> TrackedValue[Any]:
        left = self._eval_expr(node.left)
        result = True
        current = left
        for op, comparator in zip(node.ops, node.comparators):
            right = self._eval_expr(comparator)
            merged = current.provenance.merge(right.provenance)
            if isinstance(op, ast.Eq):
                result = result and (current.value == right.value)
            elif isinstance(op, ast.NotEq):
                result = result and (current.value != right.value)
            elif isinstance(op, ast.Lt):
                result = result and (current.value < right.value)
            elif isinstance(op, ast.Gt):
                result = result and (current.value > right.value)
            elif isinstance(op, ast.LtE):
                result = result and (current.value <= right.value)
            elif isinstance(op, ast.GtE):
                result = result and (current.value >= right.value)
            else:
                raise InterpreterError(f"Unsupported comparison: {type(op).__name__}")
            current = TrackedValue(current.value, merged)
        return TrackedValue(result, current.provenance)

    def _eval_boolop(self, node: ast.BoolOp) -> TrackedValue[Any]:
        values = [self._eval_expr(v) for v in node.values]
        prov = values[0].provenance
        for v in values[1:]:
            prov = prov.merge(v.provenance)
        if isinstance(node.op, ast.And):
            result = all(v.value for v in values)
        elif isinstance(node.op, ast.Or):
            result = any(v.value for v in values)
        else:
            raise InterpreterError(f"Unsupported bool op: {type(node.op).__name__}")
        return TrackedValue(result, prov)

    def _eval_list(self, node: ast.List) -> TrackedValue[Any]:
        items = [self._eval_expr(elt) for elt in node.elts]
        if not items:
            return TrackedValue.from_user([])
        prov = items[0].provenance
        for item in items[1:]:
            prov = prov.merge(item.provenance)
        return TrackedValue([i.value for i in items], prov)

    def _eval_dict(self, node: ast.Dict) -> TrackedValue[Any]:
        result: dict[Any, Any] = {}
        prov = TrackedValue.from_user(None).provenance
        for key_node, val_node in zip(node.keys, node.values):
            if key_node is None or val_node is None:
                continue
            key = self._eval_expr(key_node)
            val = self._eval_expr(val_node)
            prov = prov.merge(key.provenance).merge(val.provenance)
            result[key.value] = val.value
        return TrackedValue(result, prov)

    def _eval_subscript(self, node: ast.Subscript) -> TrackedValue[Any]:
        container = self._eval_expr(node.value)
        index = self._eval_expr(node.slice) if not isinstance(node.slice, ast.Constant) else TrackedValue.from_user(node.slice.value)
        if isinstance(node.slice, ast.Constant):
            index = TrackedValue.from_user(node.slice.value)
        try:
            result = container.value[index.value]
        except (KeyError, IndexError, TypeError) as exc:
            raise InterpreterError(_redact_error(str(exc), container.provenance)) from exc
        return TrackedValue(result, container.provenance.merge(index.provenance))

    def _eval_joined_str(self, node: ast.JoinedStr) -> TrackedValue[Any]:
        parts: list[str] = []
        prov = TrackedValue.from_user(None).provenance
        for val in node.values:
            if isinstance(val, ast.Constant):
                parts.append(str(val.value))
            elif isinstance(val, ast.FormattedValue):
                evaluated = self._eval_expr(val.value)
                prov = prov.merge(evaluated.provenance)
                parts.append(str(evaluated.value))
        return TrackedValue("".join(parts), prov)

    def _eval_call(self, node: ast.Call) -> TrackedValue[Any]:
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == "quarantine_parse":
                return self._call_quarantine_parse(node)
            return self._call_tool(func_name, node)
        raise InterpreterError("Only direct function calls are supported")

    def _call_quarantine_parse(self, node: ast.Call) -> TrackedValue[Any]:
        if self.quarantined_llm is None:
            raise InterpreterError("Quarantined LLM not configured")
        if len(node.args) < 1:
            raise InterpreterError("quarantine_parse requires untrusted content argument")

        content_arg = self._eval_expr(node.args[0])
        schema_name = "DocumentSummary"
        for kw in node.keywords:
            if kw.arg == "schema":
                schema_val = self._eval_expr(kw.value)
                schema_name = str(schema_val.value).strip("'\"")

        wrapped = wrap_untrusted(str(content_arg.value))
        parsed = self.quarantined_llm.parse(wrapped, schema_name)
        prov = Provenance(
            sources=frozenset({Source.TOOL}),
            allowed_readers=frozenset({Capability.INTERNAL, Capability.QUARANTINED}),
        )
        return TrackedValue(parsed, prov.merge(content_arg.provenance))

    def _call_tool(self, func_name: str, node: ast.Call) -> TrackedValue[Any]:
        if func_name not in self.tools:
            raise InterpreterError(f"Unknown tool: {func_name}")

        args = [self._eval_expr(a) for a in node.args]
        kwargs = {kw.arg: self._eval_expr(kw.value) for kw in node.keywords if kw.arg}

        all_inputs = args + list(kwargs.values())
        combined_prov = (
            all_inputs[0].provenance
            if all_inputs
            else TrackedValue.from_user(None).provenance
        )
        for inp in all_inputs[1:]:
            combined_prov = combined_prov.merge(inp.provenance)

        action = _infer_action(func_name, kwargs, args)
        if self.enforce_policy:
            decision = check_policy(
                task_description=self._context.task_description,
                action=action,
                data_provenance=combined_prov,
                user_confirmed=self._context.user_confirmed,
                enforce_provenance=self.enforce_provenance,
            )
            if not decision.allowed:
                if decision.requires_user_confirmation:
                    raise PolicyHalt(decision.reason, decision)
                raise SecurityViolation(decision.reason)

        try:
            raw_args = [a.value for a in args]
            raw_kwargs = {k: v.value for k, v in kwargs.items()}
            result = self.tools[func_name](*raw_args, **raw_kwargs)
        except Exception as exc:
            raise InterpreterError(_redact_error(str(exc), combined_prov)) from exc

        tool_prov = Provenance(
            sources=frozenset({Source.TOOL}),
            allowed_readers=frozenset({Capability.INTERNAL}),
        )
        return TrackedValue(result, tool_prov.merge(combined_prov))


def _infer_action(func_name: str, kwargs: dict[str, TrackedValue[Any]], args: list[TrackedValue[Any]]) -> str:
    if func_name in ("send_email", "post_message", "http_post", "external_send"):
        recipient = kwargs.get("to") or kwargs.get("recipient") or (args[0] if args else None)
        if recipient:
            return f"{func_name}:{recipient.value}"
        return func_name
    return func_name


def _redact_error(message: str, provenance: Provenance) -> str:
    """Redact error details when they may leak untrusted tool output."""
    if provenance.is_untrusted():
        return "Operation failed due to untrusted data (details redacted)"
    return message
