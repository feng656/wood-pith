from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    rule: str
    path: str
    line: int
    message: str


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _names(node: ast.AST) -> set[str]:
    return {item.id.lower() for item in ast.walk(node) if isinstance(item, ast.Name)}


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: Path, solver_file: bool) -> None:
        self.path = path
        self.solver_file = solver_file
        self.findings: list[AuditFinding] = []

    def add(self, node: ast.AST, severity: str, rule: str, message: str) -> None:
        self.findings.append(
            AuditFinding(severity, rule, str(self.path), int(getattr(node, "lineno", 1)), message)
        )

    def visit_Import(self, node: ast.Import) -> None:
        if self.solver_file:
            for alias in node.names:
                if "evaluation" in alias.name or "contribution" in alias.name:
                    self.add(node, "ERROR", "SOLVER_IMPORT_BOUNDARY", alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if self.solver_file and ("evaluation" in module or "contribution" in module):
            self.add(node, "ERROR", "SOLVER_IMPORT_BOUNDARY", module)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.solver_file and node.name in {"fit", "locate", "estimate"}:
            names = [argument.arg.lower() for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
            forbidden = [name for name in names if name in {"gt", "truth", "ground_truth", "pith_gt"}]
            if forbidden:
                self.add(
                    node,
                    "ERROR",
                    "GT_SOLVER_SIGNATURE",
                    f"solver signature contains ground-truth arguments: {forbidden}",
                )
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted(node.func)
        if name.endswith("load"):
            for keyword in node.keywords:
                if keyword.arg == "allow_pickle" and isinstance(keyword.value, ast.Constant):
                    if keyword.value.value is True:
                        self.add(node, "ERROR", "PICKLE_ARRAY", "allow_pickle=True is forbidden")
        if name.endswith("clip") and node.args:
            variables = _names(node.args[0])
            suspect = variables & {"pith", "pith_xy", "center", "candidate", "raw_center"}
            if suspect:
                self.add(
                    node,
                    "ERROR",
                    "PITH_CLIPPING",
                    f"possible clipping of an outside-capable coordinate: {sorted(suspect)}",
                )
        self.generic_visit(node)


def audit_source_tree(source_root: str | Path) -> list[dict[str, object]]:
    root = Path(source_root)
    findings: list[AuditFinding] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            findings.append(
                AuditFinding(
                    "ERROR",
                    "PYTHON_SYNTAX",
                    str(path),
                    int(exc.lineno or 1),
                    str(exc),
                )
            )
            continue
        relative_parts = path.relative_to(root).parts
        solver_file = any(part in {"solver", "estimator.py", "numerics.py"} for part in relative_parts)
        visitor = _Visitor(path, solver_file)
        visitor.visit(tree)
        findings.extend(visitor.findings)
        if "contribution" in relative_parts and "predictions" in path.read_text(encoding="utf-8"):
            findings.append(
                AuditFinding(
                    "WARNING",
                    "CONTRIBUTION_OUTPUT_BOUNDARY",
                    str(path),
                    1,
                    "contribution code mentions predictions; verify it cannot overwrite baseline results",
                )
            )
    return [asdict(finding) for finding in findings]


def count_errors(findings: Iterable[dict[str, object]]) -> int:
    return sum(1 for finding in findings if finding.get("severity") == "ERROR")
