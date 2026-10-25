from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

REFERENCE_TERMS = (
    "repeat_interleave",
    "interpolate",
    "kv_cache",
    "vocos",
    "decode",
    "solver",
)
_INFERENCE_NAME = re.compile(r"infer|generat|synth|decode|render", re.IGNORECASE)
_SOLVER_CONTEXT = re.compile(
    r"\b(?:solver|step|steps|timestep|timesteps|iteration|integrat(?:e|ion)?)\b",
    re.IGNORECASE,
)


def _function_record(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    args = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args)]
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    args.extend(arg.arg for arg in node.args.kwonlyargs)
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    return {
        "name": node.name,
        "line": node.lineno,
        "args": args,
        "signature": f"{node.name}({', '.join(args)})",
        "async": isinstance(node, ast.AsyncFunctionDef),
    }


def _target_names(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [name for element in node.elts for name in _target_names(element)]
    return []


def inspect_python(path: Path, package_root: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    module_path = path.relative_to(package_root.parent).with_suffix("")
    module = ".".join(module_path.parts)
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]

    functions: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    forwards: list[dict[str, Any]] = []
    inference_functions: list[dict[str, Any]] = []
    constants: list[dict[str, Any]] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            record = _function_record(node)
            functions.append(record)
            if _INFERENCE_NAME.search(node.name):
                inference_functions.append(record)
        elif isinstance(node, ast.ClassDef):
            methods = []
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method = _function_record(member)
                    methods.append(method)
                    if member.name == "forward":
                        forwards.append({"class": node.name, **method})
                    if _INFERENCE_NAME.search(member.name):
                        inference_functions.append({"class": node.name, **method})
            classes.append({"name": node.name, "line": node.lineno, "methods": methods})
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _target_names(target):
                    if name.isupper():
                        constants.append(
                            {"name": name, "line": node.lineno, "value": ast.unparse(node.value)}
                        )
        elif isinstance(node, ast.AnnAssign):
            for name in _target_names(node.target):
                if name.isupper():
                    constants.append(
                        {
                            "name": name,
                            "line": node.lineno,
                            "value": ast.unparse(node.value) if node.value else None,
                        }
                    )

    references = {term: set() for term in REFERENCE_TERMS}
    loops: list[dict[str, Any]] = []

    def visit(node: ast.AST, context: tuple[str, ...] = ()) -> None:
        next_context = context
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            next_context = (*context, node.name)
        for child in ast.iter_child_nodes(node):
            visit(child, next_context)

        term = None
        if isinstance(node, ast.Name):
            term = node.id.lower()
        elif isinstance(node, ast.Attribute):
            term = node.attr.lower()
        if term in references:
            references[term].add(node.lineno)

        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            context_name = ".".join(context)
            iterable = ast.unparse(node.iter) if isinstance(node, (ast.For, ast.AsyncFor)) else None
            target = ast.unparse(node.target) if isinstance(node, (ast.For, ast.AsyncFor)) else ""
            loops.append(
                {
                    "line": node.lineno,
                    "context": context_name,
                    "kind": type(node).__name__,
                    "iterable": iterable,
                    "solver_related": bool(
                        _SOLVER_CONTEXT.search(context_name)
                        or _SOLVER_CONTEXT.search(iterable or "")
                        or _SOLVER_CONTEXT.search(target)
                    ),
                }
            )

    visit(tree)
    return {
        "file": path.relative_to(package_root.parent).as_posix(),
        "module": module,
        "classes": classes,
        "functions": functions,
        "forward_signatures": forwards,
        "inference_generation_functions": inference_functions,
        "top_level_constants": constants,
        "references": {term: sorted(lines) for term, lines in references.items()},
        "loops": sorted(loops, key=lambda row: row["line"]),
    }


def inspect_snapshot(snapshot: Path) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    package_root = snapshot if snapshot.name == "silencio" else snapshot / "silencio"
    if not package_root.is_dir():
        raise ValueError(f"Silencio Python package not found under {snapshot}")

    files = sorted(package_root.rglob("*.py"))
    modules = [inspect_python(path, package_root) for path in files]
    return {
        "snapshot": snapshot.name,
        "package": package_root.name,
        "python_file_count": len(files),
        "modules": modules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory the pinned Silencio Python source with AST.")
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    inventory = inspect_snapshot(args.snapshot)
    report = json.dumps(inventory, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote AST inventory for {inventory['python_file_count']} Python files to {args.output}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
