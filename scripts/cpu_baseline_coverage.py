"""Generated evidence projection for CPU-01; task documents remain the status authority."""

import ast
import re
import subprocess
from collections import Counter, defaultdict


def imports(tree):
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            rows.append((node.module, [item.name for item in node.names], node.lineno))
        elif isinstance(node, ast.Import):
            rows.extend((item.name, [], node.lineno) for item in node.names)
    return rows


def coverage_inventory(root):
    from chemworld.examples.native_capabilities import execution_inventory

    installed = execution_inventory()
    exported = {row["module"]: row["symbols"] for row in installed["public_physchem"]}
    board_path = root / "workstreams/world_foundation/NATIVE_CONVERGENCE_TODOLIST.md"
    board = board_path.read_text(encoding="utf-8")
    task_rows = []
    for line_number, line in enumerate(board.splitlines(), 1):
        match = re.match(r"\| (NATIVE-CONV-\d+[A-Z0-9]*) \|", line)
        if not match:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        task_rows.append(
            {
                "id": cells[0],
                "priority": cells[1],
                "scope": cells[2],
                "dependencies": cells[3],
                "python_science_task_status": cells[4],
                "owner": cells[5],
                "cpu_execution_status": "not_implemented",
                "source": str(board_path.relative_to(root)),
                "line": line_number,
            }
        )
    assert len({row["id"] for row in task_rows}) == len(task_rows)
    documents = [
        board_path,
        root / "workstreams/world_foundation/archive/20261023-completed/TASK_DETAILS.md",
        root / "workstreams/world_foundation/NATIVE_CONVERGENCE_ACCEPTANCE.md",
        root / "workstreams/world_foundation/NATIVE_FOUNDATION_ARCHITECTURE.md",
    ]
    documents.extend(
        sorted((root / "workstreams/world_foundation/reports").rglob("*EXPERIMENT.md"))
    )
    extension = root / "workstreams/world_foundation/archive/20261020-native-extension"
    documents.append(extension / "NATIVE_EXTENSION_TODOLIST.md")
    documents.extend(sorted((extension / "evidence").glob("*ACCEPTANCE.md")))
    document_lines = [
        (str(path.relative_to(root)), path.read_text(encoding="utf-8").splitlines())
        for path in documents
    ]
    for row in task_rows:
        row["existing_task_sections"] = [
            {"source": path, "line": index + 1, "heading": line}
            for path, lines in document_lines
            for index, line in enumerate(lines)
            if re.match(r"^#{2,5} " + re.escape(row["id"]) + r"(?:\uff1a|:|\b)", line)
        ]

    consumers = defaultdict(list)
    unresolved_reference_imports = []
    for folder in (root / "tests", root / "scripts", root / "src/chemworld/examples"):
        for path in sorted(folder.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            relative = str(path.relative_to(root)).replace("\\", "/")
            reference = relative.startswith("tests/reference/")
            for module, names, line in imports(tree):
                if module == "chemworld.physchem":
                    for name in names:
                        matches = [key for key, symbols in exported.items() if name in symbols]
                        # Python also permits direct submodule imports without facade aliases.
                        if (root / "src/chemworld/physchem" / (name + ".py")).is_file():
                            matches.append("chemworld.physchem." + name)
                        for key in matches:
                            consumers[key].append(
                                {
                                    "source": relative,
                                    "line": line,
                                    "symbols": [name],
                                    "reference_file": reference,
                                }
                            )
                        if reference and not matches:
                            unresolved_reference_imports.append(
                                {"source": relative, "line": line, "symbol": name}
                            )
                elif module.startswith(("chemworld.physchem.", "chemworld.data.")):
                    consumers[module].append(
                        {
                            "source": relative,
                            "line": line,
                            "symbols": names,
                            "reference_file": reference,
                        }
                    )
    source_rows = []
    # These are numerical/data entry points, not a count of independent scientific models.
    for folder in (root / "src/chemworld/physchem", root / "src/chemworld/data"):
        for path in sorted(folder.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            source = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source)
            module = ".".join(path.relative_to(root / "src").with_suffix("").parts)
            entries = [
                {
                    "name": node.name,
                    "kind": type(node).__name__,
                    "line": node.lineno,
                    "documentation": ast.get_docstring(node),
                }
                for node in tree.body
                if isinstance(node, ast.FunctionDef | ast.ClassDef)
                and not node.name.startswith("_")
            ]
            direct_imports = [module_name for module_name, _, _ in imports(tree)]
            numerical_calls = sorted(
                {
                    ast.unparse(node.func)
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and any(
                        token in ast.unparse(node.func).lower()
                        for token in (
                            "solve",
                            "root",
                            "jacobian",
                            "hessian",
                            "derivative",
                            "equilibrium",
                            "flash",
                            "integrat",
                            "advance",
                            "minimiz",
                            "saturat",
                            "brent",
                            "quadrature",
                        )
                    )
                }
            )
            evidence = [
                {"source": document, "line": index + 1, "text": line}
                for document, lines in document_lines
                for index, line in enumerate(lines)
                if path.name in line or ("`" + path.stem + "`") in line
            ]
            refs = [row for row in consumers[module] if row["reference_file"]]
            source_rows.append(
                {
                    "module": module,
                    "source": str(path.relative_to(root)).replace("\\", "/"),
                    "declared_description": ast.get_docstring(tree),
                    "public_definitions": entries,
                    "facade_exports": exported.get(module, []),
                    "mathematical_call_sites": numerical_calls,
                    "internal_math_imports": [
                        name
                        for name in direct_imports
                        if name.startswith(("scipy", "numpy", "chemworld.physchem"))
                    ],
                    "test_and_example_consumers": consumers[module],
                    "reference_consumer_status": "direct_import_found"
                    if refs
                    else "not_resolved_by_static_imports",
                    "existing_evidence_mentions": evidence,
                    "continuation": "see codecs, advance declarations and original task evidence",
                    "thread_safety": "not_declared",
                    "cpu_lowering": "not_implemented",
                    "scientific_status": "refer to original task/subblock qualification",
                }
            )
    main_tasks = [row for row in task_rows if re.fullmatch(r"NATIVE-CONV-\d+", row["id"])]
    assert len(main_tasks) == 56
    return {
        "schema": "chemworld-cpu01-source-projection-v1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "mode": "generated development evidence; status remains in original task records",
        "classification_limit": "Call/import matches locate existing mathematics and evidence; "
        "they do not infer scientific validity, thread safety or successful CPU lowering.",
        "task_rows": task_rows,
        "main_task_counts": dict(Counter(row["python_science_task_status"] for row in main_tasks)),
        "installed_declarations": installed,
        "source_module_count": len(source_rows),
        "source_public_definition_count": sum(
            len(row["public_definitions"]) for row in source_rows
        ),
        "source_modules": source_rows,
        "unresolved_reference_facade_imports": unresolved_reference_imports,
    }
