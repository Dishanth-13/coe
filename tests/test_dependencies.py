"""
FIX 5 regression tests: runtime code must depend only on requirements.txt; test-only packages
(httpx for FastAPI's TestClient) belong in requirements-dev.txt.
"""

import ast
import os
import re
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# import name -> distribution name, for packages whose names differ
IMPORT_TO_DIST = {"faiss": "faiss-cpu", "sklearn": "scikit-learn"}


def _requirements(path):
    names, includes = set(), []
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("-r "):
                includes.append(line[3:].strip())
                continue
            names.add(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower())
    for inc in includes:
        names |= _requirements(inc)
    return names


def _third_party_imports(paths):
    stdlib = set(sys.stdlib_module_names)
    # First-party modules: the package, the demo script, the tests package and its modules
    local = {"vecta", "demo", "tests"} | {
        os.path.splitext(f)[0] for f in os.listdir(os.path.join(ROOT, "tests")) if f.endswith(".py")
    }
    found = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            else:
                continue
            for mod in mods:
                top = mod.split(".")[0]
                if top not in stdlib and top not in local:
                    found.add(top)
    return found


def _py_files(*rel_dirs_or_files):
    out = []
    for rel in rel_dirs_or_files:
        full = os.path.join(ROOT, rel)
        if os.path.isfile(full):
            out.append(full)
        else:
            for dirpath, _, files in os.walk(full):
                out += [os.path.join(dirpath, f) for f in files if f.endswith(".py")]
    return out


class TestDependencySeparation(unittest.TestCase):

    def test_runtime_requirements_exclude_test_only_packages(self):
        runtime = _requirements("requirements.txt")
        self.assertNotIn("httpx", runtime)
        self.assertNotIn("httpx2", runtime)

    def test_dev_requirements_include_runtime_and_httpx(self):
        dev = _requirements("requirements-dev.txt")
        self.assertTrue(_requirements("requirements.txt") <= dev)
        self.assertIn("httpx", dev)

    def test_runtime_code_imports_only_runtime_requirements(self):
        runtime = _requirements("requirements.txt")
        imports = _third_party_imports(_py_files("vecta", "demo.py", "run_dashboard.py"))
        self.assertGreater(len(imports), 5)
        missing = {m for m in imports if IMPORT_TO_DIST.get(m, m).lower() not in runtime}
        self.assertEqual(missing, set(), "runtime code imports packages missing from requirements.txt")

    def test_test_code_imports_are_covered_by_dev_requirements(self):
        dev = _requirements("requirements-dev.txt")
        imports = _third_party_imports(_py_files("tests"))
        missing = {m for m in imports if IMPORT_TO_DIST.get(m, m).lower() not in dev}
        self.assertEqual(missing, set())


if __name__ == "__main__":
    unittest.main()
