#!/usr/bin/env python3
"""Checks that every hard-coded copy of the project version agrees with VERSION.txt.

VERSION.txt is the single source of truth. CMake, the Bazel pkg-config rule and
the Python package (setup.py) all read it at build time, but a few files still
carry a literal copy that must be kept in sync by hand:

  * MODULE.bazel   -- module(name = "sentencepiece", version = "...")
  * doc/bazel.md   -- the bazel_dep(...) usage example

Run from anywhere; exits non-zero and lists every mismatch it finds.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CHECKS = [
    (
        "MODULE.bazel",
        re.compile(
            r'module\s*\(\s*name\s*=\s*"sentencepiece"\s*,\s*version\s*=\s*"([^"]+)"'
        ),
    ),
    (
        "doc/bazel.md",
        re.compile(r'bazel_dep\s*\(\s*name\s*=\s*"sentencepiece"\s*,\s*version\s*=\s*"([^"]+)"'),
    ),
]

# pyproject.toml must not pin a static version; setup.py derives it.
PYPROJECT = "python/pyproject.toml"
STATIC_VERSION = re.compile(r'^\s*version\s*=\s*"', re.MULTILINE)


def main() -> int:
    expected = (REPO_ROOT / "VERSION.txt").read_text(encoding="utf-8").strip()
    if not expected:
        print("VERSION.txt is empty", file=sys.stderr)
        return 1
    print(f"VERSION.txt: {expected}")

    errors = []
    for rel, pattern in CHECKS:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        found = pattern.findall(text)
        if not found:
            errors.append(f"{rel}: could not find a sentencepiece version declaration")
            continue
        for actual in found:
            status = "ok" if actual == expected else "MISMATCH"
            print(f"{rel}: {actual} ({status})")
            if actual != expected:
                errors.append(f"{rel}: {actual} != VERSION.txt {expected}")

    pyproject = (REPO_ROOT / PYPROJECT).read_text(encoding="utf-8")
    project_section = pyproject.split("[project]", 1)[1].split("\n[", 1)[0]
    if STATIC_VERSION.search(project_section):
        errors.append(
            f"{PYPROJECT}: [project] declares a static version; "
            'use dynamic = ["version"] so setup.py reads VERSION.txt'
        )
    else:
        print(f"{PYPROJECT}: version is dynamic (ok)")

    if errors:
        print("\nVersion mismatch:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("All versions are consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
