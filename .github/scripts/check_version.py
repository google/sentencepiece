#!/usr/bin/env python3
"""Checks that every hard-coded copy of the project version agrees with VERSION.txt.

VERSION.txt is the single source of truth for the project version.

Releasing a new version
-----------------------
Edit these files by hand. This script verifies all of them, so a forgotten one
fails CI rather than shipping:

  1. VERSION.txt   -- the version itself; everything below follows from it.
  2. MODULE.bazel  -- module(name = "sentencepiece", version = "..."). Bazel
                      requires a literal here and cannot read VERSION.txt.
  3. doc/bazel.md  -- the bazel_dep(name = "sentencepiece", version = "...")
                      example telling users which release to depend on.
  4. doc/cpp.md    -- the GIT_TAG in the FetchContent_Declare example, which
                      must name the tag of the release being described.

Then tag the release as v<version> (e.g. v0.2.3) to match the GIT_TAG above.

Derived automatically -- do not edit
------------------------------------
  * CMake         -- project(VERSION) reads VERSION.txt and feeds config.h,
                     sentencepiece.pc and the CPack package name.
  * Bazel         -- the //src:config_h genrule and the sentencepiece.pc rule
                     read VERSION.txt directly.
  * Python        -- python/setup.py reads VERSION.txt and generates
                     python/src/sentencepiece/_version.py; pyproject.toml
                     declares the version as dynamic.

Deliberately not bumped
-----------------------
Documentation states when a feature first appeared, for example "v0.2.3+" in
doc/options.md and doc/piece_constraints.md, or the "version >= 0.2.3" heading
in doc/cpp.md. Those describe history and must keep their original version.

Run from anywhere; exits non-zero and lists every mismatch it finds.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# (path, pattern capturing the version, prefix the captured value carries)
CHECKS = [
    (
        "MODULE.bazel",
        re.compile(
            r'module\s*\(\s*name\s*=\s*"sentencepiece"\s*,\s*version\s*=\s*"([^"]+)"'
        ),
        "",
    ),
    (
        "doc/bazel.md",
        re.compile(
            r'bazel_dep\s*\(\s*name\s*=\s*"sentencepiece"\s*,\s*version\s*=\s*"([^"]+)"'
        ),
        "",
    ),
    (
        # The FetchContent example must point at the tag of this release.
        "doc/cpp.md",
        re.compile(r"GIT_TAG\s+v([0-9][^\s]*)"),
        "v",
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
    for rel, pattern, prefix in CHECKS:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        found = pattern.findall(text)
        if not found:
            errors.append(f"{rel}: could not find a sentencepiece version declaration")
            continue
        for actual in found:
            status = "ok" if actual == expected else "MISMATCH"
            print(f"{rel}: {prefix}{actual} ({status})")
            if actual != expected:
                errors.append(
                    f"{rel}: {prefix}{actual} != VERSION.txt {prefix}{expected}"
                )

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
