#!/usr/bin/env python3
# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# .github/scripts/ -> repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CMAKELISTS_PATH = REPO_ROOT / "CMakeLists.txt"
MODULE_BAZEL_PATH = REPO_ROOT / "MODULE.bazel"
VERSION_TXT_PATH = REPO_ROOT / "VERSION.txt"

# Security & sanity regex patterns
VALID_TAG_PATTERN = re.compile(r"^[a-zA-Z0-9_.\-]+$")
PRERELEASE_PATTERN = re.compile(
    r"[-._](rc\d*|alpha\d*|beta\d*|dev\d*|preview\d*|ea\d*)$", re.IGNORECASE
)


def is_valid_tag(tag: str) -> bool:
    """Validate tag format to prevent code injection and skip pre-releases."""
    if not tag or not isinstance(tag, str):
        return False
    # Only allow standard tag characters
    if not VALID_TAG_PATTERN.match(tag):
        print(f"Warning: Rejected tag '{tag}' with invalid characters.")
        return False
    # Reject pre-release suffixes (rc, alpha, beta, etc.)
    if PRERELEASE_PATTERN.search(tag):
        print(f"Info: Skipping pre-release tag '{tag}'.")
        return False
    return True


def get_latest_github_tag(repo_path: str) -> str:
    """Fetch the latest release tag or tag for a GitHub repository."""
    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "User-Agent": "SentencePiece-Dependency-Updater",
        "Accept": "application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Try /releases/latest first (skips drafts and pre-releases)
    release_url = f"https://api.github.com/repos/{repo_path}/releases/latest"
    req = urllib.request.Request(release_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tag = data.get("tag_name")
            if tag and is_valid_tag(tag):
                return tag
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"Warning: HTTP {e.code} fetching releases for {repo_path}: {e}")
    except Exception as e:
        print(f"Warning: Error fetching release for {repo_path}: {e}")

    # Fallback to /tags, iterating to find the first valid non-prerelease tag
    tags_url = f"https://api.github.com/repos/{repo_path}/tags?per_page=10"
    req = urllib.request.Request(tags_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data and isinstance(data, list):
                for item in data:
                    name = item.get("name")
                    if name and is_valid_tag(name):
                        return name
    except Exception as e:
        print(f"Error fetching tags for {repo_path}: {e}")

    return ""


def get_latest_bcr_version(module_name: str) -> str:
    """Fetch the latest non-yanked version from Bazel Central Registry (BCR)."""
    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "User-Agent": "SentencePiece-Dependency-Updater",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    bcr_url = f"https://raw.githubusercontent.com/bazelbuild/bazel-central-registry/main/modules/{module_name}/metadata.json"
    req = urllib.request.Request(bcr_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            versions = data.get("versions", [])
            yanked = data.get("yanked_versions", {})
            # Filter out yanked versions and pre-releases
            valid_versions = [
                v
                for v in versions
                if v not in yanked and not PRERELEASE_PATTERN.search(v)
            ]
            if valid_versions:
                return valid_versions[-1]
            elif versions:
                non_yanked = [v for v in versions if v not in yanked]
                if non_yanked:
                    return non_yanked[-1]
    except Exception as e:
        print(f"Warning: Error fetching BCR metadata for {module_name}: {e}")

    return ""


def update_cmake_content(content: str):
    # Match FetchContent_Declare blocks
    pattern = re.compile(
        r"FetchContent_Declare\s*\(\s*([a-zA-Z0-9_\-]+)\s+([\s\S]*?)\)",
        re.MULTILINE,
    )

    updates = []

    def replace_block(match):
        full_block = match.group(0)
        target_name = match.group(1)
        body = match.group(2)

        repo_match = re.search(
            r"GIT_REPOSITORY\s+https://github\.com/([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+?)(?:\.git)?(?:\s|\))",
            body,
        )
        tag_match = re.search(r"GIT_TAG\s+([^\s\)]+)", body)

        if not repo_match or not tag_match:
            return full_block

        repo_path = repo_match.group(1)
        current_tag = tag_match.group(1)

        print(f"Found CMake dependency: {target_name} ({repo_path}) @ {current_tag}")
        latest_tag = get_latest_github_tag(repo_path)

        if not latest_tag:
            print(f"  -> Could not determine latest valid tag for {repo_path}. Keeping current.")
            return full_block

        if current_tag != latest_tag:
            print(f"  -> Updating {target_name}: {current_tag} -> {latest_tag}")
            updates.append(f"CMake {target_name}: {current_tag} -> {latest_tag}")
            # Replace only the GIT_TAG in this specific block, preserving original formatting
            new_block = re.sub(
                r"(GIT_TAG\s+)" + re.escape(current_tag),
                r"\g<1>" + latest_tag,
                full_block,
                count=1,
            )
            return new_block
        else:
            print(f"  -> {target_name} is already up to date ({current_tag}).")
            return full_block

    new_content = pattern.sub(replace_block, content)
    return new_content, updates


def update_module_bazel_content(content: str):
    updates = []
    new_content = content

    # 1. Synchronize version with VERSION.txt
    if VERSION_TXT_PATH.exists():
        version_val = VERSION_TXT_PATH.read_text(encoding="utf-8").strip()
        ver_pattern = re.compile(
            r'(module\s*\(\s*name\s*=\s*"sentencepiece"\s*,\s*version\s*=\s*)"([^"]+)"',
            re.MULTILINE,
        )
        match = ver_pattern.search(new_content)
        if match and match.group(2) != version_val:
            current_ver = match.group(2)
            new_content = ver_pattern.sub(rf'\g<1>"{version_val}"', new_content, count=1)
            updates.append(f"MODULE.bazel version: {current_ver} -> {version_val}")

    # 2. Update all bazel_dep declarations from BCR
    dep_pattern = re.compile(
        r'(bazel_dep\s*\(\s*name\s*=\s*"([a-zA-Z0-9_\-]+)"\s*,\s*version\s*=\s*)"([^"]+)"',
        re.MULTILINE,
    )

    def replace_dep(match):
        prefix = match.group(1)
        mod_name = match.group(2)
        current_ver = match.group(3)

        print(f"Found Bazel dependency: {mod_name} @ {current_ver}")
        latest_ver = get_latest_bcr_version(mod_name)

        if not latest_ver:
            print(f"  -> Could not determine latest valid BCR version for {mod_name}. Keeping current.")
            return match.group(0)

        if current_ver != latest_ver:
            print(f"  -> Updating bazel_dep {mod_name}: {current_ver} -> {latest_ver}")
            updates.append(f"bazel_dep {mod_name}: {current_ver} -> {latest_ver}")
            return f'{prefix}"{latest_ver}"'
        else:
            print(f"  -> bazel_dep {mod_name} is already up to date ({current_ver}).")
            return match.group(0)

    new_content = dep_pattern.sub(replace_dep, new_content)
    return new_content, updates


def main():
    all_updates = []

    if CMAKELISTS_PATH.exists():
        cmake_content = CMAKELISTS_PATH.read_text(encoding="utf-8")
        new_cmake, cmake_updates = update_cmake_content(cmake_content)
        if cmake_updates:
            CMAKELISTS_PATH.write_text(new_cmake, encoding="utf-8")
            all_updates.extend(cmake_updates)

    if MODULE_BAZEL_PATH.exists():
        bazel_content = MODULE_BAZEL_PATH.read_text(encoding="utf-8")
        new_bazel, bazel_updates = update_module_bazel_content(bazel_content)
        if bazel_updates:
            MODULE_BAZEL_PATH.write_text(new_bazel, encoding="utf-8")
            all_updates.extend(bazel_updates)

    if all_updates:
        print("\nSuccessfully updated dependencies:")
        for u in all_updates:
            print(f"  * {u}")
        summary_file = os.getenv("GITHUB_STEP_SUMMARY_PATH", "/tmp/update_summary.txt")
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(", ".join(all_updates))
        except Exception as e:
            print(f"Warning: Could not write summary file: {e}")
    else:
        print("\nAll dependencies in CMakeLists.txt and MODULE.bazel are up to date.")


if __name__ == "__main__":
    main()
