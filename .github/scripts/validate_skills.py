#!/usr/bin/env python3
"""Validate skill structure and frontmatter.

Checks:
1. Every skill directory has a SKILL.md file
2. SKILL.md has valid YAML frontmatter per best practices:
   - name: required, <=64 chars, lowercase letters/numbers/hyphens only,
     no XML tags, no reserved words ("anthropic", "claude")
   - description: required, non-empty, <=1024 chars, no XML tags
3. All expected skill files are present
"""

import re
import sys
from pathlib import Path

import yaml

SKILLS_DIR = Path("skills/databricks-skills")
SKIP_DIRS = {"TEMPLATE"}

EXPECTED_FILES = [
    "SKILL.md",
    "1-explore-data.md",
    "2-setup-the-mmf-cluster.md",
    "3-run-mmf.md",
    "mmf_local_notebook_template.ipynb",
    "mmf_gpu_notebook_template.ipynb",
]

RESERVED_WORDS = {"anthropic", "claude"}
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
XML_TAG_RE = re.compile(r"<[^>]+>")


def parse_frontmatter(content: str) -> dict | None:
    """Extract YAML frontmatter from markdown content."""
    match = re.match(r"^---\n(.+?)\n---", content, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
    return None


def validate_name(name: str) -> list[str]:
    """Validate the name field per best practices."""
    errors = []
    if len(name) > 64:
        errors.append(f"name '{name}' exceeds 64 characters ({len(name)})")
    if not NAME_RE.match(name):
        errors.append(f"name '{name}' must contain only lowercase letters, numbers, and hyphens")
    if XML_TAG_RE.search(name):
        errors.append(f"name '{name}' must not contain XML tags")
    for word in RESERVED_WORDS:
        if word in name:
            errors.append(f"name '{name}' must not contain reserved word '{word}'")
    return errors


def validate_description(description: str) -> list[str]:
    """Validate the description field per best practices."""
    errors = []
    if not description or not description.strip():
        errors.append("description must not be empty")
    if len(description) > 1024:
        errors.append(f"description exceeds 1024 characters ({len(description)})")
    if XML_TAG_RE.search(description):
        errors.append("description must not contain XML tags")
    return errors


def main() -> int:
    errors: list[str] = []

    if not SKILLS_DIR.is_dir():
        print(f"::error::Skills directory not found: {SKILLS_DIR}")
        return 1

    skill_dirs = sorted(
        d for d in SKILLS_DIR.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith(".")
    )

    if not skill_dirs:
        print("::error::No skill directories found")
        return 1

    for skill_dir in skill_dirs:
        skill_md = skill_dir / "SKILL.md"

        # Check SKILL.md exists
        if not skill_md.exists():
            errors.append(f"{skill_dir.name}: Missing SKILL.md")
            continue

        # Check expected files
        for filename in EXPECTED_FILES:
            if not (skill_dir / filename).exists():
                errors.append(f"{skill_dir.name}: Missing expected file '{filename}'")

        # Validate frontmatter
        content = skill_md.read_text()
        frontmatter = parse_frontmatter(content)

        if frontmatter is None:
            errors.append(f"{skill_dir.name}: Invalid or missing frontmatter in SKILL.md")
            continue

        if "name" not in frontmatter:
            errors.append(f"{skill_dir.name}: Missing 'name' field in frontmatter")
        else:
            for err in validate_name(str(frontmatter["name"])):
                errors.append(f"{skill_dir.name}: {err}")

        if "description" not in frontmatter:
            errors.append(f"{skill_dir.name}: Missing 'description' field in frontmatter")
        else:
            for err in validate_description(str(frontmatter["description"])):
                errors.append(f"{skill_dir.name}: {err}")

    # Report
    if errors:
        print("Skill validation failed:\n")
        for error in errors:
            print(f"::error::{error}")
        print()
        print(f"Found {len(errors)} error(s)")
        return 1

    print(f"All {len(skill_dirs)} skill(s) validated successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())