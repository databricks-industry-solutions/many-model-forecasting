#!/usr/bin/env python3
"""Install the Many-Model Forecasting skill into a target project.

Copies skill files and configures AI coding tools (Claude Code, Cursor, Gemini CLI)
to discover them.

Usage:
    python install.py --target /path/to/project
    python install.py --target /path/to/project --tools claude cursor
    python install.py --target /path/to/project --dry-run
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

SKILL_DIR = "databricks-skills/many-model-forecasting"
SKILL_FILES = [
    "SKILL.md",
    "1-prep-and-clean-data.md",
    "2-profile-and-classify-series.md",
    "3-provision-forecasting-resources.md",
    "4-execute-mmf-forecast.md",
    "5-post-process-and-evaluate.md",
    "mmf_local_notebook_template.ipynb",
    "mmf_gpu_notebook_template.ipynb",
    "mmf_profiling_notebook_template.ipynb",
]
MARKER = "<!-- mmf-dev-kit:skills -->"
ALL_TOOLS = ["claude", "cursor", "gemini"]
REPO_RAW_URL = "https://raw.githubusercontent.com/databricks-industry-solutions/many-model-forecasting/main/skills"

SKILL_REFERENCE_BLOCK = """\
## Many-Model Forecasting Skill

This project includes the Many-Model Forecasting (MMF) skill for Databricks.
Read these files to learn the patterns before starting any forecasting task:

- `databricks-skills/many-model-forecasting/SKILL.md` — overview and workflow
- `databricks-skills/many-model-forecasting/1-prep-and-clean-data.md` — data discovery, quality checks, and cleaning
- `databricks-skills/many-model-forecasting/2-profile-and-classify-series.md` — series profiling and classification
- `databricks-skills/many-model-forecasting/3-provision-forecasting-resources.md` — cluster setup and provisioning
- `databricks-skills/many-model-forecasting/4-execute-mmf-forecast.md` — running the forecasting pipeline
- `databricks-skills/many-model-forecasting/5-post-process-and-evaluate.md` — post-processing and evaluation
- `databricks-skills/many-model-forecasting/mmf_local_notebook_template.ipynb` — local notebook template
- `databricks-skills/many-model-forecasting/mmf_gpu_notebook_template.ipynb` — GPU notebook template
- `databricks-skills/many-model-forecasting/mmf_profiling_notebook_template.ipynb` — profiling notebook template
"""

CURSOR_RULE_CONTENT = """\
---
description: Many-Model Forecasting (MMF) on Databricks
globs:
alwaysApply: false
---

# Many-Model Forecasting Skill

This project includes the Many-Model Forecasting (MMF) skill for Databricks.
Read these files to learn the patterns before starting any forecasting task:

- `databricks-skills/many-model-forecasting/SKILL.md` — overview and workflow
- `databricks-skills/many-model-forecasting/1-prep-and-clean-data.md` — data discovery, quality checks, and cleaning
- `databricks-skills/many-model-forecasting/2-profile-and-classify-series.md` — series profiling and classification
- `databricks-skills/many-model-forecasting/3-provision-forecasting-resources.md` — cluster setup and provisioning
- `databricks-skills/many-model-forecasting/4-execute-mmf-forecast.md` — running the forecasting pipeline
- `databricks-skills/many-model-forecasting/5-post-process-and-evaluate.md` — post-processing and evaluation
- `databricks-skills/many-model-forecasting/mmf_local_notebook_template.ipynb` — local notebook template
- `databricks-skills/many-model-forecasting/mmf_gpu_notebook_template.ipynb` — GPU notebook template
- `databricks-skills/many-model-forecasting/mmf_profiling_notebook_template.ipynb` — profiling notebook template
"""


def resolve_source_dir() -> Path:
    """Find skill source files relative to this script, or download from GitHub."""
    source = Path(__file__).resolve().parent / SKILL_DIR
    if source.is_dir() and any(source.iterdir()):
        return source

    # Running standalone (e.g. downloaded install.py only) — fetch from GitHub
    print("Skill files not found locally, downloading from GitHub...")
    tmp = Path(tempfile.mkdtemp(prefix="mmf-dev-kit-"))
    skill_tmp = tmp / SKILL_DIR
    skill_tmp.mkdir(parents=True)
    for name in SKILL_FILES:
        url = f"{REPO_RAW_URL}/{SKILL_DIR}/{name}"
        try:
            data = urlopen(url).read()
        except URLError as e:
            print(f"Error: failed to download {url}: {e}", file=sys.stderr)
            sys.exit(1)
        (skill_tmp / name).write_bytes(data)
        print(f"  downloaded: {name}")
    return skill_tmp


def copy_skill_files(src: Path, tgt: Path, dry_run: bool) -> list[str]:
    """Copy skill files from src to tgt. Returns list of actions taken."""
    actions = []
    tgt.mkdir(parents=True, exist_ok=True)
    for name in SKILL_FILES:
        src_file = src / name
        tgt_file = tgt / name
        if not src_file.exists():
            print(f"Warning: source file missing: {src_file}", file=sys.stderr)
            continue
        action = "would copy" if dry_run else "copied"
        if not dry_run:
            shutil.copy2(src_file, tgt_file)
        actions.append(f"  {action}: {tgt_file}")
    return actions


def _update_marker_block(file_path: Path, block_content: str, dry_run: bool) -> str:
    """Insert or replace a marker-delimited block in a file. Returns action description."""
    marked_block = f"{MARKER}\n{block_content}\n{MARKER}\n"

    if file_path.exists():
        existing = file_path.read_text()
        parts = existing.split(MARKER)
        if len(parts) >= 3:
            # Replace existing block: parts[0] is before, parts[2] is after
            new_content = f"{parts[0]}{marked_block}{parts[2].lstrip()}"
            action_verb = "would update" if dry_run else "updated"
        else:
            # Append block
            separator = "\n" if existing and not existing.endswith("\n") else ""
            new_content = f"{existing}{separator}\n{marked_block}"
            action_verb = "would update" if dry_run else "updated"
    else:
        new_content = marked_block
        action_verb = "would create" if dry_run else "created"

    if not dry_run:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(new_content)

    return f"  {action_verb}: {file_path}"


def configure_claude(target: Path, dry_run: bool) -> str:
    """Append/update a marked block in CLAUDE.md."""
    return _update_marker_block(target / "CLAUDE.md", SKILL_REFERENCE_BLOCK, dry_run)


def configure_gemini(target: Path, dry_run: bool) -> str:
    """Append/update a marked block in GEMINI.md."""
    return _update_marker_block(target / "GEMINI.md", SKILL_REFERENCE_BLOCK, dry_run)


def configure_cursor(target: Path, dry_run: bool) -> str:
    """Write .cursor/rules/many-model-forecasting.mdc."""
    rule_path = target / ".cursor" / "rules" / "many-model-forecasting.mdc"
    action_verb = "would create" if dry_run else "created"
    if rule_path.exists():
        action_verb = "would overwrite" if dry_run else "overwrote"
    if not dry_run:
        rule_path.parent.mkdir(parents=True, exist_ok=True)
        rule_path.write_text(CURSOR_RULE_CONTENT)
    return f"  {action_verb}: {rule_path}"


def main():
    parser = argparse.ArgumentParser(description="Install the Many-Model Forecasting skill into a target project.")
    parser.add_argument(
        "--target",
        type=Path,
        default=Path.cwd(),
        help="Target project directory (default: current directory)",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=ALL_TOOLS,
        default=ALL_TOOLS,
        help="Which AI tools to configure (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    args = parser.parse_args()

    target = args.target.resolve()
    source = resolve_source_dir()

    if args.dry_run:
        print("Dry run — no files will be modified.\n")

    # Copy skill files
    actions = copy_skill_files(source, target / SKILL_DIR, args.dry_run)

    # Configure tools
    tool_configurators = {
        "claude": configure_claude,
        "cursor": configure_cursor,
        "gemini": configure_gemini,
    }
    for tool in args.tools:
        actions.append(tool_configurators[tool](target, args.dry_run))

    # Summary
    print("Summary:")
    for action in actions:
        print(action)
    print(f"\nDone. {'(dry run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
