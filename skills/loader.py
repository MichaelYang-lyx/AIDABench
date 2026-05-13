"""Skills loader for agent capabilities — pipeline 2.0

Enhanced from 1.0:
- Recursive discovery: finds SKILL.md in any nested subdirectory
  (supports workflow/ and capability/ multi-level layouts)
- Extracts description from SKILL.md markdown headers (## Description)
  when no YAML frontmatter is present
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class SkillsLoader:
    """Loader for agent skills — discovers SKILL.md files recursively."""

    def __init__(self, skills_dir: Union[Path, str]):
        self.skills_dir = Path(skills_dir)

    # ── Discovery ──

    def _find_all_skill_files(self) -> List[Path]:
        """Find only top-level SKILL.md files (direct children of skills_dir).

        Sub-skills (e.g. capability/*/SKILL.md) are NOT discovered here —
        they are loaded on-demand by the model via read_file when the
        top-level workflow SKILL.md references them.
        """
        if not self.skills_dir.exists():
            return []
        results = []
        for child in sorted(self.skills_dir.iterdir()):
            if child.is_dir():
                skill_file = child / "SKILL.md"
                if skill_file.exists():
                    results.append(skill_file)
        return results

    def list_skills(self, filter_unavailable: bool = True) -> List[Dict[str, str]]:
        """List all discovered skills."""
        skills = []
        for skill_file in self._find_all_skill_files():
            skill_dir = skill_file.parent
            # Build a relative name like "workflow/excel_data_analysis_workflow"
            try:
                rel = skill_dir.relative_to(self.skills_dir)
                name = str(rel)
            except ValueError:
                name = skill_dir.name

            meta = self._extract_metadata(skill_file)
            description = meta.get("description", name)

            skill_meta = self._parse_nanobot_metadata(meta.get("metadata", ""))
            available = self._check_requirements(skill_meta)

            if not filter_unavailable or available:
                skills.append({
                    "name": name,
                    "path": str(skill_file),
                    "description": description,
                    "skill_type": meta.get("skill_type", "unknown"),
                    "available": available,
                })
        return skills

    def load_skill(self, name: str) -> Optional[str]:
        """Load a skill by name (can be nested path like 'workflow/xxx')."""
        skill_file = self.skills_dir / name / "SKILL.md"
        if skill_file.exists():
            return skill_file.read_text(encoding="utf-8")
        return None

    def load_skill_by_path(self, path: str) -> Optional[str]:
        """Load a skill by its absolute path."""
        p = Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    def load_skills_for_context(self, skill_names: List[str]) -> str:
        parts = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                content = self._strip_frontmatter(content)
                parts.append(f"### Skill: {name}\n\n{content}")
        return "\n\n---\n\n".join(parts) if parts else ""

    def build_skills_summary(self) -> str:
        """Build XML summary of all skills for system prompt injection."""
        all_skills = self.list_skills(filter_unavailable=False)
        if not all_skills:
            return "(No skills available)"

        def esc(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<skills>"]
        for s in all_skills:
            lines.append(f'  <skill available="{str(s["available"]).lower()}">')
            lines.append(f"    <name>{esc(s['name'])}</name>")
            lines.append(f"    <description>{esc(s['description'])}</description>")
            lines.append(f"    <location>{s['path']}</location>")
            lines.append(f"  </skill>")
        lines.append("</skills>")
        return "\n".join(lines)

    # ── Metadata extraction ──

    def _extract_metadata(self, skill_file: Path) -> Dict[str, Any]:
        """Extract metadata from a SKILL.md — tries frontmatter first, then markdown headers."""
        try:
            content = skill_file.read_text(encoding="utf-8")
        except Exception:
            return {}

        meta: Dict[str, Any] = {}

        # Try YAML frontmatter
        if content.startswith("---"):
            match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if match:
                for line in match.group(1).split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        meta[key.strip()] = value.strip().strip("\"'")

        # Also try markdown ## headers (works for the workflow format)
        # ## Description → description
        # ## Skill Type → skill_type
        desc_match = re.search(r"^## Description\s*\n(.+?)(?=\n## |\Z)", content, re.MULTILINE | re.DOTALL)
        if desc_match and "description" not in meta:
            desc = desc_match.group(1).strip()
            # Take first paragraph only (avoid huge descriptions)
            first_para = desc.split("\n\n")[0].strip()
            if len(first_para) > 300:
                first_para = first_para[:300] + "..."
            meta["description"] = first_para

        type_match = re.search(r"^## Skill Type\s*\n(\w+)", content, re.MULTILINE)
        if type_match and "skill_type" not in meta:
            meta["skill_type"] = type_match.group(1).strip()

        return meta

    def get_skill_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        skill_file = self.skills_dir / name / "SKILL.md"
        if skill_file.exists():
            return self._extract_metadata(skill_file)
        return None

    # ── Helpers ──

    def _strip_frontmatter(self, content: str) -> str:
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                return content[match.end():].strip()
        return content

    def _parse_nanobot_metadata(self, raw: str) -> Dict[str, Any]:
        try:
            data = json.loads(raw)
            return data.get("nanobot", {}) if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _check_requirements(self, skill_meta: Dict[str, Any]) -> bool:
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(b):
                return False
        for env in requires.get("env", []):
            if not os.environ.get(env):
                return False
        return True
