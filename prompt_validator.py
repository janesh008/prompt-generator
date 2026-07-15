"""Validation and parsing helpers for LittleNest prompt batches."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


LOCKED_SECTION_HEADINGS = [
    "MAIN_CHARACTER",
    "SUB_CHARACTER_1",
    "SUB_CHARACTER_2",
    "SUB_CHARACTER_3",
    "SUB_CHARACTER_4",
    "SUB_CHARACTER_5",
    "SUB_CHARACTER_6",
    "SUB_CHARACTER_7",
    "SUB_CHARACTER_8",
    "CHARACTER_COMBO_2",
    "CHARACTER_COMBO_3",
    "CHARACTER_COMBO_4",
    "CHARACTER_COMBO_FULL_GROUP",
    "PATTERN",
    "PROP",
    "SCENE",
    "LOGO_EMBLEM",
    "BANNER",
    "ALPHABET_NUMBER",
    "FRAME_BORDER",
]

ALWAYS_ACTIVE_SECTIONS = {"MAIN_CHARACTER", "PATTERN", "PROP", "SCENE"}
INACTIVE_NOTE = "(not applicable for this roster)"


@dataclass
class ValidationResult:
    ok: bool
    issues: List[str]
    section_counts: Dict[str, int]

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("Prompt batch validation failed:\n- " + "\n- ".join(self.issues))


def validate_prompt_batch(text: str) -> ValidationResult:
    issues: List[str] = []
    found_headings = extract_headings(text)
    found_names = [heading["name"] for heading in found_headings]

    for expected in LOCKED_SECTION_HEADINGS:
        if expected not in found_names:
            issues.append(f"Missing locked heading: ## {expected}")

    for heading in found_headings:
        if heading["name"] not in LOCKED_SECTION_HEADINGS:
            issues.append(f"Unexpected or malformed heading: {heading['raw']}")

    ordered_found = [name for name in found_names if name in LOCKED_SECTION_HEADINGS]
    if ordered_found != LOCKED_SECTION_HEADINGS:
        issues.append("Locked headings are not present in the required order.")

    if "```" in text:
        issues.append("Output contains a code fence; expected plain text only.")

    if re.search(r"\b(master_prompt|negative_prompt|batch_\d+)\s*=", text):
        issues.append("Output contains code-style wrapper variables.")

    sections = split_sections(text)
    section_counts: Dict[str, int] = {}

    for heading in LOCKED_SECTION_HEADINGS:
        body = sections.get(heading, "").strip()
        count = count_numbered_prompts(body)
        section_counts[heading] = count
        inactive = body == INACTIVE_NOTE

        if heading in ALWAYS_ACTIVE_SECTIONS and inactive:
            issues.append(f"## {heading} is marked inactive but should always be active.")

        if not inactive and count > 0 and count < 10:
            issues.append(
                f"Active section ## {heading} has only {count} prompts; "
                "active sections need at least 10."
            )

        if heading in sections and not inactive and count == 0 and body:
            issues.append(
                f"## {heading} has text but no numbered prompts. "
                "Use numbered prompt lines or the inactive note."
            )

    return ValidationResult(
        ok=len(issues) == 0,
        issues=issues,
        section_counts=section_counts,
    )


def extract_headings(text: str) -> List[Dict[str, object]]:
    headings: List[Dict[str, object]] = []
    for match in re.finditer(r"^##\s+(.+)$", text, flags=re.MULTILINE):
        headings.append(
            {
                "raw": match.group(0),
                "name": match.group(1).strip(),
                "index": match.start(),
            }
        )
    return headings


def split_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    matches = list(re.finditer(r"^##\s+(.+)$", text, flags=re.MULTILINE))

    for index, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()

    return sections


def count_numbered_prompts(section_body: str) -> int:
    return len(re.findall(r"^\d+\.\s+", section_body, flags=re.MULTILINE))


def extract_numbered_prompts(section_body: str) -> List[str]:
    """Return prompt text lines from a section body without the leading number."""

    prompts: List[str] = []
    current: List[str] = []

    for line in section_body.splitlines():
        numbered = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if numbered:
            if current:
                prompts.append(" ".join(part.strip() for part in current).strip())
            current = [numbered.group(1).strip()]
        elif current and line.strip():
            current.append(line.strip())

    if current:
        prompts.append(" ".join(part.strip() for part in current).strip())

    return prompts


def parse_prompt_batch(text: str, *, active_only: bool = True) -> Dict[str, List[str]]:
    """Split a batch into section -> prompt list.

    This is useful for feeding the generated prompts into your image generation
    loop section by section.
    """

    sections = split_sections(text)
    parsed: Dict[str, List[str]] = {}

    for heading in LOCKED_SECTION_HEADINGS:
        body = sections.get(heading, "").strip()
        if active_only and body == INACTIVE_NOTE:
            continue
        prompts = extract_numbered_prompts(body)
        if prompts or not active_only:
            parsed[heading] = prompts

    return parsed
