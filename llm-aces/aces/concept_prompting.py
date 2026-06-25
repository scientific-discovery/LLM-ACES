from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt_template(template_name: str) -> str:
    return (PROMPTS_DIR / template_name).read_text(encoding="utf-8")


def render_prompt_template(template_name: str, replacements: dict[str, object]) -> str:
    template = load_prompt_template(template_name)
    safe_replacements = {key: str(value) for key, value in replacements.items()}
    return template.format(**safe_replacements).strip() + "\n"


def extract_task_summary(spec_text: str, dim: int) -> str:
    for raw_line in spec_text.splitlines():
        line = raw_line.strip().strip('"').strip("'")
        if line:
            return line
    return (
        f"Find the governing equations of a {dim}-dimensional autonomous ODE "
        "system from observed trajectory data."
    )


def extract_spec_instruction(spec_text: str) -> str:
    stripped = spec_text.strip()
    if stripped.startswith('"""') or stripped.startswith("'''"):
        quote = stripped[:3]
        end_index = stripped.find(quote, 3)
        if end_index != -1:
            return stripped[3:end_index].strip()

    lines: list[str] = []
    for raw_line in spec_text.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _format_stat_block(name: str, arr: np.ndarray) -> str:
    return (
        f"{name}: min={np.min(arr):.6g}, max={np.max(arr):.6g}, "
        f"mean={np.mean(arr):.6g}, std={np.std(arr):.6g}"
    )


def format_queried_initial_conditions(queried_ics: Iterable[np.ndarray] | None) -> str:
    if not queried_ics:
        return "None yet."

    lines = []
    for index, u0 in enumerate(queried_ics, start=1):
        arr = np.asarray(u0, dtype=float).reshape(-1)
        coords = ", ".join(f"{value:.6g}" for value in arr)
        lines.append(f"{index}. [{coords}]")
    return "\n".join(lines)


def build_data_summary(data_dict: dict, queried_ics: Iterable[np.ndarray] | None = None) -> str:
    queried_ics_list = list(queried_ics) if queried_ics is not None else []
    t = np.asarray(data_dict["t"], dtype=float)
    u = np.asarray(data_dict["u"], dtype=float)
    du = np.asarray(data_dict["du"], dtype=float)

    if u.ndim == 1:
        u = u[:, np.newaxis]
    if du.ndim == 1:
        du = du[:, np.newaxis]

    dt = np.diff(t) if len(t) > 1 else np.array([], dtype=float)
    is_uniform = bool(len(dt) == 0 or np.allclose(dt, dt[0], rtol=1e-4, atol=1e-8))

    lines = [
        f"time_points: {len(t)}",
        f"state_dimension: {u.shape[1]}",
        f"time_range: [{np.min(t):.6g}, {np.max(t):.6g}]",
        f"time_spacing: {'approximately_uniform' if is_uniform else 'nonuniform'}",
    ]

    for idx in range(u.shape[1]):
        lines.append(_format_stat_block(f"x_{idx}", u[:, idx]))
    for idx in range(du.shape[1]):
        lines.append(_format_stat_block(f"dx_{idx}/dt", du[:, idx]))
    lines.append(f"queried_initial_condition_count: {len(queried_ics_list)}")

    return "\n".join(lines)


def format_existing_concepts(concepts: Iterable[str]) -> str:
    formatted = []
    for index, concept in enumerate(concepts, start=1):
        formatted.append(f"{index}. {concept}")
    return "\n".join(formatted) if formatted else "None"


def _extract_mechanism(text: str) -> str:
    """Extract the mechanism line from a structured concept block for deduplication."""
    for line in text.splitlines():
        m = re.match(r"^\s*mechanism\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # Fall back to first non-empty line
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def normalize_concept(text: str) -> str:
    key = _extract_mechanism(text) if "\n" in text.strip() else text
    normalized = key.strip().lower()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"^concept\s*\d+\s*:\s*", "", normalized)
    normalized = re.sub(r"^[-*]\s*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" .:;,'\"")


def parse_concept_response(response: str) -> str:
    text = response.strip()
    if not text:
        return ""

    lines = text.splitlines()

    # Strip a leading "## Concept N" / "Concept N:" header line if present
    start = 0
    if lines and re.match(r"^#{0,3}\s*concept\s*\d*\s*:?\s*$", lines[0].strip(), re.IGNORECASE):
        start = 1

    # Collect only the first concept block — stop at the next "## Concept N" header
    block_lines = []
    for line in lines[start:]:
        if block_lines and re.match(r"^#{0,3}\s*concept\s*\d+", line.strip(), re.IGNORECASE):
            break
        block_lines.append(line)

    block = "\n".join(block_lines).strip()
    if not block:
        return ""

    # If the block is just the stop token, return it as-is for is_stop_response
    first_line = block.splitlines()[0].strip()
    if re.sub(r"\s+", " ", first_line.lower()) in ("no_new_concept", "no new concept"):
        return first_line

    return block


def is_stop_response(response: str, stop_token: str) -> bool:
    # Check the first meaningful line so multi-line blocks don't bypass stop detection
    for line in response.splitlines():
        candidate = line.strip()
        if candidate:
            return normalize_concept(candidate) == normalize_concept(stop_token)
    return False


def is_duplicate_concept(candidate: str, concepts: Iterable[str]) -> bool:
    normalized_candidate = normalize_concept(candidate)
    if not normalized_candidate:
        return True
    normalized_existing = {normalize_concept(concept) for concept in concepts}
    return normalized_candidate in normalized_existing
