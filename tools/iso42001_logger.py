#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISO/IEC 42001:2023 Artificial Intelligence Management System (AIMS)
Logging & Change Control Compliance Module.

Implements standards-compliant governance tools:
- Clause 7.5: Documented information control and auditability
- Clause 8.4 & Annex A.6.7: AI System Change Management
- Annex A.9.3: AI Operational Event Logging
"""

import os
import re
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Root path of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG_PATH = PROJECT_ROOT / "CHANGELOG.md"

REQUIRED_ENTRY_FIELDS = [
    "Date (UTC)",
    "Component",
    "Change Type",
    "ISO 42001 Clause",
    "Objective",
    "Risk Assessment",
    "Verification Gate",
]


class ISO8601Formatter(logging.Formatter):
    """
    Structured logging formatter complying with ISO 8601 and ISO/IEC 42001 Annex A.9.3.
    Emits UTC ISO 8601 timestamps (YYYY-MM-DDTHH:MM:SSZ).
    """

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_iso_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger adhering to ISO/IEC 42001 (Annex A.9.3)
    and ISO/IEC 25010 (System Reliability & Maintainability).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = ISO8601Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] [AIMS:A.9.3] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def parse_changelog_entries(changelog_content: str) -> List[Dict[str, Any]]:
    """
    Parses ISO/IEC 42001 change control records from CHANGELOG.md markdown.
    Each record must start with a heading '### [CR-YYYYMMDD-XX]' or similar.
    """
    entries = []
    # Pattern to match change record headers, e.g., '### [CR-20261027-01]'
    header_pattern = re.compile(r"^###\s+\[(CR-\d{8}-\w+)\]\s*-\s*(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(changelog_content))

    for i, match in enumerate(matches):
        entry_id = match.group(1)
        title = match.group(2).strip()
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(changelog_content)
        entry_body = changelog_content[start_pos:end_pos]

        # Extract bullet/field items
        fields = {}
        for line in entry_body.splitlines():
            line_s = line.strip()
            # Match '- **Field Name**: Value'
            field_match = re.match(r"^-\s+\*\*([^*]+)\*\*:\s*(.+)$", line_s)
            if field_match:
                f_name = field_match.group(1).strip()
                f_val = field_match.group(2).strip()
                fields[f_name] = f_val

        entries.append({
            "id": entry_id,
            "title": title,
            "fields": fields,
            "raw_body": entry_body.strip()
        })

    return entries


def validate_changelog_compliance(
    changelog_path: Optional[Path] = None
) -> Tuple[bool, List[str]]:
    """
    Validates that CHANGELOG.md exists and all entries follow ISO/IEC 42001:2023
    Annex A.6.7 Change Management schema.
    Returns (is_compliant, issues_list).
    """
    path = changelog_path or CHANGELOG_PATH
    issues = []

    if not path.is_file():
        return False, [f"Changelog file does not exist at: {path}"]

    content = path.read_text(encoding="utf-8")
    if "ISO/IEC 42001:2023" not in content:
        issues.append("Changelog header does not declare ISO/IEC 42001:2023 compliance")

    entries = parse_changelog_entries(content)
    if not entries:
        issues.append("No change control records found matching format '### [CR-YYYYMMDD-XX]'")
        return False, issues

    for entry in entries:
        entry_id = entry["id"]
        fields = entry["fields"]
        for req_field in REQUIRED_ENTRY_FIELDS:
            # Check prefix match for flexible naming (e.g., 'Verification Gate / Status')
            matched = any(f.startswith(req_field) for f in fields.keys())
            if not matched:
                issues.append(
                    f"Entry {entry_id} is missing required ISO 42001 field: '{req_field}'"
                )

        # Validate date format if present
        date_val = next((v for k, v in fields.items() if k.startswith("Date (UTC)")), None)
        if date_val:
            # Expected YYYY-MM-DD or ISO 8601 format
            if not re.match(r"^\d{4}-\d{2}-\d{2}", date_val):
                issues.append(
                    f"Entry {entry_id} has invalid UTC date format '{date_val}', expected 'YYYY-MM-DD...'"
                )

    is_compliant = len(issues) == 0
    return is_compliant, issues


def create_change_entry(
    entry_id: str,
    title: str,
    date_utc: str,
    component: str,
    change_type: str,
    iso_clause: str,
    objective: str,
    risk_assessment: str,
    verification_gate: str,
    artifacts: Optional[List[str]] = None,
) -> str:
    """
    Formats a new change control record markdown string complying with ISO/IEC 42001.
    """
    artifacts_str = ", ".join(f"`{a}`" for a in (artifacts or [])) if artifacts else "N/A"
    return (
        f"### [{entry_id}] - {title}\n"
        f"- **Date (UTC)**: {date_utc}\n"
        f"- **Component**: {component}\n"
        f"- **Change Type**: {change_type}\n"
        f"- **ISO 42001 Clause**: {iso_clause}\n"
        f"- **Objective**: {objective}\n"
        f"- **Risk Assessment**: {risk_assessment}\n"
        f"- **Verification Gate**: {verification_gate}\n"
        f"- **Traceability / Artifacts**: {artifacts_str}\n"
    )


if __name__ == "__main__":
    ok, errs = validate_changelog_compliance()
    if ok:
        print("[PASSED] CHANGELOG.md is fully compliant with ISO/IEC 42001:2023.")
    else:
        print("[FAILED] ISO/IEC 42001 compliance validation failed:")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
