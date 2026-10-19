"""
Security & Credential Utilities (ISO/IEC 27001 Compliance)
Decouples secrets from application code and enforces credential masking in all logs and manifests.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlsplit, urlunsplit


def load_env_file(env_path: Optional[Path | str] = None) -> Dict[str, str]:
    """
    Parses a .env file and populates os.environ without external dependencies.
    Existing os.environ values take precedence unless overridden.
    """
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    else:
        # Standard search hierarchy: current dir, script dir, project root
        candidates.extend([
            Path.cwd() / ".env",
            Path(__file__).resolve().parent / ".env",
            Path(__file__).resolve().parent.parent.parent / ".env",
        ])

    loaded_vars: Dict[str, str] = {}
    found_file = None
    for cand in candidates:
        if cand.is_file():
            found_file = cand
            break

    if not found_file:
        return loaded_vars

    with open(found_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key not in os.environ:
                    os.environ[key] = val
                loaded_vars[key] = val

    return loaded_vars


def mask_rtsp_url(url: str) -> str:
    """
    Masks credentials in RTSP/HTTP URLs for secure audit logging (ISO/IEC 27001).
    Example:
        rtsp://admin:secret@host:554/stream -> rtsp://admin:***@host:554/stream
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        if parts.username or parts.password:
            user = parts.username or ""
            masked_netloc = f"{user}:***@{parts.hostname}"
            if parts.port:
                masked_netloc += f":{parts.port}"
            return urlunsplit((parts.scheme, masked_netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        # Fallback regex masking if URL parser fails on non-standard formats
        return re.sub(r"://([^:@]+):([^@]+)@", r"://\1:***@", url)
    return url
