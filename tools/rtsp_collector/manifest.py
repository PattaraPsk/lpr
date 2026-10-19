"""
Lineage & Manifest Catalog Manager (ISO 8601 & ISO/IEC 20547-3)
Maintains immutable transaction logs and per-segment metadata sidecars.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .models import SegmentManifest
from .validator import calculate_sha256

logger = logging.getLogger("RTSPCollector.Manifest")


class ManifestManager:
    """
    Manages central JSON-Lines manifests and sidecar metadata for complete data lineage.
    """

    def __init__(self, base_storage_dir: Path | str, manifest_filename: str = "manifest.jsonl"):
        self.base_dir = Path(base_storage_dir)
        self.manifest_file = self.base_dir / manifest_filename
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def append_segment(self, manifest: SegmentManifest, write_sidecar: bool = True) -> Path:
        """
        Appends an ingestion event to manifest.jsonl and optionally creates a sidecar JSON file.
        """
        data = manifest.to_dict()

        # 1. Append atomically to central manifest.jsonl
        with open(self.manifest_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        # 2. Write companion sidecar JSON file (e.g., video_chunk.json)
        sidecar_path = None
        if write_sidecar:
            target_video = self.base_dir / manifest.relative_path
            sidecar_path = target_video.with_suffix(".json")
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Cataloged segment '{manifest.file_name}' [Status: {manifest.quality_status}]")
        return self.manifest_file

    def audit_storage(self) -> Dict[str, Any]:
        """
        Audits all recorded entries in manifest.jsonl against disk files (ISO 25012 verification).
        Recalculates SHA-256 to ensure zero tampering or file bit-rot.
        """
        if not self.manifest_file.is_file():
            return {
                "manifest_found": False,
                "total_records": 0,
                "verified_ok": 0,
                "corrupted_or_mismatched": 0,
                "missing_files": 0,
                "details": []
            }

        total = 0
        ok_count = 0
        corrupted = 0
        missing = 0
        details = []

        with open(self.manifest_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                total += 1
                record = json.loads(line)
                rel_path = record.get("relative_path", "")
                expected_sha = record.get("sha256_checksum", "")
                file_name = record.get("file_name", "")

                target_file = self.base_dir / rel_path
                if not target_file.is_file():
                    missing += 1
                    details.append({
                        "file_name": file_name,
                        "status": "MISSING",
                        "path": str(target_file),
                        "line": line_no
                    })
                    continue

                actual_sha = calculate_sha256(target_file)
                if actual_sha == expected_sha:
                    ok_count += 1
                    details.append({
                        "file_name": file_name,
                        "status": "VALID",
                        "size_bytes": target_file.stat().st_size
                    })
                else:
                    corrupted += 1
                    details.append({
                        "file_name": file_name,
                        "status": "CHECKSUM_MISMATCH",
                        "expected_sha": expected_sha,
                        "actual_sha": actual_sha
                    })

        return {
            "manifest_found": True,
            "manifest_path": str(self.manifest_file),
            "total_records": total,
            "verified_ok": ok_count,
            "corrupted_or_mismatched": corrupted,
            "missing_files": missing,
            "integrity_score_percent": round((ok_count / total * 100), 2) if total > 0 else 0.0,
            "details": details
        }
