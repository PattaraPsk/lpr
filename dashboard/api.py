"""
FastAPI Backend for Unified Thai LPR Web Dashboard.
This backend unifies DataManipulator, PlateAugmentor, and OWOD Studio.
The code strictly follows ASD-STE100 and ISO 24495-1:2023 standards.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi import FastAPI, HTTPException, Query, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    BaseModel = object  # type: ignore

# Internal tool imports
from tools.owod.core.quality_auditor import PlateInspectionQueue, audit_dataset


# ---------------------------------------------------------------------------
# Data Models (ASD-STE100 Compliant: Clear, unambiguous naming)
# ---------------------------------------------------------------------------
if HAS_FASTAPI:
    class AnnotationUpdateRequest(BaseModel):
        """Data schema to update license plate annotation."""
        image_path: str = Field(..., description="Absolute or relative path to the image file")
        plate_text: str = Field(..., description="Full Thai license plate text")
        province: str = Field(..., description="Province name in Thai")
        plate_type: str = Field(default="Normal", description="License plate category")
        is_valid: bool = Field(default=True, description="Flag indicating annotation validity")

    class PlateRenderRequest(BaseModel):
        """Data schema to render a synthetic license plate."""
        plate_text: str = Field(default="1กก 1234", description="Plate characters to render")
        province: str = Field(default="กรุงเทพมหานคร", description="Thai province name")
        plate_type: str = Field(default="Normal", description="Plate style")
        dirt_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Dirt overlay factor")
        blur_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Gaussian blur factor")
        perspective_angle: float = Field(default=0.0, ge=-45.0, le=45.0, description="Perspective tilt")

    class AuditRequest(BaseModel):
        """Data schema to audit a dataset directory."""
        dataset_path: str = Field(default="dataset/yolo_dlt_lpr_dataset", description="Path to dataset")
        save_report: bool = Field(default=True, description="Save data quality report to disk")

    class QueueReviewRequest(BaseModel):
        """Data schema to review low-confidence crops."""
        item_id: str = Field(..., description="Unique queue record identifier")
        decision: str = Field(..., description="Review outcome: APPROVED, REJECTED, or RELABELED")
        corrected_label: Optional[str] = Field(default=None, description="Corrected plate string")


# ---------------------------------------------------------------------------
# FastAPI Application Construction
# ---------------------------------------------------------------------------
def create_app() -> Any:
    """Create and configure the unified FastAPI application instance."""
    if not HAS_FASTAPI:
        return None

    app = FastAPI(
        title="Thai LPR Unified Dashboard API",
        version="1.0.0",
        description="Unified REST API for DataManipulator, PlateAugmentor, and OWOD Studio tools.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # System Status Endpoints
    # -----------------------------------------------------------------------
    @app.get("/api/v1/system/status", tags=["System"])
    def get_system_status() -> Dict[str, Any]:
        """Return operating status and hardware availability."""
        import torch
        cuda_ok = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU"
        return {
            "status": "ONLINE",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cuda_available": cuda_ok,
            "active_device": device_name,
            "project_root": str(PROJECT_ROOT).replace("\\", "/"),
        }

    # -----------------------------------------------------------------------
    # DataManipulator Unified Endpoints
    # -----------------------------------------------------------------------
    @app.get("/api/v1/datamanipulator/provinces", tags=["DataManipulator"])
    def list_provinces() -> List[Dict[str, Any]]:
        """Return official list of all 77 Thai provinces and standard codes."""
        try:
            from tools.DataManipulator.app.provinces import PROVINCES
            return [{"id": i, "name": p} for i, p in enumerate(PROVINCES)]
        except Exception:
            return [{"id": 0, "name": "กรุงเทพมหานคร"}]

    @app.get("/api/v1/datamanipulator/plate-types", tags=["DataManipulator"])
    def list_plate_types() -> List[Dict[str, Any]]:
        """Return supported Thai vehicle plate categories."""
        try:
            from tools.DataManipulator.app.plate_types import PLATE_TYPES
            return [{"code": k, "label": str(v)} for k, v in PLATE_TYPES.items()]
        except Exception:
            return [{"code": "normal", "label": "Normal White / Black"}]

    @app.post("/api/v1/datamanipulator/annotations", tags=["DataManipulator"])
    def update_annotation(payload: AnnotationUpdateRequest) -> Dict[str, Any]:
        """Save revised plate text, province, and type for a dataset sample."""
        img_p = Path(payload.image_path)
        if not img_p.is_file():
            raise HTTPException(status_code=404, detail=f"Image file not found: {payload.image_path}")
        
        # Save sidecar metadata JSON
        meta_file = img_p.with_suffix(".meta.json")
        data = {
            "image": img_p.name,
            "plate_text": payload.plate_text,
            "province": payload.province,
            "plate_type": payload.plate_type,
            "is_valid": payload.is_valid,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return {"status": "SUCCESS", "metadata_file": str(meta_file).replace("\\", "/")}

    # -----------------------------------------------------------------------
    # PlateAugmentor Unified Endpoints
    # -----------------------------------------------------------------------
    @app.post("/api/v1/augmentor/render", tags=["PlateAugmentor"])
    def render_plate(payload: PlateRenderRequest) -> Dict[str, Any]:
        """Render a synthetic license plate image with visual augmentations."""
        try:
            from tools.PlateAugmentor.plate_renderer import PlateRenderer
            renderer = PlateRenderer()
            img = renderer.render(
                plate_text=payload.plate_text,
                province=payload.province,
                plate_type=payload.plate_type
            )
            import cv2
            _, buffer = cv2.imencode(".png", img)
            encoded = base64.b64encode(buffer).decode("utf-8")
            return {
                "status": "SUCCESS",
                "format": "image/png",
                "image_base64": encoded,
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
            }
        except Exception as err:
            return {
                "status": "FALLBACK",
                "message": f"Renderer execution notice: {err}",
                "plate_text": payload.plate_text,
                "province": payload.province
            }

    # -----------------------------------------------------------------------
    # OWOD Studio & Model Surgery Endpoints
    # -----------------------------------------------------------------------
    @app.get("/api/v1/owod/models", tags=["OWOD Studio"])
    def list_available_models() -> List[Dict[str, Any]]:
        """List available trained models in the models directory."""
        models_dir = PROJECT_ROOT / "models"
        if not models_dir.is_dir():
            return []
        results = []
        for p in models_dir.rglob("*.pt"):
            results.append({
                "name": p.name,
                "relative_path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            })
        for p in models_dir.rglob("*.pth"):
            results.append({
                "name": p.name,
                "relative_path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            })
        return results

    @app.post("/api/v1/owod/audit", tags=["OWOD Studio"])
    def run_quality_audit(payload: AuditRequest) -> Dict[str, Any]:
        """Execute ISO/IEC 5259-2 quality audit on target dataset."""
        d_path = PROJECT_ROOT / payload.dataset_path
        if not d_path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset path does not exist: {payload.dataset_path}")
        report = audit_dataset(d_path, save_report=payload.save_report)
        return report

    # -----------------------------------------------------------------------
    # Inspection Queue Endpoints (Low-confidence RTSP Crops)
    # -----------------------------------------------------------------------
    @app.get("/api/v1/audit/queue", tags=["Quality Auditor"])
    def get_inspection_queue() -> List[Dict[str, Any]]:
        """List queued plate crops with confidence below 0.85."""
        queue = PlateInspectionQueue()
        return queue.list_pending()

    @app.post("/api/v1/audit/queue/review", tags=["Quality Auditor"])
    def review_queue_item(payload: QueueReviewRequest) -> Dict[str, Any]:
        """Process human reviewer decision for a queued crop."""
        queue = PlateInspectionQueue()
        items = queue.list_pending()
        found = False
        updated_items = []
        for it in items:
            if it.get("item_id") == payload.item_id:
                it["status"] = payload.decision
                it["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                if payload.corrected_label:
                    it["corrected_label"] = payload.corrected_label
                found = True
            updated_items.append(it)

        if not found:
            raise HTTPException(status_code=404, detail="Queue item ID not found")

        # Write back updated records
        with open(queue.manifest_file, "w", encoding="utf-8") as f:
            for it in updated_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

        return {"status": "SUCCESS", "item_id": payload.item_id, "decision": payload.decision}

    return app


# Application singleton
app = create_app()


def main() -> None:
    """Launch the FastAPI server using Uvicorn."""
    if not HAS_FASTAPI:
        print("[ERROR] FastAPI is not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)
    import uvicorn
    uvicorn.run("dashboard.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
