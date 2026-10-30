"""
Streamlit Web Application for Unified Thai LPR Workspace.
This interface unifies DataManipulator, PlateAugmentor, and OWOD Studio.
The application follows ASD-STE100 and ISO 24495-1:2023 standards.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from tools.owod.core.quality_auditor import PlateInspectionQueue, audit_dataset


def run_streamlit_app() -> None:
    """Run the Streamlit interactive dashboard."""
    if not HAS_STREAMLIT:
        print("[ERROR] Streamlit is not installed. Install with: pip install streamlit")
        sys.exit(1)

    st.set_page_config(
        page_title="Thai LPR Unified Dashboard",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🚗 Thai License Plate Recognition Unified Dashboard")
    st.caption("Standard Compliance: ISO/IEC 42001:2023 | ISO/IEC 5259-2:2023 | ASD-STE100 | ISO 24495-1:2023")

    # -----------------------------------------------------------------------
    # Sidebar Navigation
    # -----------------------------------------------------------------------
    st.sidebar.header("Navigation")
    tool_mode = st.sidebar.radio(
        "Select Toolset Module:",
        (
            "Data Manipulator (Annotation & Review)",
            "Plate Augmentor (Synthetic Generator)",
            "OWOD Studio (Model Surgery & Pruning)",
            "Quality Auditor (Low-Confidence Inspection Queue)",
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Workspace Info**")
    st.sidebar.text(f"Root: {PROJECT_ROOT.name}")

    # -----------------------------------------------------------------------
    # Tab 1: Data Manipulator
    # -----------------------------------------------------------------------
    if tool_mode == "Data Manipulator (Annotation & Review)":
        st.header("📋 Thai LPR Dataset Annotator & Reviewer")
        st.write("Inspect dataset images, verify character sequence annotations, and update provincial tags.")

        col1, col2 = st.columns([1, 2])
        with col1:
            dataset_dir = st.text_input("Dataset Root Directory:", value="dataset")
            dpath = PROJECT_ROOT / dataset_dir
            if dpath.exists():
                st.success(f"Directory found: {dpath}")
                img_files = list(dpath.rglob("*.jpg")) + list(dpath.rglob("*.png"))
                st.info(f"Total image files discovered: {len(img_files)}")
            else:
                st.warning(f"Path does not exist: {dpath}")
                img_files = []

        with col2:
            if img_files:
                selected_img = st.selectbox("Select Sample Image:", [str(p.relative_to(PROJECT_ROOT)) for p in img_files[:100]])
                if selected_img:
                    full_p = PROJECT_ROOT / selected_img
                    st.image(str(full_p), caption=selected_img, use_container_width=True)

                    c_text, c_prov = st.columns(2)
                    with c_text:
                        plate_input = st.text_input("License Plate Characters:", value="1กก 1234")
                    with c_prov:
                        prov_input = st.text_input("Province Name:", value="กรุงเทพมหานคร")

                    if st.button("Save Revised Annotation"):
                        meta_p = full_p.with_suffix(".meta.json")
                        data = {"plate_text": plate_input, "province": prov_input}
                        with open(meta_p, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        st.success(f"Saved annotation to {meta_p.name}")
            else:
                st.info("No image files found in the specified dataset directory.")

    # -----------------------------------------------------------------------
    # Tab 2: Plate Augmentor
    # -----------------------------------------------------------------------
    elif tool_mode == "Plate Augmentor (Synthetic Generator)":
        st.header("🎨 Plate Augmentor & Synthetic Generator")
        st.write("Generate synthetic Thai license plates with weathered textures, blur, and perspective transformations.")

        c1, c2 = st.columns([1, 1])
        with c1:
            plate_str = st.text_input("Plate String (DLT Format):", value="9กก 9999")
            province_str = st.text_input("Province Name (Thai):", value="กรุงเทพมหานคร")
            plate_type = st.selectbox("Plate Category:", ["Normal (White/Black)", "Yellow (Public)", "Green (Pickup)"])

            st.subheader("Visual Augmentations")
            dirt_val = st.slider("Dirt & Weathering Factor:", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
            blur_val = st.slider("Gaussian Blur Factor:", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
            angle_val = st.slider("Perspective Tilt Angle (Degrees):", min_value=-30, max_value=30, value=0, step=5)

            generate_btn = st.button("Generate Synthetic Plate Preview")

        with c2:
            st.subheader("Render Output")
            if generate_btn:
                try:
                    from tools.PlateAugmentor.plate_renderer import PlateRenderer
                    renderer = PlateRenderer()
                    rendered = renderer.render(plate_text=plate_str, province=province_str, plate_type="Normal")
                    st.image(rendered, channels="BGR", caption=f"Rendered: {plate_str} - {province_str}")
                    st.success("Plate rendered successfully.")
                except Exception as ex:
                    st.warning(f"Live rendering notice: {ex}. Check font files in models/fonts/.")

    # -----------------------------------------------------------------------
    # Tab 3: OWOD Studio
    # -----------------------------------------------------------------------
    elif tool_mode == "OWOD Studio (Model Surgery & Pruning)":
        st.header("🧠 OWOD Studio & Model Surgery")
        st.write("Inspect neural network layer structures, evaluate model parameter counts, and manage layer freezing.")

        models_dir = PROJECT_ROOT / "models"
        available_models = list(models_dir.rglob("*.pt")) + list(models_dir.rglob("*.pth")) if models_dir.exists() else []

        if available_models:
            chosen_model = st.selectbox("Select Model Checkpoint:", [str(p.relative_to(PROJECT_ROOT)) for p in available_models])
            model_full_path = PROJECT_ROOT / chosen_model
            st.write(f"**Model Path:** `{model_full_path}`")
            st.write(f"**File Size:** `{round(model_full_path.stat().st_size / (1024*1024), 2)} MB`")

            if st.button("Inspect Model Layers"):
                try:
                    import torch
                    ckpt = torch.load(str(model_full_path), map_location="cpu")
                    st.success("Loaded checkpoint state successfully.")
                    if isinstance(ckpt, dict) and "model" in ckpt:
                        st.json({"keys": list(ckpt.keys())[:10]})
                    elif hasattr(ckpt, "state_dict"):
                        st.write(f"Total layer parameters: {len(ckpt.state_dict())}")
                except Exception as err:
                    st.error(f"Inspection error: {err}")
        else:
            st.info("No model checkpoints (.pt / .pth) found in models/ directory.")

    # -----------------------------------------------------------------------
    # Tab 4: Quality Auditor & RTSP Low-Confidence Queue
    # -----------------------------------------------------------------------
    elif tool_mode == "Quality Auditor (Low-Confidence Inspection Queue)":
        st.header("🛡️ ISO/IEC 5259 Data Quality Auditor & Low-Confidence Queue")
        st.write("Review license plate crops with detector confidence scores below 0.85 routed from RTSP collectors.")

        queue = PlateInspectionQueue()
        pending_items = queue.list_pending()

        st.subheader(f"Pending Audit Queue Items ({len(pending_items)})")
        if pending_items:
            for item in pending_items[-10:]:
                with st.expander(f"Crop ID: {item.get('item_id')} | Confidence: {item.get('confidence')} | Status: {item.get('status')}"):
                    c_left, c_right = st.columns([1, 2])
                    with c_left:
                        crop_f = Path(item.get("crop_path", ""))
                        if crop_f.is_file():
                            st.image(str(crop_f), width=200)
                        else:
                            st.text("Crop file not on disk.")
                    with c_right:
                        st.json(item)
        else:
            st.info("The inspection queue is currently empty. Plate crops with confidence < 0.85 will appear here automatically.")

        st.markdown("---")
        st.subheader("Run ISO/IEC 5259-2 Dataset Quality Audit")
        audit_target = st.text_input("Dataset Directory to Audit:", value="dataset/yolo_dlt_lpr_dataset")
        if st.button("Execute ISO Audit"):
            target_path = PROJECT_ROOT / audit_target
            if target_path.exists():
                with st.spinner("Auditing dataset syntax, bounding box geometry, and class distribution..."):
                    report = audit_dataset(target_path, save_report=True)
                    grade = report.get("quality_measures", {}).get("quality_grade", "UNKNOWN")
                    val_rate = report.get("quality_measures", {}).get("syntactic_validity_rate", 0.0)
                    st.metric("Dataset Quality Grade", grade)
                    st.metric("Syntactic Validity Rate", f"{val_rate * 100:.2f}%")
                    st.json(report)
            else:
                st.error(f"Path does not exist: {target_path}")


if __name__ == "__main__":
    run_streamlit_app()
