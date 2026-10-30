#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Performance HUD & Thai Unicode Font Renderer.
Compliant with ISO/IEC 10646 & Unicode 15.0 NFC.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.alpr.config import THAI_FONT_CANDIDATES
from tools.rtsp_collector.security import mask_rtsp_url

_CACHED_FONTS: Dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
_BADGE_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
_BADGE_CACHE_MAX = 200


def get_thai_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Loads and caches TrueType font supporting Thai Unicode characters."""
    if size in _CACHED_FONTS:
        return _CACHED_FONTS[size]

    for p in THAI_FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                font = ImageFont.truetype(p, size)
                _CACHED_FONTS[size] = font
                return font
            except Exception:
                continue

    default_font = ImageFont.load_default()
    _CACHED_FONTS[size] = default_font
    return default_font


def render_thai_badge(
    text: str,
    subtext: Optional[str] = None,
    font_size: int = 18,
    sub_font_size: int = 14,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int, int] = (15, 23, 42, 230),  # Dark navy translucent
    border_color: Optional[Tuple[int, int, int]] = (56, 189, 248),  # Sky blue accent
    padding: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Renders high-quality Thai Unicode text badge with anti-aliasing and returns
    (bgr_patch, alpha_mask) for rapid overlay onto OpenCV frames without full-image PIL conversion.
    """
    cache_key = f"{text}_{subtext}_{font_size}_{sub_font_size}_{bg_color}_{border_color}"
    if cache_key in _BADGE_CACHE:
        return _BADGE_CACHE[cache_key]

    font_main = get_thai_font(font_size)
    font_sub = get_thai_font(sub_font_size) if subtext else None

    # Measure text bounding boxes
    dummy = Image.new("RGBA", (1, 1))
    d_dummy = ImageDraw.Draw(dummy)

    bb_main = d_dummy.textbbox((0, 0), text, font=font_main)
    w_main = bb_main[2] - bb_main[0]
    h_main = bb_main[3] - bb_main[1]

    w_sub, h_sub = 0, 0
    if subtext and font_sub:
        bb_sub = d_dummy.textbbox((0, 0), subtext, font=font_sub)
        w_sub = bb_sub[2] - bb_sub[0]
        h_sub = bb_sub[3] - bb_sub[1]

    total_w = max(w_main, w_sub) + padding * 2 + 12
    total_h = h_main + (h_sub + 4 if subtext else 0) + padding * 2 + 6

    patch = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)

    # Draw rounded rectangle background
    draw.rounded_rectangle(
        [(0, 0), (total_w - 1, total_h - 1)],
        radius=6,
        fill=bg_color,
        outline=border_color,
        width=1 if border_color else 0
    )

    # Draw main text (e.g. Thai plate & province)
    draw.text((padding + 4, padding + 2), text, font=font_main, fill=text_color)

    # Draw subtext (e.g. vehicle type & orientation)
    if subtext and font_sub:
        draw.text(
            (padding + 4, padding + h_main + 6),
            subtext,
            font=font_sub,
            fill=(203, 213, 225)  # Light slate gray
        )

    patch_np = np.array(patch)
    bgr_patch = cv2.cvtColor(patch_np, cv2.COLOR_RGBA2BGR)
    alpha_mask = patch_np[:, :, 3] / 255.0

    if len(_BADGE_CACHE) >= _BADGE_CACHE_MAX:
        # Evict oldest entry
        _BADGE_CACHE.pop(next(iter(_BADGE_CACHE)))

    _BADGE_CACHE[cache_key] = (bgr_patch, alpha_mask)
    return bgr_patch, alpha_mask


def overlay_badge(
    frame: np.ndarray,
    badge_bgr: np.ndarray,
    alpha: np.ndarray,
    x: int,
    y: int
) -> None:
    """Blends pre-rendered badge patch onto frame in-place with bounds checking."""
    fh, fw = frame.shape[:2]
    bh, bw = badge_bgr.shape[:2]

    # Constrain coordinates to image boundary
    x = max(0, min(x, fw - bw))
    y = max(0, min(y, fh - bh))

    sub_frame = frame[y:y + bh, x:x + bw]
    alpha_3d = alpha[:, :, np.newaxis]
    frame[y:y + bh, x:x + bw] = (1.0 - alpha_3d) * sub_frame + alpha_3d * badge_bgr


def draw_hud(
    display_frame: np.ndarray,
    orig_w: int,
    orig_h: int,
    fps: float,
    det_ms: float,
    ocr_ms: float,
    tracked_dets: List[Dict[str, Any]],
    vehicles: Optional[List[Tuple[List[float], str, float]]] = None,
    roi_box: Optional[List[int]] = None,
    roi_enabled: bool = False,
    is_drawing_roi: bool = False,
    roi_drag_start: Optional[Tuple[int, int]] = None,
    roi_drag_current: Optional[Tuple[int, int]] = None,
    show_vehicles: bool = True,
    show_boxes: bool = True,
    show_hud: bool = True,
    is_paused: bool = False,
    rotate_180: bool = False,
    coarse_to_fine: bool = True,
    device: str = "cpu",
    source_raw: str = "",
    playlist: Optional[List[Path]] = None,
    current_video_idx: int = 0,
    toast_msg: Optional[str] = None,
    toast_expire: float = 0.0,
    active_tracks_count: int = 0,
    recent_recognized_log: Optional[List[Dict[str, Any]]] = None
) -> np.ndarray:
    """Renders edge-AI telemetry header, Thai plate annotations, and recent ticker."""
    dh, dw = display_frame.shape[:2]
    scale_x = dw / float(orig_w)
    scale_y = dh / float(orig_h)

    # 0. Draw Active ROI Zone (if defined)
    active_roi = roi_box if (roi_enabled and roi_box is not None) else None
    if active_roi is not None:
        rx1, ry1, rx2, ry2 = active_roi
        drx1 = max(0, min(dw - 1, int(rx1 * scale_x)))
        dry1 = max(0, min(dh - 1, int(ry1 * scale_y)))
        drx2 = max(drx1 + 10, min(dw, int(rx2 * scale_x)))
        dry2 = max(dry1 + 10, min(dh, int(ry2 * scale_y)))

        # Dim outside area by 40% for visual isolation
        roi_cutout = display_frame[dry1:dry2, drx1:drx2].copy()
        display_frame[:] = (display_frame * 0.60).astype(np.uint8)
        display_frame[dry1:dry2, drx1:drx2] = roi_cutout

        # Glowing neon amber/gold border around ROI
        cv2.rectangle(display_frame, (drx1, dry1), (drx2, dry2), (0, 215, 255), 2, cv2.LINE_AA)
        c_len = min(22, max(8, (drx2 - drx1) // 6))
        cv2.line(display_frame, (drx1, dry1), (drx1 + c_len, dry1), (0, 255, 255), 3, cv2.LINE_AA)
        cv2.line(display_frame, (drx1, dry1), (drx1, dry1 + c_len), (0, 255, 255), 3, cv2.LINE_AA)
        cv2.line(display_frame, (drx2, dry2), (drx2 - c_len, dry2), (0, 255, 255), 3, cv2.LINE_AA)
        cv2.line(display_frame, (drx2, dry2), (drx2, dry2 - c_len), (0, 255, 255), 3, cv2.LINE_AA)

        roi_tag = f"ACTIVE ROI ({rx2 - rx1}x{ry2 - ry1})"
        cv2.putText(display_frame, roi_tag, (drx1 + 6, max(dry1 - 8, 20)), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 215, 255), 1, cv2.LINE_AA)

    # 0.5. Live mouse drag preview when drawing ROI
    if is_drawing_roi and roi_drag_start and roi_drag_current:
        x1, y1 = roi_drag_start
        x2, y2 = roi_drag_current
        px1, py1 = min(x1, x2), min(y1, y2)
        px2, py2 = max(x1, x2), max(y1, y2)
        cv2.rectangle(display_frame, (px1, py1), (px2, py2), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"DRAWING ROI: {px2 - px1}x{py2 - py1}", (px1 + 4, max(py1 - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # 0.75. Draw Vehicle Bounding Boxes (Cars, Motorcycles, Trucks, Buses)
    if show_vehicles and vehicles:
        veh_color_map = {
            "Motorcycle": (255, 0, 200),  # Neon Magenta
            "Car": (240, 180, 0),         # Sky Blue / Cyan
            "Truck": (0, 220, 100),       # Lime Green
            "Bus": (0, 140, 255),         # Amber Orange
        }
        for v_box, v_cls, v_conf in vehicles:
            vx1, vy1, vx2, vy2 = v_box
            dvx1 = int(vx1 * scale_x)
            dvy1 = int(vy1 * scale_y)
            dvx2 = int(vx2 * scale_x)
            dvy2 = int(vy2 * scale_y)

            v_color = veh_color_map.get(v_cls, (180, 180, 180))
            cv2.rectangle(display_frame, (dvx1, dvy1), (dvx2, dvy2), v_color, 2, cv2.LINE_AA)

            # Corner accent lines
            vc_len = min(20, max(8, (dvx2 - dvx1) // 5))
            cv2.line(display_frame, (dvx1, dvy1), (dvx1 + vc_len, dvy1), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(display_frame, (dvx1, dvy1), (dvx1, dvy1 + vc_len), (255, 255, 255), 2, cv2.LINE_AA)

            # Vehicle pill label
            v_tag = f"[{v_cls.upper()[:4]}] {v_cls} {int(v_conf * 100)}%"
            (tw, th), _ = cv2.getTextSize(v_tag, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
            tag_y1 = max(42, dvy1 - th - 8)
            tag_y2 = tag_y1 + th + 6
            tag_x2 = dvx1 + tw + 10
            cv2.rectangle(display_frame, (dvx1, tag_y1), (tag_x2, tag_y2), (15, 23, 42), -1)
            cv2.rectangle(display_frame, (dvx1, tag_y1), (tag_x2, tag_y2), v_color, 1)
            cv2.putText(display_frame, v_tag, (dvx1 + 5, tag_y2 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # 1. Draw Plate Bounding Boxes & Thai Badges
    if show_boxes:
        for det in tracked_dets:
            x1, y1, x2, y2 = det["box"]
            dx1, dy1, dx2, dy2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

            # Modern Neon Cyan Box with corner accent lines
            color_primary = (255, 200, 0) if det.get("orientation") == "Front" else (0, 220, 255)
            cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), color_primary, 2, cv2.LINE_AA)

            # Corner accent brackets
            corner_len = min(16, max(6, (dx2 - dx1) // 4))
            cv2.line(display_frame, (dx1, dy1), (dx1 + corner_len, dy1), (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(display_frame, (dx1, dy1), (dx1, dy1 + corner_len), (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(display_frame, (dx2, dy2), (dx2 - corner_len, dy2), (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(display_frame, (dx2, dy2), (dx2, dy2 - corner_len), (255, 255, 255), 3, cv2.LINE_AA)

            # Prepare Thai text badge
            trk_id = det.get("track_id", 0)
            plate_txt = det.get("consensus_plate") or det.get("ocr", {}).get("full_plate") or "..."
            iso_prov = det.get("iso_province", "")
            veh_info = f"{det.get('veh_type', 'Vehicle')} • {det.get('orientation', 'Unknown')}"

            main_badge_txt = f"[#{trk_id}] {plate_txt}"
            if iso_prov:
                main_badge_txt += f" ({iso_prov})"

            conf_val = det.get("ocr", {}).get("confidence", 0.0)
            sub_badge_txt = f"{veh_info}  (Conf: {int(conf_val * 100)}%)"

            badge_bgr, alpha = render_thai_badge(
                text=main_badge_txt,
                subtext=sub_badge_txt,
                font_size=18,
                sub_font_size=13,
                border_color=color_primary
            )
            overlay_badge(display_frame, badge_bgr, alpha, dx1, max(42, dy1 - badge_bgr.shape[0] - 6))

    # 2. Top Telemetry Bar
    if show_hud:
        bar_h = 42
        overlay = display_frame[:bar_h, :].copy()
        cv2.rectangle(display_frame, (0, 0), (dw, bar_h), (15, 23, 42), -1)
        cv2.addWeighted(display_frame[:bar_h, :], 0.88, overlay, 0.12, 0, display_frame[:bar_h, :])
        cv2.line(display_frame, (0, bar_h), (dw, bar_h), (56, 189, 248), 1, cv2.LINE_AA)

        # Status pill (LIVE / PAUSED)
        status_txt = "PAUSED" if is_paused else "LIVE"
        status_bg = (0, 140, 255) if is_paused else (0, 200, 80)
        cv2.rectangle(display_frame, (12, 8), (80, 34), status_bg, -1, cv2.LINE_AA)
        cv2.putText(display_frame, status_txt, (18, 26), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Telemetry text string
        if playlist and len(playlist) > 1:
            cur_name = Path(source_raw).name
            if len(cur_name) > 24:
                cur_name = cur_name[:12] + "..." + cur_name[-9:]
            source_display = f"VDO [{current_video_idx + 1}/{len(playlist)}] {cur_name}"
        else:
            source_display = mask_rtsp_url(str(source_raw))
            if len(source_display) > 35:
                source_display = "..." + source_display[-32:]

        roi_info = ""
        if roi_enabled and roi_box is not None:
            roi_info = f"  |  ROI: {roi_box[2] - roi_box[0]}x{roi_box[3] - roi_box[1]}"

        moto_info = ""
        if vehicles:
            m_cnt = sum(1 for _, v_cls, _ in vehicles if v_cls == "Motorcycle")
            if m_cnt > 0:
                moto_info = f"  |  Moto: {m_cnt}"

        strat_tag = "C2F" if coarse_to_fine else "DIR"
        telemetry_txt = (
            f"FPS: {fps:4.1f}  |  "
            f"Lat: {det_ms + ocr_ms:4.1f}ms (YOLO: {det_ms:.1f}ms | OCR: {ocr_ms:.1f}ms)  |  "
            f"Strat: {strat_tag}  |  "
            f"Device: {device.upper()}  |  "
            f"Src: {source_display}  |  "
            f"Tracks: {active_tracks_count}"
            f"{moto_info}"
            f"{roi_info}"
        )
        cv2.putText(display_frame, telemetry_txt, (92, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (241, 245, 249), 1, cv2.LINE_AA)

        # Flip indicator
        if rotate_180:
            cv2.rectangle(display_frame, (dw - 120, 8), (dw - 12, 34), (79, 70, 229), -1, cv2.LINE_AA)
            cv2.putText(display_frame, "180° FLIP", (dw - 112, 26), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Bottom Controls / Shortcuts Bar
        bot_bar_h = 30
        bot_y = dh - bot_bar_h
        bot_overlay = display_frame[bot_y:, :].copy()
        cv2.rectangle(display_frame, (0, bot_y), (dw, dh), (15, 23, 42), -1)
        cv2.addWeighted(display_frame[bot_y:, :], 0.85, bot_overlay, 0.15, 0, display_frame[bot_y:, :])

        hotkey_str = "[F] C2F/Dir Mode   [Drag Mouse] ROI   [X] Clear ROI   [N/P] Next/Prev   [V] Vehicles   [SPACE] Pause   [Q] Quit"
        cv2.putText(display_frame, hotkey_str, (16, dh - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (148, 163, 184), 1, cv2.LINE_AA)

        # 4. Video Switch Toast Banner
        if toast_msg and time.time() < toast_expire:
            toast_bgr, t_alpha = render_thai_badge(
                text=toast_msg,
                subtext="Hotkeys: [N] Next Clip | [P] Previous Clip | [SPACE] Pause",
                font_size=18,
                sub_font_size=12,
                bg_color=(15, 23, 42, 240),
                border_color=(56, 189, 248)
            )
            tbw = toast_bgr.shape[1]
            tx = max(10, (dw - tbw) // 2)
            ty = bar_h + 14
            overlay_badge(display_frame, toast_bgr, t_alpha, tx, ty)

        # 5. Recent Detections Mini Ticker
        if recent_recognized_log:
            ticker_y = bar_h + 10
            for item in list(recent_recognized_log)[-3:]:
                log_text = f"[#{item['id']}] {item['plate']} ({item['iso']})"
                log_sub = f"{item['veh']} • {item['time']}"
                badge_bgr, alpha = render_thai_badge(
                    text=log_text,
                    subtext=log_sub,
                    font_size=15,
                    sub_font_size=12,
                    bg_color=(30, 41, 59, 210),
                    border_color=(100, 116, 139)
                )
                bw = badge_bgr.shape[1]
                overlay_badge(display_frame, badge_bgr, alpha, dw - bw - 12, ticker_y)
                ticker_y += badge_bgr.shape[0] + 6

    return display_frame
