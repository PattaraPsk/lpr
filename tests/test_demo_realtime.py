"""
Test Suite for Real-Time Thai LPR Demo (demo_realtime.py).
Compliant with ISO/IEC/IEEE 29119 Software Testing Standards.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.demo_realtime import (
    render_thai_badge,
    overlay_badge,
    TrackedPlate,
    find_default_sample_video,
    RealtimeLPRDemo
)


class TestRealtimeLPRDemo(unittest.TestCase):
    """Verifies components of the real-time ALPR edge demo."""

    def test_render_thai_badge(self):
        """Verifies Thai Unicode text badge generation and caching."""
        plate_text = "ขฉ 2235 ศรีสะเกษ"
        subtext = "Car • Front"
        bgr_patch, alpha_mask = render_thai_badge(
            text=plate_text,
            subtext=subtext,
            font_size=18,
            sub_font_size=13
        )
        self.assertIsInstance(bgr_patch, np.ndarray)
        self.assertIsInstance(alpha_mask, np.ndarray)
        self.assertEqual(len(bgr_patch.shape), 3)
        self.assertEqual(bgr_patch.shape[2], 3)
        self.assertEqual(alpha_mask.shape[:2], bgr_patch.shape[:2])
        self.assertTrue(bgr_patch.shape[0] > 10)
        self.assertTrue(bgr_patch.shape[1] > 20)

        # Second call should hit the cache (same object reference)
        bgr_cached, alpha_cached = render_thai_badge(
            text=plate_text,
            subtext=subtext,
            font_size=18,
            sub_font_size=13
        )
        self.assertIs(bgr_patch, bgr_cached)
        self.assertIs(alpha_mask, alpha_cached)

    def test_overlay_badge_bounds_safe(self):
        """Verifies overlaying badge onto frame handles boundaries cleanly."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bgr_patch, alpha = render_thai_badge("1กข 9999 กรุงเทพมหานคร")

        # Test normal inside position
        overlay_badge(frame, bgr_patch, alpha, 50, 50)

        # Test edge / boundary clamping (should not raise IndexError)
        overlay_badge(frame, bgr_patch, alpha, 630, 475)
        overlay_badge(frame, bgr_patch, alpha, -10, -10)

    def test_tracked_plate_lifecycle(self):
        """Verifies TrackedPlate tracking and consensus resolution."""
        dummy_crop = np.zeros((30, 100, 3), dtype=np.uint8)
        initial_ocr = {
            "d3": "ขฉ",
            "d4": "2235",
            "province": "ศรีสะเกษ",
            "full_plate": "ขฉ 2235 ศรีสะเกษ",
            "confidence": 0.95
        }

        trk = TrackedPlate(
            track_id=1,
            box=[100.0, 100.0, 200.0, 150.0],
            conf=0.92,
            frame_idx=0,
            crop_bgr=dummy_crop,
            ocr_result=initial_ocr,
            vehicle_type="Car",
            orientation="Front"
        )

        self.assertEqual(trk.track_id, 1)
        self.assertEqual(trk.consensus_plate, "ขฉ 2235 ศรีสะเกษ")
        self.assertEqual(trk.iso_province, "TH-33")
        self.assertEqual(trk.vehicle_type, "Car")
        self.assertEqual(trk.orientation, "Front")

        # Update with another observation
        next_ocr = {
            "d3": "ขฉ",
            "d4": "2235",
            "province": "ศรีสะเกษ",
            "full_plate": "ขฉ 2235 ศรีสะเกษ",
            "confidence": 0.98
        }
        trk.update(
            box=[102.0, 101.0, 202.0, 151.0],
            conf=0.94,
            frame_idx=1,
            crop_bgr=dummy_crop,
            ocr_result=next_ocr
        )

        self.assertEqual(trk.hits, 2)
        self.assertEqual(trk.best_conf, 0.98)
        self.assertEqual(trk.consensus_plate, "ขฉ 2235 ศรีสะเกษ")

    def test_find_default_sample_video(self):
        """Checks sample video auto-discovery."""
        vid = find_default_sample_video()
        self.assertIsNotNone(vid, "Should find at least one sample video in dataset")
        self.assertTrue(vid.exists())
        self.assertTrue(vid.stat().st_size > 1024)

    def test_demo_headless_run(self):
        """Runs 5 frames of RealtimeLPRDemo in headless mode to guarantee pipeline stability."""
        sample_vid = find_default_sample_video()
        demo = RealtimeLPRDemo(
            source=str(sample_vid),
            conf_thresh=0.25,
            display_width=640,
            display_height=360,
            headless=True,
            max_frames=5
        )
        demo.run()
        self.assertTrue(True)

    def test_video_playlist_switching(self):
        """Verifies dynamic video playlist discovery and next/previous video switching."""
        sample_vid = find_default_sample_video()
        demo = RealtimeLPRDemo(
            source=str(sample_vid),
            display_width=640,
            display_height=360,
            headless=True
        )
        self.assertTrue(len(demo.playlist) >= 2, "Should have multiple videos in sample folder")
        init_idx = demo.current_video_idx

        # Switch to next video
        ok_next = demo.switch_to_next_video()
        self.assertTrue(ok_next)
        self.assertEqual(demo.current_video_idx, (init_idx + 1) % len(demo.playlist))

        # Switch back to previous video
        ok_prev = demo.switch_to_prev_video()
        self.assertTrue(ok_prev)
        self.assertEqual(demo.current_video_idx, init_idx)

        demo.stream.stop()

    def test_roi_drawing_and_subframe_processing(self):
        """Verifies interactive ROI definition and sub-frame inference bounds."""
        import cv2
        sample_vid = find_default_sample_video()
        demo = RealtimeLPRDemo(
            source=str(sample_vid),
            display_width=640,
            display_height=360,
            headless=True,
            roi_box=[100, 100, 800, 600]
        )
        self.assertEqual(demo.roi_box, [100, 100, 800, 600])
        self.assertTrue(demo.roi_enabled)

        # Simulate mouse drag callback
        demo.last_orig_shape = (1080, 1920)
        demo._mouse_callback(cv2.EVENT_LBUTTONDOWN, 50, 50, 0, None)
        demo._mouse_callback(cv2.EVENT_MOUSEMOVE, 200, 150, 0, None)
        demo._mouse_callback(cv2.EVENT_LBUTTONUP, 200, 150, 0, None)

        self.assertIsNotNone(demo.roi_box)
        self.assertEqual(len(demo.roi_box), 4)
        self.assertTrue(demo.roi_box[2] > demo.roi_box[0])
        self.assertTrue(demo.roi_box[3] > demo.roi_box[1])

        # Test Right-click to clear ROI
        demo._mouse_callback(cv2.EVENT_RBUTTONDOWN, 100, 100, 0, None)
        self.assertIsNone(demo.roi_box)

        demo.stream.stop()

    def test_motorcycle_detection_and_matching(self):
        """Verifies motorcycle plate association and rear orientation assignment."""
        sample_vid = find_default_sample_video()
        demo = RealtimeLPRDemo(
            source=str(sample_vid),
            display_width=640,
            display_height=360,
            headless=True
        )
        # Mock vehicle detections: Motorcycle bounding box [400, 300, 600, 700]
        vehicles = [
            ([400.0, 300.0, 600.0, 700.0], "Motorcycle", 0.91),
            ([50.0, 200.0, 350.0, 600.0], "Car", 0.88),
        ]
        orientations = []  # No orientation from front/rear detector

        # Plate located at bottom of motorcycle fender [470, 650, 530, 730]
        plate_box = [470.0, 650.0, 530.0, 730.0]
        veh_type, orient = demo._match_vehicle_and_orientation(plate_box, orientations, vehicles)

        self.assertEqual(veh_type, "Motorcycle")
        # In Thailand, motorcycles strictly only have rear plates
        self.assertEqual(orient, "Rear")

        demo.stream.stop()

    def test_vehicle_hud_rendering(self):
        """Verifies that vehicle bounding boxes and labels render cleanly in HUD."""
        sample_vid = find_default_sample_video()
        demo = RealtimeLPRDemo(
            source=str(sample_vid),
            display_width=640,
            display_height=360,
            headless=True,
            show_vehicles=True
        )
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        vehicles = [
            ([200.0, 150.0, 450.0, 320.0], "Motorcycle", 0.89),
            ([50.0, 100.0, 180.0, 250.0], "Car", 0.95),
        ]
        tracked_dets = [
            {
                "box": [300.0, 260.0, 380.0, 310.0],
                "conf": 0.92,
                "crop": np.zeros((30, 60, 3), dtype=np.uint8),
                "ocr": {"full_plate": "1กข 1234 กรุงเทพมหานคร", "confidence": 0.94},
                "veh_type": "Motorcycle",
                "orientation": "Rear",
                "track_id": 1,
                "consensus_plate": "1กข 1234 กรุงเทพมหานคร",
                "iso_province": "TH-10"
            }
        ]
        hud_frame = demo._draw_hud(
            display_frame=frame,
            orig_w=640,
            orig_h=360,
            fps=30.0,
            det_ms=12.5,
            ocr_ms=8.2,
            tracked_dets=tracked_dets,
            vehicles=vehicles
        )
        self.assertIsInstance(hud_frame, np.ndarray)
        self.assertEqual(hud_frame.shape, (360, 640, 3))
        # Ensure frame has been drawn on (not all zeros)
        self.assertTrue(np.any(hud_frame > 0))

        demo.stream.stop()

    def test_coarse_to_fine_detection_strategy(self):
        """Verifies coarse-to-fine vehicle cascade execution vs direct mode."""
        sample_vid = find_default_sample_video()
        # 1. Coarse-to-fine mode
        demo_c2f = RealtimeLPRDemo(
            source=str(sample_vid),
            display_width=640,
            display_height=360,
            headless=True,
            coarse_to_fine=True,
            max_frames=5
        )
        self.assertTrue(demo_c2f.coarse_to_fine)
        demo_c2f.run()

        # 2. Direct mode
        demo_direct = RealtimeLPRDemo(
            source=str(sample_vid),
            display_width=640,
            display_height=360,
            headless=True,
            coarse_to_fine=False,
            max_frames=5
        )
        self.assertFalse(demo_direct.coarse_to_fine)
        demo_direct.run()


if __name__ == "__main__":
    unittest.main()
