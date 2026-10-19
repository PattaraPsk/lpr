"""
Comprehensive Unit & Integration Test Suite for OWOD License Plate Studio & YOLO Pruning.
"""

import sys
import unittest
import numpy as np
import cv2
import tempfile
import shutil
from pathlib import Path

# Add tools and owod directory to path
owod_dir = Path(__file__).resolve().parent
tools_dir = owod_dir.parent
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))
if str(owod_dir) not in sys.path:
    sys.path.insert(0, str(owod_dir))

from owod.core.media_scanner import MediaScanner, MediaItem
from owod.core.video_processor import VideoProcessor
from owod.core.owod_engine import OWODDetector, Detection
from owod.core.dataset_generator import DatasetGenerator
from owod.core.model_surgery import ModelSurgery

class TestOWODStudio(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock directory structure with subfolders, images, and a mock video
        self.sub1 = self.temp_dir / "camera_front" / "batch1"
        self.sub2 = self.temp_dir / "camera_rear"
        self.sub1.mkdir(parents=True, exist_ok=True)
        self.sub2.mkdir(parents=True, exist_ok=True)
        
        # Create test mock images
        img1 = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Draw mock plate rectangle
        cv2.rectangle(img1, (200, 300), (440, 380), (255, 255, 255), -1)
        cv2.putText(img1, "1AB-1234", (220, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.imwrite(str(self.sub1 / "car1.jpg"), img1)
        cv2.imwrite(str(self.sub1 / "car2.png"), img1)
        cv2.imwrite(str(self.sub2 / "truck.jpg"), img1)
        
        # Create a tiny mock video
        self.video_path = self.sub2 / "sample_dashcam.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.video_path), fourcc, 10.0, (320, 240))
        for _ in range(25):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.rectangle(frame, (100, 100), (220, 140), (255, 255, 255), -1)
            out.write(frame)
        out.release()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_media_scanner(self):
        scanner = MediaScanner()
        items = scanner.scan_directory(self.temp_dir)
        self.assertGreaterEqual(len(items), 4)
        
        stats = scanner.get_stats()
        self.assertEqual(stats['total_roots'], 1)
        self.assertEqual(stats['total_images'], 3)
        self.assertEqual(stats['total_videos'], 1)
        self.assertEqual(stats['selected_files'], 4)
        
        # Test adding second root directory from a different location
        second_root = Path(tempfile.mkdtemp())
        cv2.imwrite(str(second_root / "car_root2.jpg"), np.ones((100, 200, 3), dtype=np.uint8) * 50)
        scanner.add_directory(second_root)
        
        stats2 = scanner.get_stats()
        self.assertEqual(stats2['total_roots'], 2)
        self.assertEqual(stats2['total_images'], 4)
        self.assertEqual(stats2['selected_files'], 5)
        
        # Test collision-safe unique_identifier
        item = scanner.items[-1]
        self.assertTrue(item.unique_identifier.startswith(second_root.name))
        
        # Test Deselect All
        scanner.set_all_selected(False)
        self.assertEqual(scanner.get_stats()['selected_files'], 0)
        
        # Test Select Images Only
        scanner.set_type_selected('image', True)
        self.assertEqual(scanner.get_stats()['selected_images'], 4)
        self.assertEqual(scanner.get_stats()['selected_videos'], 0)
        
        # Test Invert
        scanner.invert_selection()
        self.assertEqual(scanner.get_stats()['selected_images'], 0)
        self.assertEqual(scanner.get_stats()['selected_videos'], 1)
        
        # Test removing second root
        scanner.remove_directory(second_root)
        self.assertEqual(scanner.get_stats()['total_roots'], 1)
        shutil.rmtree(second_root, ignore_errors=True)

    def test_video_processor(self):
        info = VideoProcessor.get_video_info(self.video_path)
        self.assertTrue(info['valid'])
        self.assertEqual(info['width'], 320)
        self.assertEqual(info['height'], 240)
        self.assertEqual(info['frames'], 25)
        
        frames = list(VideoProcessor.extract_frames(self.video_path, sample_step=10))
        self.assertGreaterEqual(len(frames), 2)
        idx, ts, rgb = frames[0]
        self.assertEqual(rgb.shape, (240, 320, 3))

    def test_owod_detector_and_dataset_gen(self):
        detector = OWODDetector("yolov8s-worldv2.pt", prompt="license plate")
        self.assertTrue(detector.is_loaded)
        
        test_img = np.ones((480, 640, 3), dtype=np.uint8) * 100
        cv2.rectangle(test_img, (150, 250), (450, 350), (255, 255, 255), -1)
        cv2.putText(test_img, "9999", (180, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        detections = detector.predict(test_img)
        self.assertIsInstance(detections, list)
        
        # Test drawing
        drawn = detector.draw_detections(test_img, detections)
        self.assertEqual(drawn.shape, test_img.shape)
        
        # Test dataset generator
        out_dataset = self.temp_dir / "generated_yolo_ds"
        gen = DatasetGenerator(out_dataset, train_ratio=0.8, save_crops=True, save_visualizations=True)
        
        # Add mock detection if none
        if not detections:
            detections = [Detection(x1=150, y1=250, x2=450, y2=350, confidence=0.88, label="license plate")]
            
        success, split = gen.save_sample("test_frame", test_img, detections)
        self.assertTrue(success)
        
        yaml_path = gen.generate_data_yaml()
        self.assertTrue(yaml_path.exists())
        self.assertTrue((out_dataset / "images" / split).exists())
        self.assertTrue((out_dataset / "labels" / split).exists())

    def test_model_surgery(self):
        surgery = ModelSurgery("yolov8s.pt")
        self.assertTrue(surgery.is_loaded)
        self.assertGreater(len(surgery.layers), 10)
        
        # Test layer freezing
        frozen = surgery.freeze_layers([0, 1, 2, 3, 4])
        self.assertEqual(frozen, [0, 1, 2, 3, 4])
        self.assertTrue(surgery.layers[0].is_frozen)
        self.assertFalse(surgery.layers[8].is_frozen)
        
        # Test pruning
        surgery.prune_layers([2, 4], amount=0.35, method="l1_unstructured")
        self.assertTrue(surgery.layers[2].is_pruned)
        
        # Test slicing
        orig_count = len(surgery.layers)
        surgery.slice_model(keep_up_to_layer=12)
        self.assertEqual(len(surgery.layers), 13)
        self.assertLess(len(surgery.layers), orig_count)
        
        # Test saving checkpoint
        ckpt_out = self.temp_dir / "pruned_model.pt"
        saved = surgery.save_checkpoint(ckpt_out)
        self.assertTrue(Path(saved).exists() or Path(str(saved) + ".info.txt").exists())

if __name__ == "__main__":
    unittest.main()
