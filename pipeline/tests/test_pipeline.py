"""
tests/test_pipeline.py
End-to-end unit test verifying pipeline initialization and frame inference.
"""
import unittest
import numpy as np
from src.pipeline.lpr_pipeline import LPRPipeline


class TestPipeline(unittest.TestCase):
    def test_pipeline_initialization_and_inference(self):
        pipeline = LPRPipeline("configs/config.default.yaml")
        self.assertIsNotNone(pipeline.detector)
        self.assertIsNotNone(pipeline.recognizer)

        # Test on dummy frame (zero crashes / graceful handling)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results, vis = pipeline.process_frame(dummy_frame, frame_id=1, render_vis=True)

        self.assertIsInstance(results, list)
        self.assertIsNotNone(vis)
        self.assertEqual(vis.shape, dummy_frame.shape)


if __name__ == "__main__":
    unittest.main()
