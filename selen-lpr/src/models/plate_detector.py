"""
Pure PyTorch License Plate Detector Runtime.
Zero Ultralytics dependency. Runs native TorchScript (.torchscript) or ONNX with torchvision.ops.nms.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
import torch
import torchvision


class PureTorchPlateDetector:
    """
    Standalone plate detector using pure PyTorch TorchScript or ONNX.
    Zero dependency on the 'ultralytics' library.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu"
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = None
        self.backend = "none"

        if weights_path:
            p = Path(weights_path)
            if p.suffix == ".torchscript" and p.exists():
                print(f"[INFO] PureTorchDetector: Loading TorchScript model from {p}")
                self.model = torch.jit.load(str(p), map_location=self.device)
                self.model.eval()
                self.backend = "torchscript"
            elif p.suffix == ".onnx" and p.exists():
                print(f"[INFO] PureTorchDetector: Loading ONNX model from {p}")
                import onnxruntime as ort
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
                self.model = ort.InferenceSession(str(p), providers=providers)
                self.backend = "onnx"

    def _letterbox(self, img: np.ndarray, new_shape: int = 640) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        shape = img.shape[:2]
        r = min(new_shape / shape[0], new_shape / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (dw, dh)

    def detect_in_roi(
        self,
        frame: np.ndarray,
        roi_box: Tuple[int, int, int, int]
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Detects plate crops inside a vehicle ROI.
        roi_box: (rx, ry, rw, rh) in frame pixel coordinates.
        Returns: [(crop_bgr, (global_x, global_y, global_w, global_h), conf), ...]
        """
        rx, ry, rw, rh = roi_box
        roi_img = frame[ry:ry + rh, rx:rx + rw]
        if roi_img.size == 0:
            return []

        # If model is loaded, run neural inference
        if self.backend == "torchscript":
            lb_img, ratio, (pad_w, pad_h) = self._letterbox(roi_img, self.imgsz)
            tensor = torch.from_numpy(lb_img).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds = self.model(tensor)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]

            # preds shape: [1, num_classes + 4, num_anchors]
            preds = preds.squeeze(0).permute(1, 0)
            boxes_cxcywh = preds[:, :4]
            scores = preds[:, 4:].max(dim=1)[0]

            mask = scores > self.conf_threshold
            boxes_cxcywh = boxes_cxcywh[mask]
            scores = scores[mask]

            if len(scores) == 0:
                # Direct plate fallback if ROI is already a plate
                return self._fallback_crop_check(frame, rx, ry, rw, rh)

            # Convert cx, cy, w, h -> x1, y1, x2, y2
            x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
            y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
            x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
            y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

            # Pure PyTorch NMS
            keep = torchvision.ops.nms(boxes_xyxy, scores, self.iou_threshold)
            boxes_xyxy = boxes_xyxy[keep].cpu().numpy()
            scores = scores[keep].cpu().numpy()

            results = []
            for box, score in zip(boxes_xyxy, scores):
                # Un-pad and rescale to ROI coordinates
                bx1 = max(0, int((box[0] - pad_w) / ratio))
                by1 = max(0, int((box[1] - pad_h) / ratio))
                bx2 = min(rw, int((box[2] - pad_w) / ratio))
                by2 = min(rh, int((box[3] - pad_h) / ratio))

                # Global frame coordinates
                gx1, gy1 = rx + bx1, ry + by1
                gw, gh = bx2 - bx1, by2 - by1

                crop = frame[gy1:gy1 + gh, gx1:gx1 + gw]
                if crop.size > 0:
                    results.append((crop, (gx1, gy1, gw, gh), float(score)))

            return results if results else self._fallback_crop_check(frame, rx, ry, rw, rh)

        # Fallback heuristic / direct crop check
        return self._fallback_crop_check(frame, rx, ry, rw, rh)

    def _fallback_crop_check(
        self,
        frame: np.ndarray,
        rx: int,
        ry: int,
        rw: int,
        rh: int
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        ar = rw / max(rh, 1)
        if 1.05 <= ar <= 4.2 and min(rw, rh) >= 30:
            crop = frame[ry:ry + rh, rx:rx + rw]
            return [(crop, (rx, ry, rw, rh), 0.90)]
        return []
