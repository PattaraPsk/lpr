"""
Thai License Plate OCR Reader using Model a1.3.4.1 (Pure PyTorch VisionAttentionOCR).
Extracts d3 (category letters/prefix), d4 (digits), and province with comparison and replacement helpers.
"""

import os
import sys
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from PIL import Image

# Ensure UTF-8 string handling
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .provinces import match_province_value, normalize_thai_script

# =============================================================================
# Pure PyTorch VisionAttentionOCR Architecture (GRCNN + BiLSTM + Attention Decoder)
# =============================================================================

if HAS_TORCH:
    class GRCL_unit(nn.Module):
        def __init__(self, output_channel: int):
            super().__init__()
            self.BN_gfu = nn.BatchNorm2d(output_channel)
            self.BN_grx = nn.BatchNorm2d(output_channel)
            self.BN_fu = nn.BatchNorm2d(output_channel)
            self.BN_rx = nn.BatchNorm2d(output_channel)
            self.BN_Gx = nn.BatchNorm2d(output_channel)

        def forward(self, wgf_u, wgr_x, wf_u, wr_x):
            G_first_term = self.BN_gfu(wgf_u)
            G_second_term = self.BN_grx(wgr_x)
            G = torch.sigmoid(G_first_term + G_second_term)

            x_first_term = self.BN_fu(wf_u)
            x_second_term = self.BN_Gx(self.BN_rx(wr_x) * G)
            x = F.relu(x_first_term + x_second_term)
            return x

    class GRCL(nn.Module):
        def __init__(self, input_channel: int, output_channel: int, num_iteration: int = 5, kernel_size: int = 3, pad: int = 1):
            super().__init__()
            self.wgf_u = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
            self.wgr_x = nn.Conv2d(output_channel, output_channel, 1, 1, 0, bias=False)
            self.wf_u = nn.Conv2d(input_channel, output_channel, kernel_size, 1, pad, bias=False)
            self.wr_x = nn.Conv2d(output_channel, output_channel, kernel_size, 1, pad, bias=False)
            self.BN_x_init = nn.BatchNorm2d(output_channel)

            self.num_iteration = num_iteration
            self.GRCL = nn.Sequential(*[GRCL_unit(output_channel) for _ in range(num_iteration)])

        def forward(self, input):
            wgf_u = self.wgf_u(input)
            wf_u = self.wf_u(input)
            x = F.relu(self.BN_x_init(wf_u))
            for i in range(self.num_iteration):
                x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))
            return x

    class RCNN_FeatureExtractor(nn.Module):
        def __init__(self, input_channel: int = 1, output_channel: int = 512):
            super().__init__()
            self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                                   int(output_channel / 2), output_channel]
            self.ConvNet = nn.Sequential(
                nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                GRCL(self.output_channel[0], self.output_channel[0], num_iteration=5, kernel_size=3, pad=1),
                nn.MaxPool2d(2, 2),
                GRCL(self.output_channel[0], self.output_channel[1], num_iteration=5, kernel_size=3, pad=1),
                nn.MaxPool2d(2, (2, 1), (0, 1)),
                GRCL(self.output_channel[1], self.output_channel[2], num_iteration=5, kernel_size=3, pad=1),
                nn.MaxPool2d(2, (2, 1), (0, 1)),
                nn.Conv2d(self.output_channel[2], self.output_channel[3], 2, 1, 0, bias=False),
                nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True)
            )

        def forward(self, input):
            return self.ConvNet(input)

    class BidirectionalLSTM(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, output_size: int):
            super().__init__()
            self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
            self.linear = nn.Linear(hidden_size * 2, output_size)

        def forward(self, input):
            self.rnn.flatten_parameters()
            recurrent, _ = self.rnn(input)
            output = self.linear(recurrent)
            return output

    class AttentionCell(nn.Module):
        def __init__(self, input_size: int = 512, hidden_size: int = 512, num_embeddings: int = 187):
            super().__init__()
            self.i2h = nn.Linear(input_size, hidden_size, bias=False)
            self.h2h = nn.Linear(hidden_size, hidden_size)
            self.score = nn.Linear(hidden_size, 1, bias=False)
            self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
            self.hidden_size = hidden_size

        def forward(self, prev_hidden, batch_H, char_onehots):
            batch_H_proj = self.i2h(batch_H)
            prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
            e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))
            alpha = F.softmax(e, dim=1)
            context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)
            concat_context = torch.cat([context, char_onehots], 1)
            cur_hidden = self.rnn(concat_context, prev_hidden)
            return cur_hidden, alpha

    class Attention(nn.Module):
        def __init__(self, input_size: int = 512, hidden_size: int = 512, num_classes: int = 187):
            super().__init__()
            self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
            self.hidden_size = hidden_size
            self.num_classes = num_classes
            self.generator = nn.Linear(hidden_size, num_classes)

        def _char_to_onehot(self, input_char, onehot_dim: int = 187):
            input_char = input_char.unsqueeze(1)
            batch_size = input_char.size(0)
            one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(input_char.device)
            one_hot = one_hot.scatter_(1, input_char, 1)
            return one_hot

        def forward(self, batch_H, text=None, is_train=False, batch_max_length: int = 25):
            batch_size = batch_H.size(0)
            num_steps = batch_max_length + 1
            device = batch_H.device

            targets = torch.LongTensor(batch_size).fill_(0).to(device)
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)
            hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                      torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

            return probs

    class VisionAttentionOCR(nn.Module):
        """
        Pure PyTorch Standalone OCR Model (GRCNN + BiLSTM + Attention Decoder).
        Compatible with a1.3.4.1 checkpoint weights.
        """
        def __init__(self, num_classes: int = 187, input_channel: int = 1, output_channel: int = 512, hidden_size: int = 512, batch_max_length: int = 25):
            super().__init__()
            self.batch_max_length = batch_max_length
            self.FeatureExtraction = RCNN_FeatureExtractor(input_channel, output_channel)
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(output_channel, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
            )
            self.Prediction = Attention(hidden_size, hidden_size, num_classes)

        def forward(self, x):
            visual_feature = self.FeatureExtraction(x)
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
            visual_feature = visual_feature.squeeze(3)
            contextual_feature = self.SequenceModeling(visual_feature)
            probs = self.Prediction(contextual_feature.contiguous(), is_train=False, batch_max_length=self.batch_max_length)
            return probs


# =============================================================================
# Authentic 187-Class Decoding List for a1.3.4.1
# =============================================================================

EASYOCR_DECODING_LIST: List[str] = [
    '', '', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '!', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-',
    'ก', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'ช', 'ฌ', 'ญ', '_', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ',
    'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร',
    'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ',
    'กระบี่', 'กรุงเทพมหานคร', 'กาญจนบุรี', 'กาฬสินธุ์', 'กำแพงเพชร', 'ขอนแก่น', 'จันทบุรี',
    'ฉะเชิงเทรา', 'ชลบุรี', 'ชัยนาท', 'ชัยภูมิ', 'ชุมพร', ' ', 'เชียงราย', 'เชียงใหม่',
    'ตรัง', 'ตราด', 'ตาก', 'นครนายก', 'นครปฐม', 'นครพนม', 'นครราชสีมา', 'นครศรีธรรมราช',
    'นครสวรรค์', 'นนทบุรี', 'นราธิวาส', 'น่าน', 'บึงกาฬ', 'บุรีรัมย์', 'ปทุมธานี',
    'ประจวบคีรีขันธ์', 'ปราจีนบุรี', 'ปัตตานี', 'พระนครศรีอยุธยา', 'พะเยา', 'พังงา', 'พัทลุง',
    'พิจิตร', 'พิษณุโลก', 'เพชรบุรี', 'เพชรบูรณ์', 'แพร่', 'ภูเก็ต', 'มหาสารคาม', 'มุกดาหาร',
    'แม่ฮ่องสอน', 'ยโสธร', 'ยะลา', 'ร้อยเอ็ด', 'ระนอง', 'ระยอง', 'ราชบุรี', 'ลพบุรี',
    'ลำปาง', 'ลำพูน', 'เลย', 'ศรีสะเกษ', 'สกลนคร', 'สงขลา', 'สตูล', 'สมุทรปราการ',
    'สมุทรสงคราม', 'สมุทรสาคร', 'สระแก้ว', 'สระบุรี', 'สิงห์บุรี', 'สุโขทัย', 'สุพรรณบุรี',
    'สุราษฎร์ธานี', 'สุรินทร์', 'หนองคาย', 'หนองบัวลำภู', 'อ่างทอง', 'อำนาจเจริญ', 'อุดรธานี',
    'อุตรดิตถ์', 'อุทัยธานี', 'อุบลราชธานี', 'เบตง', 'ไม่ทราบจังหวัด',
    '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙', 'ฤ', 'ฦ', 'ข', 'ซ', 'þ', 'ÿ'
]


def decode_tokens_to_plate(tokens: List[int]) -> Tuple[str, str, str, str]:
    """
    Decodes argmax token indices into (d3, d4, province, full_plate).
    1. Prunes after EOS token index 1 ([s]).
    2. Splits segments on delimiter token index 60 ('_').
    3. Maps segments into prefix d3, digits d4, and province.
    4. Normalizes and validates province using match_province_value.
    """
    if not tokens:
        return "", "", "", ""

    # 1. Prune EOS token (index 1)
    if 1 in tokens:
        tokens = tokens[:tokens.index(1)]

    # 2. Split on delimiter token 60 ('_')
    segments: List[List[int]] = []
    current_segment: List[int] = []

    for t in tokens:
        if t == 60:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
        elif t >= 2:
            current_segment.append(t)

    if current_segment:
        segments.append(current_segment)

    decoded_parts: List[str] = []
    for seg in segments:
        text = "".join([EASYOCR_DECODING_LIST[t] for t in seg if t < len(EASYOCR_DECODING_LIST)])
        decoded_parts.append(text.strip())

    if len(decoded_parts) >= 3:
        d3 = decoded_parts[0]
        d4 = decoded_parts[1]
        province = decoded_parts[2]
    elif len(decoded_parts) == 2:
        d3 = decoded_parts[0]
        d4 = decoded_parts[1]
        province = ""
    elif len(decoded_parts) == 1:
        raw = decoded_parts[0]
        # Regex fallback
        m = re.match(r"^([1-9]?[ก-ฮ]{1,3})\s*(\d{1,4})(.*)$", raw)
        if m:
            d3 = m.group(1).strip()
            d4 = m.group(2).strip()
            province = m.group(3).strip()
        else:
            letters = re.findall(r"[1-9]?[ก-ฮ]{1,3}", raw)
            digits = re.findall(r"\d+", raw)
            d3 = letters[0] if letters else ""
            d4 = digits[-1] if digits else ""
            province = ""
    else:
        d3, d4, province = "", "", ""

    # Obsolete consonant resolution
    d3 = d3.replace("ฃ", "ข").replace("ฅ", "ค")

    # Province matching & normalization
    if province and province != "ไม่ทราบจังหวัด":
        province = match_province_value(province)
    elif province == "ไม่ทราบจังหวัด":
        province = ""

    full_plate = f"{d3} {d4} {province}".strip()
    return d3, d4, province, full_plate


# =============================================================================
# PlateReader Singleton & Predictor
# =============================================================================

class PlateReader:
    _instance: Optional["PlateReader"] = None

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        if not HAS_TORCH:
            self.model = None
            self.weights_path = None
            self.device = None
            self.error_msg = "PyTorch is not installed in the environment."
            return

        self.error_msg = None
        self.weights_path = self._resolve_weights_path(weights_path)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.img_h = 64
        self.img_w = 600
        self.model: Optional[VisionAttentionOCR] = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        if self.weights_path and self.weights_path.exists():
            self._load_model()
        else:
            self.error_msg = f"Reader checkpoint not found: {self.weights_path}"

    @classmethod
    def get_instance(cls, weights_path: Optional[str] = None) -> "PlateReader":
        if cls._instance is None:
            cls._instance = cls(weights_path=weights_path)
        return cls._instance

    def _resolve_weights_path(self, explicit_path: Optional[str] = None) -> Optional[Path]:
        if explicit_path and Path(explicit_path).is_file():
            return Path(explicit_path).resolve()

        # Potential paths to a1.3.4.1 best_norm_ED.pth
        app_dir = Path(__file__).resolve().parent
        candidates = [
            app_dir.parents[2] / "models" / "reader" / "a1.3.4.1" / "best_norm_ED.pth",
            app_dir.parents[1] / "models" / "reader" / "a1.3.4.1" / "best_norm_ED.pth",
            Path("models/reader/a1.3.4.1/best_norm_ED.pth"),
            Path("models/reader/thai_plate_reader.pth"),
            Path("best_norm_ED.pth"),
            app_dir.parents[2] / "archive" / "models" / "ocr" / "thai_plate_attn_ocr.pt",
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()

        return candidates[0]

    def _load_model(self):
        try:
            self.model = VisionAttentionOCR(num_classes=187, batch_max_length=25)
            ckpt = torch.load(str(self.weights_path), map_location=self.device)
            if isinstance(ckpt, dict):
                clean_sd = {k.replace("module.", ""): v for k, v in ckpt.items()}
                self.model.load_state_dict(clean_sd)
            self.model.to(self.device)
            self.model.eval()
            self.error_msg = None
        except Exception as e:
            self.model = None
            self.error_msg = f"Failed to load weights from {self.weights_path}: {e}"

    def preprocess_image(self, pil_img: Image.Image) -> torch.Tensor:
        """Authentic NormalizePAD transform matching a1.3.4.1."""
        gray = pil_img.convert("L")
        w, h = gray.size
        ratio = w / float(max(h, 1))
        resized_w = min(int(np.ceil(self.img_h * ratio)), self.img_w)
        resized_crop = gray.resize((resized_w, self.img_h), Image.BICUBIC)

        t = transforms.ToTensor()(resized_crop)
        t.sub_(0.5).div_(0.5)

        pad = torch.zeros(1, self.img_h, self.img_w, dtype=torch.float32)
        pad[:, :, :resized_w] = t
        if self.img_w != resized_w:
            pad[:, :, resized_w:] = t[:, :, resized_w - 1].unsqueeze(2).expand(1, self.img_h, self.img_w - resized_w)

        return pad.unsqueeze(0)

    def predict(self, image_input, use_cache: bool = True) -> Dict[str, Any]:
        """
        Runs a1.3.4.1 OCR model on image_path or PIL Image.
        Returns dict:
        {
            "d3": str,
            "d4": str,
            "province": str,
            "full_plate": str,
            "confidence": float,
            "tokens": List[int],
            "error": Optional[str]
        }
        """
        if self.model is None:
            return {
                "d3": "", "d4": "", "province": "", "full_plate": "",
                "confidence": 0.0, "tokens": [],
                "error": self.error_msg or "Model not initialized"
            }

        cache_key = None
        if isinstance(image_input, (str, Path)):
            cache_key = str(Path(image_input).resolve())
            if use_cache and cache_key in self._cache:
                return self._cache[cache_key]
            try:
                pil_img = Image.open(image_input)
            except Exception as e:
                return {
                    "d3": "", "d4": "", "province": "", "full_plate": "",
                    "confidence": 0.0, "tokens": [],
                    "error": f"Failed to open image: {e}"
                }
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            return {
                "d3": "", "d4": "", "province": "", "full_plate": "",
                "confidence": 0.0, "tokens": [],
                "error": f"Unsupported input type: {type(image_input)}"
            }

        try:
            tensor = self.preprocess_image(pil_img).to(self.device)
            with torch.no_grad():
                preds = self.model(tensor)
                # Compute confidence from softmax
                probs = F.softmax(preds, dim=-1)
                max_probs, indices = probs.max(dim=-1)
                
                token_list = indices[0].cpu().numpy().tolist()
                prob_list = max_probs[0].cpu().numpy().tolist()

            d3, d4, province, full_plate = decode_tokens_to_plate(token_list)
            
            # Compute average confidence over valid tokens
            valid_probs = [p for p in prob_list if p > 0.05]
            conf = float(np.mean(valid_probs)) if valid_probs else 0.0

            result = {
                "d3": d3,
                "d4": d4,
                "province": province,
                "full_plate": full_plate,
                "confidence": round(conf, 4),
                "tokens": token_list,
                "error": None
            }

            if cache_key:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            return {
                "d3": "", "d4": "", "province": "", "full_plate": "",
                "confidence": 0.0, "tokens": [],
                "error": f"Inference error: {e}"
            }
