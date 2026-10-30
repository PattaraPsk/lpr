# -*- coding: utf-8 -*-
import os
import cv2

def crop_characters():
    img_path = r"C:/Users/WS4090/.gemini/antigravity/brain/773ff2cd-67ac-4d2b-aa3c-714ee8828ac6/.user_uploaded/media_1792992418038_0e40a83a.jpg"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    
    h_img, w_img, _ = img.shape

    out_dirs = [
        r"D:\Projects-anti\lpr\cropped_chars",
        r"C:\Users\WS4090\.gemini\antigravity\brain\773ff2cd-67ac-4d2b-aa3c-714ee8828ac6\cropped_chars"
    ]

    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

    # Coordinates for ฉ ต 4 5 5 4
    boxes = [
        ("char_1_ฉ", "1_cha", 5, 5, 15, 22),
        ("char_2_ต", "2_to", 25, 5, 13, 22),
        ("char_3_4", "3_4", 49, 5, 12, 22),
        ("char_4_5", "4_5", 64, 5, 11, 22),
        ("char_5_5", "5_5", 78, 5, 12, 22),
        ("char_6_4", "6_4", 93, 5, 12, 22)
    ]

    for thai_name, ascii_name, x, y, bw, bh in boxes:
        crop = img[y:y+bh, x:x+bw]
        
        # Also padded version (1px margin)
        pad = 1
        y1_p = max(0, y - pad)
        y2_p = min(h_img, y + bh + pad)
        x1_p = max(0, x - pad)
        x2_p = min(w_img, x + bw + pad)
        crop_padded = img[y1_p:y2_p, x1_p:x2_p]

        for d in out_dirs:
            # cv2.imwrite with unicode paths on Windows can fail with imwrite directly,
            # so we use cv2.imencode + python open for safety and unicode support.
            for name, crop_img in [(thai_name, crop), (f"{thai_name}_padded", crop_padded),
                                  (ascii_name, crop), (f"{ascii_name}_padded", crop_padded)]:
                target_path = os.path.join(d, f"{name}.png")
                success, encoded_img = cv2.imencode(".png", crop_img)
                if success:
                    with open(target_path, "wb") as f:
                        f.write(encoded_img.tobytes())

    print("All crops saved successfully with unicode & ascii naming.")

if __name__ == "__main__":
    crop_characters()
