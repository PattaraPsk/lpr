#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, glob
import cv2, numpy as np

sys.stdout.reconfigure(encoding='utf-8')
out_dir = r"C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b\inspect_candidates"
os.makedirs(out_dir, exist_ok=True)

# Clean out old files
for f in os.listdir(out_dir):
    try:
        os.remove(os.path.join(out_dir, f))
    except Exception:
        pass

no_candidates = [
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_สณ_332_กรุงเทพมหานคร\cur_AK43_WhiteBlackCar_สณ_332_กรุงเทพมหานคร.jpg", "สณ 332 (cur_AK43)"),
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_วณ_315_กรุงเทพมหานคร\cur_AK00_WhiteBlackCar_วณ_315_กรุงเทพมหานคร.jpg", "วณ 315 (cur_AK00)"),
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_วณ_40_กรุงเทพมหานคร\cur_AN24_WhiteBlackCar_วณ_40_กรุงเทพมหานคร.jpg", "วณ 40 (cur_AN24)"),
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_ศณ_32_กรุงเทพมหานคร\cur_AK27_WhiteBlackCar_ศณ_32_กรุงเทพมหานคร.jpg", "ศณ 32 (cur_AK27)"),
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_1กณ_27_กรุงเทพมหานคร\cur_AE73_WhiteBlackCar_1กณ_27_กรุงเทพมหานคร.jpg", "1กณ 27 (cur_AE73)"),
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_ญณ_1411_กรุงเทพมหานคร\cur_AD01_WhiteBlackCar_ญณ_1411_กรุงเทพมหานคร.jpg", "ญณ 1411 (cur_AD01)"),
    (r"d:\Projects-anti\lpr\dataset\labeled\labeled_gemini_flash\images\plate_024_sedan_ษณ_3938_กรุงเทพมหานคร.jpg", "ษณ 3938 (plate_024)")
]

tho_candidates = [
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_ญฒ_5225_กรุงเทพมหานคร\cur_AU76_WhiteBlackCar_ญฒ_5225_กรุงเทพมหานคร.jpg", "ญฒ 5225 (cur_AU76)"),
    (r"d:\Projects-anti\lpr\dataset\verified\curated_lpr_dataset\plates\AB14_WhiteBlackCar_ฌฒ_1168_กรุงเทพมหานคร.jpg", "ฌฒ 1168 (AB14)"),
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_5ขฒ_3555_กรุงเทพมหานคร\cur_AL45_WhiteBlackCar_5ขฒ_3555_กรุงเทพมหานคร.jpg", "5ขฒ 3555 (cur_AL45)"),
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_2ขฒ_7789_กรุงเทพมหานคร\cur_AJ11_WhiteBlackCar_2ขฒ_7789_กรุงเทพมหานคร.jpg", "2ขฒ 7789 (cur_AJ11)"),
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_3กฒ_1_กรุงเทพมหานคร\cur_AE75_WhiteBlackCar_3กฒ_1_กรุงเทพมหานคร.jpg", "3กฒ 1 (cur_AE75)"),
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_3กฒ_3030_กรุงเทพมหานคร\cur_AM69_WhiteBlackCar_3กฒ_3030_กรุงเทพมหานคร.jpg", "3กฒ 3030 (cur_AM69)"),
    (r"d:\Projects-anti\lpr\dataset\labeled\labeled_gemini_flash\images\plate_012_pickup_2ฒร_1387_กรุงเทพมหานคร.jpg", "2ฒร 1387 (plate_012)")
]

do_candidates = [
    (r"d:\Projects-anti\lpr\dataset\verified\curated_lpr_dataset\plates\AB25_WhiteBlackCar_ฎธ_118_กรุงเทพมหานคร.jpg", "ฎธ 118 (AB25)"),
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_ฎฉ_199_กรุงเทพมหานคร\cur_AE68_WhiteBlackCar_ฎฉ_199_กรุงเทพมหานคร.jpg", "ฎฉ 199 (cur_AE68)"),
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_ฎศ_635_กรุงเทพมหานคร\cur_AY64_WhiteBlackCar_ฎศ_635_กรุงเทพมหานคร.jpg", "ฎศ 635 (cur_AY64)"),
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_ฎห_377_กรุงเทพมหานคร\cur_AM04_WhiteBlackCar_ฎห_377_กรุงเทพมหานคร.jpg", "ฎห 377 (cur_AM04)"),
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_ฎล_250_กรุงเทพมหานคร\cur_AG99_WhiteBlackCar_ฎล_250_กรุงเทพมหานคร.jpg", "ฎล 250 (cur_AG99)"),
    (r"d:\Projects-anti\lpr\dataset\verified\curated_lpr_dataset\plates\AA80_car_ฎข_1122_กรุงเทพมหานคร.jpg", "ฎข 1122 (AA80)"),
    (r"d:\Projects-anti\lpr\dataset\dataset-approved\202610261140\WhiteBlackCar_2กฎ_258_กรุงเทพมหานคร\cur_AJ52_WhiteBlackCar_2กฎ_258_กรุงเทพมหานคร.jpg", "2กฎ 258 (cur_AJ52)"),
    (r"d:\Projects-anti\lpr\dataset\labeled\labeled_gemini_flash\images\plate_019_sedan_ศฎ_2900_กรุงเทพมหานคร.jpg", "ศฎ 2900 (plate_019)")
]

for grp_name, cand_list in [('no', no_candidates), ('tho', tho_candidates), ('do', do_candidates)]:
    print(f"=== Group: {grp_name} ===")
    for idx, (path, title) in enumerate(cand_list):
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            continue
        im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        out_name = f"{grp_name}_{idx}.jpg"
        out_p = os.path.join(out_dir, out_name)
        _, buf = cv2.imencode(".jpg", im)
        buf.tofile(out_p)
        print(f"  [OK] Saved {out_name} ({w}x{h}): {title}")
