import os
import glob
import cv2

print("=== 1. LPR Videos ===")
for v in glob.glob("dataset/LPR-20261019T045913Z-1-001/LPR/*.*"):
    cap = cv2.VideoCapture(v)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {os.path.basename(v)}: {w}x{h}, {fps:.1f} fps, {count} frames ({count/max(fps,1):.1f}s)")
    cap.release()

print("\n=== 2. Extra_Videos ===")
for v in glob.glob("dataset/Extra_Videos-20261019T045943Z-1-001/Extra_Videos/*.*"):
    if v.endswith(('.mp4', '.avi', '.mkv', '.mov')):
        cap = cv2.VideoCapture(v)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  {os.path.basename(v)}: {w}x{h}, {fps:.1f} fps, {count} frames ({count/max(fps,1):.1f}s)")
        cap.release()

print("\n=== 3. Raw Video Record ===")
for v in glob.glob("dataset/Raw Video Record/*.*"):
    if v.endswith(('.mp4', '.avi', '.mkv', '.mov')):
        cap = cv2.VideoCapture(v)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  {os.path.basename(v)}: {w}x{h}, {fps:.1f} fps, {count} frames ({count/max(fps,1):.1f}s)")
        cap.release()
