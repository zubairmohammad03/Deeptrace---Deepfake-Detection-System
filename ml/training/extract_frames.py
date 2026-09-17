#!/usr/bin/env python
# ml/training/extract_frames.py
"""
Extract frames from video datasets for training.

Usage:
    python extract_frames.py --input data/faceforensics/videos/real --output data/faceforensics/real
    python extract_frames.py --input data/faceforensics/videos/fake --output data/faceforensics/fake --max_per_video 20
"""
import argparse
import os
from pathlib import Path
import cv2
from tqdm import tqdm


def extract_frames(video_path: str, output_dir: str, max_frames: int = 10, img_size: int = 256):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0

    # Sample evenly across video
    indices = [int(i * total / max_frames) for i in range(max_frames)]
    saved = 0
    stem = Path(video_path).stem

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (img_size, img_size))
        out_path = os.path.join(output_dir, f"{stem}_f{idx:05d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",          required=True, help="Input video directory")
    p.add_argument("--output",         required=True, help="Output frames directory")
    p.add_argument("--max_per_video",  type=int, default=10)
    p.add_argument("--img_size",       type=int, default=256)
    p.add_argument("--extensions",     nargs="+", default=[".mp4", ".avi", ".mov"])
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    videos = [f for f in Path(args.input).rglob("*") if f.suffix.lower() in args.extensions]
    print(f"Found {len(videos)} videos in {args.input}")

    total_frames = 0
    for vid in tqdm(videos):
        n = extract_frames(str(vid), args.output, args.max_per_video, args.img_size)
        total_frames += n

    print(f"\nExtracted {total_frames} frames to {args.output}")


if __name__ == "__main__":
    main()
