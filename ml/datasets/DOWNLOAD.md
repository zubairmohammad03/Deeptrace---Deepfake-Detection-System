# Dataset Download Guide

Place all datasets under `ml/data/` with this structure:

```
ml/data/
  faceforensics/real/   ← extracted frames from original videos
  faceforensics/fake/   ← extracted frames from manipulated videos
  celebdf/real/
  celebdf/fake/
  dfdc/real/
  dfdc/fake/
  wilddeepfake/real/
  wilddeepfake/fake/
```

---

## 1. FaceForensics++ (Most Important)
- **Size**: ~16GB compressed
- **Request access**: https://github.com/ondyari/FaceForensics
- After download, extract frames using `ml/training/extract_frames.py`

## 2. Celeb-DF v2
- **Size**: ~2.1GB
- **Download**: https://github.com/yuezunli/celeb-deepfakeforensics
- Direct Google Drive link available on their GitHub

## 3. DFDC (DeepFake Detection Challenge)
- **Size**: ~500GB full / use preview subset (~10GB)
- **Preview subset**: https://www.kaggle.com/c/deepfake-detection-challenge/data
- Requires Kaggle account: `kaggle competitions download -c deepfake-detection-challenge`

## 4. WildDeepfake
- **Size**: ~10GB
- **Download**: https://github.com/deepfakeinthewild/deepfake-in-the-wild
- Request form on their GitHub page

---

## Quick Start with Small Subset (Recommended for Development)

For fast prototyping, use only FaceForensics++ with 1000 samples per class:

```bash
# After downloading FF++, run frame extraction:
python ml/training/extract_frames.py \
  --input data/faceforensics/videos \
  --output data/faceforensics \
  --max_per_video 10
```

Then train with:
```bash
python ml/training/train.py --max_per_source 1000 --epochs 5
```
