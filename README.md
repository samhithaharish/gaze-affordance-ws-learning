# Gaze-Conditioned Weakly-Supervised Affordance Understanding

This project performs unsupervised temporal segmentation of egocentric videos using CLIP and KMeans, simulating gaze-conditioned attention to later model affordance understanding.

## 📁 Project Structure
- `scripts/` — Python scripts for each core stage
- `frames/` — Extracted frames from input video
- `features/` — Saved CLIP embeddings
- `clusters/` — Cluster outputs from KMeans

## 🧠 Models Used
- [OpenAI CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32)

## 🧪 Quick Start

```bash
# Clone this repository
git clone https://github.com/yourusername/gaze-affordance-ws-learning.git
cd gaze-affordance-ws-learning

# Install dependencies
pip install -r requirements.txt

# Extract video frames
python scripts/extract_frames.py

# Generate CLIP features
python scripts/extract_features.py

# Run KMeans clustering
python scripts/cluster_frames.py
```

## 📌 Next Steps
- Add gaze simulation module to simulate attention focus
- Add affordance tagging for action interpretation
- Integrate GPT-4 (or similar LLM) for semantic question answering

## ✨ Credits
Inspired by Prof. Yezhou Yang’s work on egocentric visual understanding and weak supervision.
