# Gaze-Conditioned Weakly-Supervised Affordance Understanding

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project performs unsupervised temporal segmentation of egocentric videos using CLIP embeddings and KMeans clustering. It lays the groundwork for gaze-conditioned affordance understanding, inspired by Yezhou Yang’s research on egocentric action understanding.

---

## 📁 Project Structure

```bash
gaze-affordance-ws-learning/
├── scripts/
│   ├── extract_frames.py        # Extracts frames from video
│   ├── extract_features.py      # Embeds frames using CLIP
│   └── cluster_frames.py        # Performs KMeans segmentation
├── frames/                      # Extracted frame images
├── features/                    # CLIP 512D embeddings (.npy)
├── clusters/                    # KMeans cluster labels
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignored files
└── README.md
```

---

## 🧠 Models Used

- [OpenAI CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32)

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gaze-affordance-ws-learning.git
cd gaze-affordance-ws-learning

# Install dependencies
pip install -r requirements.txt
```

Recommended versions:
```txt
torch==2.1.0
transformers==4.35.0
scikit-learn==1.3.0
opencv-python==4.8.0.76
matplotlib==3.8.0
Pillow==10.0.0
```

---

## 🚀 Run the Full Pipeline

```bash
# 1. Extract frames from a sample video
python scripts/extract_frames.py

# 2. Generate CLIP features for each frame
python scripts/extract_features.py

# 3. Cluster frames into action segments
python scripts/cluster_frames.py
```

---

## 📌 Next Steps

- [ ] Add gaze simulation (center-based attention mask)
- [ ] Add affordance tagging for each action cluster
- [ ] Integrate GPT-4 to generate semantic Q&A from segments

---

## ✨ Credits

Built with inspiration from [Prof. Yezhou Yang](https://yezhouyang.engineering.asu.edu/)'s work in egocentric visual understanding and weak supervision.

