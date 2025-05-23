from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import os

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_features(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features[0].cpu().numpy()

def extract_all_clip_features(frame_dir="frames", save_path="features/clip_features.npy"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    features = []
    frame_paths = sorted(os.listdir(frame_dir))
    for filename in frame_paths:
        vec = get_clip_features(os.path.join(frame_dir, filename))
        features.append(vec)
    features_np = np.stack(features)
    np.save(save_path, features_np)
    print(f"✅ Saved {len(features_np)} feature vectors of size {features_np.shape[1]} to '{save_path}'")
    return features_np
