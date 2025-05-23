from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os

def cluster_clip_features(feature_path="features/clip_features.npy", n_clusters=6, save_path="clusters/frame_clusters.npy"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    features = np.load(feature_path)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    np.save(save_path, cluster_labels)
    print(f"✅ Clustered into {n_clusters} segments and saved to '{save_path}'")
    return cluster_labels

def visualize_clusters(cluster_labels):
    plt.figure(figsize=(15, 2))
    plt.plot(cluster_labels, marker='o')
    plt.title("Frame Cluster Assignments Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Cluster ID")
    plt.grid(True)
    plt.show()
