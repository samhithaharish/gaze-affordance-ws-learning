import cv2
import os

def extract_frames(video_path, output_dir="frames", frame_skip=5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    i, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_skip == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        i += 1
    cap.release()
    print(f"✅ Extracted and saved {saved} frames to '{output_dir}'")
