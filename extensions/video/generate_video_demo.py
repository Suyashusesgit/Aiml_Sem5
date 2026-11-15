"""
Demonstration Video and GIF Generator for Project Triathlon
Demonstrates temporal anti-flicker stability via EMA smoothing and hysteresis.

Generates:
    - results/video_demo_anti_flicker.gif
    - results/video_demo_anti_flicker.mp4
"""

import os
import sys
import glob
import cv2
import numpy as np
from PIL import Image
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.video_pipeline import VideoStreamProcessor
from src.features import extract_features, FEATURE_NAMES
from src.methods import METHODS
from src.multi_degradation import high_boost_sharpen, dcp_dehaze
import joblib

ALL_METHODS = {
    **METHODS,
    "high_boost_sharpen": high_boost_sharpen,
    "dcp_dehaze": dcp_dehaze,
}

def load_base_images(target_size=(360, 240)):
    """Loads benchmark samples across diverse winning methods (Gamma, HistEq, Retinex)."""
    sample_ids = ["102", "101", "109", "1"]
    imgs = []
    for sid in sample_ids:
        candidates = [
            f"data/lol_dataset/our485/low/{sid}.png",
            f"data/lol_dataset/eval15/low/{sid}.png",
            f"data/lol_v1/our485/low/{sid}.png",
            f"data/lol_v1/eval15/low/{sid}.png",
        ]
        p = next((c for c in candidates if os.path.exists(c)), None)
        if p:
            img = cv2.imread(p)
            if img is not None:
                imgs.append(cv2.resize(img, target_size, interpolation=cv2.INTER_AREA))
    
    if len(imgs) < 2:
        # Fallback if specific files not found
        candidates = sorted(glob.glob("data/lol_dataset/our485/low/*.png") + glob.glob("data/lol_v1/our485/low/*.png"))
        for p in candidates[:4]:
            img = cv2.imread(p)
            if img is not None:
                imgs.append(cv2.resize(img, target_size, interpolation=cv2.INTER_AREA))
                
    return imgs

def generate_video_sequence(base_images, total_frames=60):
    """Synthesizes a realistic fluctuating low-light sequence with light pulses and sensor noise."""
    frames = []
    n_base = len(base_images)
    
    for i in range(total_frames):
        # Cyclical interpolation across base images
        t = (i / total_frames) * (n_base - 1)
        idx = int(t)
        alpha = t - idx
        img1 = base_images[min(idx, n_base - 1)]
        img2 = base_images[min(idx + 1, n_base - 1)]
        blended = cv2.addWeighted(img1, 1.0 - alpha, img2, alpha, 0)
        
        # Add dynamic illuminance fluctuation (simulating flickering streetlights / passing cars)
        oscillation = np.sin(i * 0.45) * 8.0 + np.sin(i * 1.2) * 5.0
        frame_flt = np.clip(blended.astype(np.float32) + oscillation, 0, 255)
        
        # Add slight sensor noise jitter
        noise = np.random.normal(0, 1.8, frame_flt.shape)
        noisy_frame = np.clip(frame_flt + noise, 0, 255).astype(np.uint8)
        frames.append(noisy_frame)
        
    return frames

def build_demo_video():
    os.makedirs("results", exist_ok=True)
    base_imgs = load_base_images((360, 240))
    raw_frames = generate_video_sequence(base_imgs, total_frames=60)
    
    model = joblib.load("models/rf_selector.joblib")
    stabilized_pipeline = VideoStreamProcessor(
        model=model,
        switch_penalty=0.15,
        alpha=0.25
    )
    
    out_frames_bgr = []
    out_frames_pil = []
    
    raw_prev_method = None
    raw_switch_count = 0
    
    stab_prev_method = None
    stab_switch_count = 0
    
    w, h = 360, 240
    canvas_w = w * 3 + 40
    canvas_h = h + 120
    
    print("Processing 60 frames through raw vs stabilized pipeline...")
    
    for frame_idx, frame in enumerate(raw_frames):
        # 1. Raw instantaneous selection (unmitigated)
        feats = extract_features(frame, include_engineered=True)
        X = np.array([[feats[k] for k in FEATURE_NAMES]])
        raw_method = model.predict(X)[0]
        if raw_prev_method is not None and raw_method != raw_prev_method:
            raw_switch_count += 1
        raw_prev_method = raw_method
        raw_enhanced = ALL_METHODS[raw_method](frame)
        
        # 2. Stabilized selection (EMA + Markovian hysteresis)
        stab_enhanced, stab_method, stab_meta = stabilized_pipeline.process_frame(frame)
        if stab_prev_method is not None and stab_method != stab_prev_method:
            stab_switch_count += 1
        stab_prev_method = stab_method
        
        # 3. Render Composite Canvas
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (15, 23, 42) # Slate background
        
        # Header text
        cv2.putText(canvas, "PROJECT TRIATHLON · STREAMING VIDEO ANTI-FLICKER EVALUATION", (20, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Frame: {frame_idx+1:02d}/60 | Luminance: {feats['brightness']:.1f} | 30 FPS", (canvas_w - 320, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)
        
        # Place 3 Panels
        # Panel 1: Low-Light Input
        x1 = 15
        y_top = 50
        canvas[y_top:y_top+h, x1:x1+w] = frame
        cv2.rectangle(canvas, (x1, y_top), (x1+w, y_top+h), (51, 65, 85), 2)
        cv2.putText(canvas, "1. Input Stream (Fluctuating)", (x1 + 8, y_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Panel 2: Raw Frame-by-Frame (Flickering)
        x2 = x1 + w + 10
        canvas[y_top:y_top+h, x2:x2+w] = raw_enhanced
        is_flicker_frame = (raw_switch_count > stab_switch_count)
        border_col = (0, 0, 220) if is_flicker_frame else (71, 85, 105)
        cv2.rectangle(canvas, (x2, y_top), (x2+w, y_top+h), border_col, 2)
        cv2.putText(canvas, f"2. Raw Frame-by-Frame: {raw_method.upper()}", (x2 + 8, y_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (56, 189, 248), 1, cv2.LINE_AA)
        if is_flicker_frame:
            cv2.putText(canvas, "FLICKERING", (x2 + w - 100, y_top + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Panel 3: Stabilized Pipeline (EMA + Hysteresis)
        x3 = x2 + w + 10
        canvas[y_top:y_top+h, x3:x3+w] = stab_enhanced
        cv2.rectangle(canvas, (x3, y_top), (x3+w, y_top+h), (34, 197, 94), 2)
        cv2.putText(canvas, f"3. Stabilized: {stab_method.upper()}", (x3 + 8, y_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (74, 222, 128), 1, cv2.LINE_AA)
        cv2.putText(canvas, "STABLE", (x3 + w - 75, y_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (34, 197, 94), 2, cv2.LINE_AA)
        
        # Bottom Telemetry HUD
        hud_y = y_top + h + 30
        cv2.putText(canvas, f"Unmitigated Jitter Switches: {raw_switch_count}", (x2 + 10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (239, 68, 68), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Stabilized Switches: {stab_switch_count} (Hysteresis Active)", (x3 + 10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (34, 197, 94), 1, cv2.LINE_AA)
        
        out_frames_bgr.append(canvas)
        out_frames_pil.append(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
        
    # Save as Animated GIF
    gif_path = "results/video_demo_anti_flicker.gif"
    out_frames_pil[0].save(
        gif_path,
        save_all=True,
        append_images=out_frames_pil[1:],
        duration=50,  # 20 FPS
        loop=0
    )
    print(f"Saved animated GIF: {gif_path} ({os.path.getsize(gif_path) / 1024:.1f} KB)")
    
    # Save as MP4 Video
    mp4_path = "results/video_demo_anti_flicker.mp4"
    fourcc_fn: Any = getattr(cv2, "VideoWriter_fourcc", getattr(cv2.VideoWriter, "fourcc", None))
    fourcc = fourcc_fn(*"mp4v") if callable(fourcc_fn) else 0x7634706d
    vw = cv2.VideoWriter(mp4_path, fourcc, 20.0, (canvas_w, canvas_h))
    for f in out_frames_bgr:
        vw.write(f)
    vw.release()
    print(f"Saved MP4 video: {mp4_path} ({os.path.getsize(mp4_path) / 1024:.1f} KB)")
    print(f"Result: Raw jittered {raw_switch_count} times vs Stabilized switched only {stab_switch_count} times!")

if __name__ == "__main__":
    build_demo_video()
