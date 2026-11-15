# Video Generalization & Temporal Anti-Flicker Pipeline

This extension implements real-time video stream enhancement with temporal consistency guarantees, preventing algorithm-switching flicker and strobing across consecutive video frames.

**Target Hardware**: Real-time CPU execution (>100 FPS on commodity edge CPU).  
**Dependencies**: OpenCV (`cv2`), NumPy. Zero deep learning dependencies required.

---

## Components

1. **`video_pipeline.py`**:
   - `TemporalFeatureFilter`: Exponential Moving Average (EMA) feature smoothing with motion-gated sparse keyframing ($\alpha=0.20$, keyframe interval 5).
   - `MarkovianHysteresisGate`: Transition penalty cost function preventing rapid algorithm toggling between adjacent frames.
   - `FastTemporalInvertDehaze`: Atmospheric transmission map caching across correlated frames.
   - `VideoEnhancementStream`: Production streaming video pipeline.
2. **`generate_video_demo.py`**:
   - Synthesizes dynamic exposure sweeps and tests naive per-frame selection versus temporally smoothed selection.
   - Computes Inter-Frame Flicker Rates (IFR) and generates comparison video.
3. **`video_demo_anti_flicker.mp4`**:
   - Rendered demonstration showing raw video, naive per-frame selection (flickering), and the stabilized anti-flicker output.

---

## How to Run (CPU)

```bash
# 1. Run unit test suite for video temporal smoothing
python3 -m unittest tests/test_video_pipeline.py

# 2. Generate side-by-side anti-flicker demonstration video (CPU)
python3 extensions/video/generate_video_demo.py \
    --input data/lol_dataset/eval15/low \
    --output extensions/video/video_demo_anti_flicker.mp4 \
    --fps 30

# 3. Stream process an arbitrary video file (CPU)
python3 -c "
from extensions.video.video_pipeline import VideoEnhancementStream
stream = VideoEnhancementStream(alpha=0.2, keyframe_interval=5)
print('Video stream processor initialized on CPU.')
"
```

---
*Maintained under Project Triathlon Extensions Architecture.*
