# Latency Benchmarks & Edge Hardware Profiling

This extension profiles inference latency, memory consumption, throughput, and Pareto regret-latency trade-offs for embedded and edge computer vision deployments.

**Target Hardware**: Commodity CPU and embedded processors (ARM Cortex, x86_64).  
**Dependencies**: OpenCV (`cv2`), NumPy, Pandas, scikit-learn, joblib. Optional: `onnxruntime`.

---

## Components

1. **`edge_hardware_profile.py`**:
   - Profiles 8-dimensional feature extraction time over 500 iterations.
   - Measures CPU inference latency of Logistic Regression and Random Forest.
   - Compares selector throughput against deep neural network baselines (Zero-DCE, RetinexNet, Restormer).
   - Generates `results/figures/fig9_edge_profiling.png`.
2. **`latency_aware_selection.py`**:
   - Implements Pareto frontier optimization balancing computational budget ($\tau$) against decision regret.
   - Computes regret-latency frontier across CPU operating points.
3. **`edge_hardware_profile.csv` & `latency_regret_frontier.csv`**:
   - Empirical latency profiles across CPU platforms.

---

## How to Run (CPU)

```bash
# 1. Run 500-iteration hardware profiling on CPU
python3 extensions/latency/edge_hardware_profile.py \
    --image data/lol_dataset/eval15/low/1.png \
    --model models/selector.joblib \
    --num_runs 500

# 2. Run latency-constrained Pareto frontier benchmark
python3 extensions/latency/latency_aware_selection.py \
    --data data/lol_dataset/our485 \
    --budget_ms 2.0
```

---
*Maintained under Project Triathlon Extensions Architecture.*
