# ASTCV — Data Engineering, Versioning & Experiment Pipeline
**Lead:** Aswin K N (M3 - Data Engineer)  
**Project:** Adaptive Screen-Timestamped Computer Vision (ASTCV)  
**Advisors & Core Team:**  
- **M1 (ML & Physics Modeler):** Amal Krishna J  
- **M2 (CV Engineer & Demo Builder):** Abhijith Krishna P S  
- **M3 (Data Engineer):** Aswin K N  
- **Faculty Guide:** Leo Francis P  

---

## 📌 Subsystem Purpose
This subsystem is the backbone data engineering repository for the ASTCV anti-spoofing and liveness platform. It provides:
1. **Multi-Source Benchmark Ingestion:** Preprocessing, face cropping, and standardization across 6 target datasets (**FaceForensics++**, **Celeb-DF v2**, **DFDC**, **UADFV**, **FFHQ**, **VoxCeleb2**).
2. **Session Storage & Versioning:** Frame-accurate session tracking linking screen emitter challenge patterns to high-resolution camera capture timestamps.
3. **M1-M2-M3 Data Contract:** Pydantic v2 schemas validating screen challenges, camera telemetry, and session manifests.
4. **Experiment & Telemetry Tracking:** Centralized MLflow tracking for preprocessing audits, cross-device jitter benchmarks, and FAR/FRR metrics.

---

## 📂 Repository Layout
```
astcv-pipeline/
├── configs/
│   └── pipeline_config.yaml         # Centralized configuration paths and parameters
├── data/
│   ├── raw/                         # 6 Benchmark datasets (FF++, Celeb-DF, DFDC, UADFV, FFHQ, VoxCeleb2)
│   ├── processed/                   # Standardized 256x256 RGB crops with manifest.jsonl
│   ├── sessions/                    # Captured WebRTC sessions from M2 (cross-device tests)
│   └── metadata/                    # Datasets status, checksums, sanity reports
├── docs/
│   ├── DATASET_ACCESS_GUIDE.md      # Access URLs, forms, email templates for all 6 datasets
│   ├── DATA_CONTRACT_SPECIFICATION.md # Joint M1-M2-M3 telemetry schema and data contract
│   └── M3_EXECUTION_GUIDE.md        # Day-by-Day operational guide for Days 1–20
├── schemas/
│   ├── data_contract.py             # Pydantic v2 schema definitions
│   └── data_contract.json           # JSON Schema export for frontend & algorithms
├── src/
│   ├── datasets/
│   │   ├── dataset_manager.py       # Registry & status audit tracker
│   │   └── generate_mock_samples.py # Day-1 synthetic data generator for immediate testing
│   ├── preprocessing/
│   │   ├── frame_extractor.py       # Target-FPS video frame extraction
│   │   ├── face_detector.py         # Multi-detector face cropping with margin
│   │   ├── standardizer.py          # Unified dataset normalization engine
│   │   └── sanity_checker.py        # Label, count, and quality audit suite
│   ├── tracking/
│   │   └── mlflow_tracker.py        # MLflow experiment logger
│   └── integration/
│       └── m2_session_receiver.py   # M2 WebRTC session receiver & jitter calculator
├── tests/
│   └── test_pipeline.py             # Pytest automated test suite
└── requirements.txt
```

---

## 🚀 Quickstart Guide

### 1. Initial Setup
```bash
python -m pip install -r requirements.txt
```

### 2. Verify Pipeline on Synthetic Samples
```bash
# Generate mock video/image samples for all 6 datasets
python src/datasets/generate_mock_samples.py

# Ingest and standardize to canonical 256x256 crops
python src/preprocessing/standardizer.py

# Run first-pass sanity check (labels, counts, quality)
python src/preprocessing/sanity_checker.py
```

### 3. Run Cross-Device Jitter Benchmark (M2 Integration)
```bash
python src/integration/m2_session_receiver.py
```

### 4. Run Automated Unit & Integration Tests
```bash
python -m pytest tests/test_pipeline.py -v
```

### 5. Launch MLflow Dashboard
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open `http://localhost:5000` to visualize ingestion runs, jitter percentiles, and session artifacts.
