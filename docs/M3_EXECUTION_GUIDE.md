# ASTCV — Phase 1 Operational Execution Manual (M3: Aswin K N)
**Role:** Data Engineer & Pipeline Architect  
**Duration:** Days 1–20 (Weeks 1–3)  
**Primary Deliverables:**
1. Structured Data Storage, Versioning & Git Repository
2. Dataset Ingestion & Preprocessing Suite for 6 Target Benchmarks
3. MLflow Experiment Tracking for Ingestion & Jitter Telemetry
4. Cross-Device Jitter Test Ingestion (< 50ms P95 latency across >= 2 devices)
5. Documented Data Contract (joint sign-off with M1 Amal & M2 Abhijith)

---

## Part 1: Day-by-Day Execution Plan

### Week 1 (Days 1–7): Pipeline Foundation & Access Approvals
* **Day 1: Pipeline Initialization & Form Submissions**
  - Verify workspace: `cd C:\Users\Ramdas\.gemini\antigravity\scratch\astcv-pipeline`
  - Run `python schemas/data_contract.py` to regenerate JSON schemas if schema changes.
  - Run `python src/datasets/generate_mock_samples.py` to seed synthetic clips.
  - **Critical Bottleneck Action:** Follow `docs/DATASET_ACCESS_GUIDE.md`:
    - Submit FaceForensics++ Google Form (TUM).
    - Email signed Celeb-DF v2 agreement (with Prof. Leo Francis).
    - Pull DFDC Preview set and accept Kaggle rules.
    - Register for VoxCeleb2 on Oxford VGG.
* **Day 2: Collaboration Sync (M1 & M2)**
  - Share `schemas/data_contract.json` with M2 (Abhijith) for his WebRTC screen pattern emitter.
  - Confirm with M1 (Amal) that 256x256 RGB face crops with 1.3x margin preserve sufficient forehead and jawline for his Phong specular and Lambertian diffuse models.
* **Day 3: Storage & Versioning Setup**
  - Track directory manifests in Git.
  - Initialize DVC for raw dataset versioning:
    ```bash
    dvc init
    dvc add data/raw
    git add data/raw.dvc .dvc .dvcignore
    git commit -m "feat(data): initialize DVC tracking for raw datasets"
    ```
* **Days 4–7: Buffer & Access Follow-up**
  - Run `python src/datasets/dataset_manager.py` daily to audit incoming files.
  - Follow up on TUM and Oxford VGG approval emails.

---

### Week 2 (Days 8–14): Dataset Ingestion & Preprocessing
* **Days 8–10: Ingest Approved Datasets**
  - As approval links arrive, download videos into respective subfolders in `data/raw/`:
    - `data/raw/faceforensics/`
    - `data/raw/celeb_df_v2/`
    - `data/raw/dfdc/`
    - `data/raw/voxceleb2/`
* **Days 11–12: Run Batch Standardization**
  - Execute canonical face cropping and alignment:
    ```bash
    python src/preprocessing/standardizer.py
    ```
  - Inspect output frames in `data/processed/{dataset_name}/` and `manifest.jsonl`.
* **Days 13–14: First-Pass Sanity Audit**
  - Run the automated sanity check suite:
    ```bash
    python src/preprocessing/sanity_checker.py
    ```
  - Review `data/metadata/sanity_check_report.md` to ensure zero corruptions, zero dimension mismatches, and balanced real vs fake distributions.

---

### Week 3 (Days 15–20): Experiment Tracking & M2 Integration
* **Day 15: Launch MLflow UI**
  - Start the MLflow tracking server:
    ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
    ```
  - View logged runs in your browser at `http://localhost:5000`.
* **Day 16: Log Preprocessing Audit to MLflow**
  - Run ingestion logger:
    ```bash
    python -c "from src.preprocessing.sanity_checker import SanityChecker; from src.tracking.mlflow_tracker import ASTCVTracker; c = SanityChecker(); t = ASTCVTracker(); t.log_dataset_ingestion(c.audit_all(), 'data/metadata/sanity_check_report.md')"
    ```
* **Days 17–18: Cross-Device Jitter Integration with M2**
  - Receive capture sessions from Abhijith (M2):
    - Device 1: Laptop workstation (Windows / macOS webcam)
    - Device 2: Smartphone (Android / iOS front camera)
  - Ingest sessions using:
    ```bash
    python src/integration/m2_session_receiver.py
    ```
  - Check that P95 latency is $\le 50\text{ ms}$ and frame drop rate is $< 5\%$.
* **Days 19–20: Joint Data Contract Sign-Off & Buffer**
  - Review `docs/DATA_CONTRACT_SPECIFICATION.md` jointly with Amal (M1) and Abhijith (M2).
  - Confirm all Phase 1 deliverables are committed to the Git repository.

---

## Part 2: Quick Command Reference

| Action | Command |
|:---|:---|
| Scan Dataset Storage | `python src/datasets/dataset_manager.py` |
| Generate Mock Datasets | `python src/datasets/generate_mock_samples.py` |
| Standardize Datasets | `python src/preprocessing/standardizer.py` |
| Run First-Pass Sanity Check | `python src/preprocessing/sanity_checker.py` |
| Ingest M2 Capture Session | `python src/integration/m2_session_receiver.py` |
| Run Test Suite | `python -m pytest tests/test_pipeline.py -v` |
| Launch MLflow Dashboard | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
