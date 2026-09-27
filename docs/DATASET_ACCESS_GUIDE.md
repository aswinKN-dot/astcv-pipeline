# ASTCV — Dataset Access & Ingestion Protocol (M3: Aswin K N)
**Document Version:** 1.0.0  
**Phase:** Phase 1 (Days 1–7)  
**Target Milestone:** Full Access Clearance & Baseline Ingestion

---

## 1. Executive Summary & Access Bottleneck Strategy
Academic datasets for facial forensics and deepfake detection require strict compliance with ethics agreements, institutional verification, and faculty endorsement. Because access approvals can take **3 to 14 days**, application requests must be submitted on **Day 1**.

The ASTCV evaluation and baseline benchmark relies on 6 benchmark datasets:
1. **FaceForensics++ (FF++)** — Benchmark for multi-method manipulation detection.
2. **Celeb-DF (v2)** — High-visual-quality deepfakes with subtle temporal and boundary artifacts.
3. **Deepfake Detection Challenge (DFDC)** — Highly diverse in-the-wild facial video dataset.
4. **UADFV** — Foundational eye-blinking and facial artifact dataset.
5. **Flickr-Faces-HQ (FFHQ)** — High-resolution (1024x1024) genuine human face baseline.
6. **VoxCeleb2** — Audio-visual biometric and temporal identity reference.

---

## 2. Dataset Application Matrix & Step-by-Step Instructions

### Dataset 1: FaceForensics++ (FF++)
* **Institution:** Technical University of Munich (TUM) & Google
* **Access Type:** Form Submission & Verification via Google Form / Academic Email
* **Direct Access URL:** [TUM FaceForensics GitHub](https://github.com/ondyari/FaceForensics)
* **Application Link:** Submit the [FaceForensics Access Request Form](https://docs.google.com/forms/d/e/1FAIpQLSdRR5UBxEW6asEHZCF347oc-vGkqqWnks48m3Zcq9pxp0o4nQ/viewform)
* **Requirements:**
  - Academic email ID (`.edu`, `.ac.in`, etc.).
  - Advisor/Guide Name: **Leo Francis P**.
  - Project Title: *ASTCV: Adaptive Screen-Timestamped Computer Vision for Deepfake & Injection Attack Defense*.
* **Download Method:**
  Once approved, TUM will send the python download script `download-FaceForensics.py`.
  Command format:
  ```bash
  python download-FaceForensics.py ./data/raw/faceforensics -d Deepfakes Face2Face FaceSwap NeuralTextures original -c c23 -t videos
  ```
* **Storage Footprint:** ~80 GB (for c23 compressed set; ~400 GB for raw c0).

---

### Dataset 2: Celeb-DF (v2)
* **Institution:** SUNY Buffalo & Temple University (Yuezun Li, Xin Yang, Pu Sun, Honggang Qi, Siwei Lyu)
* **Access Type:** Application Form Agreement via Google Drive / Baidu
* **Direct Access URL:** [Celeb-DF GitHub Repository](https://github.com/yuezunli/celeb-deepfakeforensics)
* **Application Procedure:**
  1. Download and read the [Celeb-DF Agreement Form](https://github.com/yuezunli/celeb-deepfakeforensics/blob/master/Celeb-DF-v2-agreement.pdf).
  2. Complete the form with team lead / faculty advisor signature.
  3. Email the scanned PDF to the dataset maintainers (Dr. Siwei Lyu / Dr. Yuezun Li).
* **Storage Footprint:** ~11.5 GB (590 original YouTube videos + 5,639 deepfake synthesized videos).

---

### Dataset 3: Deepfake Detection Challenge (DFDC)
* **Institution:** Meta (Facebook AI), AWS, Partnership on AI
* **Access Type:** Kaggle Competition / AWS Open Data Agreement
* **Direct Access URL:** [Kaggle DFDC Dataset](https://www.kaggle.com/c/deepfake-detection-challenge/data) / [AWS Registry of Open Data](https://registry.opendata.aws/deepfake-detection-challenge/)
* **Application Procedure:**
  1. Log in to Kaggle with your academic account.
  2. Accept the DFDC competition rules and terms of use.
  3. Download via Kaggle CLI:
     ```bash
     kaggle competitions download -c deepfake-detection-challenge -p ./data/raw/dfdc
     ```
  *(Note: DFDC full dataset is ~470 GB. For Phase 1 Day 1-7, use the **DFDC Preview Set** (~4.5 GB) for immediate pipeline verification).*

---

### Dataset 4: UADFV
* **Institution:** University at Albany (SUNY)
* **Access Type:** Open Research Download / GitHub Release
* **Direct Access URL:** [UADFV Repository](https://github.com/danmohaha/WIFS2018_In_I_Trust)
* **Application Procedure:** Direct download via Google Drive or repository link provided in the WIFS paper.
* **Storage Footprint:** ~1.2 GB (49 real videos + 49 fake videos).

---

### Dataset 5: Flickr-Faces-HQ (FFHQ)
* **Institution:** NVIDIA Research (Tero Karras et al.)
* **Access Type:** Open Research (Creative Commons BY-NC-SA 4.0)
* **Direct Access URL:** [NVIDIA FFHQ GitHub](https://github.com/NVlabs/ffhq-dataset)
* **Application Procedure:**
  Direct programmatic download via official NVIDIA download script or Google Drive mirror:
  ```bash
  python src/datasets/download_ffhq.py --target_dir ./data/raw/ffhq --subset thumb128
  ```
* **Storage Footprint:** 1024x1024 PNG images (~89 GB full), or thumbnails subset (128x128 ~1.9 GB).

---

### Dataset 6: VoxCeleb2
* **Institution:** Visual Geometry Group (VGG), University of Oxford (Arsha Nagrani, Joon Son Chung, Andrew Zisserman)
* **Access Type:** Academic Registration Form (Instant to 24-hr turnaround)
* **Direct Access URL:** [Oxford VGG VoxCeleb Website](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
* **Application Procedure:**
  1. Register at the Oxford VGG VoxCeleb portal using university credentials.
  2. Receive the temporary HTTP username and password.
  3. Download the audio-video dev parts using `wget` or `curl`:
     ```bash
     curl -u username:password -O https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_mp4.zip
     ```
* **Storage Footprint:** ~150 GB for dev split.

---

## 3. Immediate Action Plan for M3 (Days 1–3)

| Task ID | Action Item | Assignee | Deliverable | Status Target |
|:---|:---|:---|:---|:---|
| **M3-A1** | Fill & submit FaceForensics++ Google Form | Aswin | Screenshot of confirmation | Day 1 |
| **M3-A2** | Fill Celeb-DF v2 Agreement with Prof. Leo Francis | Aswin & Prof. Leo | Signed PDF emailed | Day 1 |
| **M3-A3** | Accept Kaggle DFDC rules & pull Preview dataset | Aswin | DFDC preview extracted | Day 2 |
| **M3-A4** | Pull UADFV & FFHQ thumbnails | Aswin | Raw files in `data/raw/` | Day 2 |
| **M3-A5** | Register on Oxford VGG for VoxCeleb2 | Aswin | Credentials received | Day 2 |
| **M3-A6** | Run Synthetic Mock Generator for pipeline tests | Aswin | Pipeline verified end-to-end | Day 3 |

---

## 4. Formal Email / Application Statement Template
When applying to TUM (FF++) and SUNY (Celeb-DF), use this standardized statement:

> **Project Name:** ASTCV: Adaptive Screen-Timestamped Computer Vision  
> **Institution:** Department of Computer Science & Engineering  
> **Faculty Guide:** Leo Francis P  
> **Student Researcher (Data Lead):** Aswin K N  
> **Purpose of Request:**  
> "We are conducting academic research on hardware-agnostic display-camera calibration, high-resolution hardware timestamp synchronization, and active challenge-response reflectance models to differentiate authentic biological human interaction from virtual camera video injection and deepfake spoofing. The requested dataset will be used exclusively by our academic research team for non-commercial benchmarking of temporal consistency and detection baseline evaluation under strict adherence to your terms of use. The data will be stored on a secure, restricted-access workstation."
