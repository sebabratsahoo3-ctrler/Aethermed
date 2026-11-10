# 🏥 AetherMed 4.0 — Clinical Digital Twin & Precision TDM Platform

![CI Test Suite](https://github.com/Sebabrat-Sahoo-2005/aethermed/actions/workflows/ci.yml/badge.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.150-009688.svg?logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14.2-black.svg?logo=next.js)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C.svg?logo=pytorch)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**AetherMed 4.0** is an advanced Clinical Decision Support and In-Silico Clinical Trial Optimization platform designed for Narrow Therapeutic Index (NTI) drugs, including **Warfarin, Vancomycin, and Digoxin**.

The platform combines pharmacokinetic modeling, artificial intelligence, precision therapeutic drug monitoring (TDM), organ toxicity prediction, reinforcement learning, and literature-grounded clinical reporting into a unified digital twin environment.

Its architecture integrates a high-performance **FastAPI backend** with a modern **Next.js 14 frontend**, delivering an interactive clinical cockpit for patient simulation, personalized drug analysis, dosing policy evaluation, and virtual clinical trials.

> **Disclaimer:** AetherMed 4.0 is a research and educational prototype. It is not a certified medical device and must not be used to make real-world clinical or medication decisions.

---

## 🏛️ System Architecture

AetherMed 4.0 follows a modular architecture consisting of a unified frontend, backend API, and four core computational engines.

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                              AETHERMED 4.0                                  │
│                                                                              │
│                    CLINICAL DIGITAL TWIN COCKPIT                            │
│                         Next.js 14 + Tailwind                              │
│                                                                              │
│   Live Telemetry | Patient Profiles | PK Curves | Organ Risk | OOD Safety  │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                              REST API / CORS
                              localhost:8000
                                   │
┌──────────────────────────────────▼───────────────────────────────────────────┐
│                            FASTAPI BACKEND                                  │
│                                                                              │
│   /api/health | /api/cohort | /api/simulate | /api/trial/simulate           │
└──────────────┬────────────────┬────────────────┬─────────────────────────────┘
               │                │                │
       ┌───────▼──────┐ ┌───────▼──────┐ ┌───────▼──────┐ ┌─────────────────┐
       │ Physics-ODE  │ │ Heterogeneous│ │ Offline RL   │ │ Clinical RAG   │
       │              │ │ GNN          │ │              │ │                 │
       │ Neural PK    │ │ Organ Stress │ │ Conservative │ │ ChromaDB        │
       │ Multi-Dose   │ │ Propagation  │ │ Q-Learning   │ │ Literature      │
       │ Bayesian TDM │ │ CQR          │ │ OPE          │ │ Citations       │
       └──────────────┘ └──────────────┘ └──────────────┘ └─────────────────┘
```

### Core Computational Modules

| Module | Technology | Primary Function |
|---|---|---|
| Pharmacokinetic Engine | Physics-Informed Neural ODE | Drug concentration and exposure simulation |
| Organ Toxicity Engine | Heterogeneous GNN | Multi-organ toxicity risk prediction |
| Dosing Optimization | Conservative Q-Learning | Offline reinforcement learning for dosing research |
| Uncertainty Engine | Conformal Prediction + Mahalanobis Distance | Prediction uncertainty and OOD detection |
| Clinical Knowledge Engine | ChromaDB + RAG | Literature-grounded clinical reporting |
| Frontend | Next.js 14 | Interactive digital twin visualization |

---

## ✨ Key Capabilities

### 1. Physiologically Consistent Virtual Patient Cohort

AetherMed generates a synthetic cohort of 1,000 virtual patients using clinically relevant demographic, biochemical, and pharmacogenomic variables.

Key features:

- Deterministic 2021 CKD-EPI creatinine equation for eGFR estimation.
- Physiological coupling between serum creatinine, age, and sex.
- Biological variability modeled using Gaussian noise: N(0, 3.5).
- eGFR values bounded between 5 and 140 mL/min/1.73 m².
- Pharmacogenomic variables including `CYP2C9` and `VKORC1`.
- Additional biochemical parameters including serum potassium and albumin.
- Non-degenerate patient cohort stored at `data/processed/cohort.parquet`.

### 2. Literature-Grounded Pharmacokinetic Drug Library

The pharmacokinetic engine supports three narrow therapeutic index drugs:

| Drug | Primary Clinical Monitoring Parameter | Reference Target |
|---|---|---|
| Warfarin | INR | 2.0–3.0 |
| Vancomycin | AUC24 | 400–600 mg·h/L |
| Digoxin | Serum concentration | 0.5–2.0 ng/mL |

Features include:

- Drug-specific pharmacokinetic profiles.
- Multi-compartment pharmacokinetic modeling.
- Repeated-dose superposition simulation.
- Multi-dose regimens including q8h, q12h, and q24h.
- Simulation horizons ranging from 24 to 168 hours.
- Mass conservation validation with ΔM < 10⁻⁵.
- Bayesian Maximum A Posteriori (MAP) estimation for individual pharmacokinetic parameter personalization.

The Bayesian TDM module estimates individual clearance (ηCL) and volume of distribution (ηV1) using observed blood concentration measurements.

### 3. Drug-Aware Conservative Q-Learning

The reinforcement learning module implements Conservative Q-Learning (CQL) for offline dosing policy research.

Capabilities:

- Continuous normalized action-space mapping.
- Drug-specific therapeutic dosing boundaries.
- Organ toxicity-aware action dampening.
- Critical toxicity guard when organ risk exceeds 0.80.
- Offline policy evaluation against physician-standard dosing baselines.

The evaluation framework incorporates:

- Doubly Robust (DR) estimation.
- Weighted Importance Sampling (WIS).
- Fitted Q Evaluation (FQE).
- Effective Sample Size (ESS).

### 4. Adaptive Conformal Uncertainty and OOD Detection

AetherMed integrates uncertainty quantification and out-of-distribution detection to improve the transparency of model predictions.

**Conformal Prediction (CQR):**

- Covariate-adaptive prediction intervals.
- 90% conformal uncertainty intervals.
- Interval adaptation based on organ stress and patient characteristics.

**Out-of-Distribution Detection:**

- Covariance-aware Mahalanobis distance.
- Per-feature z-score attribution.
- Detection of unusual patient profiles.
- FDA-informed single-dose and daily-dose safety guards.

### 5. Literature-Grounded Clinical RAG

The clinical knowledge engine retrieves relevant information from:

- FDA prescribing information.
- DailyMed drug documentation.
- PubMed scientific literature.

Technology stack:

- ChromaDB vector database.
- Embedding-based semantic retrieval.
- Grounded clinical response generation.
- Five-section clinical consultation reports.
- Automatic lexical claim verification.
- Interactive citation cards.

The system targets greater than 95% lexical claim coverage during its verification process.

### 6. Interactive Clinical Digital Twin Cockpit

The Next.js 14 frontend provides a unified interface for exploring patient digital twins and simulation results.

Major interface components:

- High-density pharmacokinetic concentration-time curves.
- Therapeutic target-window visualization.
- Dynamic six-node, sixteen-edge heterogeneous graph visualization.
- Organ-specific toxicity monitoring.
- Patient-level uncertainty and OOD indicators.
- Live backend telemetry.
- PyTorch device information.
- ChromaDB retrieval statistics.
- In-Silico Population Trial simulator.

The population trial simulator enables comparative evaluation of:

1. Physician-standard dosing.
2. CQL-guided dosing.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Frontend | Next.js 14, React, TypeScript, Tailwind CSS |
| Backend | Python, FastAPI, Uvicorn |
| Deep Learning | PyTorch |
| Pharmacokinetics | Physics-Informed Neural ODE |
| Graph Learning | Heterogeneous Graph Neural Networks |
| Reinforcement Learning | Conservative Q-Learning |
| Uncertainty Quantification | Conformal Prediction, CQR |
| OOD Detection | Mahalanobis Distance |
| Retrieval-Augmented Generation | ChromaDB, Sentence Embeddings |
| Visualization | Recharts |
| Data Processing | NumPy, Pandas, PyArrow |
| Testing | Pytest |
| Deployment | Docker, Docker Compose |

---

## 🚀 Installation and Quickstart

### Prerequisites

Ensure the following software is installed:

- Python 3.10 or later.
- Node.js 18 or later.
- npm.
- Git.
- Docker and Docker Compose (optional).

### 1. Clone the Repository

```bash
git clone https://github.com/Sebabrat-Sahoo-2005/aethermed.git

cd aethermed
```

### 2. Start Using the Unified Launcher (Recommended)

```bash
chmod +x run_platform.sh

./run_platform.sh
```

The unified launcher starts the backend and frontend services.

### 3. Manual Backend Setup

Create and activate a Python virtual environment.

**Linux / macOS:**

```bash
python3 -m venv venv

source venv/bin/activate
```

**Windows:**

```powershell
python -m venv venv

venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI backend:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 4. Manual Frontend Setup

Open another terminal:

```bash
cd frontend

npm install

npm run dev
```

The Next.js development server will start on port 3000.

### 5. Docker Deployment

To build and launch the application using Docker Compose:

```bash
docker compose up --build
```

---

## 🌐 Application Endpoints

Once the application is running, access the following endpoints:

| Service | URL |
|---|---|
| Clinical Digital Twin Cockpit | http://localhost:3000 |
| FastAPI Swagger Documentation | http://localhost:8000/docs |
| FastAPI ReDoc Documentation | http://localhost:8000/redoc |
| Backend Health & Telemetry | http://localhost:8000/api/health |
| Synthetic Patient Cohort | http://localhost:8000/api/cohort |

### Primary API Routes

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Backend health and system telemetry |
| GET | `/api/cohort` | Retrieve synthetic patient cohort |
| POST | `/api/simulate` | Execute patient-level simulation |
| POST | `/api/trial/simulate` | Execute in-silico population trial |

**Legacy Dashboard (Fallback):**

```bash
streamlit run src/dashboard/app.py
```

---

## 🧪 Testing

AetherMed includes an automated testing suite covering its core computational modules and backend functionality.

Run the complete test suite:

```bash
pytest tests/ -v
```

The project currently reports **88 passing tests** across unit and integration test modules.

Test coverage includes:

| Test Module | Testing Focus |
|---|---|
| `test_schema.py` | Data validation and schema integrity |
| `test_rag.py` | Retrieval and clinical knowledge grounding |
| `test_rl_policy.py` | Reinforcement learning policy validation |
| `test_aethermed4.py` | AetherMed 4.0 integration testing |

---

## 📁 Repository Structure

```text
aethermed/
│
├── backend/
│   └── main.py
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── public/
│   └── package.json
│
├── data/
│   └── processed/
│       └── cohort.parquet
│
├── src/
│   └── dashboard/
│       └── app.py
│
├── tests/
│   ├── test_schema.py
│   ├── test_rag.py
│   ├── test_rl_policy.py
│   └── test_aethermed4.py
│
├── requirements.txt
├── run_platform.sh
├── docker-compose.yml
└── README.md
```

*Note: This is a representative project structure. Actual directories and filenames may vary with the repository implementation.*

---

## 🔬 Research and Clinical Scope

AetherMed 4.0 explores the integration of computational pharmacology, artificial intelligence, and clinical decision support through a unified digital twin architecture.

Its primary research objectives include:

- Individualized pharmacokinetic parameter estimation.
- Precision therapeutic drug monitoring.
- Multi-organ toxicity prediction.
- Offline reinforcement learning for dosing policy research.
- Uncertainty-aware clinical prediction.
- Out-of-distribution patient detection.
- Literature-grounded clinical knowledge retrieval.
- Comparative in-silico clinical trial evaluation.

The platform currently focuses on Warfarin, Vancomycin, and Digoxin as representative narrow therapeutic index medications.

---

## ⚠️ Disclaimer

AetherMed 4.0 is an experimental research platform intended for academic research, algorithm development, and in-silico simulation.

- Synthetic patient data does not represent actual clinical patients.
- Model predictions are not validated prescriptions.
- Reinforcement learning outputs must not be interpreted as clinical dosing recommendations.
- Literature retrieval and automated claim verification do not guarantee clinical correctness.
- The platform is not intended for diagnosis, treatment, or independent clinical decision-making.

All clinical applications require appropriate validation, regulatory review, and qualified medical supervision.

---

## 📄 License

This project is distributed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Project Repository

**AetherMed 4.0 — Clinical Digital Twin & Precision TDM Platform**

[GitHub Repository](https://github.com/Sebabrat-Sahoo-2005/aethermed)

