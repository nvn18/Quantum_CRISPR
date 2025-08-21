# Quantum-Enhanced CRISPR & Targeted Nanobot Delivery 🚀

[![Research](https://img.shields.io/badge/Type-Research-blue)](#)
[![Domain](https://img.shields.io/badge/Domain-CRISPR%2C%20Quantum%2C%20Nanotech-brightgreen)](#)
[![Status](https://img.shields.io/badge/Status-Unpublished-orange)](#)
[![Notebook](https://img.shields.io/badge/Code-Jupyter%20Notebooks-informational)](#)

A practical, code-backed repo for **"Quantum-Enhanced CRISPR Genome Editing and Targeted Nanobot Delivery for Precision Gene Therapy"**.
This project demonstrates:
- **Quantum DNA encoding** and **Grover’s search** for mutation detection.
- **ML models (SVM, LSTM)** for off-target prediction.
- A **conceptual nanobot delivery simulation**.
- Clean visuals + a recruiter-friendly structure.

> Full paper: [`paper/Quantum_Enhanced_CRISPR.pdf`](paper/Quantum_Enhanced_CRISPR.pdf)

---

## 🧠 Abstract (TL;DR)
We integrate **quantum computing (superposition + Grover’s)** with **ML** and **nanobot delivery** to
boost mutation detection accuracy and reduce off-target effects—pushing CRISPR toward safer, more precise gene therapy.

Read the 1-page version: [`abstract/Abstract.md`](abstract/Abstract.md)

---

---
## 🏗️ Architecture (High Level)

flowchart LR
    A["QUNACR9 / Public DNA Data"] --> B["Preprocess & Encode DNA"]
    B --> C["Quantum Superposition (H, CNOT)"]
    C --> D["Grover's Search for Mutations"]
    D --> E["ML Off-target (SVM/LSTM)"]
    E --> F["Nanobot Delivery Simulation"]
    F --> G["Results & Reports"]

## 📦 Repository Layout
```
Quantum-CRISPR-Research/
│── README.md
│── paper/Quantum_Enhanced_CRISPR.pdf
│── abstract/Abstract.md
│── datasets/sample_dna_sequences.csv
│── code/
│    ├── 1_quantum_dna_encoding.ipynb
│    ├── 2_grover_mutation_detection.ipynb
│    ├── 3_ml_offtarget_rf_svm_lstm.ipynb
│    └── 4_nanobot_delivery_simulation.ipynb
│── images/ (place any figures here)
```
---

---

## ⚙️ Quickstart (Local)
1. **Create env**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install --upgrade pip
   ```
2. **Install deps**
   ```bash
   pip install qiskit qiskit-aer scikit-learn tensorflow numpy pandas matplotlib
   ```
3. **Run notebooks**
   ```bash
   pip install jupyter
   jupyter notebook code/
   ```

> If TensorFlow is heavy on your machine, run only SVM first. Qiskit code uses local simulators.
> GPU (optional) speeds up the LSTM.

---

## 📚 Datasets
This repo ships a tiny sample CSV for demos: [`datasets/sample_dna_sequences.csv`](datasets/sample_dna_sequences.csv).  
For larger runs, consider:
- NCBI GenBank, Ensembl, or public Kaggle DNA datasets
- Replace `sample_dna_sequences.csv` with your data keeping **the same columns**

**Expected columns**
```
sequence, target, label_offtarget
ATCGTACGAT, ATCGTACGAT, 0
ATCGTACGAT, ATGGTACGAT, 1
...
```

---

## 🧪 Notebooks Overview
- **1_quantum_dna_encoding.ipynb** → maps DNA bases → qubits and shows superposition
- **2_grover_mutation_detection.ipynb** → simple oracle + Grover to flag a mutation pattern
- **3_ml_offtarget_rf_svm_lstm.ipynb** → SVM baseline + LSTM sequence model
- **4_nanobot_delivery_simulation.ipynb** → conceptual 2D navigation + payload trigger

---

## 📈 Results (demo)
- Encodings verified on simulator.
- Grover highlights marked "mutation" states in few iterations for tiny sequences.
- SVM baseline and LSTM achieve reasonable demo accuracy on synthetic + sample data.
.

---

## 📄 Citation
If you reference this repo:
```
V. N. V. N. Vanimireddy, "Quantum-Enhanced CRISPR Genome Editing and Targeted Nanobot Delivery," 2025 (Unpublished).
```

---

## 🧰 CLI Usage (MGIE-style)
```bash
python code/run_quantum.py --sequence ATCG
python code/run_quantum.py --grover --n 3
python code/train_ml.py --model svm --dataset datasets/sample_dna_sequences.csv
python code/train_ml.py --model lstm --epochs 5
python code/simulate_nanobot.py --grid_size 50
```

---

## 📊 Datasets

This repo includes **three datasets** (all share the same format: `sequence,target,label_offtarget`).

- `sample_dna_sequences.csv` → 200 rows (quick demo)
- `dna_sequences_medium.csv` → 5,000 rows (training/medium scale)
- `dna_sequences_large.csv` → 50,000 rows (large-scale experiment)

**Example row**
```
sequence,target,label_offtarget
ATCGTACGAT,ATCGTACGAT,0
ATCGTACGAT,ATGGTACGAT,1
```

---

## ✅ Example Outputs (when you run the repo)

### Quantum Encoding (Qiskit)
```
Qubits: 4
Statevector (first 8 amps): [1.+0.j 0.+0.j 0.+0.j ...]
Circuit diagram saved → images/quantum_encoding_circuit.png
```

### Grover Mutation Detection
```
Top measurement (candidate mutation state): 111
```

### SVM Model
```
SVM accuracy: ~0.85
Confusion matrix saved → outputs/svm_confusion_matrix.png
Metrics JSON → outputs/svm_metrics.json
```

### LSTM Model
```
Epoch 1/5 ... Epoch 5/5
LSTM test accuracy: ~0.90
Training curve saved → outputs/lstm_training_curve.png
```

### Nanobot Simulation
- Navigation path plotted → images/nanobot_path.png
- Trajectory CSV saved → outputs/nanobot_traj.csv
```



---

## 📦 Installation (Full)
```bash
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 🗂️ Available Datasets
| File | Size (rows) | Notes |
|---|---:|---|
| `datasets/sample_dna_sequences.csv` | 200 | Quick demo |
| `datasets/dna_sequences_medium.csv` | 5,000 | Medium-scale training |
| `datasets/dna_sequences_large.csv` | 50,000 | Larger-scale training |

> All datasets share the same schema: `sequence,target,label_offtarget`

