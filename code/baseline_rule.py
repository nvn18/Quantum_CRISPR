#!/usr/bin/env python
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def hamming(a, b):
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

def run(csv_path, outdir, threshold=0):
    df = pd.read_csv(csv_path)
    y_true = df['label_offtarget'].values.astype(int)
    y_pred = np.array([1 if hamming(a,b) > threshold else 0 for a,b in zip(df['sequence'], df['target'])], dtype=int)

    acc = (y_true == y_pred).mean()
    # Confusion matrix
    cm = np.zeros((2,2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    Path(outdir).mkdir(parents=True, exist_ok=True)
    # Save confusion matrix plot
    plt.figure()
    plt.imshow(cm)
    plt.title("Rule-based Baseline Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.savefig(Path(outdir)/"baseline_confusion_matrix.png", dpi=200, bbox_inches="tight")

    with open(Path(outdir)/"baseline_metrics.json", "w") as f:
        json.dump({"accuracy": float(acc), "threshold": threshold, "counts": cm.tolist()}, f, indent=2)

    print("Baseline accuracy:", acc)
    print("Saved metrics and plot to", outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rule-based baseline using Hamming distance")
    p.add_argument("--dataset", type=str, default="datasets/sample_dna_sequences.csv")
    p.add_argument("--outdir", type=str, default="outputs/baseline")
    p.add_argument("--threshold", type=int, default=0, help="Predict off-target if Hamming > threshold")
    args = p.parse_args()
    run(args.dataset, args.outdir, threshold=args.threshold)
