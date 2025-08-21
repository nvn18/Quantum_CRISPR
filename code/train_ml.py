#!/usr/bin/env python
import argparse, sys, json
from pathlib import Path
import numpy as np, pandas as pd

def one_hot_seq(seq, max_len=30):
    vocab = {'A':0,'T':1,'C':2,'G':3}
    x = np.zeros((max_len, 4), dtype=np.float32)
    for i, ch in enumerate(seq[:max_len]):
        if ch in vocab:
            x[i, vocab[ch]] = 1.0
    return x

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = np.stack([one_hot_seq(s) for s in df['sequence'].values])  # (N,L,4)
    y = df['label_offtarget'].values.astype(int)
    return X, y

def run_svm(X, y, outdir):
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt

    Xf = X.reshape(len(X), -1)
    X_tr, X_te, y_tr, y_te = train_test_split(Xf, y, test_size=0.2, random_state=42, stratify=y)
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    report = classification_report(y_te, pred, output_dict=True)
    cm = confusion_matrix(y_te, pred)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    # Save confusion matrix plot
    plt.figure()
    plt.imshow(cm, cmap=None)
    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.savefig(Path(outdir)/"svm_confusion_matrix.png", dpi=200, bbox_inches="tight")

    with open(Path(outdir)/"svm_metrics.json", "w") as f:
        json.dump({"acc": acc, "report": report}, f, indent=2)
    print("SVM accuracy:", acc)
    print("Saved metrics and plot to", outdir)

def run_lstm(X, y, outdir, epochs=5):
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from tensorflow.keras import layers, models
    import matplotlib.pyplot as plt

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = models.Sequential([
        layers.Input(shape=(X.shape[1], X.shape[2])),
        layers.LSTM(64),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    hist = model.fit(X_tr, y_tr, validation_split=0.2, epochs=epochs, batch_size=32, verbose=1)
    loss, acc = model.evaluate(X_te, y_te, verbose=0)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    # Save training curve
    plt.figure()
    plt.plot(hist.history['accuracy'], label='train_acc')
    plt.plot(hist.history['val_accuracy'], label='val_acc')
    plt.title("LSTM Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.savefig(Path(outdir)/"lstm_training_curve.png", dpi=200, bbox_inches="tight")

    model.save(Path(outdir)/"lstm_model.h5")
    with open(Path(outdir)/"lstm_metrics.json","w") as f:
        json.dump({"test_accuracy": float(acc)}, f, indent=2)
    print("LSTM test accuracy:", acc)
    print("Saved model, metrics, and plot to", outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train off-target predictors (SVM/LSTM)")
    p.add_argument("--dataset", type=str, default="datasets/sample_dna_sequences.csv")
    p.add_argument("--model", type=str, choices=["svm","lstm"], default="svm")
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args()

    X, y = load_data(args.dataset)
    if args.model == "svm":
        run_svm(X, y, args.outdir)
    else:
        run_lstm(X, y, args.outdir, epochs=args.epochs)
