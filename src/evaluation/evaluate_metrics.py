import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models import build_model
from src.datasets import build_dataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = "./data/hmdb51"
    annotation_dir = "./data/hmdb51_splits"
    checkpoint_path = "checkpoints/distilled_AT_T10_seed1234/best_model.pth"
    output_dir = "./docs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Dataset
    print("Loading HMDB-51 split 1 validation dataset...")
    _, val_ds, _ = build_dataset(
        dataset_type="video",
        data_dir=data_dir,
        annotation_dir=annotation_dir,
        split=1,
        num_frames=16,
        frame_size=112
    )
    
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    
    # Extract class names (sorted alphabetically to match HMDB51 label construction)
    class_names = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    
    # 2. Load Model
    print("Loading best Student model...")
    model = build_model("student", num_classes=51, width_mult=1.0)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    # 3. Evaluation Loop
    y_true = []
    y_pred = []
    y_prob = []
    
    print(f"Starting evaluation over {len(val_ds)} validation samples...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # 4. Compute Metrics
    print("Computing metrics...")
    
    # Top-1 and Top-5 Accuracy
    top1_correct = (y_pred == y_true).sum()
    
    top5_preds = np.argsort(y_prob, axis=1)[:, -5:]
    top5_correct = sum(y_true[i] in top5_preds[i] for i in range(len(y_true)))
    
    top1_acc = top1_correct / len(y_true)
    top5_acc = top5_correct / len(y_true)
    
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    
    # Classification Report (Precision, Recall, F1)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = os.path.join(output_dir, "classification_report.csv")
    report_df.to_csv(report_csv_path)
    print(f"Classification report saved to {report_csv_path}")
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix - Distilled Student (KD + AT T=10)',
           ylabel='True Class',
           xlabel='Predicted Class')
           
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=8)
                        
    fig.tight_layout()
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_plot_path, dpi=150)
    print(f"Confusion matrix plot saved to {cm_plot_path}")
    
    # Save a summary file for REPORT.md
    summary_path = os.path.join(output_dir, "aggregate_metrics.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Top-1 Accuracy: {top1_acc*100:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5_acc*100:.2f}%\n")
        f.write(f"Macro Precision: {report_dict['macro avg']['precision']*100:.2f}%\n")
        f.write(f"Macro Recall: {report_dict['macro avg']['recall']*100:.2f}%\n")
        f.write(f"Macro F1 Score: {report_dict['macro avg']['f1-score']*100:.2f}%\n")

if __name__ == "__main__":
    main()
