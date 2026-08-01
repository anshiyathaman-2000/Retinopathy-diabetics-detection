import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score

from dataset import RFMiDDataset, DiabeticRetinopathyDataset
from model import get_model


def calculate_detailed_metrics(y_true, y_pred, y_prob):
    """
    Computes Accuracy, Sensitivity, Specificity, and F1-score for each class.
    Also handles AUC-ROC calculation for classes with both positive and negative cases.
    """
    num_classes = y_true.shape[1]
    
    accuracies = []
    sensitivities = []
    specificities = []
    f1_scores = []
    aucs = []
    valid_auc_classes = 0
    
    for i in range(num_classes):
        true_c = y_true[:, i]
        pred_c = y_pred[:, i]
        prob_c = y_prob[:, i]
        
        # True Positives, True Negatives, False Positives, False Negatives
        tp = np.sum((true_c == 1) & (pred_c == 1))
        tn = np.sum((true_c == 0) & (pred_c == 0))
        fp = np.sum((true_c == 0) & (pred_c == 1))
        fn = np.sum((true_c == 1) & (pred_c == 0))
        
        # Accuracy
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        accuracies.append(acc)
        
        # Sensitivity (Recall / True Positive Rate)
        sens = tp / (tp + fn + 1e-8)
        sensitivities.append(sens)
        
        # Specificity (True Negative Rate)
        spec = tn / (tn + fp + 1e-8)
        specificities.append(spec)
        
        # F1-score
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        f1_scores.append(f1)
        
        # AUC-ROC (only defined if there are both positive and negative samples in true labels)
        if len(np.unique(true_c)) == 2:
            auc = roc_auc_score(true_c, prob_c)
            aucs.append(auc)
            valid_auc_classes += 1
        else:
            aucs.append(0.5)  # Default fallback representation
            
    # Macro averages
    macro_acc = np.mean(accuracies)
    macro_sens = np.mean(sensitivities)
    macro_spec = np.mean(specificities)
    macro_f1 = np.mean(f1_scores)
    
    # Average AUC over classes where it was valid to calculate
    macro_auc = np.mean([aucs[i] for i in range(num_classes) if len(np.unique(y_true[:, i])) == 2])
    
    return {
        "accuracies": accuracies,
        "sensitivities": sensitivities,
        "specificities": specificities,
        "f1_scores": f1_scores,
        "aucs": aucs,
        "macro_acc": macro_acc,
        "macro_sens": macro_sens,
        "macro_spec": macro_spec,
        "macro_f1": macro_f1,
        "macro_auc": macro_auc
    }

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    all_targets = []
    all_logits = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            loss = criterion(logits, targets)
            
            running_loss += loss.item() * inputs.size(0)
            
            all_targets.append(targets.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    
    all_targets = np.concatenate(all_targets, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    # Probabilities via Sigmoid
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))
    # Binary predictions (threshold = 0.5)
    all_preds = (all_probs >= 0.5).astype(np.float32)
    
    # Calculate macro/micro F1
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_targets, all_preds, average='micro', zero_division=0)
    
    # Calculate Hamming loss
    hl = hamming_loss(all_targets, all_preds)
    
    # Detailed metrics
    metrics = calculate_detailed_metrics(all_targets, all_preds, all_probs)
    metrics["loss"] = epoch_loss
    metrics["f1_macro"] = f1_macro
    metrics["f1_micro"] = f1_micro
    metrics["hamming_loss"] = hl
    
    return metrics, all_targets, all_preds

def main(model_name="hybrid", epochs=15, batch_size=8, lr=1e-4, lr_backbone=1e-5, dataset_name="archive2"):
    static_dir = "/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/paper/static"
    os.makedirs(static_dir, exist_ok=True)
    
    # Detect device: MPS for Apple Silicon, CUDA for NVIDIA GPU, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU acceleration (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    if dataset_name == "archive2":
        base_dir = "/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/paper/archive 2/retino"
        print(f"Loading archive 2 Diabetic Retinopathy dataset from {base_dir}")
        
        train_dataset = DiabeticRetinopathyDataset(base_dir, split="train", target_size=(256, 256), is_train=True)
        val_dataset = DiabeticRetinopathyDataset(base_dir, split="valid", target_size=(256, 256), is_train=False)
        test_dataset = DiabeticRetinopathyDataset(base_dir, split="test", target_size=(256, 256), is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        label_names = train_dataset.label_cols
        num_classes = train_dataset.num_classes
        num_eval_samples = len(val_dataset)
        
        # Compute class weights
        train_labels = np.array(train_dataset.labels, dtype=np.float32)
        pos_counts = np.sum(train_labels, axis=0)
        neg_counts = len(train_labels) - pos_counts
    else:
        csv_path = "/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/paper/Evaluation_Set/RFMiD_Validation_Labels.csv"
        img_dir = "/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/paper/Evaluation_Set/Validation"
        
        # Read labels to perform train-test split
        df_labels = pd.read_csv(csv_path)
        num_samples = len(df_labels)
        indices = list(range(num_samples))
        
        # 80% train, 20% validation split
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
        print(f"Dataset split: {len(train_indices)} training samples, {len(val_indices)} validation samples")
        
        # Create datasets & loaders
        train_dataset = RFMiDDataset(csv_path, img_dir, indices=train_indices, target_size=(256, 256), is_train=True)
        val_dataset = RFMiDDataset(csv_path, img_dir, indices=val_indices, target_size=(256, 256), is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        label_names = train_dataset.label_cols
        num_classes = train_dataset.num_classes
        num_eval_samples = len(val_indices)
        
        # Compute class pos_weight to handle imbalance
        train_labels = df_labels.iloc[train_indices][label_names].values.astype(np.float32)
        pos_counts = np.sum(train_labels, axis=0)
        neg_counts = len(train_labels) - pos_counts
    
    # Initialize selected model
    model = get_model(model_name, num_classes=num_classes)
    model = model.to(device)
    
    pos_weight = []
    for pos, neg in zip(pos_counts, neg_counts):
        if pos > 0:
            weight = neg / pos
        else:
            weight = 1.0
        # Clip weights to a reasonable range (1.0 to 25.0) to avoid extreme gradients
        weight = min(max(weight, 1.0), 25.0)
        pos_weight.append(weight)
        
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    
    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    
    # Discriminative learning rates: backbone uses lower lr, head uses higher lr
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = any(term in name for term in ["cnn_backbone", "backbone", "resnet_backbone", "inception_backbone", "patch_proj"])
        if is_backbone:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr}
    ], weight_decay=1e-4)
    
    # LR Scheduler (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1_macro": [],
        "val_f1_micro": [],
        "val_hamming_loss": []
    }
    
    best_f1 = 0.0
    best_epoch = 0
    start_time = time.time()
    
    print("\nStarting Training Loop...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        
        # Evaluate
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["val_f1_micro"].append(val_metrics["f1_micro"])
        history["val_hamming_loss"].append(val_metrics["hamming_loss"])
        
        epoch_duration = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val F1-Macro: {val_metrics['f1_macro']:.4f} | "
              f"Val F1-Micro: {val_metrics['f1_micro']:.4f} | "
              f"Hamming Loss: {val_metrics['hamming_loss']:.4f} | "
              f"Time: {epoch_duration:.1f}s")
        
        # Save best model
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            checkpoint_name = f"best_model_{model_name}.pth"
            torch.save(model.state_dict(), os.path.join(static_dir, checkpoint_name))
            if model_name == "hybrid":
                torch.save(model.state_dict(), os.path.join(static_dir, "best_model.pth"))
            
    total_time = time.time() - start_time
    print(f"\nTraining Complete! Best Val F1-Macro: {best_f1:.4f} at Epoch {best_epoch}")
    print(f"Total time elapsed: {total_time/60:.2f} minutes")
    
    # Load the best model for final evaluation
    checkpoint_name = f"best_model_{model_name}.pth"
    checkpoint_path = os.path.join(static_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path) and model_name == "hybrid":
        checkpoint_path = os.path.join(static_dir, "best_model.pth")
        
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded best model checkpoint ({os.path.basename(checkpoint_path)}) for final evaluation.")
        
    final_metrics, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    
    # Plot and save loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss", marker='o')
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(static_dir, f"loss_plot_{model_name}.png"), bbox_inches='tight')
    if model_name == "hybrid":
        plt.savefig(os.path.join(static_dir, "loss_plot.png"), bbox_inches='tight')
    plt.close()
    print(f"Saved loss curve plot to static/loss_plot_{model_name}.png")
    
    # Generate final evaluation report and write to file
    import shutil
    report_path = os.path.join(static_dir, f"evaluation_results_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write("========================================================\n")
        f.write(f"{model_name.upper()} MODEL FINAL EVALUATION REPORT\n")
        f.write("========================================================\n\n")
        f.write(f"Total Samples evaluated: {num_eval_samples}\n")
        f.write(f"Hamming Loss: {final_metrics['hamming_loss']:.5f}\n")
        f.write(f"Global F1-Macro: {final_metrics['f1_macro']:.5f}\n")
        f.write(f"Global F1-Micro: {final_metrics['f1_micro']:.5f}\n")
        f.write(f"Mean Accuracy: {final_metrics['macro_acc']:.5f}\n")
        f.write(f"Mean Sensitivity: {final_metrics['macro_sens']:.5f}\n")
        f.write(f"Mean Specificity: {final_metrics['macro_spec']:.5f}\n")
        f.write(f"Mean AUC-ROC: {final_metrics['macro_auc']:.5f}\n\n")
        
        f.write("PER-CLASS EVALUATION DETAILS:\n")
        f.write("---------------------------------------------------------------------------------\n")
        f.write(f"{'Label Name':<15} | {'Accuracy':<10} | {'Sensitivity':<12} | {'Specificity':<12} | {'F1-Score':<10} | {'AUC-ROC':<10}\n")
        f.write("---------------------------------------------------------------------------------\n")
        for i in range(num_classes):
            name = label_names[i]
            acc = final_metrics["accuracies"][i]
            sens = final_metrics["sensitivities"][i]
            spec = final_metrics["specificities"][i]
            f1 = final_metrics["f1_scores"][i]
            auc = final_metrics["aucs"][i]
            
            # Print if valid (i.e. has support in validation set)
            f.write(f"{name:<15} | {acc:<10.4f} | {sens:<12.4f} | {spec:<12.4f} | {f1:<10.4f} | {auc:<10.4f}\n")
        f.write("---------------------------------------------------------------------------------\n")
        
    print(f"Saved evaluation results report to: {report_path}")
    if model_name == "hybrid":
        shutil.copy(report_path, os.path.join(static_dir, "evaluation_results.txt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-Transformer-LSTM Retinal Classification Model")
    parser.add_argument("--model", type=str, default="hybrid", choices=["hybrid", "resnet50", "vit", "cnn_transformer", "cnn_lstm", "cnn_inception"], help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for newly initialized layers")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Learning rate for fine-tuning the backbone")
    parser.add_argument("--dataset", type=str, default="archive2", choices=["archive2", "rfmid"], help="Dataset to train on")
    args = parser.parse_args()
    
    main(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, lr_backbone=args.lr_backbone, dataset_name=args.dataset)
