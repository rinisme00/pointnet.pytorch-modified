#!/usr/bin/env python3
import argparse
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")      # use a non‑GUI backend
import matplotlib.pyplot as plt

# Add the project root to ``sys.path`` so that our local modules can be
# imported when this script is executed directly from within the ``utils``
# directory.
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pointnet.model import PointNetDenseCls, feature_transform_regularizer
from utils.custom_dataset import CustomDataset


def smooth_labels(target: torch.Tensor, num_classes: int, smoothing: float = 0.1) -> torch.Tensor:
    assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0, 1)."
    with torch.no_grad():
        true_dist = torch.zeros(target.size(0), num_classes, device=target.device)
        if num_classes > 1:
            true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    return true_dist


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train PointNet segmentation model with val/test splits")
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='seg', help='output folder for checkpoints')
    parser.add_argument('--model', type=str, default='', help='path to a pretrained model to initialise from')
    parser.add_argument('--dataset', type=str, required=True, help='dataset root directory containing train/val/test')
    parser.add_argument('--feature_transform', action='store_true', help='use feature transform module')
    parser.add_argument('--augment', action='store_true', help='apply data augmentation during training')
    parser.add_argument('--num_classes', type=int, default=None, help='number of segmentation classes (auto‑inferred if omitted)')
    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay factor')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing factor (0 = no smoothing)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='maximum norm for gradient clipping')
    parser.add_argument('--manualSeed', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--class_weights', type=str, default='none', help="Class weighting: 'none' (default), 'auto' (inverse-frequency on train set), "         "or a comma-separated list like '1.0,2.5,3.0'.")
    return parser.parse_args()

def compute_miou(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    ious = []
    for i in range(targets.shape[0]):
        part_ious = []
        for part in range(num_classes):
            I = np.sum((predictions[i] == part) & (targets[i] == part))
            U = np.sum((predictions[i] == part) | (targets[i] == part))
            iou = I / float(U) if U > 0 else 1.0
            part_ious.append(iou)
        ious.append(np.mean(part_ious))
    return float(np.mean(ious))

def compute_class_weights(dataset, num_classes: int, epsilon: float = 1e-6, normalize: str = 'mean') -> torch.Tensor:
    import numpy as np
    counts = np.zeros(num_classes, dtype=np.int64)

    for i in range(len(dataset)):
        _, seg = dataset[i]  # seg: (N,) numpy array or torch tensor
        if hasattr(seg, 'numpy'):
            seg = seg.numpy()
        seg = seg.astype(np.int64, copy=False)
        u, c = np.unique(seg, return_counts=True)
        for k, v in zip(u, c):
            if 0 <= int(k) < num_classes:
                counts[int(k)] += int(v)

    total = counts.sum()
    if total == 0:
        raise RuntimeError("No labels found when computing class weights.")

    freqs = counts / float(total)
    inv = 1.0 / (freqs + epsilon)

    if normalize == 'mean':
        inv = inv / inv.mean()
    elif normalize == 'max':
        inv = inv / inv.max()

    w = torch.tensor(inv, dtype=torch.float32)
    print(f"[ClassWeights] counts={counts.tolist()} freqs={freqs.tolist()} weights={inv.tolist()}")
    return w



def get_device() -> torch.device:
    """Return GPU device if available, else CPU."""
    if torch.cuda.is_available():
        print(f"[Device] CUDA is available. Training on GPU: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda:0')
    else:
        print("[Device] CUDA not available. Training on CPU.")
        return torch.device('cpu')


def main() -> None:
    opt = parse_args()
    print(opt)

    # Set random seeds for reproducibility
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.manualSeed)

    # Verify dataset structure
    if not os.path.exists(opt.dataset):
        raise FileNotFoundError(f"Dataset path not found: {opt.dataset}")
    train_dir = os.path.join(opt.dataset, 'train')
    val_dir = os.path.join(opt.dataset, 'val')
    test_dir = os.path.join(opt.dataset, 'test')
    for subdir in ('train', 'val', 'test'):
        split_path = os.path.join(opt.dataset, subdir)
        if not os.path.isdir(split_path):
            raise ValueError(f"Dataset must contain '{subdir}' subdirectory: missing {split_path}")

    # Load datasets
    train_dataset = CustomDataset(root_dir=opt.dataset, split='train', augment=opt.augment)
    val_dataset = CustomDataset(root_dir=opt.dataset, split='val', augment=False)
    test_dataset = CustomDataset(root_dir=opt.dataset, split='test', augment=False)
    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples and {len(test_dataset)} test samples.")

    # Determine number of classes
    num_classes: Optional[int] = opt.num_classes or getattr(train_dataset, 'num_classes', None)
    if num_classes is None:
        raise ValueError("Number of classes could not be inferred; please provide --num_classes")
    print(f"Using {num_classes} segmentation classes.")

    weights = None
    if opt.class_weights.lower() != 'none':
        if opt.class_weights.strip().lower() == 'auto':
            weights = compute_class_weights(train_dataset, num_classes).to('cpu')  # move to device later
        else:
            # Parse manual weights like "1.0,2.0,3.0"
            parts = [float(x) for x in opt.class_weights.split(',')]
            assert len(parts) == num_classes, f"--class_weights has {len(parts)} values, but num_classes={num_classes}"
            weights = torch.tensor(parts, dtype=torch.float32)
            print(f"[ClassWeights] manual={parts}")

    # Create criterion (note: if weights provided, we disable label smoothing)
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to('cuda' if torch.cuda.is_available() else 'cpu') if weights is not None else None)
    use_soft_labels = (opt.label_smoothing > 0) and (weights is None)
    if opt.label_smoothing > 0 and weights is not None:
        print("[ClassWeights] Detected class weights; ignoring label_smoothing to avoid conflicting objectives.")

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers
    )

    # Model
    classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)
    if opt.model:
        classifier.load_state_dict(torch.load(opt.model, map_location='cpu'))
    device = get_device()
    classifier.to(device)

    # Optimiser and scheduler
    optimizer = optim.Adam(
        classifier.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.nepoch, eta_min=1e-6
    )

    best_val_miou: float = 0.0

    train_loss_history: list[float] = []
    train_acc_history: list[float] = []
    lr_history: list[float] = []

    # Training/validation loop
    for epoch in range(1, opt.nepoch + 1):
        epoch_start = time.time()
        classifier.train()
        running_loss = 0.0
        running_correct = 0
        running_points = 0

        # Training loop with progress bar
        for points, target in tqdm(train_loader, desc=f"Epoch {epoch} Training", ncols=80):
            points = points.transpose(2, 1).to(device)  # reshape to (B, 3, N)
            target = target.to(device)  # (B, N)

            optimizer.zero_grad()
            pred, trans, trans_feat = classifier(points)
            pred = pred.view(-1, num_classes)  # (B*N, C)
            target_flat = target.view(-1)

            # Compute loss with optional label smoothing
            if use_soft_labels:
                true_dist = smooth_labels(target_flat, num_classes, smoothing=opt.label_smoothing)
                log_pred = F.log_softmax(pred, dim=1)
                loss = torch.mean(torch.sum(-true_dist * log_pred, dim=1))
            else:
                loss = criterion(pred, target_flat)

            # Feature transform regulariser
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), opt.grad_clip)
            optimizer.step()

            _, preds = pred.max(1)
            running_correct += preds.eq(target_flat).sum().item()
            running_points += target_flat.size(0)
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_acc = running_correct / running_points if running_points > 0 else 0.0

        # Validation loop
        classifier.eval()
        val_ious = []
        with torch.no_grad():
            for points, target in tqdm(val_loader, desc=f"Epoch {epoch} Validation", ncols=80):
                points = points.transpose(2, 1).to(device)
                target = target.to(device)
                pred, _, _ = classifier(points)
                pred_choice = pred.argmax(2).cpu().numpy()
                target_np = target.cpu().numpy()
                val_ious.append(compute_miou(pred_choice, target_np, num_classes))
        val_miou = float(np.mean(val_ious)) if val_ious else 0.0

        # Track the best validation mIoU and optionally save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch}/{opt.nepoch} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val mIoU: {val_miou:.4f} | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        lr_history.append(current_lr)
        # Optional: also track validation mIoU
        try:
            val_miou_history.append(val_miou)
        except NameError:
            val_miou_history = [val_miou]

        # Save checkpoint for every epoch
        os.makedirs(opt.outf, exist_ok=True)
        ckpt_path = os.path.join(opt.outf, f"model_epoch_{epoch}.pth")
        torch.save(classifier.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # After training, optionally evaluate on the test set
    print(f"\nFinished training. Best validation mIoU: {best_val_miou:.4f}")
    print("Evaluating on test set…")
    classifier.eval()
    test_ious = []
    with torch.no_grad():
        for points, target in tqdm(test_loader, desc="Testing", ncols=80):
            points = points.transpose(2, 1).to(device)
            target = target.to(device)
            pred, _, _ = classifier(points)
            pred_choice = pred.argmax(2).cpu().numpy()
            target_np = target.cpu().numpy()
            test_ious.append(compute_miou(pred_choice, target_np, num_classes))
    test_miou = float(np.mean(test_ious)) if test_ious else 0.0
    print(f"Test mIoU: {test_miou:.4f}")

    def plot_training_metrics(losses, accs, lrs, val_mious, out_dir):
        import os
        os.makedirs(out_dir, exist_ok=True)
        epochs = range(1, len(losses) + 1)
        fig, axes = plt.subplots(4, 1, figsize=(7, 11), sharex=True)

        axes[0].plot(epochs, losses, marker='o'); axes[0].set_ylabel('Loss'); axes[0].set_title('Training Loss'); axes[0].grid(True)
        axes[1].plot(epochs, accs, marker='o');   axes[1].set_ylabel('Accuracy'); axes[1].set_title('Training Accuracy'); axes[1].grid(True)
        axes[2].plot(epochs, lrs, marker='o');    axes[2].set_ylabel('Learning Rate'); axes[2].set_title('LR Schedule'); axes[2].grid(True)
        axes[3].plot(epochs, val_mious, marker='o'); axes[3].set_xlabel('Epoch'); axes[3].set_ylabel('Val mIoU'); axes[3].set_title('Validation mIoU'); axes[3].grid(True)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'training_metrics.png'), dpi=300)
        plt.close(fig)
        print(f"Saved training metrics plot to {os.path.join(out_dir, 'training_metrics.png')}")

    # call the plotting function using the collected histories and your output directory
    plot_training_metrics(train_loss_history, train_acc_history, lr_history, val_miou_history, opt.outf)

if __name__ == '__main__':
    main()