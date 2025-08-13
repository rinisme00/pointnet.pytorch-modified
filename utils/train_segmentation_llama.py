#!/usr/bin/env python3
"""
Training script for PointNet segmentation with LLaMA backbone integration,
label smoothing, checkpoint on every epoch, and improved logging.
"""
import os
import sys
import time
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# Compute the project root (…/pointnet.pytorch-modified) and ensure imports work
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports after adjusting sys.path
from custom_dataset import CustomDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
from llama_backbone.encoder import LLaMAEncoder


def smooth_labels(target, num_classes, smoothing=0.1):
    """
    Returns a smoothed label tensor (shape [N, num_classes]).
    """
    with torch.no_grad():
        true_dist = torch.zeros(target.size(0), num_classes, device=target.device)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    return true_dist


def parse_args():
    parser = argparse.ArgumentParser(description="Train PointNet segmentation model with LLaMA backbone")
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='seg', help='output folder')
    parser.add_argument('--model', type=str, default='', help='pretrained model path')
    parser.add_argument('--dataset', type=str, required=True, help='dataset root directory')
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform module")
    parser.add_argument('--augment', action='store_true', help="use data augmentation during training")
    parser.add_argument('--num_classes', type=int, default=None, help='number of segmentation classes (auto-inferred if omitted)')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay factor')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing factor (0 = no smoothing)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='max norm for gradient clipping')
    parser.add_argument('--manualSeed', type=int, default=42, help='random seed for reproducibility')
    return parser.parse_args()


def main():
    opt = parse_args()
    print(opt)

    # Reproducibility
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.manualSeed)

    # Verify dataset structure
    if not os.path.exists(opt.dataset):
        raise FileNotFoundError(f"Dataset path not found: {opt.dataset}")
    train_dir = os.path.join(opt.dataset, 'train')
    test_dir = os.path.join(opt.dataset, 'test')
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise ValueError(f"Dataset must contain 'train' and 'test' subdirectories")

    # Load datasets
    train_dataset = CustomDataset(root_dir=opt.dataset, split='train', augment=opt.augment)
    test_dataset  = CustomDataset(root_dir=opt.dataset, split='test',  augment=False)
    print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples.")

    # Determine number of classes
    num_classes = opt.num_classes or getattr(train_dataset, 'num_classes', None)
    if num_classes is None:
        raise ValueError("Number of classes could not be inferred; please provide --num_classes")
    print(f"Using {num_classes} classes.")

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset,  batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)

    # Models: PointNet + LLaMA backbone encoder
    classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)
    if opt.model:
        classifier.load_state_dict(torch.load(opt.model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.to(device)

    # Instantiate LLaMA encoder (fine-tunable via LoRA)
    llama_enc = LLaMAEncoder().to(device)

    # Optimizer & scheduler (include LLaMA parameters)
    optimizer = optim.Adam(
        list(classifier.parameters()) + list(llama_enc.parameters()),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.nepoch, eta_min=1e-6
    )

    best_miou = 0.0

    for epoch in range(1, opt.nepoch + 1):
        epoch_start = time.time()
        classifier.train(); llama_enc.train()
        running_loss = 0.0
        running_correct = 0
        running_points = 0

        # Training loop
        for points, target in tqdm(train_loader, desc=f"Epoch {epoch} Training", ncols=80):
            points = points.transpose(2, 1).to(device)
            target = target.to(device)

            # Prepare text prompts for LLaMA—replace with your actual prompts
            prompts = ["Segment this object based on 3D geometry."] * points.size(0)
            text_emb = llama_enc(prompts)  # shape: (B, hidden_dim)

            optimizer.zero_grad()
            # Forward: pass both point cloud and text embedding
            pred, trans, trans_feat = classifier(points, text_emb)
            pred = pred.view(-1, num_classes)
            target_flat = target.view(-1)

            # Loss
            if opt.label_smoothing > 0:
                true_dist = smooth_labels(target_flat, num_classes, smoothing=opt.label_smoothing)
                log_pred = F.log_softmax(pred, dim=1)
                loss = torch.mean(torch.sum(-true_dist * log_pred, dim=1))
            else:
                loss = F.cross_entropy(pred, target_flat)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001

            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(classifier.parameters()) + list(llama_enc.parameters()), opt.grad_clip
            )
            optimizer.step()

            _, preds = pred.max(1)
            running_correct += preds.eq(target_flat).sum().item()
            running_points += target_flat.size(0)
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_acc = running_correct / running_points

        # Evaluation loop
        classifier.eval(); llama_enc.eval()
        ious = []
        with torch.no_grad():
            for points, target in tqdm(test_loader, desc=f"Epoch {epoch} Testing ", ncols=80):
                points = points.transpose(2, 1).to(device)
                target = target.to(device)
                prompts = ["Segment this object based on 3D geometry."] * points.size(0)
                text_emb = llama_enc(prompts)
                pred, _, _ = classifier(points, text_emb)
                pred_choice = pred.argmax(2).cpu().numpy()
                target_np = target.cpu().numpy()
                for i in range(target_np.shape[0]):
                    part_ious = []
                    for part in range(num_classes):
                        I = np.sum((pred_choice[i] == part) & (target_np[i] == part))
                        U = np.sum((pred_choice[i] == part) | (target_np[i] == part))
                        iou = I / float(U) if U > 0 else 1
                        part_ious.append(iou)
                    ious.append(np.mean(part_ious))
        test_miou = np.mean(ious)

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch}/{opt.nepoch} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Test mIoU: {test_miou:.4f} | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        # Save checkpoint for every epoch
        os.makedirs(opt.outf, exist_ok=True)
        ckpt_path = os.path.join(opt.outf, f"model_epoch_{epoch}.pth")
        torch.save({
            'pointnet': classifier.state_dict(),
            'llama': llama_enc.state_dict()
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        best_miou = max(best_miou, test_miou)

    print(f"\nBest Test mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    main()