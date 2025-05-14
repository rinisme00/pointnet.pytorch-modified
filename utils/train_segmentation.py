import os, sys
# assume this file lives in <project_root>/utils/…
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

#!/usr/bin/env python3
import argparse
import os
import random
import torch
import torch.optim as optim
import torch.utils.data
from custom_dataset import CustomDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def smooth_labels(target, num_classes, smoothing=0.1):
    """
    Returns a smoothed label tensor (of shape [N, num_classes]) for label smoothing.
    The correct label gets value 1-smoothing, and the others get smoothing/(num_classes-1).
    """
    with torch.no_grad():
        true_dist = torch.zeros(target.size(0), num_classes, device=target.device)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1 - smoothing)
    return true_dist

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--augment', action='store_true', help="use data augmentation")
parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay factor')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing factor (0 = no smoothing)')
parser.add_argument('--grad_clip', type=float, default=1.0, help='maximum norm for gradient clipping')
opt = parser.parse_args()
print(opt)

# Set random seed for reproducibility
opt.manualSeed = random.randint(1, 10000)
print("Random Seed:", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if not os.path.exists(opt.dataset):
    raise ValueError(f"Dataset path not found: {opt.dataset}")

train_path = os.path.join(opt.dataset, 'train')
test_path = os.path.join(opt.dataset, 'test')
if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise ValueError(f"Dataset structure incorrect. Expected 'train' and 'test' directories in {opt.dataset}")

# Load datasets
train_dataset = CustomDataset(root_dir=opt.dataset, split='train', augment=opt.augment)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
test_dataset = CustomDataset(root_dir=opt.dataset, split='test', augment=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)

num_classes = opt.num_classes
print(f'Number of classes: {num_classes}')

classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)
if opt.model:
    classifier.load_state_dict(torch.load(opt.model))
classifier = classifier.cuda()

optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.nepoch, eta_min=1e-6)

best_miou = 0.0
for epoch in range(opt.nepoch):
    classifier.train()
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    for points, target in train_dataloader:
        points = points.transpose(2, 1).cuda()
        target = target.cuda()

        optimizer.zero_grad()
        pred, trans, trans_feat = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1)
        if opt.label_smoothing > 0:
            true_dist = smooth_labels(target, num_classes, smoothing=opt.label_smoothing)
            log_pred = F.log_softmax(pred, dim=1)
            loss = torch.mean(torch.sum(-true_dist * log_pred, dim=1))
        else:
            loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), opt.grad_clip)
        optimizer.step()

        pred_choice = pred.argmax(1)
        total_correct += pred_choice.eq(target).sum().item()
        total_points += target.size(0)
        total_loss += loss.item()

    train_loss = total_loss / len(train_dataloader)
    train_acc = total_correct / float(total_points)
    scheduler.step()
    print(f"Epoch [{epoch+1}/{opt.nepoch}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    classifier.eval()
    shape_ious = []
    with torch.no_grad():
        for points, target in test_dataloader:
            points = points.transpose(2, 1).cuda()
            target = target.cuda()
            pred, _, _ = classifier(points)
            pred_choice = pred.argmax(2)
            pred_np = pred_choice.cpu().numpy()
            target_np = target.cpu().numpy()
            for shape_idx in range(target_np.shape[0]):
                part_ious = []
                for part in range(num_classes):
                    I = np.sum((pred_np[shape_idx]==part) & (target_np[shape_idx]==part))
                    U = np.sum((pred_np[shape_idx]==part) | (target_np[shape_idx]==part))
                    iou = I / float(U) if U > 0 else 1
                    part_ious.append(iou)
                shape_ious.append(np.mean(part_ious))
    test_miou = np.mean(shape_ious)
    print(f"Epoch [{epoch+1}/{opt.nepoch}] Test mIoU: {test_miou:.4f}")

    # Save checkpoint for EVERY epoch
    save_path = os.path.join(opt.outf, f'model_epoch_{epoch+1}.pth')
    os.makedirs(opt.outf, exist_ok=True)
    torch.save(classifier.state_dict(), save_path)
    print(f"Model saved to {save_path} with Test mIoU: {test_miou:.4f}")

    # Optionally update best_miou if needed for logging purposes
    if test_miou > best_miou:
        best_miou = test_miou

print(f"\nBest Test mIoU: {best_miou:.4f}")