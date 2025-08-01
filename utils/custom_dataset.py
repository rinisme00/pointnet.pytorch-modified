#!/usr/bin/env python3
"""
Custom PyTorch Dataset for point cloud segmentation with optional class inference and enhanced documentation.
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

def _load_pts(path: str) -> np.ndarray:
    """Load a point cloud from a .pts file."""
    return np.loadtxt(path)  # shape (N, 3)

def _load_seg(path: str) -> np.ndarray:
    """Load segmentation labels from a .seg file."""
    return np.loadtxt(path, dtype=int)

class CustomDataset(Dataset):
    """
    Dataset for point cloud segmentation.

    Args:
        root_dir (str): Root directory containing 'train' and 'test' subfolders.
        split (str): 'train' or 'test'.
        num_points (int): Number of points to sample per cloud.
        augment (bool): Apply data augmentation if True and split='train'.
        dropout_ratio (float): Fraction of points to drop in augmentation.
        infer_num_classes (bool): If True, auto-infer number of classes from data.
        num_classes (int, optional): Force a specific class count.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        num_points: int = 2048,
        augment: bool = False,
        dropout_ratio: float = 0.1,
        infer_num_classes: bool = True,
        num_classes: int = None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.augment = augment and split == 'train'
        self.dropout_ratio = dropout_ratio

        # Validate directories
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Expected directory not found: {split_dir}")
        self.split_dir = split_dir

        # Gather point and segmentation files
        self.files = sorted(
            [f for f in os.listdir(split_dir) if f.endswith('.pts')]
        )
        if not self.files:
            raise ValueError(f"No .pts files found in {split_dir}")
        if len(self.files) == 1 and split == 'test':
            print("Warning: Only one .pts file found in test split.")

        # Class count inference
        if infer_num_classes:
            self.num_classes = self._infer_num_classes()
        else:
            self.num_classes = num_classes
        if self.num_classes is None:
            raise ValueError(
                "Number of classes could not be determined; "
                "provide num_classes or use infer_num_classes=True"
            )

        print(f"Loaded {len(self.files)} samples for split='{split}', num_classes={self.num_classes}")

    def _infer_num_classes(self) -> int:
        """Scan all .seg files to determine number of unique classes."""
        labels_set = set()
        for fname in self.files:
            seg_path = os.path.join(self.split_dir, fname.replace('.pts', '.seg'))
            labels = _load_seg(seg_path)
            labels_set.update(np.unique(labels).tolist())
        # Classes assumed to be 0..C-1 or 1..C
        max_label = max(labels_set)
        return max_label + 1

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        # Load data
        pts_path = os.path.join(self.split_dir, self.files[idx])
        seg_path = pts_path.replace('.pts', '.seg')
        points = _load_pts(pts_path)
        labels = _load_seg(seg_path)

        # Resample to fixed size
        N = points.shape[0]
        if N >= self.num_points:
            choice = np.random.choice(N, self.num_points, replace=False)
        else:
            choice = np.random.choice(N, self.num_points, replace=True)
        points = points[choice]
        labels = labels[choice]

        # Augmentation
        if self.augment:
            points = self._augment(points)

        # Normalize to unit sphere
        points = self._normalize(points)

        # Convert to tensors
        return (
            torch.tensor(points, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )

    def _augment(self, points: np.ndarray) -> np.ndarray:
        """Apply random augmentations: rotation, scaling, jitter, dropout."""
        # Rotation
        theta = random.uniform(0, 2 * np.pi)
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,            1]
        ])
        points = points.dot(rot)

        # Flip
        if random.random() < 0.5:
            points[:,0] *= -1

        # Translation
        trans = np.random.uniform(-0.1, 0.1, (1,3))
        points += trans

        # Scaling
        points *= random.uniform(0.8, 1.2)

        # Jitter
        points += np.random.normal(0, 0.02, points.shape)

        # Dropout
        if random.random() < 0.5:
            drop_n = int(points.shape[0] * self.dropout_ratio)
            idx = np.random.choice(points.shape[0], drop_n, replace=False)
            points[idx] = np.mean(points, axis=0)
        return points

    def _normalize(self, points: np.ndarray) -> np.ndarray:
        """Center and scale point cloud to unit sphere."""
        centroid = points.mean(axis=0)
        pts = points - centroid
        m = np.sqrt((pts ** 2).sum(axis=1)).max()
        if m > 0:
            pts /= m
        return pts
