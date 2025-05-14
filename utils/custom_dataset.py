#!/usr/bin/env python3
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random

class CustomDataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=2048, augment=False, dropout_ratio=0.1):
        """
        Parameters:
          root_dir     : Root directory for dataset.
          split        : 'train' or 'test'
          num_points   : Number of points to resample.
          augment      : Whether to perform data augmentation.
          dropout_ratio: Fraction of points to randomly drop in augmentation.
        """
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        self.num_points = num_points
        self.augment = augment
        self.dropout_ratio = dropout_ratio

        # Ensure directories exist
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        if not os.path.exists(self.split_dir):
            raise ValueError(f"Split directory not found: {self.split_dir} (expected {split} folder).")

        # Get all .pts files
        self.files = [f for f in os.listdir(self.split_dir) if f.endswith('.pts')]
        if len(self.files) == 0:
            raise ValueError(f"No .pts files found in {self.split_dir}.")
        if len(self.files) == 1 and split == 'test':
            print("Warning: Only 1 .pts file found in test set. Proceeding anyway.")

        print(f"Loaded {len(self.files)} point cloud files for {split} split from {self.split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pts_file = os.path.join(self.split_dir, self.files[idx])
        seg_file = pts_file.replace('.pts', '.seg')
        if not os.path.exists(seg_file):
            raise FileNotFoundError(f"Segmentation file missing: {seg_file}")

        # Load point cloud and labels
        points = np.loadtxt(pts_file)  # shape: (N, 3)
        # Load segmentation labels as provided (assumed to be one label per point)
        labels = np.loadtxt(seg_file, dtype=int)
        # Optional: if your labels are 1-indexed, they should be converted outside
        # here in your conversion script or here; for now we assume they are already correctly processed.

        # Resample points: if the input is larger than num_points, randomly select; if smaller, allow replacement.
        if points.shape[0] >= self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.num_points, replace=True)
        points = points[choice, :]
        labels = labels[choice]

        if self.augment and self.split == 'train':
            points = self.augment_points(points)

        points = self.normalize_points(points)

        return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def augment_points(self, points):
        """Perform aggressive augmentation on the input point cloud."""
        # Random rotation about z-axis
        theta = random.uniform(0, 2 * np.pi)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta),  np.cos(theta), 0],
                               [0, 0, 1]])
        points = np.dot(points, rot_matrix)

        # Random flip on x-axis
        if random.random() < 0.5:
            points[:, 0] = -points[:, 0]

        # Random translation: shift by up to ±0.1 in each axis
        translation = np.random.uniform(-0.1, 0.1, (1, 3))
        points = points + translation

        # Random scaling
        scale = random.uniform(0.8, 1.2)
        points = points * scale

        # Random jittering with increased noise
        jitter = np.random.normal(0, 0.03, points.shape)
        points = points + jitter

        # Random dropout of points to simulate missing data 
        if random.random() < 0.5:
            num_drop = int(points.shape[0] * self.dropout_ratio)
            drop_idx = np.random.choice(points.shape[0], num_drop, replace=False)
            # Set dropped points to the centroid (or add random noise)
            centroid = np.mean(points, axis=0)
            points[drop_idx] = centroid

        return points

    def normalize_points(self, points):
        """Center and scale the point cloud to unit sphere."""
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
        return points
