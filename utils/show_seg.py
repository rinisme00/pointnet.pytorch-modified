#!/usr/bin/env python3
"""
Simplified visualization and export for PointNet segmentation.
Shows a side-by-side plot of ground truth vs. prediction
and exports a new .pts file with predicted class labels per point.
"""
import os
import sys
import random
import argparse
import json
import numpy as np
import torch
from torch.autograd import no_grad
import matplotlib
from matplotlib import cm, pyplot as plt

# Setup project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from custom_dataset import CustomDataset
from pointnet.model import PointNetDenseCls


def build_colour_map(k: int):
    """Return an HSV-based RGB colormap for k classes."""
    cmap = matplotlib.colormaps.get_cmap("hsv")
    colours = np.array([cmap(i / k)[:3] for i in range(k)], dtype=np.float32)
    return colours



def vis_matplotlib(xyz, gt_rgb, pred_rgb, save_path=None):
    """Plot GT vs. prediction side-by-side, work around numpy.may_share_memory bug."""
    if save_path:
        matplotlib.use("Agg")
    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("Ground Truth")
    # convert to lists here:
    ax1.scatter(
        xyz[:,0].tolist(),
        xyz[:,1].tolist(),
        xyz[:,2].tolist(),
        s=2,
        c=gt_rgb.tolist()
    )
    ax1.axis("off")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title("Prediction")
    ax2.scatter(
        xyz[:,0].tolist(),
        xyz[:,1].tolist(),
        xyz[:,2].tolist(),
        s=2,
        c=pred_rgb.tolist()
    )
    ax2.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization: {save_path}")
    else:
        plt.show()




def save_pts_with_labels(xyz: np.ndarray, labels: np.ndarray, out_path: str):
    """Export xyz and label per point into .pts file."""
    with open(out_path, 'w') as f:
        for (x, y, z), lbl in zip(xyz, labels):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {lbl}\n")
    print(f"Exported points with labels: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, help='path to .pth checkpoint')
    p.add_argument('--dataset', required=True, help='root dataset dir with train/ and test/')
    p.add_argument('--split', choices=['train','test'], default='test', help='which split to sample')
    p.add_argument('--idx', type=int, default=-1, help='sample index, -1=random')
    p.add_argument('--num_points', type=int, default=2048, help='points per sample (must match training)')
    p.add_argument('--cpu', action='store_true', help='force CPU even if CUDA available')
    p.add_argument('--save', metavar='OUT.png', help='path to save visualization image')
    p.add_argument('--export', metavar='OUT.pts', help='path to export predicted labels (.pts)')
    return p.parse_args()


def main():
    args = parse_args()
    print(json.dumps(vars(args), indent=2))

    # Device
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    # Load dataset
    ds = CustomDataset(args.dataset, split=args.split, num_points=args.num_points, augment=False)
    if len(ds) == 0:
        sys.exit(f"No samples found in split '{args.split}'")

    idx = random.randrange(len(ds)) if args.idx < 0 else args.idx
    points, gt_lbl = ds[idx]
    xyz = points.numpy()
    gt_arr = gt_lbl.numpy()

    # Load model
    state = torch.load(os.path.expanduser(args.model), map_location=device)
    state_dict = state.get('state_dict', state) if isinstance(state, dict) else state
    # Infer num_classes
    num_classes = state_dict.get('conv4.weight', None)
    if num_classes is not None:
        k = state_dict['conv4.weight'].shape[0]
    else:
        k = int(gt_lbl.max()) + 1

    # Color maps
    colours = build_colour_map(k)
    gt_rgb = colours[gt_arr]

    # Build network
    net = PointNetDenseCls(k=k, feature_transform=True).to(device)
    missing, _ = net.load_state_dict(state_dict, strict=False)
    if missing:
        print('Load warnings, missing keys:', missing)
    net.eval()

    # Inference
    with no_grad():
        logits, _, _ = net(points.to(device).unsqueeze(0).transpose(2,1))
        pred_lbl = logits.argmax(2).squeeze(0).cpu().numpy()
    pred_rgb = colours[pred_lbl]

    # Visualization
    vis_matplotlib(xyz, gt_rgb, pred_rgb, save_path=args.save)

    # Export labeled points
    if args.export:
        save_pts_with_labels(xyz, pred_lbl, args.export)

if __name__ == '__main__':
    main()
