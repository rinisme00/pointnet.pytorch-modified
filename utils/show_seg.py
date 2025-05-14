#!/usr/bin/env python3
"""
Visualise & export a PointNet segmentation checkpoint (.pth) with debug prints.

Flags:
  --model           path/to/checkpoint.pth
  --dataset         root dir with train/ and test/
  --split           train or test (default: test)
  --idx             sample index to load (default: -1=random)
  --num_points      points/sample (must match training; default=2048)
  --cpu             force CPU
  --save OUT.png    save GT/vs/pred scatter via Matplotlib
  --export PREFIX   write PREFIX_pred_label0…_pred_labelK.pts
  --export_gt PF    write PREFIX_gt_label… plus gt_edges/unassigned
  --export_edges PF write PREFIX_edges.pts for segmentation boundaries
  --export_unassigned PF write PREFIX_unassigned.pts for low-confidence
  --knn K           neighbors for edge detection (default: 10)
  --conf_thresh T   threshold for unassigned (max-softmax < T; default: 0.0)

Usage sample:
  python show_seg.py \
    --model /.../model_epoch_897.pth \
    --dataset /.../training_dataset/ \
    --split test --idx 3 \
    --save results/vis_3.png \
    --export results/model_epoch_897 \
    --export_gt results/model_epoch_897_gt \
    --export_edges results/model_epoch_897 \
    --export_unassigned results/model_epoch_897 \
    --conf_thresh 0.1
"""
from __future__ import print_function
import os, sys, json, random, argparse, warnings, pathlib
import numpy as np
import torch
from torch.autograd import no_grad
import matplotlib
from matplotlib import cm, pyplot as plt

# project imports
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from custom_dataset import CustomDataset
from pointnet.model   import PointNetDenseCls

# helpers
def build_colour_map(k:int):
    cmap = cm.get_cmap("hsv", k)
    return np.asarray([cmap(i)[:3] for i in range(k)], dtype=np.float32)

def vis_matplotlib(xyz, gt_rgb, pred_rgb, save_path=None):
    fig = plt.figure(figsize=(6,3))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("Ground-truth")
    ax1.scatter(*xyz.T, s=2, c=gt_rgb); ax1.axis("off")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title("Prediction")
    ax2.scatter(*xyz.T, s=2, c=pred_rgb); ax2.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ saved figure → {save_path}")
    else:
        plt.show()

def save_pts(xyz:np.ndarray, out_path:pathlib.Path):
    with open(out_path, "w") as f:
        for x,y,z in xyz:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    print(f"✓ Wrote → {out_path}")

# argument parsing
def parse_args():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model",    required=True)
    p.add_argument("--dataset",  required=True)
    p.add_argument("--split",    default="test", choices=["train","test"])
    p.add_argument("--idx",      type=int, default=-1,
                   help="-1=random, else sample index")
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--cpu",      action="store_true")
    p.add_argument("--save",     metavar="OUT.png",
                   help="save scatter comparison")
    p.add_argument("--export",   metavar="PREFIX",
                   help="write PREFIX_pred_label0…_pred_labelK.pts")
    p.add_argument("--export_gt", metavar="PREFIX",
                   help="write PREFIX_gt_label… plus gt_edges/unassigned")
    p.add_argument("--export_edges",      metavar="PREFIX",
                   help="write PREFIX_edges.pts (segmentation boundaries)")
    p.add_argument("--export_unassigned", metavar="PREFIX",
                   help="write PREFIX_unassigned.pts (low-confidence)")
    p.add_argument("--knn",      type=int, default=10,
                   help="neighbors for edge detection")
    p.add_argument("--conf_thresh", type=float, default=0.0,
                   help="threshold for unassigned (max-softmax < T)")
    return p.parse_args()

# main
def main():
    args = parse_args()
    print(json.dumps(vars(args), indent=2))

    if args.save:
        matplotlib.use("Agg")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # load dataset
    ds = CustomDataset(args.dataset, split=args.split,
                       num_points=args.num_points, augment=False)
    if len(ds)==0:
        sys.exit("Empty split – check your folders.")
    idx = random.randrange(len(ds)) if args.idx<0 else args.idx
    pts, gt_lbl = ds[idx]
    xyz = pts.numpy(); gt_arr = gt_lbl.numpy()

    # DEBUG: unique GT labels
    print("DEBUG: unique ground-truth labels:", np.unique(gt_arr))

    # load checkpoint
    ckpt = torch.load(os.path.expanduser(args.model), map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # DEBUG: checkpoint keys & class count
    if "conv4.weight" in state_dict:
        print("DEBUG: conv4.weight shape =", state_dict["conv4.weight"].shape)
        num_cls = state_dict["conv4.weight"].shape[0]
    else:
        print("DEBUG: no conv4.weight key; keys:", list(state_dict.keys()))
        num_cls = int(gt_lbl.max().item()) + 1
    colours = build_colour_map(num_cls)

    net = PointNetDenseCls(k=num_cls, feature_transform=True).to(device)
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("⚠ load_state_dict warnings:")
        print("  missing:   ", missing)
        print("  unexpected:", unexpected)
    net.eval()

    # forward
    with no_grad():
        logits,_,_ = net(pts.to(device).unsqueeze(0).transpose(2,1))
        pred_lbl = logits.argmax(2).squeeze(0).cpu().numpy()
        conf     = torch.softmax(logits,2).squeeze(0).max(1)[0].cpu().numpy()

    # DEBUG: unique predicted labels
    print("DEBUG: unique predicted labels:", np.unique(pred_lbl))

    # visualize
    gt_rgb   = colours[gt_arr]
    pred_rgb = colours[pred_lbl]
    vis_matplotlib(xyz, gt_rgb, pred_rgb, save_path=args.save)

    # export pred
    if args.export:
        base = pathlib.Path(args.export).expanduser()
        for cls in range(num_cls):
            mask = (pred_lbl == cls)
            save_pts(xyz[mask], base.with_name(f"{base.name}_pred_label{cls}.pts"))

    # export gt
    if args.export_gt:
        base = pathlib.Path(args.export_gt).expanduser()
        for cls in np.unique(gt_arr):
            save_pts(xyz[gt_arr==cls], base.with_name(f"{base.name}_gt_label{cls}.pts"))
        if 0 in gt_arr:
            save_pts(xyz[gt_arr==0], base.with_name(f"{base.name}_gt_edges.pts"))
        if np.any(gt_arr<0):
            save_pts(xyz[gt_arr<0], base.with_name(f"{base.name}_gt_unassigned.pts"))

    # export edges
    if args.export_edges:
        diff = xyz[:,None,:] - xyz[None,:,:]
        d2   = np.sum(diff*diff, axis=2)
        nbrs = np.argsort(d2, axis=1)[:,1:1+args.knn]
        mask = np.any(pred_lbl[nbrs] != pred_lbl[:,None], axis=1)
        save_pts(xyz[mask], pathlib.Path(args.export_edges)
                 .with_name(f"{pathlib.Path(args.export_edges).name}_edges.pts"))

    # export unassigned
    if args.export_unassigned:
        ua = (conf < args.conf_thresh)
        save_pts(xyz[ua], pathlib.Path(args.export_unassigned)
                 .with_name(f"{pathlib.Path(args.export_unassigned).name}_unassigned.pts"))

if __name__=="__main__":
    main()
