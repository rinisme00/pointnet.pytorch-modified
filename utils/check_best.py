import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
"""
Evaluate every .pth in --ckpt_dir, rank by mIoU, and copy the best one to
best_model_miou.pth

Usage:
  python utils/check_best.py \
        --ckpt_dir  /storage/.../seg \
        --dataset   /storage/.../training_dataset \
        --num_classes 7
"""
import argparse, glob, json, os, shutil, warnings

import numpy as np
import torch, torch.utils.data
from custom_dataset import CustomDataset
from pointnet.model import PointNetDenseCls


# ──────────────────── helpers ──────────────────────────────────────────────────
def load_state_dict(ckpt_path):
    """Return the raw state‑dict saved in ckpt_path (independent of format)."""
    raw = torch.load(ckpt_path, map_location="cpu")

    # 1) plain state‑dict
    if isinstance(raw, dict) and all(isinstance(k, str) for k in raw.keys()):
        if "weight" in next(iter(raw)) or "feat" in next(iter(raw)):
            return raw

    # 2) dict that wraps the real state‑dict
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "net", "model"):
            if key in raw:
                return raw[key]

    # 3) full torch.nn.Module (rare)
    if hasattr(raw, "state_dict"):
        return raw.state_dict()

    raise RuntimeError(f"Un‑recognised checkpoint format: {ckpt_path}")


@torch.no_grad()
def evaluate(net, loader, k):
    net.eval()
    miou_all, acc_all = [], []

    for pts, lbl in loader:
        pts = pts.transpose(2, 1).cuda(non_blocking=True)
        lbl = lbl.cuda(non_blocking=True)

        pred, _, _ = net(pts)
        pred_choice = pred.argmax(2)

        p = pred_choice.cpu().numpy()
        t = lbl.cpu().numpy()
        for s in range(t.shape[0]):
            part_ious = []
            correct = (p[s] == t[s]).sum()
            total = t.shape[1]
            for part in range(k):
                I = np.logical_and(p[s] == part, t[s] == part).sum()
                U = np.logical_or (p[s] == part, t[s] == part).sum()
                part_ious.append(1.0 if U == 0 else I / float(U))
            miou_all.append(np.mean(part_ious))
            acc_all .append(correct / total)

    return float(np.mean(miou_all)), float(np.mean(acc_all))


# ──────────────────── main ─────────────────────────────────────────────────────
def main(args):
    test_set = CustomDataset(root_dir=args.dataset, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batchSize, shuffle=False, num_workers=args.workers
    )

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pth")))
    if not ckpts:
        raise RuntimeError(f"No .pth files found in {args.ckpt_dir}")

    results = []
    for ckpt in ckpts:
        try:
            state_dict = load_state_dict(ckpt)
        except RuntimeError as e:
            warnings.warn(str(e))
            continue

        # auto‑detect whether feature_transform was on when this checkpoint
        # was created
        use_ft = any(k.startswith("feat.fstn") for k in state_dict.keys())
        net = PointNetDenseCls(
            k=args.num_classes, feature_transform=use_ft
        ).cuda()

        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if missing:
            # Something is seriously mismatched (different #classes etc.)
            warnings.warn(
                f"Skip {os.path.basename(ckpt)}: shape mismatch ({len(missing)} missing keys)"
            )
            continue

        miou, acc = evaluate(net, test_loader, args.num_classes)
        results.append(dict(ckpt=os.path.basename(ckpt), miou=miou, acc=acc))
        print(f"{os.path.basename(ckpt):30s}  mIoU={miou:.4f}  acc={acc:.4f}")

    if not results:
        raise RuntimeError("None of the checkpoints could be evaluated.")

    # ── pick the best by mIoU ──────────────────────────────────────────────────
    results.sort(key=lambda d: d["miou"], reverse=True)
    best = results[0]
    print("\nBest checkpoint (by mIoU)")
    print(json.dumps(best, indent=2))

    # copy/symlink for convenience
    src = os.path.join(args.ckpt_dir, best["ckpt"])
    dst = os.path.join(args.ckpt_dir, "best_model_miou.pth")
    try:
        os.remove(dst)
    except FileNotFoundError:
        pass
    shutil.copy2(src, dst)
    print(f"Copied best model to {dst}")

    # dump full table
    with open(os.path.join(args.ckpt_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)


# ──────────────────── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="directory with .pth files")
    ap.add_argument("--dataset", required=True, help="root dataset folder")
    ap.add_argument("--num_classes", type=int, default=7)
    ap.add_argument("--batchSize", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    main(args)

