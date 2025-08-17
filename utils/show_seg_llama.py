#!/usr/bin/env python3
"""
show_seg_llama.py

Visualize LLaMA+SegHead segmentation checkpoints produced by
utils/train_segmentation_llama.py on a .pts/.seg dataset.

- Supports splits: dataset/{train,test}/*.{pts,seg} or a flat folder of pairs.
- Applies the same normalization & sample/pad policy as training (no aug).
- Visualizes Ground Truth vs Prediction, and can export predicted labels.

Usage:
  python show_seg_llama.py \
    --model seg_llama/epochs/epoch_0098.pth \
    --dataset /path/to/dataset/split \
    --split test --idx 0 \
    --points 2048 --save out.png --export pred.pts

Options:
  --binary_collapse   Collapse {1,2}->1 (broken) and 0->0 (unbroken) for display/export
  --no_normalize      Disable centering & unit-sphere scaling (not recommended)
  --bf16              Use bfloat16 compute if supported (else fp16 on CUDA, fp32 on CPU)
  --cpu               Force CPU
  --seed              Sampling seed for deterministic subsampling/padding
"""

import os, sys, glob, argparse, json, random
import numpy as np
import torch
from torch.autograd import no_grad
import matplotlib
from matplotlib import pyplot as plt

# ---------- Project imports ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    # PointNet stem defined in the training script
    from train_segmentation_llama import PointNetStem
    from llama_backbone.encoder import LLaMAEncoder
    from llama_backbone.heads import SegHead
except Exception as e:
    raise RuntimeError(
        "Could not import project modules. "
        "Run from the project root where llama_backbone/* and train_segmentation_llama.py are available."
    ) from e


# ---------- Utils ----------
def normalize_unit_sphere(P: np.ndarray) -> np.ndarray:
    """Center to mean and scale to unit max radius (like training)."""
    xyz = P[:, :3]
    center = xyz.mean(axis=0, keepdims=True)
    xyz = xyz - center
    scale = np.linalg.norm(xyz, axis=1).max()
    if scale < 1e-6:
        scale = 1e-6
    P = P.copy()
    P[:, :3] = xyz / scale
    return P

def sample_pad(P: np.ndarray, S: np.ndarray, n_ctx: int, rng: np.random.Generator):
    """Subsample or pad to n_ctx points; return (P,S,mask)."""
    N = P.shape[0]
    if S.shape[0] != N:
        m = min(N, S.shape[0])
        P, S = P[:m], S[:m]
        N = m
    if n_ctx <= 0:
        mask = np.ones(N, dtype=np.int64)
        return P, S, mask
    if N >= n_ctx:
        idx = rng.choice(N, n_ctx, replace=False)
        mask = np.ones(n_ctx, dtype=np.int64)
    else:
        pad = n_ctx - N
        idx_real = np.arange(N)
        idx_pad = rng.choice(N, pad, replace=True) if N > 0 else np.zeros(pad, dtype=np.int64)
        idx = np.concatenate([idx_real, idx_pad])
        mask = np.concatenate([np.ones(N, dtype=np.int64), np.zeros(pad, dtype=np.int64)])
    P = P[idx]
    S = S[idx]
    return P, S, mask

def build_colour_map(k: int):
    cmap = matplotlib.colormaps.get_cmap("hsv")
    cols = np.array([cmap(i / max(1, k))[:3] for i in range(k)], dtype=np.float32)
    return cols

def vis_pair(xyz, gt_rgb, pred_rgb, save_path=None):
    if save_path:
        matplotlib.use("Agg")
    fig = plt.figure(figsize=(6.5, 3.25))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("Ground Truth")
    ax1.scatter(xyz[:, 0].tolist(), xyz[:, 1].tolist(), xyz[:, 2].tolist(), s=2, c=gt_rgb.tolist())
    ax1.axis("off")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title("Prediction")
    ax2.scatter(xyz[:, 0].tolist(), xyz[:, 1].tolist(), xyz[:, 2].tolist(), s=2, c=pred_rgb.tolist())
    ax2.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[save] {save_path}")
    else:
        plt.show()

def save_pts_with_labels(xyz: np.ndarray, labels: np.ndarray, out_path: str):
    with open(out_path, "w") as f:
        for (x, y, z), l in zip(xyz, labels):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(l)}\n")
    print(f"[export] {out_path}")

def find_pairs(root):
    dtrain, dtest = os.path.join(root, "train"), os.path.join(root, "test")

    def list_pairs(d):
        pts = sorted(glob.glob(os.path.join(d, "*.pts")))
        bases = {os.path.splitext(os.path.basename(p))[0] for p in pts}
        out = []
        for b in sorted(bases):
            p = os.path.join(d, b + ".pts")
            s = os.path.join(d, b + ".seg")
            if os.path.exists(p) and os.path.exists(s):
                out.append((p, s))
        return out

    if os.path.isdir(dtrain) and os.path.isdir(dtest):
        return {"train": list_pairs(dtrain), "test": list_pairs(dtest)}
    # flat folder fallback
    pts = sorted(glob.glob(os.path.join(root, "*.pts")))
    bases = {os.path.splitext(os.path.basename(p))[0] for p in pts}
    flat = []
    for b in sorted(bases):
        p = os.path.join(root, b + ".pts")
        s = os.path.join(root, b + ".seg")
        if os.path.exists(p) and os.path.exists(s):
            flat.append((p, s))
    return {"train": flat, "test": flat}

def infer_num_classes_from_ckpt(ckpt: dict):
    if "num_classes" in ckpt and isinstance(ckpt["num_classes"], (int, np.integer)):
        return int(ckpt["num_classes"])
    if "seg_head" in ckpt and isinstance(ckpt["seg_head"], dict):
        for k, w in ckpt["seg_head"].items():
            if k.endswith(".weight") and hasattr(w, "shape") and getattr(w, "ndim", 0) == 2:
                return int(w.shape[0])
    return None

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="checkpoint .pth from train_segmentation_llama.py")
    ap.add_argument("--dataset", required=True, help="dataset root (with train/test or flat .pts/.seg)")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--idx", type=int, default=-1, help="sample index; -1=random")
    ap.add_argument("--points", type=int, default=1024, help="N context points (<=0 = all)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no_normalize", action="store_true")
    ap.add_argument("--binary_collapse", action="store_true", help="map {1,2}->1; 0->0 for display/export")
    ap.add_argument("--save", metavar="OUT.png", help="save visualization to file")
    ap.add_argument("--export", metavar="OUT.pts", help="export predicted labels into .pts (x y z label)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    print(json.dumps(vars(args), indent=2))

    # --- Load ckpt FIRST (read arch flags from it) ---
    ckpt = torch.load(os.path.expanduser(args.model), map_location="cpu")
    if not isinstance(ckpt, dict) or ("seg_head" not in ckpt and "peft_state_dict" not in ckpt):
        raise RuntimeError("This script expects a LLaMA+SegHead checkpoint saved by train_segmentation_llama.py")

    # Infer arch hyperparams (with fallbacks)
    in_dim        = int(ckpt.get("in_dim", 3))
    fourier_k     = int(ckpt.get("fourier_k", 16))
    model_or_path = ckpt.get("model_or_path", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    lora_r        = int(ckpt.get("lora_r", 16))
    lora_alpha    = int(ckpt.get("lora_alpha", 32))
    lora_dropout  = float(ckpt.get("lora_dropout", 0.05))

    # --- arch-from-ckpt (must be before any `if use_stem:`) ---
    use_stem      = bool(ckpt.get("use_pointnet_stem", False))
    pn_local_dim  = int(ckpt.get("pn_local_dim", 0) or 0)
    stem_sd       = ckpt.get("stem", None)
    use_feature_transform = bool(stem_sd) and any(k.startswith("fstn.") for k in stem_sd.keys())
    enc_in_dim    = in_dim + (pn_local_dim if use_stem else 0)

    # Device & dtype (AFTER reading flags, BEFORE building modules)
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        bf16_ok = False
        compute_dtype = torch.float32
    else:
        device = torch.device("cuda")
        bf16_ok = torch.cuda.is_bf16_supported()
        compute_dtype = (torch.bfloat16 if (args.bf16 and bf16_ok) else torch.float16)

    # Build PointNet stem (now use_stem is defined)
    stem = None
    if use_stem:
        stem = PointNetStem(local_dim=pn_local_dim, use_feature_transform=use_feature_transform).to(device)
        if stem_sd is not None:
            stem.load_state_dict(stem_sd, strict=True)
        stem.eval()

    # Dataset discovery & sample selection
    pairs = find_pairs(args.dataset)
    files = pairs[args.split]
    if not files:
        raise FileNotFoundError(f"No .pts/.seg pairs under split '{args.split}' at {args.dataset}")
    idx = random.randrange(len(files)) if args.idx < 0 else args.idx
    idx = max(0, min(idx, len(files) - 1))
    pts_path, seg_path = files[idx]
    print(f"[data] using {args.split}[{idx}]: {os.path.basename(pts_path)}")

    # Load a single sample
    P = np.loadtxt(pts_path, dtype=np.float32, ndmin=2)  # [N, F]
    S = np.loadtxt(seg_path, dtype=np.int64,   ndmin=1)  # [N]
    if P.ndim != 2 or P.shape[1] < in_dim:
        raise ValueError(f"{pts_path} has shape {P.shape}; expected at least {in_dim} features/point.")
    P = P[:, :in_dim]
    # Keep a copy of original coords if you ever want to plot them
    # xyz_orig = P[:, :3].copy()

    # Preprocess (like training but no augmentation)
    if not args.no_normalize:
        P = normalize_unit_sphere(P)

    # Sample/pad to args.points
    rng = np.random.default_rng(args.seed)
    P_s, S_s, M = sample_pad(P, S, args.points, rng)
    # ignore negative labels and pads
    S_s = np.where((S_s >= 0), S_s, -100)
    S_s = np.where(M == 1, S_s, -100)

    # Determine num_classes (prefer ckpt info)
    k = infer_num_classes_from_ckpt(ckpt)
    if k is None:
        vmax = 0
        limit = min(len(files), 64)
        for i in range(limit):
            _, sfile = files[i]
            lab = np.loadtxt(sfile, dtype=np.int64, ndmin=1)
            if lab.size > 0:
                vmax = max(vmax, int(lab.max()))
        k = int(vmax + 1)
    print(f"[info] inferred num_classes={k}")

    # Build encoder + head with the EXACT arch used in training
    encoder = LLaMAEncoder(
        model_or_path=model_or_path,
        in_dim=enc_in_dim,             # critical: 3 + pn_local_dim if stem used
        fourier_k=fourier_k,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        load_in_4bit=True,
        compute_dtype=compute_dtype,
    ).to(device)

    seg_head = SegHead(d_model=encoder.d_model, num_classes=k).to(device)

    # Load weights (LoRA adapters, point tokenizer, seg head)
    peft_sd = ckpt.get("peft_state_dict", None)
    if peft_sd:
        try:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(encoder.llm, peft_sd)
        except Exception:
            encoder.llm.load_state_dict(peft_sd, strict=False)
    if "point_tok" in ckpt:
        encoder.point_tok.load_state_dict(ckpt["point_tok"], strict=True)
    if "seg_head" in ckpt:
        seg_head.load_state_dict(ckpt["seg_head"], strict=True)

    # Align dtype of head to encoder/backbone param dtype if possible
    try:
        bb_param = next((p for p in encoder.parameters() if p.requires_grad), None)
        if bb_param is not None:
            seg_head.to(device=device, dtype=bb_param.dtype)
    except StopIteration:
        pass

    # --- Build model inputs (concat stem features if used) ---
    P_base = torch.from_numpy(P_s).to(device).unsqueeze(0).float()               # [1, N, in_dim] (>=3)
    M_t = torch.from_numpy((M == 1).astype(np.bool_)).to(device).unsqueeze(0)    # [1, N]
    S_t = torch.from_numpy(S_s).to(device)                                       # [N] (stats only)

    if use_stem and stem is not None:
        # stem consumes xyz only
        F_loc, _ = stem(P_base[..., :3])                                         # [1, N, pn_local_dim]
        P_in = torch.cat([P_base, F_loc.to(P_base.dtype)], dim=-1)               # [1, N, enc_in_dim]
    else:
        P_in = P_base

    # Inference
    if device.type == "cuda":
        ac = torch.amp.autocast(device_type="cuda", dtype=(torch.bfloat16 if (args.bf16 and bf16_ok) else torch.float16))
    else:
        ac = torch.cpu.amp.autocast(enabled=False)

    with no_grad(), ac:
        H = encoder(P_in, M_t)                          # [1, N, d]
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        H = H.to(next(seg_head.parameters()).dtype)
        logits = seg_head(H)                            # [1, N, k]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        logits = logits.clamp_(-30, 30)

    preds = logits.argmax(-1).squeeze(0).detach().cpu().numpy()  # [N]
    valid = (S_s != -100)

    # Stats
    if valid.any():
        acc = float((preds[valid] == S_s[valid]).mean())
    else:
        acc = float("nan")
    u_gt, c_gt = np.unique(S_s[S_s >= 0], return_counts=True) if (S_s >= 0).any() else ([], [])
    u_pd, c_pd = np.unique(preds, return_counts=True)
    print(f"[stats] gt hist: {dict(zip(u_gt.tolist(), c_gt.tolist()))}")
    print(f"[stats] pd hist: {dict(zip(u_pd.tolist(), c_pd.tolist()))}")
    print(f"[stats] (subsample) accuracy: {acc:.4f}")

    # Optional collapse to binary for display/export
    if args.binary_collapse:
        # 0 -> 0 (unbroken), {1,2,...}->1 (broken)
        preds_disp = (preds != 0).astype(np.int64)
        gt_disp = np.where(S_s >= 0, (S_s != 0).astype(np.int64), S_s)
        k_disp = 2
    else:
        preds_disp = preds
        gt_disp = S_s
        k_disp = k

    # Colors & visualization (on normalized/sampled coords for fidelity)
    colours = build_colour_map(k_disp)
    gt_col_idx = np.clip(np.where(gt_disp >= 0, gt_disp, 0), 0, k_disp - 1)
    pd_col_idx = np.clip(np.where(preds_disp >= 0, preds_disp, 0), 0, k_disp - 1)
    gt_rgb = colours[gt_col_idx]
    pd_rgb = colours[pd_col_idx]
    vis_xyz = P_s[:, :3]
    vis_pair(vis_xyz, gt_rgb, pd_rgb, save_path=args.save)

    # Export predicted labels (.pts with x y z lbl)
    if args.export:
        save_pts_with_labels(vis_xyz, preds_disp, args.export)


if __name__ == "__main__":
    main()