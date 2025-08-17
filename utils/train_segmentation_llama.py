#!/usr/bin/env python3
# utils/train_segmentation_llama.py
# TinyLlama (4-bit) + optional PointNet stem → per-point segmentation head
# Parity flags with train_segmentation.py: --batchSize, --nepoch, --outf, --augment,
# --feature_transform, --label_smoothing, --lr

import os, sys, json, time, argparse, random
from dataclasses import asdict, dataclass
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# Ensure project root on path
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---- External simplified dataset (required) ----
HAS_EXT_DATASET, EXT_IMPORT_ERR = False, None
try:
    from utils.custom_dataset import CustomDataset as ExtCustomDataset  # when run from project root
    HAS_EXT_DATASET = True
except Exception:
    try:
        from custom_dataset import CustomDataset as ExtCustomDataset      # when cwd==utils/
        HAS_EXT_DATASET = True
    except Exception as e:
        EXT_IMPORT_ERR = e

# --- TinyLlama wrapper + segmentation head ---
from llama_backbone.encoder import LLaMAEncoder   # (B,N,F) -> (B,N,d_model)
from llama_backbone.heads import SegHead          # (B,N,d_model) -> (B,N,C)


# ======================= IoU utils =======================
class IoUMeter:
    """Accumulate per-class intersection/union and compute mIoU."""
    def __init__(self, num_classes: int, ignore_index: int = -100, device: torch.device | str | None = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.reset()

    def to(self, device):
        self.device = torch.device(device)
        self.inter = self.inter.to(self.device)
        self.union = self.union.to(self.device)
        return self

    def reset(self):
        self.inter = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self.union = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.device != self.device:
            self.to(preds.device)
        preds = preds.to(self.device)
        target = target.to(self.device)

        if self.ignore_index is not None and self.ignore_index >= 0:
            valid = target != self.ignore_index
            preds = preds[valid]
            target = target[valid]

        for c in range(self.num_classes):
            pred_c = preds == c
            target_c = target == c
            inter = (pred_c & target_c).sum()
            union = pred_c.sum() + target_c.sum() - inter
            self.inter[c] += inter
            self.union[c] += union

    def compute(self, present_only: bool = False):
        inter = self.inter.float()
        union = self.union.float()
        if present_only:
            mask = union > 0
            # Keep full length; mark absent classes as NaN for readability
            iou_per_class = torch.full_like(union, float("nan"))
            iou_per_class[mask] = inter[mask] / union[mask]
            miou = iou_per_class[mask].mean().item() if mask.any() else float("nan")
            return miou, iou_per_class.tolist()
        else:
            iou_per_class = inter / union.clamp_min(1)
            miou = iou_per_class.mean().item()
            return miou, iou_per_class.tolist()

    @staticmethod
    @torch.no_grad()
    def run_eval_pass(encoder, seg_head, loader, device, cfg, weights, scaler_ctx, stem=None):
        if stem is not None:
            stem.eval()
        encoder.eval()
        seg_head.eval()

        iou_meter = IoUMeter(num_classes=cfg.num_classes, ignore_index=-100, device=device).to(device)
        total_loss = 0.0
        total_steps = 0
        total_correct = 0
        total_valid = 0

        for P, S, M in loader:
            P = P.to(device).float()
            S = S.to(device).long()
            M = M.to(device).bool()

            pn_reg = 0.0
            if stem is not None:
                F_loc, reg = stem(P[..., :3])
                P = torch.cat([P, F_loc], dim=-1)
                pn_reg = reg  # not used in eval loss

            with scaler_ctx:
                H = encoder(P, M)
                H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
                H = H.to(next(seg_head.parameters()).dtype)

                logits = seg_head(H).clamp_(-30, 30)
                C = logits.size(-1)
                logits_flat = logits.reshape(-1, C).float()
                target_flat = S.reshape(-1)
                valid = target_flat != -100

                if valid.any():
                    if cfg.label_smoothing > 0:
                        Cn = logits_flat.size(-1)
                        with torch.no_grad():
                            true_dist = torch.full_like(logits_flat[valid], cfg.label_smoothing / (Cn - 1))
                            true_dist.scatter_(1, target_flat[valid].unsqueeze(1), 1.0 - cfg.label_smoothing)
                        logp = F.log_softmax(logits_flat[valid], dim=1)
                        ce = torch.mean(torch.sum(-true_dist * logp, dim=1))
                    else:
                        ce = F.cross_entropy(logits_flat[valid], target_flat[valid], weight=weights, reduction="mean")

                    if torch.isfinite(ce):
                        total_loss += float(ce.item())
                        total_steps += 1

                    preds = logits_flat[valid].argmax(-1)
                    total_correct += (preds == target_flat[valid]).sum().item()
                    total_valid += int(valid.sum().item())
                    iou_meter.update(preds, target_flat[valid])

        loss = (total_loss / total_steps) if total_steps > 0 else float("nan")
        acc  = (total_correct / total_valid) if total_valid > 0 else float("nan")
        miou, per_class = iou_meter.compute(present_only=cfg.present_only_miou)
        return {"loss": loss, "acc": acc, "mIoU": miou, "per_class": per_class}


# ======================= PointNet stem (optional) =======================
def feature_transform_regularizer(trans):
    # ||I - A A^T||_F^2
    d = trans.size(-1)
    I = torch.eye(d, device=trans.device, dtype=trans.dtype).unsqueeze(0)
    loss = torch.mean(torch.norm(I - torch.matmul(trans, trans.transpose(2, 1)), dim=(1, 2))**2)
    return loss

class STN3d(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden*2), nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden*2, 64), nn.ReLU(True),
            nn.Linear(64, 9)
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):  # x: (B,N,3)
        B, N, _ = x.shape
        h = self.mlp1(x)                           # (B,N,2h)
        g = torch.max(h, dim=1, keepdim=False)[0]  # (B,2h)
        out = self.fc(g).view(B, 3, 3)
        I = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
        return out + I

class STNkd(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.mlp1 = nn.Sequential(
            nn.Linear(k, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, k*k)
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, f):  # f: (B,N,k)
        B, N, k = f.shape
        h = self.mlp1(f)                           # (B,N,128)
        g = torch.max(h, dim=1, keepdim=False)[0]  # (B,128)
        out = self.fc(g).view(B, k, k)
        I = torch.eye(k, device=f.device, dtype=f.dtype).unsqueeze(0)
        return out + I

class PointNetStem(nn.Module):
    """Produce per-point local features to concatenate with raw coords before LLaMAEncoder."""
    def __init__(self, local_dim=64, use_feature_transform=True):
        super().__init__()
        self.use_feature_transform = use_feature_transform
        self.stn3 = STN3d(hidden=64)
        self.mlp_pts = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(True),
            nn.Linear(64, local_dim), nn.ReLU(True)
        )
        self.fstn = STNkd(local_dim) if use_feature_transform else None
        self.mlp_out = nn.Sequential(
            nn.Linear(local_dim, local_dim), nn.ReLU(True)
        )

    def forward(self, P):  # P: (B,N,3)
        reg = 0.0
        T3 = self.stn3(P)                          # (B,3,3)
        P_aligned = torch.bmm(P, T3)               # (B,N,3)
        F_loc = self.mlp_pts(P_aligned)            # (B,N,d)
        if self.use_feature_transform:
            Tf = self.fstn(F_loc)                  # (B,d,d)
            F_loc = torch.bmm(F_loc, Tf)
            reg = feature_transform_regularizer(Tf) + feature_transform_regularizer(T3)
        F_out = self.mlp_out(F_loc)
        return F_out, reg


# ======================= Training =======================
@dataclass
class TrainConfig:
    dataset: str
    outdir: str
    model_or_path: str
    num_classes: int
    in_dim: int = 3
    points: int = 2048
    batch_size: int = 16
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    workers: int = 4
    val_ratio: float = 0.05
    seed: int = 42
    bf16: bool = False
    grad_accum_steps: int = 1
    clip_grad_norm: float = 1.0

    # LLaMA / PEFT
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    fourier_k: int = 16
    load_in_4bit: bool = True

    # New options
    label_mode: str = "3class"          # "3class" | "broken_binary" | "fragment_binary"
    use_class_weight: bool = False
    present_only_miou: bool = False
    augment: bool = True
    # PointNet stem
    use_pointnet_stem: bool = True
    pn_local_dim: int = 64
    feature_transform: bool = False
    ft_lambda: float = 1e-3
    # Loss
    label_smoothing: float = 0.0
    # Sampling
    oversample_fragment: int = 1        # >1 to oversample files containing class=2 in raw seg


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Root with 'train/' and 'test/' subfolders containing .pts/.seg pairs")
    ap.add_argument("--outdir", "--outf", dest="outdir", type=str, default="seg_llama", help="output folder")
    ap.add_argument("--model_or_path", type=str, required=True,
                    help="e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0 or local path")
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--in_dim", type=int, default=3)
    ap.add_argument("--points", type=int, default=2048, help="number of points per cloud")
    ap.add_argument("--batch_size", "--batchSize", dest="batch_size", type=int, default=16)
    ap.add_argument("--epochs", "--nepoch", dest="epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_ratio", type=float, default=0.05)  # unused with external split, kept for parity
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)

    # LLaMA / PEFT / quant
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--fourier_k", type=int, default=16)
    ap.add_argument("--no_4bit", action="store_true", help="disable 4-bit loading")

    # Dataset / labels
    ap.add_argument("--label_mode", choices=["3class", "broken_binary", "fragment_binary"],
                    default="3class",
                    help="3class: 0/1/2; broken_binary: {1,2}->1 vs 0; fragment_binary: 2->1 vs {0,1}->0")
    ap.add_argument("--use_class_weight", action="store_true")
    ap.add_argument("--present_only_miou", action="store_true")
    ap.add_argument("--augment", action="store_true", help="use data augmentation on train split")
    # PointNet stem
    ap.add_argument("--use_pointnet_stem", action="store_true", help="prepend PointNet stem features")
    ap.add_argument("--pn_local_dim", type=int, default=64)
    ap.add_argument("--feature_transform", action="store_true", help="enable feature transform + regularizer")
    ap.add_argument("--ft_lambda", type=float, default=1e-3)
    # Loss
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    # Sampling
    ap.add_argument("--oversample_fragment", type=int, default=1)

    args = ap.parse_args()
    return args


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def check_device(tag="startup"):
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        print(f"[{tag}] device=cuda:{dev} | {props.name} | VRAM={props.total_memory/1024**3:.1f}GB | BF16_supported={torch.cuda.is_bf16_supported()}")
    else:
        print(f"[{tag}] device=cpu | CUDA not available")


def fmt_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def smooth_ce_loss(logits, target, smoothing: float):
    # logits: (M,C), target: (M,), smoothing in [0,1)
    if smoothing <= 0.0:
        return F.cross_entropy(logits, target, reduction="mean")
    C = logits.size(-1)
    with torch.no_grad():
        true_dist = torch.full_like(logits, smoothing / (C - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    logp = F.log_softmax(logits, dim=1)
    return torch.mean(torch.sum(-true_dist * logp, dim=1))


# -------- Label mapping + collate for external dataset --------
def _map_labels(S: torch.Tensor, mode: str) -> torch.Tensor:
    """Map raw labels to chosen mode. Keeps ignore_index=-100 for negatives."""
    S = torch.where(S >= 0, S, torch.full_like(S, -100))
    if mode == "broken_binary":
        pos = S >= 0
        S = S.clone()
        S[pos] = (S[pos] != 0).long()    # {1,2}->1 ; 0->0
    elif mode == "fragment_binary":
        pos = S >= 0
        S = S.clone()
        S[pos] = (S[pos] == 2).long()    # 2->1 ; {0,1}->0
    return S

def make_ext_collate(label_mode: str):
    """Adapter collate for ExtCustomDataset: add mask M=1 and map labels."""
    def _collate(batch):
        # batch: list of (points[N,3], labels[N])
        Ps, Ss = zip(*batch)
        P = torch.stack([torch.as_tensor(p, dtype=torch.float32) for p in Ps], dim=0)  # (B,N,3)
        S = torch.stack([torch.as_tensor(s, dtype=torch.long) for s in Ss], dim=0)     # (B,N)
        S = _map_labels(S, label_mode)
        M = torch.ones_like(S, dtype=torch.bool)                                       # all valid (fixed-length sampling)
        return P, S, M
    return _collate


def main():
    if not HAS_EXT_DATASET:
        raise ImportError(f"Failed to import utils/custom_dataset.py: {EXT_IMPORT_ERR}\n"
                          "This script now relies on the external CustomDataset with 'train/' and 'test/' splits.")

    args = parse_args()
    cfg = TrainConfig(
        dataset=args.dataset,
        outdir=args.outdir,
        model_or_path=args.model_or_path,
        num_classes=args.num_classes,
        in_dim=args.in_dim,
        points=args.points,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        workers=args.workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        bf16=args.bf16,
        grad_accum_steps=args.grad_accum_steps,
        clip_grad_norm=args.clip_grad_norm,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        fourier_k=args.fourier_k,
        load_in_4bit=(not args.no_4bit),
        label_mode=args.label_mode,
        use_class_weight=args.use_class_weight,
        present_only_miou=args.present_only_miou,
        augment=args.augment,
        use_pointnet_stem=args.use_pointnet_stem,
        pn_local_dim=args.pn_local_dim,
        feature_transform=args.feature_transform,
        ft_lambda=args.ft_lambda,
        label_smoothing=args.label_smoothing,
        oversample_fragment=args.oversample_fragment,
    )

    # Binary modes force 2 classes
    if cfg.label_mode in ("broken_binary", "fragment_binary"):
        cfg.num_classes = 2

    set_seed(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)
    with open(os.path.join(cfg.outdir, "args.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print("[cfg]", json.dumps(asdict(cfg), indent=2))

    # ---------------- Data (external dataset only) ----------------
    split_train = os.path.join(cfg.dataset, "train")
    split_test  = os.path.join(cfg.dataset, "test")
    if not (os.path.isdir(split_train) and os.path.isdir(split_test)):
        raise FileNotFoundError(f"Expected '{split_train}' and '{split_test}' with .pts/.seg pairs")

    train_set = ExtCustomDataset(
        root_dir=cfg.dataset, split="train",
        num_points=(cfg.points if cfg.points and cfg.points > 0 else 2048),
        augment=cfg.augment,
        infer_num_classes=False, num_classes=cfg.num_classes
    )
    val_set = ExtCustomDataset(
        root_dir=cfg.dataset, split="test",
        num_points=(cfg.points if cfg.points and cfg.points > 0 else 2048),
        augment=False,
        infer_num_classes=False, num_classes=cfg.num_classes
    )
    print(f"[data] train={len(train_set)}  val={len(val_set)}")

    # Oversample files that contain raw label 2 (fragment), if requested
    sampler = None
    if cfg.oversample_fragment and cfg.oversample_fragment > 1:
        sample_weights = []
        seg_files = [os.path.join(train_set.split_dir, f.replace(".pts", ".seg")) for f in train_set.files]
        for seg_path in seg_files:
            try:
                raw_S = np.loadtxt(seg_path, dtype=np.int64, ndmin=1)
                has_frag = bool(np.any(raw_S == 2))
            except Exception:
                has_frag = False
            sample_weights.append(cfg.oversample_fragment if has_frag else 1.0)
        if sample_weights:
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    collate_fn = make_ext_collate(cfg.label_mode)

    # ---------------- Dataloaders ----------------
    drop_last = len(train_set) >= cfg.batch_size

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )

    # dedicated evaluation loader over the full train split
    train_eval_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    if len(train_loader) == 0 and len(train_eval_loader) == 0:
        raise RuntimeError(
            f"Empty training split: len(train_set)={len(train_set)}, batch_size={cfg.batch_size}. "
            f"Use --batchSize <= {max(1,len(train_set))} or add more samples."
        )

    # ---------------- Device & AMP ----------------
    check_device("before_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if (cfg.bf16 and torch.cuda.is_bf16_supported()) else torch.float16
        try:
            scaler_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)  # torch>=2.0
        except TypeError:
            scaler_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)
    else:
        scaler_ctx = nullcontext()

    use_fp16_scaler = (device.type == "cuda") and not (cfg.bf16 and torch.cuda.is_bf16_supported())
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_scaler)
    compute_dtype = torch.bfloat16 if (cfg.bf16 and torch.cuda.is_bf16_supported()) else torch.float16

    # ---------------- Model ----------------
    stem = None
    enc_in_dim = cfg.in_dim
    if cfg.use_pointnet_stem:
        stem = PointNetStem(local_dim=cfg.pn_local_dim, use_feature_transform=cfg.feature_transform).to(device)
        enc_in_dim = cfg.in_dim + cfg.pn_local_dim

    encoder = LLaMAEncoder(
        model_or_path=cfg.model_or_path,
        in_dim=enc_in_dim,
        fourier_k=cfg.fourier_k,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        load_in_4bit=cfg.load_in_4bit,
        compute_dtype=compute_dtype,
    ).to(device)

    seg_head = SegHead(d_model=encoder.d_model, num_classes=cfg.num_classes)
    head_dtype = (torch.bfloat16 if (cfg.bf16 and torch.cuda.is_bf16_supported())
                  else (torch.float16 if device.type == "cuda" else torch.float32))
    seg_head.to(device=device, dtype=head_dtype)

    params = []
    if stem is not None:
        params += list(stem.parameters())
    params += list(seg_head.parameters())
    params += [p for p in encoder.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    # ---------------- Class weights (optional) ----------------
    weights = None
    if cfg.use_class_weight:
        with torch.no_grad():
            counts = torch.zeros(cfg.num_classes, dtype=torch.long)
            seg_files = [os.path.join(train_set.split_dir, f.replace(".pts", ".seg")) for f in train_set.files]
            for sp in seg_files:
                try:
                    arr = torch.from_numpy(np.loadtxt(sp, dtype=np.int64, ndmin=1))
                    arr = arr[arr >= 0]
                    if arr.numel():
                        if cfg.label_mode == "broken_binary":
                            # {1,2}->1 ; 0->0
                            arr = torch.where(arr != 0, torch.tensor(1, dtype=arr.dtype), torch.tensor(0, dtype=arr.dtype))
                        elif cfg.label_mode == "fragment_binary":
                            # 2->1 ; {0,1}->0
                            arr = (arr == 2).long()
                        vals, c = torch.unique(arr, return_counts=True)
                        counts[vals] += c
                except Exception:
                    pass
        freqs = counts.float().clamp_min(1)
        weights = (freqs.sum() / freqs)
        weights = (weights / weights.mean()).to(device=device, dtype=torch.float32)
        print(f"[class weights] counts={counts.tolist()}  weights={[round(x,4) for x in weights.tolist()]}")


    # ---------------- Train ----------------
    epochs_dir = os.path.join(cfg.outdir, "epochs"); os.makedirs(epochs_dir, exist_ok=True)
    best_miou = -1.0
    best_path = os.path.join(cfg.outdir, "best.pth")
    global_start = time.time()

    def save_ckpt(ep, metrics):
        ckpt = {
            "epoch": ep,
            **metrics,
            # Arch / cfg snapshot
            "model_or_path": cfg.model_or_path,
            "in_dim": cfg.in_dim,
            "fourier_k": cfg.fourier_k,
            "num_classes": cfg.num_classes,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "label_mode": cfg.label_mode,
            "use_pointnet_stem": cfg.use_pointnet_stem,
            "pn_local_dim": cfg.pn_local_dim,
            # Weights
            "stem": (stem.state_dict() if stem is not None else None),
            "point_tok": encoder.point_tok.state_dict(),
            "seg_head": seg_head.state_dict(),
            "class_weights": (weights.tolist() if weights is not None else None),
            "peft_state_dict": None,
        }
        try:
            from peft import get_peft_model_state_dict
            ckpt["peft_state_dict"] = {k: v.cpu() for k, v in get_peft_model_state_dict(encoder.llm).items()}
        except Exception:
            ckpt["peft_state_dict"] = {k: v.detach().cpu() for k, v in encoder.llm.named_parameters() if v.requires_grad}

        ep_path = os.path.join(epochs_dir, f"epoch_{ep:04d}.pth")
        torch.save(ckpt, ep_path)
        torch.save(ckpt, os.path.join(cfg.outdir, "latest.pth"))
        return ep_path

    last_train = last_val = {}

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        if stem is not None: stem.train()
        encoder.train(); seg_head.train()
        total_loss = 0.0; n_steps = 0
        train_correct = 0; train_total = 0
        train_iou_meter = IoUMeter(num_classes=cfg.num_classes, ignore_index=-100, device=device).to(device)

        optimizer.zero_grad(set_to_none=True)
        iterable = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]", leave=False) if tqdm else train_loader

        for step, (P,S,M) in enumerate(iterable, start=1):
            P = P.to(device, non_blocking=True).float()   # (B,N,3)
            S = S.to(device, non_blocking=True).long()
            M = M.to(device, non_blocking=True).bool()

            pn_reg = 0.0
            if stem is not None:
                # only use xyz for stem
                F_loc, reg = stem(P[..., :3])
                P = torch.cat([P, F_loc], dim=-1)
                pn_reg = reg

            with scaler_ctx:
                H = encoder(P, M)                                # (B,N,d)
                H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
                H = H.to(next(seg_head.parameters()).dtype)

                logits = seg_head(H)                             # (B,N,C)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-30, 30)

                C = logits.size(-1)
                logits_flat = logits.reshape(-1, C).float()
                target_flat = S.reshape(-1)
                valid = target_flat != -100
                valid_count = valid.sum()

                if valid_count > 0:
                    preds = logits_flat[valid].argmax(-1)
                    train_correct += (preds == target_flat[valid]).sum().item()
                    train_total   += int(valid_count.item())
                    train_iou_meter.update(preds, target_flat[valid])

                    ce = (smooth_ce_loss(logits_flat[valid], target_flat[valid], cfg.label_smoothing)
                          if cfg.label_smoothing > 0 else
                          F.cross_entropy(logits_flat[valid], target_flat[valid], weight=weights, reduction="mean"))
                else:
                    ce = torch.zeros((), device=logits.device, dtype=logits_flat.dtype)

                loss = ce + (cfg.ft_lambda * pn_reg if cfg.feature_transform and stem is not None else 0.0)
                loss = loss / cfg.grad_accum_steps

            if not torch.isfinite(loss):
                if tqdm is None:
                    print(f"[warn] non-finite loss at epoch {epoch} step {step} — skipping batch")
                optimizer.zero_grad(set_to_none=True); continue

            if use_fp16_scaler:
                scaler.scale(loss).backward()
                if step % cfg.grad_accum_steps == 0:
                    if cfg.clip_grad_norm and cfg.clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad_norm)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if cfg.clip_grad_norm and cfg.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad_norm)
                if step % cfg.grad_accum_steps == 0:
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * cfg.grad_accum_steps
            n_steps += 1

            if epoch == 1 and step == 1:
                print("label range (incl pads):", int(S.min().item()), int(S.max().item()),
                      "| valid_count:", int((S != -100).sum().item()), "of", S.numel())

            if tqdm is not None:
                elapsed = time.time() - epoch_start
                iterable.set_postfix(loss=f"{(total_loss/max(1,n_steps)):.4f}", elapsed=fmt_time(elapsed))

        scheduler.step()
        train_loss = total_loss / max(1, n_steps)
        train_acc = (train_correct / train_total) if train_total > 0 else float("nan")
        train_miou, train_per_class_iou = train_iou_meter.compute(present_only=cfg.present_only_miou)

        need_fallback = (n_steps == 0) or (not np.isfinite(train_loss)) or (not np.isfinite(train_acc)) or np.isnan(train_miou)

        if need_fallback:
            print("[info] training stats invalid for this epoch -> running eval pass on train split")
            tr_eval = IoUMeter.run_eval_pass(
                encoder=encoder, seg_head=seg_head, loader=train_eval_loader,
                device=device, cfg=cfg, weights=weights, scaler_ctx=scaler_ctx, stem=stem
            )
            train_loss = tr_eval["loss"]
            train_acc  = tr_eval["acc"]
            train_miou = tr_eval["mIoU"]
            train_per_class_iou = tr_eval["per_class"]

        last_train = dict(train_loss=train_loss, train_acc=train_acc, train_mIoU=train_miou,
                          train_per_class_iou=train_per_class_iou)

        # -------- Validation --------
        if stem is not None: stem.eval()
        encoder.eval(); seg_head.eval()
        val_loss = 0.0; val_steps = 0; correct = 0; total = 0
        iou_meter = IoUMeter(num_classes=cfg.num_classes, ignore_index=-100, device=device).to(device)
        viter = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [val]", leave=False) if tqdm else val_loader

        with torch.no_grad():
            for P,S,M in viter:
                P = P.to(device).float(); S = S.to(device).long(); M = M.to(device).bool()
                pn_reg = 0.0
                if stem is not None:
                    F_loc, reg = stem(P[..., :3])
                    P = torch.cat([P, F_loc], dim=-1)
                    pn_reg = reg

                with scaler_ctx:
                    H = encoder(P, M)
                    H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
                    H = H.to(next(seg_head.parameters()).dtype)
                    logits = seg_head(H).clamp_(-30, 30)

                    C = logits.size(-1)
                    logits_flat = logits.reshape(-1, C).float()
                    target_flat = S.reshape(-1)
                    valid = target_flat != -100
                    if valid.any():
                        ce = (smooth_ce_loss(logits_flat[valid], target_flat[valid], cfg.label_smoothing)
                              if cfg.label_smoothing > 0 else
                              F.cross_entropy(logits_flat[valid], target_flat[valid], weight=weights, reduction="mean"))
                        if torch.isfinite(ce):
                            val_loss += float(ce.item()); val_steps += 1
                            preds = logits_flat[valid].argmax(-1)
                            correct += (preds == target_flat[valid]).sum().item()
                            total += int(valid.sum().item())
                            iou_meter.update(preds, target_flat[valid])

        val_loss = (val_loss / val_steps) if val_steps > 0 else float("nan")
        val_acc  = (correct / total) if total > 0 else float("nan")
        val_miou, per_class_iou = iou_meter.compute(present_only=cfg.present_only_miou)
        last_val = dict(val_loss=val_loss, val_acc=val_acc, val_mIoU=val_miou, val_per_class_iou=per_class_iou)

        epoch_time = time.time() - epoch_start
        print(
            f"[Epoch {epoch}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  train_mIoU={train_miou:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_mIoU={val_miou:.4f}  "
            f"time={fmt_time(epoch_time)}"
        )

        # Formatting handles 'nan' fine
        print(f"[train] IoU per class: {', '.join(f'{x:.3f}' if isinstance(x, float) else str(x) for x in train_per_class_iou)}")
        print(f"[val]   IoU per class: {', '.join(f'{x:.3f}' if isinstance(x, float) else str(x) for x in per_class_iou)}")

        # Save checkpoint each epoch + track best
        ep_path = save_ckpt(epoch, {**last_train, **last_val})
        print(f"[ckpt] saved {ep_path}")
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(torch.load(os.path.join(cfg.outdir, "latest.pth")), best_path)
            print(f"[ckpt] new best mIoU={best_miou:.4f} → {best_path}")

    total_time = time.time() - global_start
    print(f"[final] best_val_mIoU={best_miou:.4f}")
    print(f"[done] total_time={fmt_time(total_time)}  dir={cfg.outdir}")


if __name__ == "__main__":
    main()