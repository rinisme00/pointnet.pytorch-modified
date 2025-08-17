#!/usr/bin/env python3
import os, argparse
import numpy as np

def must_pair(base_path: str):
    pts, seg = base_path + ".pts", base_path + ".seg"
    if not (os.path.exists(pts) and os.path.exists(seg)):
        raise FileNotFoundError(f"Missing pair: {pts} / {seg}")
    return pts, seg

def resolve_base(src: str, file_arg: str) -> str:
    """
    --file can be:
      - basename without extension (e.g., 'fragment') -> joined with --src
      - path without extension (e.g., /x/y/fragment)
      - path to .pts or .seg (we strip the extension)
    """
    base = file_arg
    if base.endswith(".pts") or base.endswith(".seg"):
        base = os.path.splitext(base)[0]
    if not os.path.isabs(base):
        base = os.path.join(src, base)
    return base

def save_pts_seg(base_out: str, P: np.ndarray, S: np.ndarray):
    os.makedirs(os.path.dirname(base_out), exist_ok=True)
    np.savetxt(base_out + ".pts", P, fmt="%.6f")
    np.savetxt(base_out + ".seg", S.astype(np.int64), fmt="%d")

def label_stats(S: np.ndarray):
    uniq, cnt = np.unique(S, return_counts=True)
    return dict(zip(uniq.tolist(), cnt.tolist()))

def main():
    ap = argparse.ArgumentParser(
        description="Split ONE .pts/.seg pair by points into train/test according to a ratio."
    )
    ap.add_argument("--src", required=True,
                    help="Root folder for relative --file. Not used if --file is absolute.")
    ap.add_argument("--dst", required=True,
                    help="Output dataset root; will create dst/split/train and dst/split/test")
    ap.add_argument("--file", required=True,
                    help="Basename OR path to the .pts/.seg pair to split "
                         "(e.g. 'fragment' or '/abs/path/fragment' or '/abs/path/fragment.pts').")
    ap.add_argument("--ratio", type=float, default=0.5,
                    help="Fraction of points to go into TRAIN (rest go to TEST). Default 0.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_name", type=str, default=None,
                    help="Output stem name in train/ (default: '<basename>_train').")
    ap.add_argument("--test_name", type=str, default=None,
                    help="Output stem name in test/  (default: '<basename>_test').")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    out_train = os.path.join(dst, "split", "train")
    out_test  = os.path.join(dst, "split", "test")
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    base = resolve_base(src, args.file)
    pts_path, seg_path = must_pair(base)

    # Derive basename and output stems
    basename = os.path.basename(base)
    train_stem = args.train_name or f"{basename}_train"
    test_stem  = args.test_name  or f"{basename}_test"

    # Load
    P = np.loadtxt(pts_path, dtype=np.float32, ndmin=2)
    S = np.loadtxt(seg_path, dtype=np.int64,   ndmin=1)
    if P.ndim != 2 or S.ndim != 1 or P.shape[0] != S.shape[0]:
        raise ValueError(f"Shape mismatch: P{P.shape} vs S{S.shape} in {pts_path}/{seg_path}")

    # Split indices
    n = P.shape[0]
    k = int(round(max(0.0, min(1.0, args.ratio)) * n))
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    idx_train, idx_test = idx[:k], idx[k:]

    P_tr, S_tr = P[idx_train], S[idx_train]
    P_te, S_te = P[idx_test],  S[idx_test]

    # Save
    save_pts_seg(os.path.join(out_train, train_stem), P_tr, S_tr)
    save_pts_seg(os.path.join(out_test,  test_stem),  P_te, S_te)

    # Stats
    print("\n[Split one file done]")
    print("Source pair:", pts_path, "|", seg_path)
    print("Train out :", os.path.join(out_train, train_stem + ".{pts,seg}"))
    print("Test  out :", os.path.join(out_test,  test_stem  + ".{pts,seg}"))
    print(f"Counts -> train: {len(S_tr)}  test: {len(S_te)}")
    print("Label stats -> train:", label_stats(S_tr), " | test:", label_stats(S_te))

if __name__ == "__main__":
    main()
