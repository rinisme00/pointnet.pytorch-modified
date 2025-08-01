#!/usr/bin/env python3
"""
Chuyển 4 .pcd → out/*.pts & .seg   (Conda env đã kích hoạt)
"""

import sys, os, open3d as o3d, numpy as np

# -------- mapping file → nhãn --------
LABEL_MAP = {
    "full.pcd": 0,
    "broken.pcd": 0,
    "suface.pcd": 1,
    "fragment.pcd": 2,
}
N_POINTS = 2048          # ví dụ 2048 nếu muốn cố định
# -------------------------------------

in_dir  = sys.argv[1] if len(sys.argv) > 1 else "."
out_dir = sys.argv[2] if len(sys.argv) > 2 else "./out"
os.makedirs(out_dir, exist_ok=True)

for fname, lbl in LABEL_MAP.items():
    fpath = os.path.join(in_dir, fname)
    if not os.path.isfile(fpath):
        print(f"[WARN] Không thấy {fpath}, bỏ qua")
        continue

    pcd = o3d.io.read_point_cloud(fpath)
    pts = np.asarray(pcd.points, dtype=np.float32)

    if N_POINTS:
        choice = np.random.choice(len(pts), N_POINTS, replace=len(pts)<N_POINTS)
        pts = pts[choice]

    seg = np.full((pts.shape[0],), lbl, dtype=np.int32)

    base = os.path.splitext(fname)[0]
    np.savetxt(os.path.join(out_dir, base + ".pts"), pts, fmt="%.6f")
    np.savetxt(os.path.join(out_dir, base + ".seg"), seg, fmt="%d")

    print(f"[OK] {fname}  →  {base}.pts / .seg  ({pts.shape[0]} pts)")

print("Đã lưu:", os.path.abspath(out_dir))
