import os
import numpy as np
from plyfile import PlyData

PREFERRED_NAMES = ["Label", "label"]  # exact names first

def pick_label_field(vertex_dtype_names):
    names = list(vertex_dtype_names)
    # 1) exact preferred names
    for k in PREFERRED_NAMES:
        if k in names:
            return k
    # 2) otherwise, any field containing 'label' (case-insensitive)
    cands = [n for n in names if 'label' in n.lower()]
    if not cands:
        return None
    # Heuristic: choose the one with the shortest name (usually the final one),
    # break ties by putting names without 'scalar_' first.
    cands.sort(key=lambda s: (len(s), s.startswith('scalar_')))
    return cands[0]

def normalise_labels(arr, tol=1e-6):
    arr = np.asarray(arr, dtype=np.float64)
    # Map {-1, +1} → {0, 1}
    uniq = np.unique(np.round(arr, 6))
    if set(uniq).issubset({-1.0, 1.0}):
        arr = 0.5 * (arr + 1.0)
    # Snap tiny float noise to nearest integer
    arr = np.rint(arr).astype(np.int32)
    return arr

def convert_ply_to_pts_seg(ply_path):
    basename = os.path.splitext(os.path.basename(ply_path))[0]
    out_dir = os.path.dirname(ply_path)
    pts_path = os.path.join(out_dir, f"{basename}.pts")
    seg_path = os.path.join(out_dir, f"{basename}.seg")

    ply = PlyData.read(ply_path)
    v = ply['vertex'].data
    names = v.dtype.names

    label_field = pick_label_field(names)
    if label_field is None:
        raise ValueError(f"No label field found in: {ply_path}. Available: {names}")

    xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
    labels_raw = np.asarray(v[label_field])

    labels = normalise_labels(labels_raw)

    # (Optional) collapse to binary: map 2→1
    # labels[labels == 2] = 1

    # Safety: ensure labels are only 0/1/2 after normalisation
    bad = np.setdiff1d(np.unique(labels), np.array([0,1,2]))
    if bad.size:
        print(f"⚠ Warning: unexpected labels {bad} in {basename}.seg; they will be kept as-is.")

    np.savetxt(pts_path, xyz, fmt="%.6f")
    np.savetxt(seg_path, labels, fmt="%d")

    u, c = np.unique(labels, return_counts=True)
    print(f"✓ {basename}: field='{label_field}' → classes {dict(zip(u.tolist(), c.tolist()))}")

def batch_convert_ply_in_folder(folder_path):
    ply_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.ply')]
    if not ply_files:
        print("⚠ No .ply files found.")
        return
    for f in ply_files:
        try:
            convert_ply_to_pts_seg(os.path.join(folder_path, f))
        except Exception as e:
            print(f"⚠ Error on {f}: {e}")

if __name__ == "__main__":
    folder = "convert"
    batch_convert_ply_in_folder(folder)