#!/usr/bin/env python3
import os, sys
import numpy as np

PREFERRED_NAMES = ["Label", "label"]  # exact names first

def pick_label_field(vertex_prop_names):
    """
    Choose a label-like field:
    1) exact 'Label' / 'label'
    2) anything containing 'label' (case-insensitive)
       break ties by: shorter name first, and prefer names not starting with 'scalar_'
    """
    names = list(vertex_prop_names)
    for k in PREFERRED_NAMES:
        if k in names:
            return k
    cands = [n for n in names if 'label' in n.lower()]
    if not cands:
        return None
    # Prefer the cleanest-looking name
    cands.sort(key=lambda s: (len(s), s.startswith('scalar_')))
    return cands[0]

def normalise_labels(arr, tol=1e-6):
    """Robust label normalizer for CloudCompare variants."""
    arr = np.asarray(arr, dtype=np.float64)
    uniq = np.unique(np.round(arr, 6))

    # Map {-1, +1} → {0,1}
    if set(uniq).issubset({-1.0, 1.0}):
        arr = 0.5 * (arr + 1.0)

    # After possible map, snap to ints
    arr = np.rint(arr).astype(np.int32)
    uniq = np.unique(arr)

    # Common CloudCompare patterns
    if set(uniq).issubset({0, 2}):
        arr[arr == 2] = 1
    elif set(uniq).issubset({1, 2}):
        arr = arr - 1  # {1,2} → {0,1}
    elif uniq.size == 1 and uniq[0] == 2:
        arr[:] = 1

    return arr

def _parse_ascii_header(fp):
    """
    Parse ASCII PLY header, return:
    - n_vert: vertex count (int)
    - vprops: list of vertex property names in order
    """
    line = fp.readline()
    if not line or line.strip() != 'ply':
        raise ValueError('Not a PLY file (missing "ply" magic).')

    fmt = fp.readline()
    if not fmt or 'format ascii' not in fmt.strip():
        raise ValueError('Only ASCII PLY is supported by this reader.')

    vprops = []
    n_vert = None
    cur_elem = None

    while True:
        line = fp.readline()
        if not line:
            raise ValueError('Unexpected EOF while reading header.')
        s = line.strip()
        if s == 'end_header':
            break
        if not s:
            continue
        toks = s.split()

        if toks[0] == 'comment':
            continue

        if toks[:2] == ['element', 'vertex']:
            n_vert = int(toks[2])
            cur_elem = 'vertex'
            continue

        if toks[:2] == ['element', 'face']:
            cur_elem = 'face'  # we will ignore face section completely
            continue

        if toks[0] == 'element':
            cur_elem = toks[1]  # some other element; we will ignore its data later
            continue

        if toks[0] == 'property' and cur_elem == 'vertex':
            # property <type> <name>  OR  property list <count_type> <index_type> <name>
            # We only need the name (last token).
            vprops.append(toks[-1])

    if n_vert is None or not vprops:
        raise ValueError('No "element vertex" found or no vertex properties.')

    return n_vert, vprops

def convert_ply_to_pts_seg_ascii(ply_path):
    """
    ASCII-only, vertex-only reader. Reads exactly n_vert rows after header,
    ignores everything else (faces, edges, etc.).
    """
    basename = os.path.splitext(os.path.basename(ply_path))[0]
    out_dir = os.path.dirname(ply_path)
    pts_path = os.path.join(out_dir, f"{basename}.pts")
    seg_path = os.path.join(out_dir, f"{basename}.seg")

    with open(ply_path, 'r', encoding='utf-8', errors='replace') as f:
        n_vert, vprops = _parse_ascii_header(f)

        # Find indices for x,y,z and label
        try:
            xi, yi, zi = vprops.index('x'), vprops.index('y'), vprops.index('z')
        except ValueError:
            raise ValueError(f"{basename}: Missing x/y/z in vertex properties: {vprops}")

        label_field = pick_label_field(vprops)
        if label_field is None:
            raise ValueError(f"{basename}: No label-like field in vertex properties: {vprops}")
        li = vprops.index(label_field)

        # Read exactly n_vert non-empty lines as vertex rows
        xyz = np.empty((n_vert, 3), dtype=np.float32)
        lab = np.empty((n_vert,), dtype=np.float64)  # float first; normalize later

        read_rows = 0
        while read_rows < n_vert:
            line = f.readline()
            if not line:
                raise ValueError(f"{basename}: Unexpected EOF in vertex block at row {read_rows}.")
            s = line.strip()
            if not s:
                continue  # skip blank lines inside vertex block (rare)
            vals = s.split()
            # A strict PLY should have len(vals) == len(vprops)
            # Be tolerant as long as we can index the needed columns
            need = max(xi, yi, zi, li)
            if len(vals) <= need:
                raise ValueError(
                    f"{basename}: Vertex row {read_rows} too short ({len(vals)} cols) "
                    f"for required index {need}. Row: {vals}"
                )
            try:
                xyz[read_rows, 0] = float(vals[xi])
                xyz[read_rows, 1] = float(vals[yi])
                xyz[read_rows, 2] = float(vals[zi])
                lab[read_rows]    = float(vals[li])
            except Exception as e:
                raise ValueError(f"{basename}: Failed to parse vertex row {read_rows}: {e}")
            read_rows += 1

    labels = normalise_labels(lab)

    # Optional: enforce binary collapse for {0,2} style files (already handled in normalise_labels)
    # labels[labels == 2] = 1

    # Save
    np.savetxt(pts_path, xyz, fmt="%.6f")
    np.savetxt(seg_path, labels.astype(np.int32), fmt="%d")

    # Report
    u, c = np.unique(labels, return_counts=True)
    counts = dict(zip(u.tolist(), c.tolist()))
    print(f"✓ {basename}: field='{label_field}' → classes {counts}")

def batch_convert_ply_in_folder(folder_path):
    ply_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.ply')]
    if not ply_files:
        print("⚠ No .ply files found.")
        return
    for f in sorted(ply_files):
        path = os.path.join(folder_path, f)
        try:
            convert_ply_to_pts_seg_ascii(path)
        except Exception as e:
            print(f"⚠ Error on {f}: {e}")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "/storage/student6/cup_dataset/sauce"
    batch_convert_ply_in_folder(folder)
