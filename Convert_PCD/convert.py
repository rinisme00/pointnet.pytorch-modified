import os
import re
import glob
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from collections import Counter

# -----------------------------
# Utility Functions
# -----------------------------
def read_pcd(file_path):
    """Reads a .pcd file using Open3D and returns the point cloud as a NumPy array."""
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

def write_pts(points, output_file):
    """Writes the point coordinates (N x 3) to a .pts file."""
    np.savetxt(output_file, points, fmt="%.6f")

def write_seg(seg_data, output_file):
    """Writes segmentation labels to a .seg file.
       (Here, each point gets a single integer label.)
    """
    with open(output_file, "w") as f:
        for item in seg_data:
            f.write(str(item) + "\n")

# -----------------------------
# Processing Functions
# -----------------------------
def process_face_file(file_path, output_dir):
    """
    Processes a standard (face) .pcd file:
      - Reads the point cloud.
      - Extracts the face label from the filename (expects format: brick_partXX_Y.pcd).
      - **Subtracts 1 so that labels become zero-indexed.**
      - Writes a .pts file with the point coordinates.
      - Writes a .seg file with the repeated zero-indexed face label for each point.
    Returns the points array and the zero-indexed face label.
    """
    points = read_pcd(file_path)
    # Subtract 1 here so that a file named brick_partXX_Y.pcd produces label Y-1.
    m = re.match(r'.*_(\d+)\.pcd$', os.path.basename(file_path))
    label = int(m.group(1)) - 1 if m else 0

    base_name = os.path.basename(file_path).replace(".pcd", "")
    pts_file = os.path.join(output_dir, base_name + ".pts")
    seg_file = os.path.join(output_dir, base_name + ".seg")

    write_pts(points, pts_file)
    seg_labels = [label] * points.shape[0]
    write_seg(seg_labels, seg_file)
    
    return points, label

def assign_edge_labels_improved(edge_points, face_clouds):
    """
    Improved method for converting an edges .pcd file.
    For each edge point:
      - For each available face (already zero-indexed) compute the Euclidean distance via a KDTree.
      - Assign the label of the face cloud with the minimal distance.
    This distance–based selection should yield smoother, more consistent labels.
    """
    kdtrees = {}
    for label, points in face_clouds.items():
        if points.shape[0] > 0:
            kdtrees[label] = KDTree(points)
        else:
            kdtrees[label] = None

    assigned_labels = []
    for pt in edge_points:
        best_label = None
        best_distance = float('inf')
        for label, tree in kdtrees.items():
            if tree is not None:
                d, _ = tree.query(pt)
                if d < best_distance:
                    best_distance = d
                    best_label = label
        if best_label is None:
            best_label = -1
        assigned_labels.append(best_label)
    return assigned_labels

def process_edges_file(file_path, face_clouds, output_dir, threshold=0.05):
    """
    Processes an edges .pcd file:
      - Reads the edge point cloud.
      - Uses the improved assign_edge_labels_improved() function to assign a single label per edge point.
      - Writes the .pts file with point coordinates.
      - Writes the .seg file with the assigned labels.
    """
    edge_points = read_pcd(file_path)
    seg_labels = assign_edge_labels_improved(edge_points, face_clouds)
    base_name = os.path.basename(file_path).replace(".pcd", "")
    pts_file = os.path.join(output_dir, base_name + ".pts")
    seg_file = os.path.join(output_dir, base_name + ".seg")

    write_pts(edge_points, pts_file)
    write_seg(seg_labels, seg_file)

def assign_nearest_labels(unassigned_points, face_clouds):
    """
    For each unassigned point, finds the nearest neighbor among all segmented face points
    using a KDTree and assigns the corresponding face label.
    Returns a list of labels.
    """
    all_points = []
    all_labels = []
    for label, points in face_clouds.items():
        all_points.append(points)
        all_labels.extend([label] * points.shape[0])
    if not all_points:
        return [-1] * unassigned_points.shape[0]
    all_points = np.vstack(all_points)
    tree = KDTree(all_points)
    distances, indices = tree.query(unassigned_points)
    nearest_labels = [all_labels[idx] for idx in indices]
    return nearest_labels

def process_unassigned_file(file_path, face_clouds, output_dir):
    """
    Processes an unassigned points .pcd file:
      - Reads the point cloud.
      - Uses assign_nearest_labels() to assign a face label via the nearest neighbor.
      - Writes the .pts file with point coordinates.
      - Writes the .seg file with the assigned labels.
    """
    unassigned_points = read_pcd(file_path)
    seg_labels = assign_nearest_labels(unassigned_points, face_clouds)
    base_name = os.path.basename(file_path).replace(".pcd", "")
    pts_file = os.path.join(output_dir, base_name + ".pts")
    seg_file = os.path.join(output_dir, base_name + ".seg")

    write_pts(unassigned_points, pts_file)
    write_seg(seg_labels, seg_file)

# -----------------------------
# Main Function
# -----------------------------
def main():
    """
    Main routine:
      1. Scans for all .pcd files in the input directory.
      2. Groups files by brick part (e.g. brick_part01, brick_part02, etc.).
      3. Processes each group:
           - Face files are converted (using process_face_file, now zero-indexing labels).
           - Edge files (if present) are processed (using process_edges_file and the improved distance–based labeling).
           - Unassigned points (if present) are processed (using process_unassigned_file).
    """
    input_dir = "/storage/student6/dev/Convert_PCD/in"
    output_dir = "/storage/student6/dev/Convert_PCD/out/train"

    os.makedirs(output_dir, exist_ok=True)

    pcd_pattern = os.path.join(input_dir, "cube_part*.pcd")
    files = glob.glob(pcd_pattern)
    if not files:
        print("No .pcd files found in:", input_dir)
        return

    # Group files by brick part using a regex on the filename.
    part_groups = {}
    for file in files:
        m = re.match(r'(cube_part\d+)_.*\.pcd$', os.path.basename(file))
        if m:
            group = m.group(1)
            part_groups.setdefault(group, []).append(file)

    for part, files in part_groups.items():
        print("Processing:", part)
        face_clouds = {}
        # Process face files (those not containing "edges" or "unassigned_points")
        for file in files:
            if ("edges" not in file) and ("unassigned_points" not in file):
                points, label = process_face_file(file, output_dir)
                face_clouds[label] = points

        # Process edges files
        for file in files:
            if "edges" in file:
                print("  Processing edges file:", file)
                process_edges_file(file, face_clouds, output_dir)

        # Process unassigned points files
        for file in files:
            if "unassigned_points" in file:
                print("  Processing unassigned points file:", file)
                process_unassigned_file(file, face_clouds, output_dir)

if __name__ == "__main__":
    main()