import os
import numpy as np
from plyfile import PlyData

def convert_ply_to_pts_seg(ply_path):
    basename = os.path.splitext(os.path.basename(ply_path))[0]
    pts_path = os.path.join(os.path.dirname(ply_path), f"{basename}.pts")
    seg_path = os.path.join(os.path.dirname(ply_path), f"{basename}.seg")

    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']

    # Lấy tọa độ
    xyz = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)

    # Tìm trường chứa nhãn
    label_field = None
    for name in vertices.data.dtype.names:
        if 'label' in name.lower():
            label_field = name
            break
    if label_field is None:
        raise ValueError(f"No label field found in: {ply_path}")

    labels = np.array(vertices[label_field], dtype=int)

    # Ghi file
    np.savetxt(pts_path, xyz, fmt="%.6f")
    np.savetxt(seg_path, labels, fmt="%d")
    print(f"✓ Converted: {os.path.basename(ply_path)} → {basename}.pts / .seg")

def batch_convert_ply_in_folder(folder_path):
    ply_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]
    if not ply_files:
        print("⚠ Không tìm thấy file .ply nào trong thư mục.")
        return

    for ply_file in ply_files:
        try:
            convert_ply_to_pts_seg(os.path.join(folder_path, ply_file))
        except Exception as e:
            print(f"⚠ Lỗi khi xử lý {ply_file}: {e}")

# === SỬ DỤNG ===
if __name__ == '__main__':
    folder = "convert"  # 🔁 Thay bằng thư mục chứa các .ply
    batch_convert_ply_in_folder(folder)
