import math
from pathlib import Path

def rotate_points_around_center(points, angle_rad):
    """
    将点列表绕其几何中心顺时针旋转 angle_rad 弧度。
    points: [(x1,y1), (x2,y2), ...] 归一化坐标
    """
    # 计算中心
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotated = []
    for x, y in points:
        dx = x - cx
        dy = y - cy
        # 顺时针旋转
        new_dx = dx * cos_a + dy * sin_a
        new_dy = -dx * sin_a + dy * cos_a
        rotated.append((cx + new_dx, cy + new_dy))
    return rotated

def rotate_obb_line(line, angle_deg=90):
    parts = line.strip().split()
    if len(parts) != 9:
        return line.strip()
    class_id = parts[0]
    coords = list(map(float, parts[1:]))
    points = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
    angle_rad = math.radians(angle_deg)
    new_points = rotate_points_around_center(points, angle_rad)
    flat = [coord for p in new_points for coord in p]
    return f"{class_id} " + " ".join(f"{c:.6f}" for c in flat)

def process_folder(input_dir, output_dir, angle_deg=90, overwrite=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in input_dir.glob("*.txt"):
        if not overwrite and (output_dir / txt_file.name).exists():
            print(f"跳过: {txt_file.name}")
            continue
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        new_lines = [rotate_obb_line(line, angle_deg) for line in lines if line.strip()]
        out_path = output_dir / txt_file.name
        with open(out_path, 'w') as f:
            f.write("\n".join(new_lines))
        print(f"已处理: {txt_file.name}")

if __name__ == "__main__":
    INPUT_DIR = r"D:\Study\10_dataset\HRSID_RBOX_CPLX73\labels\train_og"
    OUTPUT_DIR = r"D:\Study\10_dataset\HRSID_RBOX_CPLX73\labels\train"
    ANGLE_DEG = 90
    OVERWRITE = True

    process_folder(INPUT_DIR, OUTPUT_DIR, ANGLE_DEG, OVERWRITE)