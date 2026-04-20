import math
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_xml_to_yolo_obb(xml_path, output_path=None, angle_sign=1, angle_offset=0):
    """
    将XML旋转框标注转换为YOLO OBB格式。

    Args:
        xml_path: XML文件路径
        output_path: 输出txt文件路径（默认与XML同目录，扩展名.txt）
        angle_sign: 角度符号修正，1表示使用原角度，-1表示取反
        angle_offset: 角度偏移（弧度），用于修正基准方向（如需要+π/2或-π/2）
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 图像尺寸
    size_elem = root.find('size')
    width = int(size_elem.find('width').text)
    height = int(size_elem.find('height').text)

    lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name.lower() != 'ship':  # 可根据需要修改类别
            continue

        robndbox = obj.find('robndbox')
        cx = float(robndbox.find('cx').text)
        cy = float(robndbox.find('cy').text)
        w = float(robndbox.find('w').text)
        h = float(robndbox.find('h').text)
        angle = float(robndbox.find('angle').text)

        # 应用符号修正和偏移
        angle = angle_sign * angle + angle_offset

        # 矩形四个角点相对于中心点的坐标（未旋转）
        half_w = w / 2.0
        half_h = h / 2.0
        rel_points = [
            (-half_w, -half_h),  # 左上
            (half_w, -half_h),  # 右上
            (half_w, half_h),  # 右下
            (-half_w, half_h)  # 左下
        ]

        # 旋转矩阵（逆时针）
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        abs_points = []
        for xr, yr in rel_points:
            x = cx + xr * cos_a - yr * sin_a
            y = cy + xr * sin_a + yr * cos_a
            abs_points.append((x, y))

        # 归一化
        norm_points = [(x / width, y / height) for x, y in abs_points]
        flat = [coord for point in norm_points for coord in point]

        # 类别ID（假设只有ship，映射为0）
        line = f"0 " + " ".join(f"{p:.6f}" for p in flat)
        lines.append(line)

    if output_path is None:
        output_path = Path(xml_path).with_suffix('.txt')
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))


def batch_convert(input_dir, output_dir=None, angle_sign=1, angle_offset=0):
    input_dir = Path(input_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_dir

    for xml_file in input_dir.glob("*.xml"):
        txt_file = output_dir / (xml_file.stem + ".txt")
        convert_xml_to_yolo_obb(xml_file, txt_file, angle_sign, angle_offset)
        print(f"Converted: {xml_file.name} -> {txt_file.name}")


if __name__ == "__main__":
    # ========== 用户配置 ==========
    XML_DIR = r"D:\Study\10_dataset\HRSID_RBox_Annotations\HRSID_RBox_Annotations"  # XML所在文件夹
    OUTPUT_DIR = r"D:\Study\10_dataset\HRSID_RBOX_CPLX73\labels\all"  # 输出YOLO标注文件夹
    ANGLE_SIGN = -1  # 尝试设为 -1 修正方向
    ANGLE_OFFSET =  -math.pi/2 # 若仍偏90°，可设为 math.pi/2 或 -math.pi/2
    # ============================

    batch_convert(XML_DIR, OUTPUT_DIR, ANGLE_SIGN, ANGLE_OFFSET)