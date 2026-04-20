import os
import sys
from pathlib import Path


def replace_classes_in_txt(file_path, old_names, new_name):
    """
    将标注文件中指定的类别名替换为新类别名
    :param file_path: 标注文件路径
    :param old_names: 需要替换的旧类别名列表，如 ['Dredger', 'Container']
    :param new_name: 新类别名，如 'ship'
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified = False
    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # DOTA格式：parts[0]是顶点数，parts[1:9]是坐标，parts[-2]是类别名，parts[-1]是难度标记
        if len(parts) < 10:
            # 格式不符合预期，保留原样
            new_lines.append(line)
            continue
        class_name = parts[-2]
        if class_name in old_names:
            parts[-2] = new_name
            modified = True
        new_lines.append(' '.join(parts))

    if modified:
        # 写回原文件（建议先备份）
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print(f"已更新: {file_path}")
    else:
        print(f"无变化: {file_path}")


def main():
    # 设置标注文件所在根目录（根据你的实际路径修改）
    root_dir = r"D:\Study\10_dataset\SRSDD\labels\test_original"  # 通常标签在 labels 下，包含 train_original 和 val_original 子目录
    if not os.path.exists(root_dir):
        print(f"目录不存在: {root_dir}")
        sys.exit(1)

    old_names = ['Dredger', 'Container','Cell-Container','Fishing','LawEnforce','ore-oil']  # 需要替换的旧类别名
    new_name = 'ship'  # 新类别名

    # 递归查找所有 .txt 文件
    for txt_file in Path(root_dir).rglob("*.txt"):
        print(f"处理: {txt_file}")
        replace_classes_in_txt(txt_file, old_names, new_name)


if __name__ == "__main__":
    main()