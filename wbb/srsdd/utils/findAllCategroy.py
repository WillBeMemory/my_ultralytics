import os
from pathlib import Path

# 修改为你的标签根目录（例如 D:/Study/10_dataset/SRSDD/labels）
label_root = Path(r"D:\Study\10_dataset\SRSDD\labels")

# 递归查找所有 .txt 文件
all_classes = set()
for txt_file in label_root.rglob("*.txt"):
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # DOTA 格式：最后一列是难度，倒数第二列是类别名
            if len(parts) >= 10:
                class_name = parts[-2]
                all_classes.add(class_name)

print("找到的所有类别：")
for i, cls in enumerate(sorted(all_classes)):
    print(f"{i+1}. {cls}")
print(f"类别总数：{len(all_classes)}")