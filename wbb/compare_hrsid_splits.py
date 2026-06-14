"""比较原始HRSID数据集(COCO格式)与HRSID_YOLO数据集的划分逻辑差异"""

import json
import os

# ============ 路径定义 ============
ORIGINAL_BASE = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_JPG"
YOLO_BASE = r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO"

# ============ 1. 读取原始COCO标注获取train/test图片列表 ============
print("=" * 70)
print("步骤1: 读取原始COCO标注文件")
print("=" * 70)

with open(os.path.join(ORIGINAL_BASE, "annotations", "train2017.json"), "r", encoding="utf-8") as f:
    coco_train = json.load(f)
with open(os.path.join(ORIGINAL_BASE, "annotations", "test2017.json"), "r", encoding="utf-8") as f:
    coco_test = json.load(f)

original_train_names = set()
for img in coco_train["images"]:
    fname = os.path.splitext(img["file_name"])[0]
    original_train_names.add(fname)

original_test_names = set()
for img in coco_test["images"]:
    fname = os.path.splitext(img["file_name"])[0]
    original_test_names.add(fname)

print(f"原始 train2017.json 中图片数: {len(original_train_names)}")
print(f"原始 test2017.json 中图片数: {len(original_test_names)}")
print(f"原始总计: {len(original_train_names) + len(original_test_names)}")

# 检查原始train/test是否有重叠
overlap_original = original_train_names & original_test_names
print(f"原始train与test重叠数: {len(overlap_original)}")

# ============ 2. 读取HRSID_YOLO的train/val图片列表 ============
print("\n" + "=" * 70)
print("步骤2: 读取HRSID_YOLO数据集图片列表")
print("=" * 70)

yolo_train_dir = os.path.join(YOLO_BASE, "images", "train")
yolo_val_dir = os.path.join(YOLO_BASE, "images", "val")

yolo_train_files = os.listdir(yolo_train_dir)
yolo_val_files = os.listdir(yolo_val_dir)

yolo_train_names = set(os.path.splitext(f)[0] for f in yolo_train_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')))
yolo_val_names = set(os.path.splitext(f)[0] for f in yolo_val_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')))

print(f"HRSID_YOLO train 图片数: {len(yolo_train_names)}")
print(f"HRSID_YOLO val 图片数: {len(yolo_val_names)}")
print(f"HRSID_YOLO 总计: {len(yolo_train_names) + len(yolo_val_names)}")

# 检查YOLO train/val是否有重叠
overlap_yolo = yolo_train_names & yolo_val_names
print(f"HRSID_YOLO train与val重叠数: {len(overlap_yolo)}")

# ============ 3. 交叉比较 ============
print("\n" + "=" * 70)
print("步骤3: 交叉比较两个数据集的划分")
print("=" * 70)

# 原始train → HRSID_YOLO各子集
orig_train_to_yolo_train = original_train_names & yolo_train_names
orig_train_to_yolo_val = original_train_names & yolo_val_names
orig_train_not_in_yolo = original_train_names - yolo_train_names - yolo_val_names

# 原始test → HRSID_YOLO各子集
orig_test_to_yolo_train = original_test_names & yolo_train_names
orig_test_to_yolo_val = original_test_names & yolo_val_names
orig_test_not_in_yolo = original_test_names - yolo_train_names - yolo_val_names

# HRSID_YOLO中不在原始数据集中的图片
yolo_train_not_in_orig = yolo_train_names - original_train_names - original_test_names
yolo_val_not_in_orig = yolo_val_names - original_train_names - original_test_names

print("\n--- 原始 train → HRSID_YOLO ---")
print(f"  原始train → HRSID_YOLO train: {len(orig_train_to_yolo_train)}")
print(f"  原始train → HRSID_YOLO val:   {len(orig_train_to_yolo_val)}")
print(f"  原始train → 不在HRSID_YOLO中:  {len(orig_train_not_in_yolo)}")

print("\n--- 原始 test → HRSID_YOLO ---")
print(f"  原始test → HRSID_YOLO train: {len(orig_test_to_yolo_train)}")
print(f"  原始test → HRSID_YOLO val:   {len(orig_test_to_yolo_val)}")
print(f"  原始test → 不在HRSID_YOLO中:  {len(orig_test_not_in_yolo)}")

print("\n--- HRSID_YOLO中不在原始数据集中的图片 ---")
print(f"  HRSID_YOLO train 中不在原始数据集: {len(yolo_train_not_in_orig)}")
print(f"  HRSID_YOLO val 中不在原始数据集:   {len(yolo_val_not_in_orig)}")

# ============ 4. 汇总矩阵 ============
print("\n" + "=" * 70)
print("步骤4: 划分转移矩阵")
print("=" * 70)
print(f"{'':>25} {'→ HRSID_YOLO train':>20} {'→ HRSID_YOLO val':>20} {'→ 不在YOLO中':>15}")
print(f"{'原始 train (3642)':>25} {len(orig_train_to_yolo_train):>20} {len(orig_train_to_yolo_val):>20} {len(orig_train_not_in_yolo):>15}")
print(f"{'原始 test (1962)':>25} {len(orig_test_to_yolo_train):>20} {len(orig_test_to_yolo_val):>20} {len(orig_test_not_in_yolo):>15}")

# ============ 5. 列出被移动的示例文件 ============
print("\n" + "=" * 70)
print("步骤5: 被移动的示例文件名")
print("=" * 70)

print("\n--- 从原始test移到HRSID_YOLO train的示例(最多20个) ---")
moved_test_to_train = sorted(orig_test_to_yolo_train)
for name in moved_test_to_train[:20]:
    print(f"  {name}")
if len(moved_test_to_train) > 20:
    print(f"  ... 共 {len(moved_test_to_train)} 个")

print("\n--- 从原始train移到HRSID_YOLO val的示例(最多20个) ---")
moved_train_to_val = sorted(orig_train_to_yolo_val)
for name in moved_train_to_val[:20]:
    print(f"  {name}")
if len(moved_train_to_val) > 20:
    print(f"  ... 共 {len(moved_train_to_val)} 个")

# ============ 6. 分析划分逻辑 ============
print("\n" + "=" * 70)
print("步骤6: 划分逻辑分析")
print("=" * 70)

total_yolo_train = len(yolo_train_names)
total_yolo_val = len(yolo_val_names)
train_ratio = total_yolo_train / (total_yolo_train + total_yolo_val) * 100
val_ratio = total_yolo_val / (total_yolo_train + total_yolo_val) * 100

print(f"\nHRSID_YOLO 划分比例: train {total_yolo_train} ({train_ratio:.1f}%) / val {total_yolo_val} ({val_ratio:.1f}%)")
print(f"近似比例: train:val ≈ {total_yolo_train//total_yolo_val if total_yolo_val > 0 else 'N/A'}:1")

# 检查是否是简单的重新随机划分
all_orig = original_train_names | original_test_names
all_yolo = yolo_train_names | yolo_val_names
if all_orig == all_yolo:
    print("\n✅ 两个数据集包含完全相同的图片（只是重新划分）")
else:
    missing = all_orig - all_yolo
    extra = all_yolo - all_orig
    print(f"\n❌ 两个数据集图片不完全相同")
    print(f"  原始有但YOLO没有: {len(missing)}")
    print(f"  YOLO有但原始没有: {len(extra)}")

# 分析原始test中移到YOLO train的比例
if len(original_test_names) > 0:
    test_to_yolo_train_ratio = len(orig_test_to_yolo_train) / len(original_test_names) * 100
    test_to_yolo_val_ratio = len(orig_test_to_yolo_val) / len(original_test_names) * 100
    print(f"\n原始test图片去向:")
    print(f"  → HRSID_YOLO train: {len(orig_test_to_yolo_train)}/{len(original_test_names)} ({test_to_yolo_train_ratio:.1f}%)")
    print(f"  → HRSID_YOLO val:   {len(orig_test_to_yolo_val)}/{len(original_test_names)} ({test_to_yolo_val_ratio:.1f}%)")

# 分析原始train中移到YOLO val的比例
if len(original_train_names) > 0:
    train_to_yolo_val_ratio = len(orig_train_to_yolo_val) / len(original_train_names) * 100
    train_to_yolo_train_ratio = len(orig_train_to_yolo_train) / len(original_train_names) * 100
    print(f"\n原始train图片去向:")
    print(f"  → HRSID_YOLO train: {len(orig_train_to_yolo_train)}/{len(original_train_names)} ({train_to_yolo_train_ratio:.1f}%)")
    print(f"  → HRSID_YOLO val:   {len(orig_train_to_yolo_val)}/{len(original_train_names)} ({train_to_yolo_val_ratio:.1f}%)")

# ============ 7. 尝试分析文件名规律 ============
print("\n" + "=" * 70)
print("步骤7: 文件名规律分析（检查是否按场景/编号划分）")
print("=" * 70)

# 检查文件名前缀规律
def get_prefix(name, sep='_'):
    parts = name.split(sep)
    if len(parts) > 1:
        return parts[0]
    return name[:4]  # 取前4个字符

# 分析原始test→YOLO train的文件名前缀
if moved_test_to_train:
    prefixes = {}
    for name in moved_test_to_train:
        p = get_prefix(name)
        prefixes[p] = prefixes.get(p, 0) + 1
    print(f"\n原始test→YOLO train 的文件名前缀分布:")
    for p, c in sorted(prefixes.items(), key=lambda x: -x[1])[:15]:
        print(f"  前缀 '{p}': {c} 个")

# 分析原始train→YOLO val的文件名前缀
if moved_train_to_val:
    prefixes2 = {}
    for name in moved_train_to_val:
        p = get_prefix(name)
        prefixes2[p] = prefixes2.get(p, 0) + 1
    print(f"\n原始train→YOLO val 的文件名前缀分布:")
    for p, c in sorted(prefixes2.items(), key=lambda x: -x[1])[:15]:
        print(f"  前缀 '{p}': {c} 个")

# 检查文件名中的数字编号是否连续
def extract_number(name):
    import re
    nums = re.findall(r'\d+', name)
    if nums:
        return int(nums[-1])
    return None

print("\n--- 文件名编号范围分析 ---")
for label, name_set in [
    ("原始train→YOLO train", orig_train_to_yolo_train),
    ("原始train→YOLO val", orig_train_to_yolo_val),
    ("原始test→YOLO train", orig_test_to_yolo_train),
    ("原始test→YOLO val", orig_test_to_yolo_val),
]:
    numbers = [extract_number(n) for n in name_set]
    numbers = [n for n in numbers if n is not None]
    if numbers:
        print(f"  {label}: 编号范围 [{min(numbers)}, {max(numbers)}], 均值={sum(numbers)/len(numbers):.1f}")

print("\n" + "=" * 70)
print("分析完成！")
print("=" * 70)
