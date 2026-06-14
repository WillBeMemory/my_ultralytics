"""
对比官方 SSDD 数据集与本地 SSDD 数据集的划分差异
"""
import json
import os
from pathlib import Path

# 官方 SSDD COCO 标注
OFFICIAL_ANN = Path(r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\Official-SSDD-OPEN\Official-SSDD-OPEN\BBox_SSDD\coco_style\annotations")

# 本地 SSDD YOLO 格式
LOCAL_SSDD = Path(r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\SSDD")

# 1. 读取官方划分
with open(OFFICIAL_ANN / "train.json", "r") as f:
    official_train = json.load(f)
with open(OFFICIAL_ANN / "test.json", "r") as f:
    official_test = json.load(f)

official_train_files = set()
for img in official_train["images"]:
    name = Path(img["file_name"]).stem  # e.g., "000002"
    official_train_files.add(name)

official_test_files = set()
for img in official_test["images"]:
    name = Path(img["file_name"]).stem
    official_test_files.add(name)

print(f"Official SSDD:")
print(f"  train: {len(official_train_files)} images")
print(f"  test:  {len(official_test_files)} images")
print(f"  total: {len(official_train_files) + len(official_test_files)} images")
print(f"  overlap: {len(official_train_files & official_test_files)} images")

# 验证官方划分规则：尾数1和9为test
test_by_rule = set()
train_by_rule = set()
for name in official_train_files | official_test_files:
    last_digit = name[-1]
    if last_digit in ('1', '9'):
        test_by_rule.add(name)
    else:
        train_by_rule.add(name)

print(f"\nRule check (last digit 1,9 = test):")
print(f"  Rule test set: {len(test_by_rule)}")
print(f"  Official test set: {len(official_test_files)}")
print(f"  Match: {test_by_rule == official_test_files}")

# 2. 读取本地划分
local_train_files = set()
for f in os.listdir(LOCAL_SSDD / "images" / "train"):
    local_train_files.add(Path(f).stem)

local_val_files = set()
for f in os.listdir(LOCAL_SSDD / "images" / "val"):
    local_val_files.add(Path(f).stem)

local_test_files = set()
for f in os.listdir(LOCAL_SSDD / "images" / "test"):
    local_test_files.add(Path(f).stem)

print(f"\nLocal SSDD:")
print(f"  train: {len(local_train_files)} images")
print(f"  val:   {len(local_val_files)} images")
print(f"  test:  {len(local_test_files)} images")
print(f"  total: {len(local_train_files) + len(local_val_files) + len(local_test_files)} images")

# 3. 交叉对比
all_official = official_train_files | official_test_files
all_local = local_train_files | local_val_files | local_test_files

print(f"\nOfficial total: {len(all_official)}, Local total: {len(all_local)}")
print(f"Same set of images: {all_official == all_local}")

# 转移矩阵
print(f"\nTransfer Matrix:")
print(f"{'':>20} {'-> Local train':>15} {'-> Local val':>15} {'-> Local test':>15} {'Total':>10}")
print(f"{'Official train':>20} {len(official_train_files & local_train_files):>15} {len(official_train_files & local_val_files):>15} {len(official_train_files & local_test_files):>15} {len(official_train_files):>10}")
print(f"{'Official test':>20} {len(official_test_files & local_train_files):>15} {len(official_test_files & local_val_files):>15} {len(official_test_files & local_test_files):>15} {len(official_test_files):>10}")

# 检查本地 test 是否完全来自官方 test
local_test_from_official_train = local_test_files & official_train_files
local_test_from_official_test = local_test_files & official_test_files
print(f"\nLocal test set composition:")
print(f"  From official train: {len(local_test_from_official_train)}")
print(f"  From official test:  {len(local_test_from_official_test)}")

# 检查本地 val 是否完全来自官方 test
local_val_from_official_train = local_val_files & official_train_files
local_val_from_official_test = local_val_files & official_test_files
print(f"\nLocal val set composition:")
print(f"  From official train: {len(local_val_from_official_train)}")
print(f"  From official test:  {len(local_val_from_official_test)}")
