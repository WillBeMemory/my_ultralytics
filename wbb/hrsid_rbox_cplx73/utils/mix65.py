import os
import random
import shutil
from pathlib import Path

def copy_images_with_ratio(src1, src2, dst, target_ratio=0.65):
    """
    将文件夹1全部复制，文件夹2随机复制一部分，使最终文件夹1的图片占总数的 target_ratio

    参数:
        src1: 文件夹1路径（全部复制）
        src2: 文件夹2路径（随机抽取）
        dst: 输出文件夹路径（自动创建）
        target_ratio: 目标占比（默认0.65）
    """
    src1_path = Path(src1)
    src2_path = Path(src2)
    dst_path = Path(dst)

    # 检查输入文件夹是否存在
    if not src1_path.exists() or not src1_path.is_dir():
        print(f"错误：文件夹1不存在或不是目录: {src1}")
        return
    if not src2_path.exists() or not src2_path.is_dir():
        print(f"错误：文件夹2不存在或不是目录: {src2}")
        return

    # 获取图片文件列表（支持常见图片格式）
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF')
    images1 = [f for f in src1_path.iterdir() if f.suffix in image_exts]
    images2 = [f for f in src2_path.iterdir() if f.suffix in image_exts]

    if not images1:
        print("文件夹1中没有找到图片文件。")
        return

    n1 = len(images1)
    # 计算所需文件夹2的图片数量，使得 n1 / (n1 + n2) ≈ target_ratio
    n2 = int(round(n1 * (1 - target_ratio) / target_ratio))

    if n2 == 0:
        print("目标比例过高，不需要文件夹2的图片。")
    else:
        print(f"文件夹1有 {n1} 张图片，需要从文件夹2中抽取 {n2} 张图片（目标占比 {target_ratio:.0%}）")

    # 检查文件夹2中的图片数量是否足够
    if n2 > len(images2):
        print(f"警告：文件夹2中只有 {len(images2)} 张图片，不足 {n2} 张，将全部复制。")
        n2 = len(images2)

    # 随机抽取
    if n2 > 0:
        selected_from2 = random.sample(images2, n2)
    else:
        selected_from2 = []

    # 创建目标文件夹
    dst_path.mkdir(parents=True, exist_ok=True)

    # 复制文件夹1的所有图片
    print("\n复制文件夹1的图片...")
    for img in images1:
        try:
            shutil.copy2(str(img), str(dst_path / img.name))
        except Exception as e:
            print(f"复制失败 {img.name}: {e}")

    # 复制选中的文件夹2图片
    if selected_from2:
        print("\n复制文件夹2的图片...")
        for img in selected_from2:
            try:
                shutil.copy2(str(img), str(dst_path / img.name))
            except Exception as e:
                print(f"复制失败 {img.name}: {e}")

    # 最终统计
    total = n1 + len(selected_from2)
    print(f"\n完成！共复制 {total} 张图片到 {dst}")
    print(f"文件夹1图片: {n1} 张 ({n1/total:.1%})")
    print(f"文件夹2图片: {len(selected_from2)} 张 ({len(selected_from2)/total:.1%})")

if __name__ == "__main__":
    # 请根据实际情况修改路径
    folder1 = r"D:\Study\10_dataset\HRSID_SPLIT\inshore"   # 全部复制的图片文件夹
    folder2 = r"D:\Study\10_dataset\HRSID_SPLIT\offshore"   # 随机抽取的图片文件夹
    folder3 = r"D:\Study\10_dataset\HRSID_SPLIT\MIX_in65_of35"   # 输出文件夹

    # 执行复制，可调整目标比例（默认0.65）
    copy_images_with_ratio(folder1, folder2, folder3, target_ratio=0.65)