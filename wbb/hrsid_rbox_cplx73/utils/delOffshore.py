import os
import sys
import glob
from pathlib import Path

def delete_matching_files(source_dir, target_dir, extensions=None):
    """
    删除目标文件夹中与源文件夹图片同名的文件

    参数:
        source_dir: 源文件夹路径（图片所在）
        target_dir: 目标文件夹路径（将删除其中的同名文件）
        extensions: 图片扩展名列表，默认常见图片格式
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF']

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # 检查文件夹是否存在
    if not source_path.exists() or not source_path.is_dir():
        print(f"错误：源文件夹不存在或不是目录: {source_dir}")
        return
    if not target_path.exists() or not target_path.is_dir():
        print(f"错误：目标文件夹不存在或不是目录: {target_dir}")
        return

    # 收集源文件夹中的图片文件名（包含扩展名）
    image_files = []
    for ext in extensions:
        image_files.extend(source_path.glob(f"*{ext}"))

    if not image_files:
        print("源文件夹中未找到任何图片文件。")
        return

    print(f"在源文件夹中找到 {len(image_files)} 个图片文件。")

    # 查找目标文件夹中要删除的文件
    to_delete = []
    for img in image_files:
        target_file = target_path / img.name
        if target_file.exists():
            to_delete.append(target_file)

    if not to_delete:
        print("目标文件夹中没有找到同名文件。")
        return

    print("\n将要删除以下文件：")
    for f in to_delete:
        print(f"  {f.name}")

    # 请求确认
    confirm = input(f"\n确认删除以上 {len(to_delete)} 个文件？(yes/no): ").strip().lower()
    if confirm != 'yes':
        print("操作已取消。")
        return

    # 执行删除
    for f in to_delete:
        try:
            f.unlink()
            print(f"已删除: {f.name}")
        except Exception as e:
            print(f"删除失败 {f.name}: {e}")

if __name__ == "__main__":
    # 设置路径
    # 源文件夹：当前文件夹（脚本所在目录）
    source_dir = r"D:\Study\10_dataset\HRSID_SPLIT\offshore"
    # 目标文件夹：需要删除同名文件的文件夹（请根据实际情况修改）
    target_dir = r"D:\Study\10_dataset\HRSID_SPLIT\inshore"

    # 如果通过命令行参数传递目标文件夹，可以取消注释下面的代码
    # if len(sys.argv) > 1:
    #     target_dir = sys.argv[1]

    delete_matching_files(source_dir, target_dir)