import os
import shutil


def copy_labels_by_images(image_folder, label_source_folder, label_target_folder, image_extensions=None):
    """
    根据图片文件夹中的文件名，从标注源文件夹复制对应的txt文件到目标文件夹。

    参数：
        image_folder: 存放已筛选图片的文件夹路径
        label_source_folder: 存放所有YOLO标注txt的源文件夹路径
        label_target_folder: 要将标注复制到的目标文件夹路径
        image_extensions: 图片文件扩展名元组，默认支持常见格式
    """
    if image_extensions is None:
        image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

    # 创建目标文件夹（如果不存在）
    os.makedirs(label_target_folder, exist_ok=True)

    # 遍历图片文件夹中的所有文件
    for filename in os.listdir(image_folder):
        # 检查是否为图片文件（根据扩展名）
        if filename.lower().endswith(image_extensions):
            # 提取文件名（不含扩展名）
            basename = os.path.splitext(filename)[0]
            # 构造对应的标注文件名（假设扩展名为.txt）
            label_filename = basename + '.txt'
            source_path = os.path.join(label_source_folder, label_filename)
            target_path = os.path.join(label_target_folder, label_filename)

            # 检查源标注文件是否存在
            if os.path.isfile(source_path):
                shutil.copy2(source_path, target_path)  # copy2保留元数据
                print(f"✅ 已复制: {label_filename}")
            else:
                print(f"⚠️ 警告：未找到对应的标注文件 {label_filename}（图片：{filename}）")

    print("处理完成！")


if __name__ == "__main__":
    # ==================== 配置区域 ====================
    # 请根据你的实际路径修改以下变量
    image_folder = r"D:\Study\10_dataset\HRSID_CPLX_6535\images\val"  # 已筛选图片所在的文件夹
    label_source_folder = r"D:\Study\10_dataset\HRSID_YOLO\labels\all"  # 存放所有YOLO标注txt的文件夹
    label_target_folder = r"D:\Study\10_dataset\HRSID_CPLX_6535\labels\val"  # 要复制标注的目标文件夹
    # ==================================================

    copy_labels_by_images(image_folder, label_source_folder, label_target_folder)