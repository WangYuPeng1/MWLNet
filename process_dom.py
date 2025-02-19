import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import json

# 文件夹路径
folder_path = "./3_DOM_28"  # 替换为你的文件夹路径

# 获取文件夹中所有的 DOM_xxx.tif 文件和 DOM_xxx.json 文件
tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

# 确保每对 .tif 和 .json 文件是对应的
for tif_file in tif_files:
    # 提取编号（例如 DOM_1.tif 中的 1）
    base_name = tif_file.split('.')[0]  # 去除文件扩展名，得到 DOM_1
    json_file = f"{base_name}.json"  # 生成对应的 json 文件名

    # 如果存在对应的 json 文件，进行处理
    if json_file in json_files:
        tif_path = os.path.join(folder_path, tif_file)
        json_path = os.path.join(folder_path, json_file)

        # Step 1: 加载原始 tif 图像并转换为灰度
        original_image = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)

        # Step 2: 使用阈值和轮廓检测获取主要边界轮廓
        _, thresh = cv2.threshold(original_image, 1, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 假设只有一个主要轮廓，提取它并转换为 (x, y) 坐标对的列表
        boundary_points = [(int(point[0][0]), int(point[0][1])) for point in contours[0]]

        # Step 3: 定义标签映射（使用 RGB 颜色）
        label_to_color = {
            "lodging-wheats": (255, 0, 0),  # 红色
            "healthy-wheats": (0, 255, 0),  # 绿色
            "background": (255, 255, 255)  # 白色背景
        }

        # Step 4: 定义图像大小，创建 RGB 背景的标签图像
        canvas_size = original_image.shape[::-1]  # (宽, 高)
        labeled_image = Image.new("RGB", canvas_size, color=(255, 255, 255))  # 白色背景（RGB）
        boundary_free_image = Image.new("RGB", canvas_size, color=(255, 255, 255))  # 白色背景（RGB）

        # Step 5: 加载 JSON 文件并绘制多边形标签
        with open(json_path) as file:
            data = json.load(file)

        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"]
            polygon = [(int(x), int(y)) for x, y in points]
            color = label_to_color.get(label, (255, 255, 255))  # 默认使用白色背景

            # 在标签图像中绘制带边界和无边界的多边形
            draw = ImageDraw.Draw(labeled_image)
            draw.polygon(polygon, fill=color)

            draw_no_boundary = ImageDraw.Draw(boundary_free_image)
            draw_no_boundary.polygon(polygon, fill=color)

        # Step 6: 创建主边界掩膜并应用于标签图像
        mask_image = Image.new("L", canvas_size, color=0)  # 初始化掩膜为全黑
        mask_draw = ImageDraw.Draw(mask_image)
        mask_draw.polygon(boundary_points, fill=255)  # 主多边形区域设为 255（白色）

        # 将掩膜应用到图像，裁剪主多边形区域
        labeled_image = Image.composite(labeled_image, Image.new("RGB", canvas_size, color=(255, 255, 255)), mask_image)
        boundary_free_image = Image.composite(boundary_free_image, Image.new("RGB", canvas_size, color=(255, 255, 255)),
                                              mask_image)

        # Step 7: 保存最终的 RGB tif 文件
        labeled_output_path = os.path.join(folder_path, f"{base_name}_labeled_rgb.tif")
        boundary_free_output_path = os.path.join(folder_path, f"{base_name}_boundary_free_rgb.tif")

        labeled_image.save(labeled_output_path, format="TIFF")
        boundary_free_image.save(boundary_free_output_path, format="TIFF")

        print(f"Processed {base_name}...")
