# 将DSM分辨率转换成1000*1000
import os
import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import reproject


# 批量处理文件夹内所有DSM文件
def resample_dsm(input_dsm_path, output_dsm_path, new_width=1000, new_height=1000):
    with rasterio.open(input_dsm_path) as src:
        # 获取原始数据的元数据
        meta = src.meta.copy()

        # 读取原始数据
        dsm_data = src.read(1)

        # 计算新的分辨率和转换矩阵
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        # 更新元数据
        meta.update({
            "width": new_width,
            "height": new_height,
            "transform": new_transform
        })

        # 重采样数据
        dsm_resampled = np.zeros((new_height, new_width), dtype=meta['dtype'])
        reproject(
            source=dsm_data,
            destination=dsm_resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=new_transform,
            dst_crs=src.crs,
            resampling=Resampling.bilinear
        )

        # 保存重采样后的数据
        with rasterio.open(output_dsm_path, 'w', **meta) as dst:
            dst.write(dsm_resampled, 1)

        return dsm_data, dsm_resampled


# 处理文件夹内所有 DSM 文件
def process_dsm_folder(folder_path, output_folder, new_width=1000, new_height=1000):
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

    for tif_file in tif_files:
        input_dsm_path = os.path.join(folder_path, tif_file)
        output_dsm_path = os.path.join(output_folder, f"resampled_{tif_file}")

        # 处理每个 DSM 文件
        try:
            dsm_data, dsm_resampled = resample_dsm(input_dsm_path, output_dsm_path, new_width, new_height)
            print(f"Processed: {tif_file}")

            # # 可视化
            # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            # axes[0].imshow(dsm_data, cmap='terrain', vmin=dsm_data.min(), vmax=dsm_data.max())
            # axes[0].set_title(f'Original DSM ({dsm_data.shape[0]}x{dsm_data.shape[1]})')
            # axes[0].axis('off')
            #
            # axes[1].imshow(dsm_resampled, cmap='terrain', vmin=dsm_resampled.min(), vmax=dsm_resampled.max())
            # axes[1].set_title(f'Resampled DSM ({new_height}x{new_width})')
            # axes[1].axis('off')
            #
            # plt.tight_layout()
            # plt.show()

        except Exception as e:
            print(f"Error processing {tif_file}: {e}")


# 设置文件夹路径
folder_path = "ISPRS_dataset/data1/2_DSM/"  # 替换为你的文件夹路径
output_folder = "ISPRS_dataset/data1/2_DSM/"  # 替换为输出文件夹路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 批量处理文件夹中的 DSM 文件
process_dsm_folder(folder_path, output_folder, new_width=1000, new_height=1000)