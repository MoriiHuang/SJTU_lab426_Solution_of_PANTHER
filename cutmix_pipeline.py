import os
import random
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import rotate, zoom, map_coordinates, gaussian_filter
from skimage.transform import resize
from typing import Tuple, List, Optional

# --------- 工具函数 ----------
def save_nifti(image_np: np.ndarray, reference_image: sitk.Image, save_path: Path) -> None:
    """保存NIfTI图像"""
    image_sitk = sitk.GetImageFromArray(image_np.astype(np.float32))
    image_sitk.CopyInformation(reference_image)
    sitk.WriteImage(image_sitk, str(save_path))

def get_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    """提取二值mask的bounding box"""
    positions = np.where(mask > 0)
    if len(positions[0]) == 0:
        return None
    z_min, z_max = positions[0].min(), positions[0].max()
    y_min, y_max = positions[1].min(), positions[1].max()
    x_min, x_max = positions[2].min(), positions[2].max()
    return z_min, z_max, y_min, y_max, x_min, x_max

def elastic_deformation_3d(image: np.ndarray, alpha: float = 1000, sigma: float = 10) -> np.ndarray:
    """3D弹性变形"""
    shape = image.shape
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(z + dz, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def random_rotation_3d(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """3D随机旋转"""
    axes = [(0,1), (0,2), (1,2)]
    angle = random.uniform(-30, 30)
    axis = random.choice(axes)
    
    image = rotate(image, angle, axes=axis, reshape=False, mode='reflect')
    mask = rotate(mask, angle, axes=axis, reshape=False, mode='constant', cval=0)
    mask = (mask > 0.5).astype(mask.dtype)
    
    return image, mask

def random_flip_3d(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """3D随机翻转"""
    axis = random.randint(0, 2)
    image = np.flip(image, axis)
    mask = np.flip(mask, axis)
    return image, mask

def random_scale_3d(image: np.ndarray, mask: np.ndarray, min_scale: float = 0.8, max_scale: float = 1.2) -> Tuple[np.ndarray, np.ndarray]:
    """3D随机缩放"""
    scale = random.uniform(min_scale, max_scale)
    orig_shape = image.shape
    
    image = zoom(image, scale, order=1)
    mask = zoom(mask, scale, order=0)
    mask = (mask > 0.5).astype(mask.dtype)
    
    # 裁剪或填充回原始尺寸
    if scale < 1.0:
        # 填充
        new_image = np.zeros(orig_shape, dtype=image.dtype)
        new_mask = np.zeros(orig_shape, dtype=mask.dtype)
        
        start = [(orig_shape[i] - image.shape[i]) // 2 for i in range(3)]
        end = [start[i] + image.shape[i] for i in range(3)]
        
        new_image[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = image
        new_mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = mask
        return new_image, new_mask
    else:
        # 裁剪
        start = [(image.shape[i] - orig_shape[i]) // 2 for i in range(3)]
        end = [start[i] + orig_shape[i] for i in range(3)]
        return image[start[0]:end[0], start[1]:end[1], start[2]:end[2]], mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

def intensity_augmentation(image: np.ndarray) -> np.ndarray:
    """强度变换增强"""
    # 随机亮度调整
    image = image * random.uniform(0.8, 1.2)
    
    # 随机对比度调整
    mean_val = np.mean(image)
    image = np.clip((image - mean_val) * random.uniform(0.7, 1.3) + mean_val, image.min(), image.max())
    
    # 随机添加高斯噪声
    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(0, 0.05), image.shape)
        image = image + noise
    
    return np.clip(image, image.min(), image.max())

def augment_patch(patch_img: np.ndarray, patch_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对病灶patch进行随机变换增强"""
    # 强度变换
    patch_img = intensity_augmentation(patch_img)
    
    # 随机选择几何变换
    transforms = []
    
    # # 1. 随机旋转
    # if random.random() > 0.5:
    #     transforms.append(random_rotation_3d)
    
    # 2. 随机缩放
    if random.random() > 0.5:
        transforms.append(random_scale_3d)
    
    # 3. 随机翻转
    if random.random() > 0.5:
        transforms.append(random_flip_3d)
    
    # 4. 随机弹性变形
    # if random.random() > 0.3:
    #     def elastic_deform(img, msk):
    #         img = elastic_deformation_3d(img, alpha=random.uniform(10, 50), sigma=random.uniform(5, 10))
    #         msk = elastic_deformation_3d(msk, alpha=random.uniform(10, 50), sigma=random.uniform(5, 10))
    #         msk = (msk > 0.5).astype(msk.dtype)
    #         return img, msk
    #     transforms.append(elastic_deform)
    
    # 应用变换
    for transform in transforms:
        patch_img, patch_mask = transform(patch_img, patch_mask)
    
    return patch_img, patch_mask
def find_paste_position(pancreas_mask, patch_shape):
    """
    从 pancreas_mask 中选择一个合适的粘贴位置，使得 patch 位于胰腺区域内。
    
    参数:
        pancreas_mask (ndarray): 胰腺二值掩膜，1 表示胰腺区域
        patch_shape (tuple): (dz, dy, dx) 粘贴区域的形状

    返回:
        (z, y, x): 粘贴区域的起始坐标（左上角）
    """
    dz, dy, dx = patch_shape
    z_size, y_size, x_size = pancreas_mask.shape

    # 找到所有胰腺区域的索引
    pancreas_indices = np.argwhere(pancreas_mask > 0)

    # 过滤掉不能成为中心点的 index（防止 patch 越界）
    valid_centers = [
        (z, y, x)
        for z, y, x in pancreas_indices
        if dz//2 <= z < z_size - dz//2 and
           dy//2 <= y < y_size - dy//2 and
           dx//2 <= x < x_size - dx//2
    ]

    if not valid_centers:
        return None  # 没有合法粘贴点

    # 随机选一个中心点
    cz, cy, cx = random.choice(valid_centers)

    # 计算粘贴起始点（左上角）
    z = cz - dz // 2
    y = cy - dy // 2
    x = cx - dx // 2

    return z, y, x

def paste_patch(base_image: np.ndarray, base_mask: np.ndarray, 
                patch_img: np.ndarray, patch_mask: np.ndarray, 
                z: int, y: int, x: int) -> Tuple[np.ndarray, np.ndarray]:
    """粘贴病灶patch到目标图像"""
    dz, dy, dx = patch_mask.shape
    base_image[z:z+dz, y:y+dy, x:x+dx] = patch_img
    base_mask[z:z+dz, y:y+dy, x:x+dx] = patch_mask
    return base_image, base_mask

def blend_patch(base_image: np.ndarray, patch_img: np.ndarray, 
                patch_mask: np.ndarray, z: int, y: int, x: int, 
                blend_ratio: float = 0.7) -> np.ndarray:
    """混合病灶patch到目标图像，使边缘更自然"""
    dz, dy, dx = patch_mask.shape
    region = base_image[z:z+dz, y:y+dy, x:x+dx]
    
    # 创建混合权重
    weights = np.zeros_like(patch_mask, dtype=np.float32)
    for i in range(3):  # 在三个维度上计算距离
        dist = np.minimum(
            np.arange(dz)[:, None, None] / dz,
            np.minimum(
                np.arange(dy)[None, :, None] / dy,
                np.arange(dx)[None, None, :] / dx
            )
        )
        dist = np.minimum(dist, 1 - dist)
        weights = np.maximum(weights, dist)
    
    weights = np.clip(weights * 5, 0, 1)  # 调整权重曲线
    blend_weights = weights * blend_ratio
    
    # 混合图像
    blended = region * (1 - blend_weights) + patch_img * blend_weights
    base_image[z:z+dz, y:y+dy, x:x+dx] = blended
    
    return base_image

# --------- 主流程 ----------
def run_cutmix_pipeline(labeled_img_dir: str, labeled_label_dir: str, 
                       unlabeled_img_dir: str, pancreas_mask_dir: str, 
                       output_dir: str, prob_large: float = 0.6, 
                       max_tumors: int = 3) -> None:
    """
    主流程函数
    
    参数:
        labeled_img_dir: 已标注图像路径
        labeled_label_dir: 已标注标签路径
        unlabeled_img_dir: 未标注图像路径
        pancreas_mask_dir: 胰腺mask路径
        output_dir: 输出路径
        prob_large: 使用大病灶的概率
        max_tumors: 每个病例最多粘贴的病灶数量
    """
    output_img_dir = Path(output_dir) / "imagesTr"
    output_label_dir = Path(output_dir) / "labelsTr"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # 所有已标注病例
    labeled_cases = sorted([f.stem for f in Path(labeled_label_dir).glob("*.mha")])
    large_cases = [c for c in labeled_cases if any(x in c for x in ['10007', '10015', '10016', '10066', '10093'])]
    unlabeled_cases = sorted([f.stem for f in Path(unlabeled_img_dir).glob("*.mha")])
    
    for case_id in tqdm(unlabeled_cases, desc="Processing Unlabeled Cases"):
        # 读取未标注图像 + 胰腺mask
        unlabeled_image = sitk.ReadImage(f"{unlabeled_img_dir}/{case_id}.mha")
        pancreas_mask = sitk.ReadImage(f"{pancreas_mask_dir}/{case_id}.nii.gz")
        unlabeled_array = sitk.GetArrayFromImage(unlabeled_image)
        pancreas_array = sitk.GetArrayFromImage(pancreas_mask)
        
        new_img = np.copy(unlabeled_array)
        new_mask = np.zeros_like(unlabeled_array)
        
        # 随机决定要粘贴的病灶数量 (1到max_tumors)
        num_tumors = random.randint(1, max_tumors)
        used_patches = []
        
        for _ in range(num_tumors):
            # 随机采样一个病灶
            if random.random() < prob_large:
                tumor_case = random.choice(large_cases)
            else:
                tumor_case = random.choice([c for c in labeled_cases if c not in large_cases])
            
            tumor_image = sitk.ReadImage(f"{labeled_img_dir}/{tumor_case+'_0000'}.mha")
            tumor_mask = sitk.ReadImage(f"{labeled_label_dir}/{tumor_case}.mha")
            tumor_array = sitk.GetArrayFromImage(tumor_image)
            mask_array = sitk.GetArrayFromImage(tumor_mask)
            mask_array[mask_array!=1] = 0  # 确保mask是二值的
            
            # 获取病灶区域
            bbox = get_bbox(mask_array)
            if bbox is None:
                continue
                
            z1, z2, y1, y2, x1, x2 = bbox
            patch_img = tumor_array[z1:z2, y1:y2, x1:x2]
            patch_mask = mask_array[z1:z2, y1:y2, x1:x2]
            
            # 应用变换增强
            patch_img, patch_mask = augment_patch(patch_img, patch_mask)
            
            # 找粘贴位置 (确保不与已有病灶重叠太多)
            for _ in range(20):  # 最多尝试20次
                paste_loc = find_paste_position(pancreas_array, patch_mask.shape)
                if paste_loc is None:
                    continue
                    
                z, y, x = paste_loc
                
                
                used_patches.append((z, y, x, patch_mask.shape[0], patch_mask.shape[1], patch_mask.shape[2]))
                
                # 粘贴 (使用混合方式使边缘更自然)
                # new_img = blend_patch(new_img, patch_img, patch_mask, z, y, x)
                # new_mask = paste_patch(new_mask, np.zeros_like(new_mask), patch_mask, patch_mask, z, y, x)[0]
                new_img, new_mask = paste_patch(new_img, new_mask, patch_img, patch_mask, z, y, x)
                break  # 成功粘贴后跳出尝试循环
        # 保存
        save_nifti(new_img, unlabeled_image, output_img_dir / f"{case_id}.mha")
        save_nifti(new_mask, unlabeled_image, output_label_dir / f"{case_id.replace('_0000','')}.mha")
    
    print(f"✅ CutMix 数据增强完成，保存路径: {output_img_dir}, {output_label_dir}")

if __name__ == "__main__":
    labeled_img_dir = "ImagesTr"  # 替换为你的已标注图像路径
    labeled_label_dir = "LabelsTr"  # 替换为你的已标注标签路径
    unlabeled_img_dir = "ImagesTr_unlabeled"  # 替换为你的未标注图像路径
    pancreas_mask_dir = "pancreas_labels"  # 替换为你的胰腺mask路径
    output_dir = "CutMix"  # 替换为你的输出路径
    
    run_cutmix_pipeline(
        labeled_img_dir=labeled_img_dir,
        labeled_label_dir=labeled_label_dir,
        unlabeled_img_dir=unlabeled_img_dir,
        pancreas_mask_dir=pancreas_mask_dir,
        output_dir=output_dir,
        prob_large=0.6,
        max_tumors=1
    )