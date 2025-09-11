#!/usr/bin/env python3
"""
医学影像处理Pipeline脚本
功能：批量处理输入文件夹中的医学影像，生成胰腺分割结果和变换后的影像
用法：python pipeline.py --input_dir /path/to/input --output_dir /path/to/output
"""

import argparse
import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from mrsegmentator import inference
from typing import Dict, Tuple

def apply_ct_transforms(ct_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    对CT数据应用三种变换：
    1. 归一化 (Normalize)
    2. 平方映射 (x^2)
    3. 立方根映射 (x^(1/3))
    """
    clipped = ct_data
    normalized = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-6)
    
    return {
        'normalized': normalized,
        'squared': np.power(normalized, 2),
        'cube_root': np.power(normalized, 1/3)
    }

def resample_img(image: sitk.Image, out_spacing: Tuple[float, float, float], is_label: bool = False) -> sitk.Image:
    """
    重采样图像到指定spacing
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, out_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(image)

def upsample_mask(
    mask_array: np.ndarray, 
    original_spacing: Tuple[float, float, float], 
    target_spacing: Tuple[float, float, float],
    fill_holes: bool = False
) -> np.ndarray:
    """
    上采样mask到目标spacing
    """
    mask_img = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    mask_img.SetSpacing(original_spacing)
    
    resampled = resample_img(mask_img, target_spacing, is_label=True)
    resampled_array = sitk.GetArrayFromImage(resampled)
    
    if fill_holes:
        resampled_array = binary_fill_holes(resampled_array)
    
    return resampled_array

def normalize_and_resample(input_path: str) -> sitk.Image:
    """
    标准化并重采样图像
    """
    image = sitk.ReadImage(input_path)
    # 这里可以添加您的标准化逻辑
    return image

def process_single_image(input_path: Path, output_dir: Path):
    """
    处理单个医学影像文件
    """
    # 创建必要的子目录
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Processing: {input_path.name}")
    
    try:
        # 读取原始图像
        itk_image = sitk.ReadImage(str(input_path))
        ct_array = sitk.GetArrayFromImage(itk_image)  # 形状为 (Z,Y,X)
        
        # 应用三种变换并保存
        transforms = apply_ct_transforms(ct_array)
        for i, (name, data) in enumerate(transforms.items()):
            new_img = sitk.GetImageFromArray(data.astype(np.float32))
            new_img.CopyInformation(itk_image)
            sitk.WriteImage(
                new_img, 
                str(temp_dir / f'{input_path.stem}_000{i}.mha')
            )
        
        # 生成低分辨率版本用于分割
        out_spacing = [s * 2 for s in itk_image.GetSpacing()]
        image_low_res = resample_img(itk_image, out_spacing=out_spacing, is_label=False)
        lowres_path = temp_dir / f"{input_path.stem}_0000.nii.gz"
        sitk.WriteImage(image_low_res, str(lowres_path))
        
        # 使用mrsegmentator进行胰腺分割
        seg_output_dir = temp_dir / "segmentation"
        seg_output_dir.mkdir(exist_ok=True)
        
        inference.infer([str(lowres_path)], str(seg_output_dir), [0])
        
        # 处理分割结果
        seg_mask_path = seg_output_dir / f"{input_path.stem}_0000_seg.nii.gz"
        if seg_mask_path.exists():
            mrseg_mask = sitk.ReadImage(str(seg_mask_path))
            mrseg_mask_array = sitk.GetArrayFromImage(mrseg_mask)
            
            # 提取胰腺mask (假设胰腺标签为7)
            pancreas_mask = (mrseg_mask_array == 7).astype(np.uint8)
            
            # 上采样到原始分辨率
            pancreas_mask_high_res = upsample_mask(
                pancreas_mask, 
                mrseg_mask.GetSpacing(), 
                itk_image.GetSpacing()
            )
            
            # 保存胰腺mask
            pancreas_mask_img = sitk.GetImageFromArray(pancreas_mask_high_res)
            pancreas_mask_img.CopyInformation(itk_image)
            mask_output_path = output_dir / f"{input_path.stem}_pancreas_mask.mha"
            sitk.WriteImage(pancreas_mask_img, str(mask_output_path))
            
            # 保存变换后的图像
            transform3_path = temp_dir / f"{input_path.stem}_0003.mha"
            transformed_img = normalize_and_resample(str(transform3_path))
            sitk.WriteImage(transformed_img, str(output_dir / f"{input_path.stem}_transformed.mha"))
            
            print(f"Successfully processed: {input_path.name}")
        else:
            print(f"Segmentation failed for: {input_path.name}")
            
    except Exception as e:
        print(f"Error processing {input_path.name}: {str(e)}")

def process_batch(input_dir: Path, output_dir: Path):
    """
    批量处理输入目录下的所有医学影像
    """
    # 支持的医学影像格式
    supported_formats = (".nii", ".nii.gz", ".mha", ".mhd", ".dcm")
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 遍历输入目录
    for input_path in input_dir.glob("*"):
        if input_path.suffix.lower() in supported_formats:
            process_single_image(input_path, output_dir)
    
    print(f"\nProcessing complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="输入医学影像目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出结果目录")
    
    args = parser.parse_args()
    
    process_batch(Path(args.input_dir), Path(args.output_dir))