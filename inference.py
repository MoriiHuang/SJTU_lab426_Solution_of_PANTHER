#  Copyright 2025 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
The following is the inference script for the baseline algorithm for Task 1 of the PANTHER challenge.

It is meant to run within a container.

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the GC platform:
https://grand-challenge.org/documentation/runtime-environment/
"""

from pathlib import Path
from threading import TIMEOUT_MAX
import time
import glob
import SimpleITK as sitk
import numpy as np
import os
import subprocess
import shutil
from scipy import ndimage
from data_utils import *
from mrsegmentator import inference
from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import warnings
warnings.filterwarnings("ignore")

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


def apply_crf_3d_slicewise(image_3d, prob_3d):
    """
    对3D图像逐层应用2D CRF
    image_3d: (D, H, W) 原始图像
    prob_3d: (C, D, H, W) 概率图 (C=2: 背景和前景)
    """
    d, h, w = image_3d.shape
    result = np.zeros((d, h, w), dtype=np.uint8)
    print(f"Applying CRF on 3D image of shape {image_3d.shape} with probabilities of shape {prob_3d.shape}")
    
    for z in range(d):
        # 准备2D数据
        img_2d = image_3d[z]
        prob_2d = prob_3d[:, z]
        
        # 创建CRF
        crf = dcrf.DenseCRF2D(w, h, 3)  # 2个类别
        
        # 设置unary potential
        unary = unary_from_softmax(prob_2d)
        crf.setUnaryEnergy(unary)
        
        # 添加空间和外观约束
        crf.addPairwiseGaussian(sxy=3, compat=3)
        
        # 推理
        q = crf.inference(5)
        result[z] = np.argmax(np.array(q).reshape((3, h, w)), axis=0)
    
    return result

def fuse_probabilities_fine(prob_modelA, prob_modelB):
    """
    融合两个模型的概率图：
    - 对label=1的概率取加权平均
    """
    # 加权平均
    weight_A = 0.75
    weight_B = 0.25
    fused_prob = np.zeros_like(prob_modelA)
    labels = prob_modelA.shape[0]  # 假设第一个维度是类别数
    for i in range(labels):
        fused_prob[i] = (weight_A * prob_modelA[i] + weight_B * prob_modelB[i])
    # 确保概率归一化
    fused_prob = fused_prob / np.sum(fused_prob, axis=0, keepdims=True)
    
    return fused_prob

class PancreaticTumorSegmentationContainer(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # input / output paths for nnUNet
        self.umamba_input_dir = Path("/opt/algorithm/umamba/input")
        self.umamba_output_dir = Path("/opt/algorithm/umamba/output")
        self.umamba2_input_dir = Path("/opt/algorithm/umamba2/input")
        self.umamba2_output_dir = Path("/opt/algorithm/umamba2/output")
        # input / output paths for predictions-model
        folders_with_mri = [folder for folder in os.listdir("/input/images") if "mri" in folder.lower()]
        if len(folders_with_mri) == 1:
            mr_ip_dir_name = folders_with_mri[0]
            print("Folder containing eval image", mr_ip_dir_name)
        else:
            print("Error: Expected one folder containing 'mri', but found", len(folders_with_mri))
            mr_ip_dir_name = 'abdominal-t1-mri' #default value
        
        self.mr_ip_dir = Path(f"/input/images/{mr_ip_dir_name}") #abdominal-t2-mri
        self.output_dir = Path("/output")
        self.output_dir_images = Path(os.path.join(self.output_dir, "images"))
        self.output_dir_seg_mask = Path(os.path.join(self.output_dir_images, "pancreatic-tumor-segmentation"))
        self.segmentation_mask = self.output_dir_seg_mask / "tumor_seg.mha"
        self.weights_path = Path("/opt/ml/model") #weights can be uploaded as a separate tarball to Grand Challenge (Algorithm > Models). The resources will be extracted to this path at runtime
        self.mrsegmentator_weights = "/opt/ml/model/weights"
        self.medsam_weights = "/opt/ml/model/medsam_lite_best.pth"
        self.umamba_path = "/opt/ml/model/umamba"
        self.mrsegmentator_input_dir = Path("/opt/algorithm/mrsegmentator/input")
        self.mrsegmentator_output_dir = Path("/opt/algorithm/mrsegmentator/output")
        os.environ["MRSEG_WEIGHTS_PATH"] = self.mrsegmentator_weights

        # ensure required folders exist
        self.output_dir_seg_mask.mkdir(exist_ok=True, parents=True)
        self.umamba_input_dir.mkdir(exist_ok=True, parents=True) #not used in the current implementation
        self.umamba_output_dir.mkdir(exist_ok=True, parents=True)
        self.umamba2_input_dir.mkdir(exist_ok=True, parents=True) #not used in the current implementation
        self.umamba2_output_dir.mkdir(exist_ok=True, parents=True)
        self.mrsegmentator_input_dir.mkdir(exist_ok=True, parents=True)
        self.mrsegmentator_output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        mha_files = glob.glob(os.path.join(self.mr_ip_dir, '*.mha'))
        # Check if any .mha files were found
        if mha_files:
            self.mr_image = mha_files[0]
        else:
            print('No mha images found in input directory')

    def run(self):
        """
        Load T1 MRI and generate segmentation of the tumor 
        """
        _show_torch_cuda_info()
        start_time = time.perf_counter()

        #1. copy the input image to umamba input directory adding _0000 suffix
        shutil.copyfile(self.mr_image, self.umamba_input_dir / "mri_0000.mha")
        shutil.copyfile(self.mr_image, self.umamba2_input_dir / "mri_0000.mha")

        #   print shape and spacing
        itk_image = sitk.ReadImage(self.mr_image)
        ct_array = sitk.GetArrayFromImage(itk_image)  # 形状为 (Z,Y,X)
        self.umamba_predict(
            input_dir=self.umamba2_input_dir,
            output_dir=self.umamba2_output_dir,
            task='30',     
        )
        # 应用变换
        transforms = apply_ct_transforms(ct_array)
            
            # 保存三种变换（_0000, _0001, _0002）
        for i, (name, data) in enumerate(transforms.items()):
            # 转换为SimpleITK图像并保留原空间信息
            new_img = sitk.GetImageFromArray(data.astype(np.float32))
            new_img.CopyInformation(itk_image)  # 复制原图的元数据（spacing, origin等）
            sitk.WriteImage(
                new_img, 
                str(self.umamba_input_dir / f'mri_000{i}.mha')
            )
        out_spacing = [itk_image.GetSpacing()[0] * 2, itk_image.GetSpacing()[1] * 2, itk_image.GetSpacing()[2] * 2]
        image_low_res = resample_img(itk_image, out_spacing  = out_spacing, is_label=False)
        sitk.WriteImage(image_low_res, str(self.mrsegmentator_input_dir / "mri_0000.nii.gz"))
        mrseg_image = os.path.join(self.mrsegmentator_input_dir, "mri_0000.nii.gz")
            #   generate pancreas mask with mrsegmentator
        inference.infer([mrseg_image], self.mrsegmentator_output_dir, [0])
            #   keep only the pancreas mask (pancreas==7)
        mrseg_mask = sitk.ReadImage(self.mrsegmentator_output_dir / "mri_0000_seg.nii.gz")

        mrseg_mask_array = sitk.GetArrayFromImage(mrseg_mask)
        pancreas_mask = mrseg_mask_array.copy()
        # pancreas_mask[pancreas_mask != 7] = 0
        # pancreas_mask[pancreas_mask == 7] = 1
        pancreas_mask_high_res = upsample_mask(pancreas_mask, mrseg_mask.GetSpacing(), itk_image.GetSpacing(), fill_holes=False)
        pancreas_mask_high_image = sitk.GetImageFromArray(pancreas_mask_high_res)
        pancreas_mask_high_image.CopyInformation(itk_image)  # Copy original image's metadata
        print(pancreas_mask_high_image.GetSize(), pancreas_mask_high_image.GetSpacing())
        sitk.WriteImage(pancreas_mask_high_image,str(self.umamba_input_dir / f'mri_0003.mha'))

        resampled_image = normalize_and_resample(str(self.umamba_input_dir / f'mri_0003.mha'))

        sitk.WriteImage(resampled_image,str(self.umamba_input_dir / f'mri_0003.mha'))

        print(f"Original image shape: {itk_image.GetSize()}, spacing: {itk_image.GetSpacing()}")
        #2. Predict with umamba
        print(f"Input image umamba:{os.listdir(self.umamba_input_dir)}")
        mr_mask_name = "mri.mha"
        mr_prob_name = "mri.npz"
        print("input dir has the following files:", os.listdir(self.umamba_input_dir))
        print("output dir has the following files:", os.listdir(self.umamba_output_dir))
        self.umamba_predict(
            input_dir=self.umamba_input_dir,
            output_dir=self.umamba_output_dir,
        )
        print(f"Output files: {os.listdir(self.umamba_output_dir)}")
        #3. Post-process umamba output
        #   read the umamba output
        tumor_mask_2 = sitk.ReadImage(self.umamba2_output_dir / mr_mask_name)
        tumor_mask_array_2 = sitk.GetArrayFromImage(tumor_mask_2)
        tumor_mask_array_2[tumor_mask_array_2 != 1] = 0  # ensure binary mask
        voxel_count = np.sum(tumor_mask_array_2)
        print("voxel count:", voxel_count)
        if voxel_count >= 2500:
            print("Start Union:")
            tumor_mask = sitk.ReadImage(self.umamba_output_dir / mr_mask_name)
            tumor_mask_array = sitk.GetArrayFromImage(tumor_mask)
            tumor_mask_array[tumor_mask_array !=1] =0
            union_mask_array = np.logical_or(tumor_mask_array == 1, tumor_mask_array_2 == 1).astype(np.uint8)
            filter_tumor_mask =  sitk.GetImageFromArray(union_mask_array)
            filter_tumor_mask.CopyInformation(tumor_mask_2)
            # sitk.WriteImage(tumor_mask, '/tmpres/tumor_mask_origin.mha')
            print(f"umamba output shape: {tumor_mask_2.GetSize()}, spacing: {tumor_mask_2.GetSpacing()}")
            #   convert to numpy array
            sitk.WriteImage(filter_tumor_mask, self.segmentation_mask)
        else:
            tumor_mask = sitk.ReadImage(self.umamba_output_dir / mr_mask_name)
            tumor_mask_array = sitk.GetArrayFromImage(tumor_mask)
            tumor_mask_array[tumor_mask_array !=1] =0
            print("unique:",np.unique(tumor_mask_array))
            filter_tumor_mask =  sitk.GetImageFromArray(tumor_mask_array)
            filter_tumor_mask.CopyInformation(tumor_mask)
            # sitk.WriteImage(tumor_mask, '/tmpres/tumor_mask_origin.mha')
            print(f"umamba output shape: {tumor_mask.GetSize()}, spacing: {tumor_mask.GetSpacing()}")
            #   convert to numpy array
            sitk.WriteImage(filter_tumor_mask, self.segmentation_mask)
        
        end_time = time.perf_counter()
        print(f"Prediction time: {end_time - start_time:.3f} seconds")
    def umamba_predict(self, input_dir, output_dir, task="31", trainer="nnUNetTrainerUMambaBot",
                    configuration="3d_fullres", checkpoint="checkpoint_best.pth", folds="all"):
        """
        Use trained nnUNet network to generate segmentation masks using umamba
        """
        # Set environment variables
        os.environ['nnUNet_results'] = str(self.umamba_path)
        cmd = [
            'nnUNetv2_predict',
            '-d', task,
            '-i', str(input_dir),
            '-o', str(output_dir),
            '-c', configuration,
            '-tr', trainer,
            '--save_probabilities',
        ]
        if folds:
            cmd.append('-f')
            # If folds is a string and contains a comma, split it; otherwise, wrap it in a list.
            fold_list = folds.split(',') if isinstance(folds, str) and ',' in folds else [folds]
            cmd.extend(fold_list)

        if checkpoint:
            cmd.append('-chk')
            cmd.append(str(checkpoint))

        cmd_str = " ".join(cmd)
        print(f"Running command: {cmd_str}")
        subprocess.check_call(cmd_str, shell=True)



        
    def predict(self, input_dir, output_dir, task="Dataset091_PantherTask2", trainer="nnUNetTrainer",
                    configuration="3d_fullres", checkpoint="checkpoint_final.pth", folds="0,1,2"):
            """
            Use trained nnUNet network to generate segmentation masks
            """

            # Set environment variables
            os.environ['nnUNet_results'] = str(self.nnunet_model_dir)

            # Run prediction script
            cmd = [
                'nnUNetv2_predict',
                '-d', task,
                '-i', str(input_dir),
                '-o', str(output_dir),
                '-c', configuration,
                '-tr', trainer,
                '--disable_progress_bar',
                '--continue_prediction'
            ]

            if folds:
                cmd.append('-f')
                # If folds is a string and contains a comma, split it; otherwise, wrap it in a list.
                fold_list = folds.split(',') if isinstance(folds, str) and ',' in folds else [folds]
                cmd.extend(fold_list)

            if checkpoint:
                cmd.append('-chk')
                cmd.append(str(checkpoint))

            cmd_str = " ".join(cmd)
            print(f"Running command: {cmd_str}")
            subprocess.check_call(cmd_str, shell=True)

    def move_checkpoints(self, source_dir, folds="0,1,2", trainer="nnUNetTrainer", task="Dataset091_PantherTask2"):
        """
        Move nnUNet checkpoints to nnUNet_results directory.
        """
        # Create the top-level destination directory if it doesn't exist
        os.makedirs(self.nnunet_model_dir, exist_ok=True)
        print(os.listdir(source_dir))
        task_name = task.split("_")[1]
        
        # Determine fold list, supporting both a comma-separated string or a single value.
        fold_list = folds.split(',') if isinstance(folds, str) and ',' in folds else [str(folds)]
        
        # Move the checkpoints
        for fold in fold_list:
            source_path = os.path.join(source_dir, f"checkpoint_best_{task_name}_fold_{fold}.pth")
            destination_path = os.path.join(self.nnunet_model_dir, task, f"{trainer}__nnUNetPlans__3d_fullres", f"fold_{fold}", "checkpoint_final.pth")
            # Create the destination directory if it doesn't exist
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            # Move the file
            try:
                shutil.copyfile(source_path, destination_path)
                print(f"Copied checkpoint for fold {fold} to {destination_path}")
            except FileNotFoundError:
                print(f"Source file not found: {source_path}")
            except Exception as e:
                print(f"Error moving checkpoint for fold {fold}: {e}")
        #copy 'dataset.json', 'plans.json'，version.json  into f"{trainer}__nnUNetPlans__3d_fullres"
        # dataset_json = os.path.join(source_dir, "dataset.json")
        # plans_json = os.path.join(source_dir, "plans.json")
        # version_json = os.path.join(source_dir, "version.json")
        # for file_name in [dataset_json, plans_json, version_json]:
        #     if os.path.exists(file_name):
        #         destination_file = os.path.join(self.nnunet_model_dir, task, f"{trainer}__nnUNetPlans__3d_fullres", os.path.basename(file_name))
        #         shutil.copyfile(file_name, destination_file)
        #         print(f"Copied {file_name} to {destination_file}")
        #     else:
        #         print(f"File not found: {file_name}")



def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print(torch.__version__)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    PancreaticTumorSegmentationContainer().run()
