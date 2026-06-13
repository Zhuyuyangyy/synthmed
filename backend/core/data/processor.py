"""
Medical Image Processing Utilities
医学影像处理工具箱
支持DICOM, NIfTI, NRRD等格式
"""
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
import SimpleITK as sitk
import nibabel as nib
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
from scipy import ndimage
from dataclasses import dataclass


@dataclass
class MedicalImageInfo:
    """医学影像元数据"""
    modality: str              # CT, MRI, XRAY, US
    body_part: str             # CHEST, HEAD, ABDOMEN, etc.
    pixel_spacing: Tuple[float, float, float]  # 体素间距
    slice_thickness: float
    shape: Tuple[int, int, int]  # (D, H, W) for 3D
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: Tuple[float, ...]
    bits_allocated: int
    bits_stored: int
    window_center: Optional[float] = None
    window_width: Optional[float] = None


class DICOMProcessor:
    """DICOM医学影像处理器"""
    
    SUPPORTED_MODALITIES = ["CT", "MR", "CR", "DX", "US", "XA", "RG"]
    
    @staticmethod
    def read_dicom_series(directory: str) -> Tuple[np.ndarray, List[MedicalImageInfo]]:
        """
        读取DICOM序列。
        
        Args:
            directory: DICOM文件目录
        Returns:
            (volume_data, metadata_list)
        """
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(directory)
        
        if not series_ids:
            raise ValueError(f"No DICOM series found in {directory}")
        
        # 读取第一个序列
        dicom_names = reader.GetGDCMSeriesFileNames(directory, series_ids[0])
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # 转为numpy
        data = sitk.GetArrayFromImage(image)  # (D, H, W)
        
        # 读取元数据
        metadata = []
        for filename in dicom_names[:len(data)]:
            try:
                dcm = pydicom.dcmread(filename)
                info = MedicalImageInfo(
                    modality=dcm.Modality if hasattr(dcm, 'Modality') else "Unknown",
                    body_part=dcm.BodyPartExamined if hasattr(dcm, 'BodyPartExamined') else "Unknown",
                    pixel_spacing=tuple(float(x) for x in dcm.PixelSpacing) if hasattr(dcm, 'PixelSpacing') else (1.0, 1.0),
                    slice_thickness=float(dcm.SliceThickness) if hasattr(dcm, 'SliceThickness') else 1.0,
                    shape=data.shape,
                    spacing=image.GetSpacing(),
                    origin=image.GetOrigin(),
                    direction=image.GetDirection(),
                    bits_allocated=int(dcm.BitsAllocated) if hasattr(dcm, 'BitsAllocated') else 16,
                    bits_stored=int(dcm.BitsStored) if hasattr(dcm, 'BitsStored') else 12
                )
                
                # Window Center/Width
                if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
                    info.window_center = float(dcm.WindowCenter)
                    info.window_width = float(dcm.WindowWidth)
                
                metadata.append(info)
            except Exception as e:
                print(f"Warning: Failed to read metadata from {filename}: {e}")
        
        return data, metadata
    
    @staticmethod
    def read_single_dicom(filename: str) -> Tuple[np.ndarray, MedicalImageInfo]:
        """读取单个DICOM文件"""
        dcm = pydicom.dcmread(filename)
        
        # 应用VOI LUT
        image = apply_voi_lut(dcm.pixel_array, dcm)
        
        # 转Hounsfield单位 (CT)
        if dcm.Modality == "CT":
            intercept = float(dcm.RescaleIntercept)
            slope = float(dcm.RescaleSlope)
            image = image * slope + intercept
        
        info = MedicalImageInfo(
            modality=dcm.Modality if hasattr(dcm, 'Modality') else "Unknown",
            body_part=dcm.BodyPartExamined if hasattr(dcm, 'BodyPartExamined') else "Unknown",
            pixel_spacing=tuple(float(x) for x in dcm.PixelSpacing) if hasattr(dcm, 'PixelSpacing') else (1.0, 1.0),
            slice_thickness=float(dcm.SliceThickness) if hasattr(dcm, 'SliceThickness') else 1.0,
            shape=image.shape,
            spacing=(1.0, 1.0, 1.0),
            origin=(0, 0, 0),
            direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
            bits_allocated=int(dcm.BitsAllocated),
            bits_stored=int(dcm.BitsStored)
        )
        
        if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
            info.window_center = float(dcm.WindowCenter)
            info.window_width = float(dcm.WindowWidth)
        
        return image, info
    
    @staticmethod
    def apply_windowing(
        image: np.ndarray,
        window_center: float,
        window_width: float
    ) -> np.ndarray:
        """应用窗宽窗位"""
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        windowed = np.clip(image, window_min, window_max)
        windowed = (windowed - window_min) / (window_max - window_min)
        
        return windowed.astype(np.float32)
    
    @staticmethod
    def read_nifti(filename: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """读取NIfTI格式"""
        img = nib.load(filename)
        data = img.get_fdata()
        return data, img
    
    @staticmethod
    def read_nrrd(filename: str) -> Tuple[np.ndarray, sitk.Image]:
        """读取NRRD格式"""
        image = sitk.ReadImage(filename)
        data = sitk.GetArrayFromImage(image)
        return data, image


class MedicalImagePreprocessor:
    """医学影像预处理器"""
    
    # CT窗宽窗位预设
    WINDOW_PRESETS = {
        "ct_brain": {"center": 40, "width": 80},
        "ct_subdural": {"center": 80, "width": 200},
        "ct_lung": {"center": -600, "width": 1600},
        "ct_soft_tissue": {"center": 50, "width": 350},
        "ct_bone": {"center": 400, "width": 1800},
        "ct_mediastinum": {"center": 50, "width": 350},
        "xray_chest": {"center": 50, "width": 350},
        "xray_soft": {"center": 40, "width": 400}
    }
    
    @staticmethod
    def normalize_ct(
        image: np.ndarray,
        mode: str = "minmax"
    ) -> np.ndarray:
        """
        归一化CT图像。
        
        Args:
            image: HU单位CT图像
            mode: minmax | zscore | unit
        """
        if mode == "minmax":
            # _clip to typical HU range
            image = np.clip(image, -1024, 3072)
            image = (image + 1024) / 4096
        elif mode == "zscore":
            image = (image - image.mean()) / (image.std() + 1e-8)
        elif mode == "unit":
            image = image / 1000.0  # HU / 1000 → [-1, 3]
        
        return image.astype(np.float32)
    
    @staticmethod
    def resize_3d(
        volume: np.ndarray,
        target_size: Tuple[int, int, int],
        mode: str = "bilinear"
    ) -> np.ndarray:
        """调整3D体数据大小"""
        from scipy.ndimage import zoom
        
        factors = (
            target_size[0] / volume.shape[0],
            target_size[1] / volume.shape[1],
            target_size[2] / volume.shape[2]
        )
        
        order = 1 if mode == "bilinear" else 0
        resized = zoom(volume, factors, order=order)
        
        return resized
    
    @staticmethod
    def extract_slice_2d(
        volume: np.ndarray,
        axis: int = 2,
        index: int = 0
    ) -> np.ndarray:
        """从3D体数据提取2D切片"""
        if axis == 0:
            return volume[index]
        elif axis == 1:
            return volume[:, index, :]
        else:
            return volume[:, :, index]
    
    @staticmethod
    def create_mask_from_threshold(
        image: np.ndarray,
        lower: float = -1024,
        upper: float = 3072,
        inside_value: int = 1
    ) -> np.ndarray:
        """从阈值创建掩码"""
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[(image >= lower) & (image <= upper)] = inside_value
        return mask
    
    @staticmethod
    def connected_component_labeling(
        mask: np.ndarray,
        connectivity: int = 26
    ) -> np.ndarray:
        """连通区域标记"""
        from scipy import ndimage
        
        structure = ndimage.generate_binary_structure(3, connectivity)
        labeled, n_features = ndimage.label(mask, structure=structure)
        
        return labeled, n_features
    
    @staticmethod
    def largest_connected_component(
        mask: np.ndarray,
        return_inverse: bool = False
    ) -> np.ndarray:
        """保留最大连通区域"""
        labeled, n = MedicalImagePreprocessor.connected_component_labeling(mask)
        
        if n == 0:
            return mask
        
        # 找最大区域
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        max_idx = np.argmax(sizes) + 1
        
        result = np.zeros_like(mask)
        result[labeled == max_idx] = 1
        
        if return_inverse:
            result = mask.copy()
            result[labeled == max_idx] = 0
        
        return result


class VolumeRenderer:
    """
    3D体绘制工具。
    用于Three.js前端可视化。
    """
    
    @staticmethod
    def create_mip(
        volume: np.ndarray,
        axis: int = 2
    ) -> np.ndarray:
        """
        创建最大密度投影(MIP)。
        
        Args:
            volume: 3D体数据
            axis: 投影轴
        Returns:
            2D投影图像
        """
        if axis == 0:
            return volume.max(axis=0)
        elif axis == 1:
            return volume.max(axis=1)
        else:
            return volume.max(axis=2)
    
    @staticmethod
    def create_orthogonal_slices(
        volume: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        创建三个正交切片。
        
        Returns:
            {axial, sagittal, coronal}切片
        """
        d, h, w = volume.shape
        
        return {
            "axial": volume[d // 2, :, :],      # 横断面
            "sagittal": volume[:, h // 2, :],   # 矢状面
            "coronal": volume[:, :, w // 2]     # 冠状面
        }
    
    @staticmethod
    def resample_to_isotropic(
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: float = 1.0
    ) -> np.ndarray:
        """重采样到各向同性"""
        from scipy.ndimage import zoom
        
        factors = (
            original_spacing[0] / target_spacing,
            original_spacing[1] / target_spacing,
            original_spacing[2] / target_spacing
        )
        
        return zoom(volume, factors, order=1)
    
    @staticmethod
    def create_2d_projections_for_web(
        volume: np.ndarray,
        num_angles: int = 36
    ) -> List[np.ndarray]:
        """
        创建用于Web 3D渲染的2D投影序列。
        
        生成一组不同角度的MIP，用于前端实现3D效果。
        """
        projections = []
        
        for i in range(num_angles):
            angle = i * 360 / num_angles
            # 简化版本：使用旋转后的MIP
            # 实际应用中需要更复杂的3D变换
            
            proj = VolumeRenderer.create_mip(volume)
            proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
            proj = (proj * 255).astype(np.uint8)
            
            projections.append(proj)
        
        return projections
    
    @staticmethod
    def export_volume_for_threejs(
        volume: np.ndarray,
        output_path: str,
        format: str = "raw"
    ):
        """
        导出体数据用于Three.js渲染。
        
        Args:
            volume: 3D numpy数组
            output_path: 输出路径
            format: raw | nifti | nrrd
        """
        if format == "raw":
            # 导出为RAW格式 + JSON元数据
            volume.tofile(output_path + ".raw")
            
            import json
            metadata = {
                "shape": volume.shape,
                "dtype": str(volume.dtype),
                "min": float(volume.min()),
                "max": float(volume.max())
            }
            
            with open(output_path + ".json", 'w') as f:
                json.dump(metadata, f)
        
        elif format == "nifti":
            nifti_img = nib.Nifti1Image(volume, np.eye(4))
            nib.save(nifti_img, output_path)
        
        elif format == "nrrd":
            img = sitk.GetImageFromArray(volume)
            sitk.WriteImage(img, output_path)


def load_medical_image(
    path: Union[str, Path],
    modality: str = None
) -> Tuple[np.ndarray, Dict]:
    """
    通用医学影像加载函数。
    
    自动检测格式并加载。
    """
    path = Path(path)
    suffix = path.suffix.lower()
    
    metadata = {}
    
    if suffix in [".nii", ".nii.gz"]:
        data, img = DICOMProcessor.read_nifti(str(path))
        metadata = {
            "affine": img.affine,
            "header": dict(img.header)
        }
    
    elif suffix in [".nrrd", ".nhdr"]:
        data, img = DICOMProcessor.read_nrrd(str(path))
        metadata = {
            "spacing": img.GetSpacing(),
            "origin": img.GetOrigin()
        }
    
    elif suffix == ".dcm":
        data, info = DICOMProcessor.read_single_dicom(str(path))
        metadata = {
            "modality": info.modality,
            "window_center": info.window_center,
            "window_width": info.window_width
        }
    
    elif path.is_dir():
        # 假设是DICOM序列
        data, infos = DICOMProcessor.read_dicom_series(str(path))
        if infos:
            metadata = {
                "modality": infos[0].modality,
                "spacing": infos[0].spacing,
                "shape": infos[0].shape
            }
    
    else:
        # 尝试作为普通图像加载
        data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        
        if data is None:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        if len(data.shape) == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        else:
            data = data.astype(np.float32)
    
    return data, metadata


def augment_medical_image(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    augmentations: List[str] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    医学影像数据增强。
    
    Args:
        image: 输入图像
        mask: 可选分割掩码
        augmentations: 要应用的数据增强列表
    """
    if augmentations is None:
        augmentations = ["flip_h", "flip_v", "rotate"]
    
    # 翻转
    if "flip_h" in augmentations:
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            if mask is not None:
                mask = np.flip(mask, axis=1).copy()
    
    if "flip_v" in augmentations:
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=0).copy()
            if mask is not None:
                mask = np.flip(mask, axis=0).copy()
    
    # 旋转 (90度)
    if "rotate" in augmentations:
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k).copy()
            if mask is not None:
                mask = np.rot90(mask, k).copy()
    
    # 亮度对比度调整
    if "brightness" in augmentations:
        alpha = 1.0 + np.random.uniform(-0.2, 0.2)  # 对比度
        beta = np.random.uniform(-0.2, 0.2)          # 亮度
        image = np.clip(image * alpha + beta, 0, 1)
    
    # 噪声
    if "noise" in augmentations and np.random.rand() > 0.7:
        noise = np.random.normal(0, 0.02, image.shape)
        image = np.clip(image + noise, 0, 1)
    
    return image, mask
