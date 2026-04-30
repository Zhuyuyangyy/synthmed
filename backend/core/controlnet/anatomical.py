"""
ControlNet with Anatomical Topology Constraints
解剖学拓扑约束控制网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.models.attention_processor import AttnProcessor


@dataclass
class AnatomicalConstraint:
    """解剖学约束配置"""
    mask: torch.Tensor                          # 分割掩码
    structure_boundaries: Optional[torch.Tensor] = None  # 结构边界
    topology_mask: Optional[torch.Tensor] = None  # 拓扑掩码
    weight: float = 1.0                        # 约束权重
    constraint_type: str = "anatomy"            # anatomy | boundary | topology


class AnatomicalControlNet:
    """
    支持解剖学拓扑约束的ControlNet。
    
    功能:
    1. 基于分割掩码的结构约束
    2. 边界感知控制
    3. 拓扑一致性保持
    """
    
    def __init__(
        self,
        base_model_path: str = "stabilityai/stable-diffusion-2-1",
        controlnet_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        
        # 加载ControlNet
        if controlnet_path:
            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
        else:
            # 使用默认ControlNet
            self.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-scribble",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
        
        self.pipe = None  # 初始化后设置
        
    def set_pipeline(self, pipe: StableDiffusionControlNetPipeline):
        """设置完整pipeline"""
        self.pipe = pipe
        
    @torch.no_grad()
    def preprocess_mask(
        self,
        mask: torch.Tensor,
        target_size: Tuple[int, int] = (512, 512)
    ) -> torch.Tensor:
        """
        预处理分割掩码。
        
        Args:
            mask: 输入掩码 (H, W) 或 (1, H, W)
            target_size: 目标尺寸
        Returns:
            预处理后的掩码 (1, H, W)
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        # 归一化
        if mask.max() > 1:
            mask = mask.float() / 255.0
        
        # 调整大小
        mask = F.interpolate(
            mask.unsqueeze(0).float(),
            size=target_size,
            mode='bilinear',
            align_corners=False
        )[0]
        
        return mask
    
    @torch.no_grad()
    def extract_boundaries(
        self,
        mask: torch.Tensor,
        kernel_size: int = 3
    ) -> torch.Tensor:
        """
        从掩码提取边界。
        
        Args:
            mask: (1, H, W) 分割掩码
            kernel_size: 形态学操作核大小
        Returns:
            边界掩码 (1, H, W)
        """
        # 使用Sobel算子提取边界
        from scipy import ndimage
        
        mask_np = mask.squeeze().cpu().numpy()
        
        # Sobel
        sobel_x = ndimage.sobel(mask_np, axis=0)
        sobel_y = ndimage.sobel(mask_np, axis=1)
        boundary = np.sqrt(sobel_x**2 + sobel_y**2)
        
        boundary = (boundary > 0.1).astype(np.float32)
        
        return torch.from_numpy(boundary).unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def create_control_image(
        self,
        constraint: AnatomicalConstraint,
        style: str = "scribble"
    ) -> torch.Tensor:
        """
        创建ControlNet控制图像。
        
        Args:
            constraint: 解剖学约束
            style: 控制风格 (scribble | canny | depth | normal)
        Returns:
            控制图像 (1, 3, H, W)
        """
        mask = constraint.mask
        
        if mask.dim() == 3:
            mask = mask[0]  # 取第一个通道
        
        h, w = mask.shape
        
        if style == "scribble":
            # 简笔画风格 - 保留结构轮廓
            # 转为二值
            binary = (mask > 0.5).float()
            
            # 提取边界
            boundary = self.extract_boundaries(binary.unsqueeze(0).unsqueeze(0))
            
            # 创建彩色编码的scribble
            control = torch.zeros(3, h, w, device=self.device)
            
            # 不同结构用不同颜色
            for struct_id in range(1, 4):  # 最多3种结构
                struct_mask = (mask == struct_id).float()
                if struct_mask.sum() > 0:
                    boundary_this = self.extract_boundaries(struct_mask.unsqueeze(0).unsqueeze(0))
                    
                    if struct_id == 1:
                        control[0] += boundary_this.squeeze()  # 红色
                    elif struct_id == 2:
                        control[1] += boundary_this.squeeze()  # 绿色
                    else:
                        control[2] += boundary_this.squeeze()  # 蓝色
            
            # 添加边界
            control = torch.clamp(control + boundary * 0.5, 0, 1)
            
        elif style == "canny":
            # Canny边缘检测
            from cv2 import Canny, cv2
            
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            edges = Canny(mask_np, 50, 150)
            
            control = torch.from_numpy(edges / 255.0).unsqueeze(0).to(self.device)
            control = torch.cat([control] * 3, dim=0)
            
        elif style == "depth":
            # 深度图风格 - 使用距离变换
            from scipy.ndimage import distance_transform_edt
            
            binary = (mask > 0.5).float().cpu().numpy()
            dist_transform = distance_transform_edt(binary)
            dist_transform = dist_transform / (dist_transform.max() + 1e-8)
            
            depth = torch.from_numpy(dist_transform).unsqueeze(0).to(self.device)
            control = torch.cat([depth] * 3, dim=0)
        
        else:
            # 默认: 灰度掩码
            control = mask.unsqueeze(0).to(self.device)
            control = torch.cat([control] * 3, dim=0)
        
        return control
    
    def compute_topology_loss(
        self,
        generated: torch.Tensor,
        constraint: AnatomicalConstraint
    ) -> torch.Tensor:
        """
        计算拓扑一致性损失。
        
        确保生成图像与输入掩码的拓扑结构一致。
        
        Args:
            generated: 生成的图像特征 (B, C, H, W)
            constraint: 解剖学约束
        Returns:
            拓扑损失标量
        """
        if constraint.topology_mask is None:
            return torch.tensor(0.0, device=self.device)
        
        # 将生成图像与拓扑掩码对齐
        mask = constraint.topology_mask.to(self.device)
        
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        
        # 计算区域一致性
        # 对于每个结构区域，计算生成图像的方差
        loss = torch.tensor(0.0, device=self.device)
        
        for struct_id in range(1, 4):
            struct_mask = (mask == struct_id).float()
            
            if struct_mask.sum() > 100:  # 确保有足够像素
                # 在该区域内计算生成图像的均值和方差
                region_mean = (generated * struct_mask).sum() / (struct_mask.sum() + 1e-8)
                region_var = ((generated - region_mean) ** 2 * struct_mask).sum() / (struct_mask.sum() + 1e-8)
                
                # 低方差表示更均匀的生成
                loss += region_var * 0.1
        
        return loss * constraint.weight
    
    def compute_boundary_loss(
        self,
        generated: torch.Tensor,
        constraint: AnatomicalConstraint
    ) -> torch.Tensor:
        """
        计算边界一致性损失。
        
        Args:
            generated: 生成的图像特征
            constraint: 解剖学约束
        Returns:
            边界损失
        """
        if constraint.structure_boundaries is None:
            return torch.tensor(0.0, device=self.device)
        
        boundaries = constraint.structure_boundaries.to(self.device)
        
        if boundaries.dim() == 2:
            boundaries = boundaries.unsqueeze(0)
        
        # 计算生成图像的梯度
        grad_x = generated[:, :, :, 1:] - generated[:, :, :, :-1]
        grad_y = generated[:, :, 1:, :] - generated[:, :, :-1, :]
        
        # 在边界处应该有强梯度
        boundaries_x = boundaries[:, :, :, 1:] if boundaries.shape[-1] > 1 else boundaries
        boundaries_y = boundaries[:, :, 1:, :] if boundaries.shape[-2] > 1 else boundaries
        
        # 边界处缺少梯度 = 损失
        loss_x = (torch.abs(grad_x) * boundaries_x).sum() / (boundaries_x.sum() + 1e-8)
        loss_y = (torch.abs(grad_y) * boundaries_y).sum() / (boundaries_y.sum() + 1e-8)
        
        return (loss_x + loss_y) * constraint.weight
    
    @torch.no_grad()
    def generate_with_constraint(
        self,
        prompt: str,
        constraint: AnatomicalConstraint,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42,
        controlnet_conditioning_scale: float = 0.8
    ) -> torch.Tensor:
        """
        使用解剖学约束生成图像。
        
        Args:
            prompt: 文本提示
            constraint: 解剖学约束
            其他参数同Stable Diffusion
        Returns:
            生成的图像
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not set. Call set_pipeline() first.")
        
        # 创建控制图像
        control_image = self.create_control_image(constraint)
        
        # 生成
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale
        )
        
        return output.images[0]


class MultiModalConsistencyLoss(nn.Module):
    """
    多模态一致性损失函数。
    
    确保生成的MRI与CT图像在空间结构上完全对齐。
    创新点：用于SCI论文。
    """
    
    def __init__(self, lambda_structure: float = 1.0, lambda_intensity: float = 0.5):
        super().__init__()
        self.lambda_structure = lambda_structure
        self.lambda_intensity = lambda_intensity
        
    def forward(
        self,
        generated_mri: torch.Tensor,
        generated_ct: torch.Tensor,
        reference_structure: torch.Tensor
    ) -> torch.Tensor:
        """
        计算多模态一致性损失。
        
        Args:
            generated_mri: 生成的MRI图像
            generated_ct: 生成的CT图像
            reference_structure: 参考结构掩码
        Returns:
            总损失
        """
        # 1. 结构一致性损失
        # 确保两种模态在相同结构区域有一致的边缘
        mri_grad_x = generated_mri[:, :, :, 1:] - generated_mri[:, :, :, :-1]
        mri_grad_y = generated_mri[:, :, 1:, :] - generated_mri[:, :, :-1, :]
        
        ct_grad_x = generated_ct[:, :, :, 1:] - generated_ct[:, :, :, :-1]
        ct_grad_y = generated_ct[:, :, 1:, :] - generated_ct[:, :, :-1, :]
        
        # 边缘一致性
        structure_loss = (
            torch.abs(mri_grad_x - ct_grad_x).mean() +
            torch.abs(mri_grad_y - ct_grad_y).mean()
        )
        
        # 2. 强度分布一致性损失
        # 同种组织在两种模态中应该有相似的空间分布
        ref_mask = reference_structure.float()
        
        # 计算参考结构区域的统计量
        ref_mean = (reference_structure * ref_mask).mean()
        ref_std = (reference_structure - ref_mean).std()
        
        mri_in_region = generated_mri * ref_mask
        ct_in_region = generated_ct * ref_mask
        
        # 归一化后的强度差异
        mri_normalized = (mri_in_region - mri_in_region.mean()) / (mri_in_region.std() + 1e-8)
        ct_normalized = (ct_in_region - ct_in_region.mean()) / (ct_in_region.std() + 1e-8)
        
        intensity_loss = torch.abs(mri_normalized - ct_normalized).mean()
        
        # 3. 总损失
        total_loss = (
            self.lambda_structure * structure_loss +
            self.lambda_intensity * intensity_loss
        )
        
        return total_loss
    
    def compute_cross_modality_consistency(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算跨模态一致性指标。
        
        Returns:
            包含各种一致性指标的字典
        """
        # 结构相似度 (SSIM-like)
        c1 = 1e-5
        c2 = 1e-4
        
        mu1 = image1.mean()
        mu2 = image2.mean()
        
        sigma1_sq = image1.var()
        sigma2_sq = image2.var()
        sigma12 = ((image1 - mu1) * (image2 - mu2)).mean()
        
        ssim = (
            (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2) /
            ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        )
        
        # 互信息
        # 简化版本：相关系数
        correlation = torch.corrcoef(
            torch.stack([
                image1.flatten(),
                image2.flatten()
            ])
        )[0, 1]
        
        # 归一化互信息
        h1 = torch.histc(image1, bins=50)
        h2 = torch.histc(image2, bins=50)
        
        h1 = h1 / (h1.sum() + 1e-8)
        h2 = h2 / (h2.sum() + 1e-8)
        
        h_joint = torch.histc(
            torch.stack([image1.flatten(), image2.flatten()], dim=1),
            bins=50
        ).reshape(50, 50)
        h_joint = h_joint / (h_joint.sum() + 1e-8)
        
        # 互信息计算
        h_x = -torch.sum(h1 * torch.log(h1 + 1e-8))
        h_y = -torch.sum(h2 * torch.log(h2 + 1e-8))
        
        nmi = 2 * correlation.abs().item()  # 简化的NMI
        
        return {
            "ssim": ssim.item(),
            "correlation": correlation.item(),
            "normalized_mutual_information": nmi
        }
