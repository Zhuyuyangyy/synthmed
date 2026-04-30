"""
Latent Diffusion Model for Medical Imaging Synthesis
基于LDM的医学影像合成核心
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
import yaml
from pathlib import Path


@dataclass
class GenerationConfig:
    """合成配置"""
    prompt: str
    negative_prompt: str = "low quality, blurry, artifacts, distorted anatomy"
    width: int = 512
    height: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = 42
    num_images: int = 1
    
    # 医学特定参数
    modality: str = "ct"  # ct, mri, xray, ultrasound
    body_part: str = "chest"
    pathology: Optional[str] = None  # nodule, tumor, fracture, etc.
    severity: float = 0.5  # 0.0-1.0 严重程度
    
    # ControlNet参数
    control_image: Optional[torch.Tensor] = None
    control_strength: float = 0.8
    controlnet_conditioning_scale: float = 0.8


@dataclass
class GeneratedImage:
    """合成影像结果"""
    image: torch.Tensor  # (C, H, W) or (H, W, C) depending on output
    prompt: str
    seed: int
    latent: Optional[torch.Tensor] = None
    metadata: Optional[Dict] = None


class LatentDiffusionModel:
    """
    Latent Diffusion Model for Medical Imaging.
    
    基于Stable Diffusion架构，针对医学影像进行优化：
    - 支持CT, MRI, X-ray, Ultrasound等多种模态
    - 支持ControlNet解剖学约束
    - 支持多模态一致性损失
    """
    
    def __init__(
        self,
        model_path: str = "stabilityai/stable-diffusion-2-1",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float16
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        
        print(f"Loading LDM model on {device}...")
        
        # Load components
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer"
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=torch_dtype
        ).to(device)
        
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch_dtype
        ).to(device)
        
        self.unet = UNet2DConditionModel.from_pretrained(
            model_path,
            subfolder="unet",
            torch_dtype=torch_dtype
        ).to(device)
        
        self.scheduler = self._create_scheduler()
        
        # Freeze all components
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        print("LDM model loaded successfully!")
        
    def _create_scheduler(self):
        """创建调度器"""
        from diffusers import DDIMScheduler, UNet2DConditionModel
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1
        )
        return scheduler
    
    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = ""
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码文本提示"""
        # Tokenize
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # Encode
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        prompt_embeds = prompt_embeds.to(dtype=self.torch_dtype)
        
        # Negative prompt
        if do_classifier_free_guidance and negative_prompt:
            uncond_tokens = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            uncond_input_ids = uncond_tokens.input_ids.to(self.device)
            negative_prompt_embeds = self.text_encoder(uncond_input_ids)[0]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.torch_dtype)
            
            # Concatenate
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        return prompt_embeds
    
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """将图像编码到潜在空间"""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        if image.shape[-1] != 512 or image.shape[-2] != 512:
            image = F.interpolate(image, size=(512, 512), mode='bilinear', align_corners=False)
        
        if image.max() > 1.0:
            image = image / 255.0
        
        if image.shape[1] != 3:
            # Convert grayscale to RGB
            image = image.repeat(1, 3, 1, 1)
        
        image = 2 * image - 1  # Normalize to [-1, 1]
        
        latents = self.vae.encode(image.to(self.torch_dtype)).latent_dist.sample()
        latents = latents * 0.18215  # Scale factor
        
        return latents
    
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """从潜在空间解码图像"""
        latents = latents / 0.18215
        
        image = self.vae.decode(latents.to(self.torch_dtype)).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        
        return image
    
    @torch.no_grad()
    def generate(
        self,
        config: GenerationConfig
    ) -> List[GeneratedImage]:
        """
        生成医学影像。
        
        Args:
            config: 合成配置
        Returns:
            List of generated images with metadata
        """
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Encode prompts
        prompt_embeds = self.encode_prompt(
            config.prompt,
            do_classifier_free_guidance=config.guidance_scale > 1.0,
            negative_prompt=config.negative_prompt
        )
        
        # Set timesteps
        self.scheduler.set_timesteps(config.num_inference_steps)
        
        # Create latent noise
        batch_size = config.num_images
        latent_shape = (
            self.unet.in_channels,
            config.height // 8,
            config.width // 8
        )
        
        latents = torch.randn(
            (batch_size, *latent_shape),
            device=self.device,
            dtype=self.torch_dtype,
            generator=torch.Generator(device=self.device).manual_seed(config.seed)
        )
        latents = latents * self.scheduler.init_noise_sigma
        
        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if config.guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds
            ).sample
            
            # Apply guidance
            if config.guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode
        images = self.decode_latents(latents)
        
        # Post-process
        results = []
        for i in range(batch_size):
            img = images[i]
            
            # Convert to HWC format
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            
            img = (img.cpu().numpy() * 255).astype(np.uint8)
            
            results.append(GeneratedImage(
                image=torch.from_numpy(img).permute(2, 0, 1) if img.shape[2] == 3 
                       else torch.from_numpy(img).unsqueeze(0),
                prompt=config.prompt,
                seed=config.seed + i,
                metadata={
                    "modality": config.modality,
                    "body_part": config.body_part,
                    "pathology": config.pathology,
                    "severity": config.severity,
                    "width": config.width,
                    "height": config.height,
                    "inference_steps": config.num_inference_steps,
                    "guidance_scale": config.guidance_scale
                }
            ))
        
        return results
    
    def fine_tune(
        self,
        train_images: List[torch.Tensor],
        train_prompts: List[str],
        output_path: str,
        num_epochs: int = 100,
        learning_rate: float = 1e-5,
        batch_size: int = 1
    ):
        """
        在医学影像数据集上微调LDM。
        
        使用Low-Rank Adaptation (LoRA)进行高效微调。
        """
        from peft import LoraConfig, get_peft_model
        
        # Unfreeze for training
        self.unet.requires_grad_(True)
        
        # Add LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1
        )
        
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=learning_rate
        )
        
        self.unet.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for i in range(0, len(train_images), batch_size):
                batch_imgs = train_images[i:i+batch_size]
                batch_prompts = train_prompts[i:i+batch_size]
                
                # Encode images to latents
                latents = []
                for img in batch_imgs:
                    latent = self.encode_image(img)
                    latents.append(latent)
                
                latents = torch.cat(latents, dim=0)
                
                # Encode prompts
                prompt_embeds = self.encode_prompt(
                    " ".join(batch_prompts),
                    do_classifier_free_guidance=False
                )
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps, (latents.shape[0],),
                    device=self.device
                )
                
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                
                # Predict noise
                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds[:latents.shape[0]]
                ).sample
                
                # Compute loss
                loss = F.mse_loss(noise_pred, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(train_images) / batch_size)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Save
        self.unet.save_pretrained(output_path)
        print(f"Fine-tuned model saved to {output_path}")
        
        self.unet.eval()
        self.unet.requires_grad_(False)


class MedicalImageSynthesizer:
    """
    医学影像合成器。
    整合LDM、ControlNet、后处理等组件。
    """
    
    def __init__(self, ldm_model: LatentDiffusionModel = None):
        self.ldm = ldm_model or LatentDiffusionModel()
        self.generation_history: List[GeneratedImage] = []
        
        # 预设模板
        self.templates = {
            "dental": {
                "ct": {
                    "prompt": "dental CT scan of teeth, panoramic view, high resolution, medical imaging, anatomical accuracy",
                    "negative_prompt": "artifacts, noise, distortion, low quality, blurry"
                },
                "cbct": {
                    "prompt": "cone beam CT of mandible, 3D reconstruction, dental implant planning, sharp bone structure",
                    "negative_prompt": "motion artifacts, metal artifacts, low dose appearance"
                }
            },
            "chest": {
                "ct": {
                    "prompt": "chest CT scan, axial slice, lung window, pulmonary nodules visible, medical imaging",
                    "negative_prompt": "artifacts, reconstruction errors, low quality"
                },
                "xray": {
                    "prompt": "chest X-ray PA view, posteroanterior projection, lung fields clear, mediastinum normal",
                    "negative_prompt": "underexposed, overexposed, patient motion, artifacts"
                },
                "mri": {
                    "prompt": "chest MRI T2 weighted, mediastinum, vascular structures, soft tissue contrast"
                }
            },
            "brain": {
                "mri": {
                    "prompt": "brain MRI FLAIR sequence, white matter, gray matter differentiation, subcortical structures",
                    "negative_prompt": "artifacts, motion, signal loss"
                },
                "ct": {
                    "prompt": "brain CT without contrast, stroke window, ventricles, sulci visible"
                }
            },
            "abdomen": {
                "ct": {
                    "prompt": "abdominal CT with contrast, portal venous phase, liver spleen pancreas visible",
                    "negative_prompt": "motion artifacts, beam hardening"
                }
            }
        }
        
    def generate_from_template(
        self,
        modality: str,
        body_part: str,
        pathology: str = None,
        num_images: int = 1,
        **kwargs
    ) -> List[GeneratedImage]:
        """从模板生成影像"""
        template = self.templates.get(body_part, {}).get(modality, {})
        
        prompt = template.get("prompt", "")
        if pathology:
            pathology_prompts = {
                "nodule": "with pulmonary nodule, round opacity, {severity} size",
                "tumor": "with mass lesion, irregular margin, {severity} enhancement",
                "fracture": "with bone fracture, discontinuity line visible",
                "bleeding": "with hemorrhage, hyperdense area, {severity} extent",
                "inflammation": "with inflammation, tissue swelling, contrast enhancement"
            }
            pathology_prompt = pathology_prompts.get(pathology, pathology)
            prompt += f", {pathology_prompt.format(severity=kwargs.get('severity', 'moderate'))}"
        
        config = GenerationConfig(
            prompt=prompt,
            negative_prompt=template.get("negative_prompt", ""),
            modality=modality,
            body_part=body_part,
            pathology=pathology,
            num_images=num_images,
            seed=kwargs.get("seed", 42),
            **kwargs
        )
        
        results = self.ldm.generate(config)
        self.generation_history.extend(results)
        
        return results
    
    def generate_with_mask(
        self,
        mask: torch.Tensor,
        target_prompt: str,
        num_images: int = 1,
        strength: float = 0.8
    ) -> List[GeneratedImage]:
        """
        基于分割掩码生成影像。
        
        Args:
            mask: 分割掩码 (1, H, W) 或 (H, W)
            target_prompt: 目标描述
            num_images: 生成数量
            strength: 控制强度
        """
        # 准备控制图像
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        # 归一化掩码到[0,1]
        if mask.max() > 1:
            mask = mask / 255.0
        
        # 调整大小到512x512
        mask_resized = F.interpolate(
            mask.unsqueeze(0).float(),
            size=(512, 512),
            mode='bilinear',
            align_corners=False
        )[0]
        
        # 创建彩色掩码用于ControlNet
        control_image = self._mask_to_control_image(mask_resized)
        
        config = GenerationConfig(
            prompt=target_prompt,
            control_image=control_image,
            control_strength=strength,
            num_images=num_images
        )
        
        results = self.ldm.generate(config)
        self.generation_history.extend(results)
        
        return results
    
    def _mask_to_control_image(self, mask: torch.Tensor) -> torch.Tensor:
        """将掩码转换为ControlNet可用的控制图像"""
        # 创建彩色编码的掩码
        # 0: 背景(黑), 1: 结构1(红), 2: 结构2(绿), 3: 结构3(蓝)
        h, w = mask.shape[1], mask.shape[2]
        
        control = torch.zeros(3, h, w)
        
        # 背景
        bg_mask = (mask[0] == 0)
        
        # 结构1 (红色)
        s1_mask = (mask[0] == 1)
        control[0][s1_mask] = 1.0
        
        # 结构2 (绿色)
        s2_mask = (mask[0] == 2)
        control[1][s2_mask] = 1.0
        
        # 结构3 (蓝色)
        s3_mask = (mask[0] == 3)
        control[2][s3_mask] = 1.0
        
        return control
    
    def batch_generate(
        self,
        configs: List[GenerationConfig],
        callback: Callable[[int, GeneratedImage], None] = None
    ) -> List[List[GeneratedImage]]:
        """
        批量生成。
        
        Args:
            configs: 配置列表
            callback: 每生成一张图片的回调
        Returns:
            所有生成结果
        """
        all_results = []
        
        for i, config in enumerate(configs):
            results = self.ldm.generate(config)
            all_results.append(results)
            
            if callback:
                for j, result in enumerate(results):
                    callback(i * config.num_images + j, result)
        
        return all_results
    
    def get_statistics(self) -> Dict:
        """获取生成统计"""
        total = len(self.generation_history)
        
        modalities = {}
        body_parts = {}
        
        for img in self.generation_history:
            if img.metadata:
                mod = img.metadata.get("modality", "unknown")
                body = img.metadata.get("body_part", "unknown")
                
                modalities[mod] = modalities.get(mod, 0) + 1
                body_parts[body] = body_parts.get(body, 0) + 1
        
        return {
            "total_generated": total,
            "by_modality": modalities,
            "by_body_part": body_parts
        }
