import timm

import torch
import torch.nn as nn
import timm

class EVAReIDEncoder(nn.Module):
    def __init__(self, model_name: str = 'eva02_base_patch14_clip_224', pretrained: bool = True):
        """
        Custom Multi-Stage Feature Extractor Module for EVA/EVA02 Vision Transformers,
        structured for Re-ID and multi-modal alignment experiments.
        """
        super().__init__()
        # num_classes=0 configures the forward_head to return the 512-dim projection
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, dynamic_img_size=True,
        dynamic_img_pad=True)
        
    @property
    def num_features(self) -> int:
        """Returns the native backbone feature width (e.g., 768 for Base)"""
        return self.backbone.num_features

    def forward(self, img_tensor: torch.Tensor):
        """
        Args:
            img_tensor (torch.Tensor): Input images of shape [Batch, 3, H, W]
        Returns:
            x11 (torch.Tensor): Intermediate tokens from block 11 [Batch, Seq_Len, Hidden_Dim]
            x12 (torch.Tensor): Final raw tokens from block 12 [Batch, Seq_Len, Hidden_Dim]
            xproj (torch.Tensor): 512-dimensional multi-modal joint embedding [Batch, 512]
        """
        # 1. Convert pixels to patch embeddings
        x = self.backbone.patch_embed(img_tensor)
        
        # Handle variations in timm position embedding naming across versions
        if hasattr(self.backbone, '_pos_embed'):
            x = self.backbone._pos_embed(x)
        else:
            x = self.backbone.pos_embed(x)
            
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        
        x11, x12 = None, None
        
        # 2. Sequentially process through internal transformer blocks
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            
            # Intercept block 11 and block 12 features (0-indexed)
            if i == 10:
                x11 = x
            elif i == 11:
                x12 = x

        # 3. Execute final layer norm and head projection down to 512 dimensions
        xproj = self.backbone.forward_head(x12)
        
        return xproj, x12, x11
