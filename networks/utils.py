import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution with instance normalization"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.InstanceNorm2d(out_channels, eps=1e-3, affine=True)
        
        # Initialize weights
        nn.init.normal_(self.depthwise.weight, mean=0, std=0.01)
        nn.init.constant_(self.depthwise.bias, 0)
        nn.init.normal_(self.pointwise.weight, mean=0, std=0.01)
        nn.init.constant_(self.pointwise.bias, 0)

    def forward(self, x):
        # Handle NaN and infinite values
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        mask = (x != -1).float()
        x = torch.where(mask.bool(), x, torch.zeros_like(x))
        
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        # Check for NaN/Inf again after convolution
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        
        x = self.norm(x)
        density = mask.mean(dim=1, keepdim=True)
        return F.leaky_relu(x * density, 0.1)

class AdaptiveBeamFeatureExtractor(nn.Module):
    """Improved beam feature extractor with adaptive fusion"""
    def __init__(self, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            DepthwiseSeparableConv(1, 64),
            DepthwiseSeparableConv(64, 128),
            nn.AvgPool2d(2),
            DepthwiseSeparableConv(128, 256),
            nn.AvgPool2d(2),
            DepthwiseSeparableConv(256, 512),
            nn.AvgPool2d(2),
            DepthwiseSeparableConv(512, out_dim),
            nn.AvgPool2d(2),
        )
    
    def forward(self, x):
        return self.net(x)

class EnhancedSpatialAwareCrossAttention(nn.Module):
    """Enhanced spatial-aware cross-attention with adaptive weights"""
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.weight_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//4),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dim//4, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, current_beam, all_beams):
        B, C, S, D = all_beams.shape
        
        # Calculate importance weights for each beam
        weights = self.weight_predictor(all_beams.mean(dim=2))  # [B, C, 1]
        weighted_beams = all_beams * weights.view(B, C, 1, 1)
        
        all_beams_flat = weighted_beams.permute(0, 2, 1, 3).reshape(B*S, C, D)
        current_beam_flat = current_beam.reshape(B*S, 1, D)
        
        out_flat, _ = self.attn(current_beam_flat, all_beams_flat, all_beams_flat)
        out = out_flat.view(B, S, D)
        
        # Residual connection to preserve original information
        return self.norm(out + current_beam)

class EnhancedBeamEncoder(nn.Module):
    """Improved beam encoder with adaptive reference selection"""
    def __init__(self, feature_dim=512, out_dim=768, num_heads=8):
        super().__init__()
        self.embded = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 1), stride=1, padding=0)
        self.feature_extractor = AdaptiveBeamFeatureExtractor(feature_dim)
        self.cross_attn = EnhancedSpatialAwareCrossAttention(feature_dim, num_heads)
        self.proj = nn.Linear(feature_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.residual = nn.Conv2d(1, 1, kernel_size=1)
        
        # Initialize residual connection with small weights
        nn.init.constant_(self.residual.weight, 0.1)
        nn.init.constant_(self.residual.bias, 0)

    def embed_num(self, images, numbers):
        """Embed beam numbers into the image tensor"""
        B, C, H, W = images.shape
        num_tensor = torch.tensor(numbers, dtype=images.dtype, device=images.device)
        num_rows = num_tensor.view(C, 1, 1).expand(C, 1, W)
        extended_images = torch.cat([images, num_rows.unsqueeze(0).expand(B, -1, -1, -1)], dim=2)
        batch_processed = extended_images.view(B * C, 1, H + 1, W)
        compressed = self.embded(batch_processed)
        return compressed.view(B, C, H, W)

    def forward(self, beam_imgs):
        B, C, H, W = beam_imgs.shape
        
        # Handle sparse input
        mask = (beam_imgs != -1).float()
        beam_imgs = torch.where(mask.bool(), beam_imgs, torch.zeros_like(beam_imgs))
        
        ################################################################################################################################################
        beam_imgs = self.embed_num(beam_imgs, [-1,0,1])  # Embed beam numbers
        ################################################################################################################################################
        
        # Feature extraction
        x = beam_imgs.view(B * C, 1, H, W)
        features = self.feature_extractor(x)
        
        # Reshape features
        _, D, h, w = features.shape
        features = features.view(B, C, D, h, w)  # [B, C, D, h, w]
        features = features.permute(0, 1, 3, 4, 2)  # [B, C, h, w, D]
        features = features.reshape(B, C, h * w, D)  # [B, C, h*w, D]
        
        # Adaptive reference beam selection (no longer fixed to middle)
        ref_idx = math.ceil(C / 2 - 1)
        current_beam = features[:, ref_idx]
        
        # Enhanced cross-attention
        attn_out = self.cross_attn(current_beam, features)
        
        # Projection and normalization
        projected = self.norm(self.proj(attn_out))
        
        # Add residual information from original beam
        residual = self.residual(beam_imgs[:,[ref_idx]])
        residual = F.interpolate(residual, size=(16,16), mode='bilinear')
        residual = residual.view(B, 16 * 16, 1)
        
        return projected + residual

class CombinedConditionEncoder(nn.Module):
    """Condition encoder using only beam data with density-aware processing"""
    def __init__(self):
        super().__init__()
        self.beam_enc = EnhancedBeamEncoder()
        
        # Density-aware projection for handling input sparsity
        self.density_proj = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(64, 768, 1),
            nn.Sigmoid()
        )

    def forward(self, beam_imgs):
        # 1. Extract beam features
        beam_feat = self.beam_enc(beam_imgs)  # [B, 256, 768]
        
        # 2. Calculate density weights (handle sparse input)
        density_mask = (beam_imgs != -1).float().mean(dim=1, keepdim=True)  # [B, 1, H, W]
        density_weight = self.density_proj(density_mask)  # [B, 768, 16, 16]
        density_weight = density_weight.flatten(2).permute(0, 2, 1)  # [B, 256, 768]
        
        # 3. Apply density weighting
        return beam_feat * density_weight