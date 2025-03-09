import torch
import torch.nn as nn
from vqa_model import MultimodalFusion

# Test data
vision_features = torch.randn(2, 1, 768)
language_features = torch.randn(2, 77, 768)
language_mask = torch.zeros(2, 77, dtype=torch.bool)

# Create fusion module
fusion = MultimodalFusion(vision_dim=768, language_dim=768, fusion_dim=512)

# Test forward pass
output = fusion(vision_features, language_features, language_mask)
print(f'Output shape: {output.shape}')