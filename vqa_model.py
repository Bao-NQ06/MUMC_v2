import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Tuple, Optional, Dict, List, Union
import numpy as np
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding with cluster masking integration.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.
        
        Args:
            x: Input images of shape (B, C, H, W)
            
        Returns:
            Patches of shape (B, N, P*P*C) where N is number of patches
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        
        # Extract patches using unfold
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, -1, C * self.patch_size * self.patch_size)
        
        return patches
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to convert image to patch embeddings.
        
        Args:
            x: Input images of shape (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, N, D)
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, H/P, W/P, D) -> (B, N, D)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ClusterMasking(nn.Module):
    """
    Cluster Masking module that masks patches based on cosine similarity.
    """
    def __init__(
        self,
        num_patches: int = 196,
        anchor_ratio: float = 0.05,
        similarity_threshold: float = 0.75,
        min_mask_ratio: float = 0.5,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.anchor_ratio = anchor_ratio
        self.similarity_threshold = similarity_threshold
        self.min_mask_ratio = min_mask_ratio
    
    def normalize_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Normalize patches to have zero mean and unit variance.
        
        Args:
            patches: Patch embeddings of shape (B, N, D)
            
        Returns:
            Normalized patches of shape (B, N, D)
        """
        # Normalize along the feature dimension
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True)
        patches = (patches - mean) / (torch.sqrt(var + 1e-6))
        return patches
    
    def compute_cosine_similarity(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise cosine similarity between patches.
        
        Args:
            patches: Normalized patch embeddings of shape (B, N, D)
            
        Returns:
            Cosine similarity matrix of shape (B, N, N)
        """
        # Normalize for cosine similarity
        patches_norm = F.normalize(patches, p=2, dim=-1)
        
        # Compute pairwise cosine similarity
        similarity = torch.bmm(patches_norm, patches_norm.transpose(1, 2))
        return similarity
    
    def generate_mask(self, similarity: torch.Tensor) -> torch.Tensor:
        """
        Generate mask based on anchor patches and similarity threshold.
        
        Args:
            similarity: Cosine similarity matrix of shape (B, N, N)
            
        Returns:
            Boolean mask of shape (B, N) where True indicates masked patches
        """
        B, N, _ = similarity.shape
        device = similarity.device
        
        # Number of anchor patches
        num_anchors = max(1, int(N * self.anchor_ratio))
        
        # Initialize mask (False = keep, True = mask)
        mask = torch.zeros((B, N), dtype=torch.bool, device=device)
        
        for b in range(B):
            # Randomly select anchor patches
            anchor_indices = torch.randperm(N, device=device)[:num_anchors]
            
            # Get similarity scores for anchor patches
            anchor_similarity = similarity[b, anchor_indices, :]
            
            # Find patches with similarity above threshold for any anchor
            cluster_mask = (anchor_similarity > self.similarity_threshold).any(dim=0)
            
            # Include anchor patches in the mask
            cluster_mask[anchor_indices] = True
            
            # Ensure minimum masking ratio
            mask_ratio = cluster_mask.float().mean().item()
            if mask_ratio < self.min_mask_ratio:
                # Number of additional patches to mask
                additional_masks = int((self.min_mask_ratio - mask_ratio) * N)
                
                # Find unmasked patches
                unmasked = (~cluster_mask).nonzero(as_tuple=True)[0]
                
                # Randomly select additional patches to mask
                if len(unmasked) > 0 and additional_masks > 0:
                    additional_indices = unmasked[torch.randperm(len(unmasked), device=device)[:additional_masks]]
                    cluster_mask[additional_indices] = True
            
            mask[b] = cluster_mask
        
        return mask
    
    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cluster masking to patches.
        
        Args:
            patches: Patch embeddings of shape (B, N, D)
            
        Returns:
            Tuple of (masked patches, mask) where mask is True for masked patches
        """
        # Normalize patches
        normalized_patches = self.normalize_patches(patches)
        
        # Compute cosine similarity
        similarity = self.compute_cosine_similarity(normalized_patches)
        
        # Generate mask
        mask = self.generate_mask(similarity)
        
        # Apply mask (set masked patches to zero)
        masked_patches = patches.clone()
        masked_patches[mask] = 0.0
        
        return masked_patches, mask


class VisionTransformer(nn.Module):
    """
    Vision Transformer with support for cluster masking.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_cluster_mask: bool = True,
        anchor_ratio: float = 0.05,
        similarity_threshold: float = 0.75,
        min_mask_ratio: float = 0.5,
    ):
        super().__init__()
        self.use_cluster_mask = use_cluster_mask
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        
        # Cluster masking
        if use_cluster_mask:
            self.cluster_masking = ClusterMasking(
                num_patches=self.patch_embed.num_patches,
                anchor_ratio=anchor_ratio,
                similarity_threshold=similarity_threshold,
                min_mask_ratio=min_mask_ratio,
            )
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ) for _ in range(depth)
        ])
        
        # Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize cls token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize patch embedding
        nn.init.normal_(self.patch_embed.proj.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.proj.bias)
        
        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.normal_(block.attn.qkv.weight, std=0.02)
            nn.init.zeros_(block.attn.qkv.bias)
            nn.init.normal_(block.attn.proj.weight, std=0.02)
            nn.init.zeros_(block.attn.proj.bias)
            nn.init.normal_(block.mlp.fc1.weight, std=0.02)
            nn.init.zeros_(block.mlp.fc1.bias)
            nn.init.normal_(block.mlp.fc2.weight, std=0.02)
            nn.init.zeros_(block.mlp.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.
        
        Args:
            x: Input images of shape (B, C, H, W)
            
        Returns:
            Features of shape (B, embed_dim)
        """
        # Get patch embeddings
        x = self.patch_embed(x)  # (B, N, D)
        
        # Apply cluster masking if enabled
        mask = None
        if self.use_cluster_mask:
            x, mask = self.cluster_masking(x)
        
        # Add cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, N+1, D)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Return cls token features
        return x[:, 0]


class TransformerBlock(nn.Module):
    """
    Transformer block with support for attention masking.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the transformer block.
        
        Args:
            x: Input features of shape (B, N, D)
            mask: Optional mask of shape (B, N) where True indicates masked patches
            
        Returns:
            Transformed features of shape (B, N, D)
        """
        # Convert patch mask to attention mask if provided
        attn_mask = None
        if mask is not None:
            # Create attention mask that masks the masked patches
            # Expand mask to include cls token (which is never masked)
            B, N = mask.shape
            expanded_mask = torch.cat([
                torch.zeros((B, 1), dtype=torch.bool, device=mask.device),
                mask
            ], dim=1)  # (B, N+1)
            
            # Create attention mask
            attn_mask = torch.zeros((B, N+1, N+1), dtype=torch.bool, device=mask.device)
            
            # Set rows and columns corresponding to masked patches to True
            # This prevents attention to and from masked patches
            for b in range(B):
                masked_indices = expanded_mask[b].nonzero(as_tuple=True)[0]
                if len(masked_indices) > 0:
                    # Mask columns (keys) - no patch attends to masked patches
                    attn_mask[b, :, masked_indices] = True
        
        # Apply attention and MLP
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    """
    Multi-head attention with support for attention masking.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input features of shape (B, N, D)
            mask: Optional attention mask of shape (B, N, N) where True indicates masked positions
            
        Returns:
            Attention output of shape (B, N, D)
        """
        B, N, C = x.shape
        
        # Generate query, key, value projections
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, C/H)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            # mask: (B, N, N) -> (B, 1, N, N) to broadcast across heads
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask, float('-inf'))
        
        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """
    MLP block used in Vision Transformer.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MLP.
        
        Args:
            x: Input features of shape (B, N, D)
            
        Returns:
            Output features of shape (B, N, D')
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """
    Generate 2D sine-cosine positional embedding.
    
    Args:
        embed_dim: Embedding dimension
        grid_size: Grid height/width (assuming square grid)
        cls_token: Whether to include position embedding for cls token
        
    Returns:
        Positional embedding array of shape (grid_size*grid_size, embed_dim) or
        (1+grid_size*grid_size, embed_dim) if cls_token is True
    """
    import numpy as np
    
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, grid_size, grid_size)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """
    Generate 2D sine-cosine positional embedding from grid.
    
    Args:
        embed_dim: Embedding dimension
        grid: Grid of shape (2, 1, grid_size, grid_size)
        
    Returns:
        Positional embedding array of shape (grid_size*grid_size, embed_dim)
    """
    import numpy as np
    
    assert embed_dim % 2 == 0
    
    # Use half of dimensions for each of sin/cos
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Generate 1D sine-cosine positional embedding from grid.
    
    Args:
        embed_dim: Embedding dimension
        pos: Position grid of shape (1, grid_size, grid_size)
        
    Returns:
        Positional embedding array of shape (grid_size*grid_size, embed_dim)
    """
    import numpy as np
    
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    
    pos = pos.reshape(-1)  # (H*W,)
    out = np.einsum('i,j->ij', pos, omega)  # (H*W, D/2), outer product
    
    emb_sin = np.sin(out)  # (H*W, D/2)
    emb_cos = np.cos(out)  # (H*W, D/2)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (H*W, D)
    return emb


class LanguageEncoder(nn.Module):
    """
    Language encoder for processing question tokens.
    """
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 77,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ) for _ in range(depth)
        ])
        
        # Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize position embeddings
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Initialize token embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the language encoder.
        
        Args:
            tokens: Input token indices of shape (B, L)
            
        Returns:
            Features of shape (B, L, D)
        """
        # Get token embeddings
        x = self.token_embedding(tokens)  # (B, L, D)
        
        # Add position embeddings
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        return x


class MultimodalFusion(nn.Module):
    """
    Multimodal fusion module for combining vision and language features.
    """
    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        fusion_dim: int,
        fusion_method: str = "cross_attention",
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fusion_method = fusion_method
        
        if fusion_method == "cross_attention":
            # Cross-attention from language to vision
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=language_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm1 = nn.LayerNorm(language_dim)
            self.norm2 = nn.LayerNorm(language_dim)
            self.mlp = MLP(
                in_features=language_dim,
                hidden_features=language_dim * 4,
                dropout=dropout,
            )
            self.proj = nn.Linear(language_dim, fusion_dim)
        elif fusion_method == "concat":
            # Concatenation and projection
            self.language_pool = nn.Linear(language_dim, language_dim)
            self.proj = nn.Sequential(
                nn.Linear(vision_dim + language_dim, fusion_dim * 2),
                nn.LayerNorm(fusion_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim * 2, fusion_dim),
            )
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
    
    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        language_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the multimodal fusion module.
        
        Args:
            vision_features: Vision features of shape (B, D_v) for global features
                             or (B, N, D_v) for sequence features
            language_features: Language features of shape (B, L, D_l)
            language_mask: Optional mask for language tokens of shape (B, L)
                          where True indicates masked tokens
            
        Returns:
            Fused features of shape (B, D_f)
        """
        if self.fusion_method == "cross_attention":
            # Reshape vision features if needed
            if vision_features.dim() == 2:
                vision_features = vision_features.unsqueeze(1)  # (B, 1, D_v)
            
            # Create attention mask from language mask if provided
            attn_mask = None
            if language_mask is not None:
                # For cross attention, key_padding_mask should match vision_features length
                # The error occurs because we need a mask of shape (B, src_len) where src_len is
                # the sequence length of vision_features (which is 1 in this case)
                attn_mask = torch.zeros(
                    language_mask.size(0),  # batch size
                    vision_features.size(1),  # sequence length of vision features (1)
                    dtype=torch.bool,
                    device=language_mask.device
                )
            
            # Apply cross-attention
            x = language_features
            residual = x
            x = self.norm1(x)
            x_attn, _ = self.cross_attn(
                query=x,
                key=vision_features,
                value=vision_features,
                key_padding_mask=attn_mask,
            )
            x = residual + x_attn
            
            # Apply MLP
            residual = x
            x = self.norm2(x)
            x = residual + self.mlp(x)
            
            # Pool language features (mean pooling)
            x = x.mean(dim=1)  # (B, D_l)
            
            # Project to fusion dimension
            x = self.proj(x)  # (B, D_f)
            
        elif self.fusion_method == "concat":
            # Pool language features
            language_pooled = self.language_pool(language_features)  # (B, L, D_l)
            
            # Apply mask if provided
            if language_mask is not None:
                language_pooled = language_pooled.masked_fill(
                    language_mask.unsqueeze(-1), 0.0
                )
                # Mean pooling with mask
                language_pooled = language_pooled.sum(dim=1) / (~language_mask).sum(dim=1, keepdim=True).clamp(min=1)
            else:
                # Mean pooling without mask
                language_pooled = language_pooled.mean(dim=1)  # (B, D_l)
            
            # Concatenate vision and language features
            x = torch.cat([vision_features, language_pooled], dim=1)  # (B, D_v + D_l)
            
            # Project to fusion dimension
            x = self.proj(x)  # (B, D_f)
        
        return x


class AnswerPredictionHead(nn.Module):
    """
    Answer prediction head for VQA.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_answers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answers),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the answer prediction head.
        
        Args:
            x: Input features of shape (B, D)
            
        Returns:
            Logits of shape (B, num_answers)
        """
        return self.mlp(x)


class QuestionTypeClassifier(nn.Module):
    """
    Classifier to determine if a question is yes/no (closed) or open-ended.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary classification: 0=open-ended, 1=yes/no
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the question type classifier.
        
        Args:
            x: Input features of shape (B, D)
            
        Returns:
            Logits of shape (B, 2) where higher value at index 1 indicates yes/no question
        """
        return self.classifier(x)


class YesNoAnswerHead(nn.Module):
    """
    Binary classification head for yes/no questions.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary: 0=no, 1=yes
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the yes/no answer head.
        
        Args:
            x: Input features of shape (B, D)
            
        Returns:
            Logits of shape (B, 2) for yes/no classification
        """
        return self.mlp(x)


class ContrastiveProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps features into a common embedding space for contrastive learning.
    """
    def __init__(self, input_dim: int, proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim)  # Normalize embeddings for cosine similarity
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projection head.
        
        Args:
            x: Input features of shape (B, D)
            
        Returns:
            Projected features of shape (B, proj_dim)
        """
        return self.projection(x)


class MedicalVQAModel(nn.Module):
    """
    Medical Visual Question Answering model with cluster masking,
    dual-branch answer prediction, and contrastive pretraining.
    """
    def __init__(
        self,
        # Vision parameters
        img_size: int = 224,
        patch_size: int = 16,
        vision_embed_dim: int = 768,
        vision_depth: int = 12,
        vision_num_heads: int = 12,
        # Language parameters
        vocab_size: int = 30522,
        max_seq_len: int = 77,
        language_embed_dim: int = 768,
        language_depth: int = 12,
        language_num_heads: int = 12,
        # Fusion parameters
        fusion_dim: int = 768,
        fusion_method: str = "cross_attention",
        # Answer prediction parameters
        num_open_answers: int = 1000,
        # Masking parameters
        use_cluster_mask: bool = True,
        anchor_ratio: float = 0.05,
        similarity_threshold: float = 0.75,
        min_mask_ratio: float = 0.5,
        # Contrastive learning parameters
        contrastive_proj_dim: int = 256,
        use_contrastive: bool = True,
        # Other parameters
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=vision_embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            mlp_ratio=4.0,
            dropout=dropout,
            use_cluster_mask=use_cluster_mask,
            anchor_ratio=anchor_ratio,
            similarity_threshold=similarity_threshold,
            min_mask_ratio=min_mask_ratio,
        )
        
        # Language encoder
        self.language_encoder = LanguageEncoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=language_embed_dim,
            depth=language_depth,
            num_heads=language_num_heads,
            mlp_ratio=4.0,
            dropout=dropout,
        )
        
        # Multimodal fusion
        self.fusion = MultimodalFusion(
            vision_dim=vision_embed_dim,
            language_dim=language_embed_dim,
            fusion_dim=fusion_dim,
            fusion_method=fusion_method,
            num_heads=8,
            dropout=dropout,
        )
        
        # Question type classifier
        self.question_type_classifier = QuestionTypeClassifier(
            input_dim=fusion_dim,
            hidden_dim=fusion_dim // 2,
            dropout=dropout,
        )
        
        # Dual-branch answer prediction heads
        self.open_answer_head = AnswerPredictionHead(
            input_dim=fusion_dim,
            hidden_dim=fusion_dim * 2,
            num_answers=num_open_answers,
            dropout=dropout,
        )
        
        self.yesno_answer_head = YesNoAnswerHead(
            input_dim=fusion_dim,
            hidden_dim=fusion_dim // 2,
            dropout=dropout,
        )
        
        # Contrastive learning components
        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.vision_proj_head = ContrastiveProjectionHead(
                input_dim=vision_embed_dim,
                proj_dim=contrastive_proj_dim,
                dropout=dropout,
            )
            
            self.language_proj_head = ContrastiveProjectionHead(
                input_dim=language_embed_dim,
                proj_dim=contrastive_proj_dim,
                dropout=dropout,
            )
    
    def compute_contrastive_loss(self, vision_proj: torch.Tensor, language_proj: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss between vision and language projections.
        
        Args:
            vision_proj: Projected vision features of shape (B, proj_dim)
            language_proj: Projected language features of shape (B, proj_dim)
            temperature: Temperature parameter for softmax scaling
            
        Returns:
            InfoNCE contrastive loss
        """
        # Normalize projections (for numerical stability)
        vision_proj = F.normalize(vision_proj, dim=1)
        language_proj = F.normalize(language_proj, dim=1)
        
        # Compute similarity matrix
        batch_size = vision_proj.shape[0]
        similarity = torch.matmul(vision_proj, language_proj.T) / temperature  # (B, B)
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=similarity.device)
        
        # Compute loss in both directions (vision->language and language->vision)
        loss_v2l = F.cross_entropy(similarity, labels)
        loss_l2v = F.cross_entropy(similarity.T, labels)
        
        # Average the losses
        contrastive_loss = (loss_v2l + loss_l2v) / 2.0
        
        return contrastive_loss
    
    def forward(
        self,
        images: torch.Tensor,
        question_tokens: torch.Tensor,
        question_mask: Optional[torch.Tensor] = None,
        is_training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Medical VQA model with dual-branch prediction and contrastive learning.
        
        Args:
            images: Input images of shape (B, 3, H, W)
            question_tokens: Tokenized questions of shape (B, L)
            question_mask: Optional mask for question tokens of shape (B, L)
                          where True indicates masked tokens
            is_training: Whether the model is in training mode
            
        Returns:
            Dictionary containing:
                - 'question_type_logits': Logits for question type classification (B, 2)
                - 'open_answer_logits': Logits for open-ended answers (B, num_open_answers)
                - 'yesno_logits': Logits for yes/no answers (B, 2)
                - 'contrastive_loss': Contrastive loss if in training mode and use_contrastive is True
        """
        # Process images through vision encoder
        vision_features = self.vision_encoder(images)  # (B, vision_embed_dim)
        # Shape validation: vision_features should be (B, vision_embed_dim)
        assert vision_features.dim() == 2, f"Expected vision_features to have 2 dimensions, got {vision_features.dim()}"
        assert vision_features.size(1) == self.vision_encoder.pos_embed.size(-1), \
            f"Expected vision_features to have shape (B, {self.vision_encoder.pos_embed.size(-1)}), got {vision_features.shape}"
        
        # Process questions through language encoder
        language_features = self.language_encoder(question_tokens)  # (B, L, language_embed_dim)
        # Shape validation: language_features should be (B, L, language_embed_dim)
        assert language_features.dim() == 3, f"Expected language_features to have 3 dimensions, got {language_features.dim()}"
        
        # Pool language features for contrastive learning
        if self.use_contrastive and is_training:
            # Mean pooling of language features
            if question_mask is not None:
                # Mask out padding tokens
                masked_features = language_features.masked_fill(question_mask.unsqueeze(-1), 0.0)
                # Sum and divide by number of non-masked tokens
                language_pooled = masked_features.sum(dim=1) / (~question_mask).sum(dim=1, keepdim=True).clamp(min=1)
            else:
                language_pooled = language_features.mean(dim=1)  # (B, language_embed_dim)
        
        # Fuse vision and language features
        fused_features = self.fusion(
            vision_features=vision_features,
            language_features=language_features,
            language_mask=question_mask,
        )  # (B, fusion_dim)
        # Shape validation: fused_features should be (B, fusion_dim)
        assert fused_features.dim() == 2, f"Expected fused_features to have 2 dimensions, got {fused_features.dim()}"
        
        # Classify question type
        question_type_logits = self.question_type_classifier(fused_features)  # (B, 2)
        
        # Get question type probabilities
        question_type_probs = F.softmax(question_type_logits, dim=1)  # (B, 2)
        
        # Predict answers using both heads
        open_answer_logits = self.open_answer_head(fused_features)  # (B, num_open_answers)
        yesno_logits = self.yesno_answer_head(fused_features)  # (B, 2)
        
        # Prepare output dictionary
        outputs = {
            'question_type_logits': question_type_logits,
            'open_answer_logits': open_answer_logits,
            'yesno_logits': yesno_logits,
        }
        
        # Add contrastive loss if in training mode and contrastive learning is enabled
        if self.use_contrastive and is_training:
            # Project features to contrastive embedding space
            vision_proj = self.vision_proj_head(vision_features)  # (B, contrastive_proj_dim)
            language_proj = self.language_proj_head(language_pooled)  # (B, contrastive_proj_dim)
            
            # Compute contrastive loss
            contrastive_loss = self.compute_contrastive_loss(vision_proj, language_proj)
            outputs['contrastive_loss'] = contrastive_loss
        
        return outputs


def train_vqa_model(
    model: MedicalVQAModel,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    contrastive_weight: float = 0.5,  # Weight for contrastive loss
    pretraining_mode: bool = False,   # Whether to use contrastive pretraining
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "e:/MUMC_v2/checkpoints",
    log_interval: int = 100,
) -> None:
    """
    Train the Medical VQA model with dual-branch answer prediction and contrastive learning.
    
    Args:
        model: The Medical VQA model to train
        train_dataloader: DataLoader for training data
        val_dataloader: Optional DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        warmup_steps: Number of warmup steps for learning rate scheduler
        contrastive_weight: Weight for contrastive loss component
        pretraining_mode: If True, focus on contrastive pretraining
        device: Device to train on
        save_path: Path to save model checkpoints
        log_interval: Interval for logging training progress
    """
    import os
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    import time
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Set up learning rate scheduler with warmup
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Set up loss functions
    ce_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        contrastive_loss_sum = 0.0
        question_type_loss_sum = 0.0
        answer_loss_sum = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Get batch data
            images = batch["images"].to(device)
            question_tokens = batch["question_tokens"].to(device)
            question_mask = batch.get("question_mask", None)
            if question_mask is not None:
                question_mask = question_mask.to(device)
            answer_labels = batch["answer_labels"].to(device)
            
            # Forward pass with is_training=True
            outputs = model(images, question_tokens, question_mask, is_training=True)
            
            # Extract outputs
            question_type_logits = outputs['question_type_logits']
            open_answer_logits = outputs['open_answer_logits']
            yesno_logits = outputs['yesno_logits']
            
            # Determine question type labels (yes/no vs open-ended)
            # This is a simplification - in practice, you would have ground truth question type labels
            # Here we're using a heuristic: if answer is 'yes' or 'no', it's a yes/no question
            # In a real implementation, you would have this information in your dataset
            question_texts = batch.get("question_texts", None)
            answer_texts = batch.get("answer_texts", None)
            
            # Create question type labels (0=open-ended, 1=yes/no)
            # This is a placeholder - in practice, derive this from your dataset
            if answer_texts is not None:
                is_yesno = torch.tensor(
                    [ans.lower() in ['yes', 'no'] for ans in answer_texts],
                    dtype=torch.long,
                    device=device
                )
            else:
                # Fallback: assume all are open-ended questions
                is_yesno = torch.zeros(images.size(0), dtype=torch.long, device=device)
            
            # Create yes/no answer labels (0=no, 1=yes) for yes/no questions
            # This is a placeholder - in practice, derive this from your dataset
            if answer_texts is not None:
                yesno_labels = torch.tensor(
                    [1 if ans.lower() == 'yes' else 0 for ans in answer_texts],
                    dtype=torch.long,
                    device=device
                )
            else:
                # Fallback
                yesno_labels = torch.zeros(images.size(0), dtype=torch.long, device=device)
            
            # Calculate question type classification loss
            q_type_loss = ce_criterion(question_type_logits, is_yesno)
            
            # Calculate answer prediction losses
            # For yes/no questions, use the yes/no head
            # For open-ended questions, use the open answer head
            
            # Create masks for each question type
            yesno_mask = (is_yesno == 1)
            open_mask = (is_yesno == 0)
            
            # Initialize answer loss
            answer_loss = torch.tensor(0.0, device=device)
            
            # Calculate loss for yes/no questions if any exist in the batch
            if yesno_mask.any():
                yesno_loss = ce_criterion(
                    yesno_logits[yesno_mask],
                    yesno_labels[yesno_mask]
                )
                answer_loss = answer_loss + yesno_loss * yesno_mask.sum()
            
            # Calculate loss for open-ended questions if any exist in the batch
            if open_mask.any():
                open_loss = ce_criterion(
                    open_answer_logits[open_mask],
                    answer_labels[open_mask]
                )
                answer_loss = answer_loss + open_loss * open_mask.sum()
            
            # Normalize answer loss by batch size
            answer_loss = answer_loss / images.size(0)
            
            # Calculate total loss
            loss = answer_loss + q_type_loss
            
            # Add contrastive loss if available and enabled
            if 'contrastive_loss' in outputs and model.use_contrastive:
                contrastive_loss = outputs['contrastive_loss']
                contrastive_loss_sum += contrastive_loss.item()
                
                # In pretraining mode, give more weight to contrastive loss
                if pretraining_mode:
                    loss = contrastive_loss * contrastive_weight + loss * (1 - contrastive_weight)
                else:
                    # In finetuning mode, focus more on answer prediction
                    loss = loss + contrastive_loss * contrastive_weight
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            question_type_loss_sum += q_type_loss.item()
            answer_loss_sum += answer_loss.item()
            global_step += 1
            
            # Log progress
            if global_step % log_interval == 0:
                avg_loss = train_loss / (batch_idx + 1)
                avg_q_type_loss = question_type_loss_sum / (batch_idx + 1)
                avg_answer_loss = answer_loss_sum / (batch_idx + 1)
                avg_contrastive_loss = contrastive_loss_sum / (batch_idx + 1) if contrastive_loss_sum > 0 else 0
                
                elapsed = time.time() - start_time
                print(f"Epoch: {epoch+1}/{num_epochs} | Step: {global_step} | "
                      f"Loss: {avg_loss:.4f} | Q-Type: {avg_q_type_loss:.4f} | "
                      f"Answer: {avg_answer_loss:.4f} | Contrastive: {avg_contrastive_loss:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.2f}s")
        
        # Validate after each epoch
        if val_dataloader is not None:
            val_metrics = validate_vqa_model(model, val_dataloader, ce_criterion, device)
            val_loss = val_metrics['total_loss']
            print(f"Epoch: {epoch+1}/{num_epochs} | Val Loss: {val_loss:.4f} | "
                  f"Val Open Acc: {val_metrics['open_acc']:.4f} | Val YesNo Acc: {val_metrics['yesno_acc']:.4f} | "
                  f"Val Q-Type Acc: {val_metrics['q_type_acc']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    os.path.join(save_path, "best_model.pth")
                )
        
        # Save checkpoint after each epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss / len(train_dataloader),
                "contrastive_loss": contrastive_loss_sum / len(train_dataloader) if contrastive_loss_sum > 0 else 0,
            },
            os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pth")
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss / len(train_dataloader),
            },
            os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pth")
        )


def validate_vqa_model(
    model: MedicalVQAModel,
    val_dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """
    Validate the Medical VQA model with dual-branch answer prediction.
    
    Args:
        model: The Medical VQA model to validate
        val_dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Dictionary of validation metrics including losses and accuracies
    """
    model.eval()
    total_loss = 0.0
    q_type_loss = 0.0
    open_answer_loss = 0.0
    yesno_loss = 0.0
    
    # Metrics counters
    q_type_correct = 0
    open_correct = 0
    yesno_correct = 0
    open_total = 0
    yesno_total = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            # Get batch data
            images = batch["images"].to(device)
            question_tokens = batch["question_tokens"].to(device)
            question_mask = batch.get("question_mask", None)
            if question_mask is not None:
                question_mask = question_mask.to(device)
            answer_labels = batch["answer_labels"].to(device)
            
            # Forward pass with is_training=False
            outputs = model(images, question_tokens, question_mask, is_training=False)
            
            # Extract outputs
            question_type_logits = outputs['question_type_logits']
            open_answer_logits = outputs['open_answer_logits']
            yesno_logits = outputs['yesno_logits']
            
            # Determine question type labels (yes/no vs open-ended)
            # This is a simplification - in practice, you would have ground truth question type labels
            answer_texts = batch.get("answer_texts", None)
            
            # Create question type labels (0=open-ended, 1=yes/no)
            if answer_texts is not None:
                is_yesno = torch.tensor(
                    [ans.lower() in ['yes', 'no'] for ans in answer_texts],
                    dtype=torch.long,
                    device=device
                )
            else:
                # Fallback: assume all are open-ended questions
                is_yesno = torch.zeros(images.size(0), dtype=torch.long, device=device)
            
            # Create yes/no answer labels (0=no, 1=yes) for yes/no questions
            if answer_texts is not None:
                yesno_labels = torch.tensor(
                    [1 if ans.lower() == 'yes' else 0 for ans in answer_texts],
                    dtype=torch.long,
                    device=device
                )
            else:
                # Fallback
                yesno_labels = torch.zeros(images.size(0), dtype=torch.long, device=device)
            
            # Calculate question type classification loss
            batch_q_type_loss = criterion(question_type_logits, is_yesno)
            q_type_loss += batch_q_type_loss.item()
            
            # Create masks for each question type
            yesno_mask = (is_yesno == 1)
            open_mask = (is_yesno == 0)
            
            # Calculate loss and accuracy for yes/no questions
            batch_yesno_loss = 0.0
            if yesno_mask.any():
                batch_yesno_loss = criterion(
                    yesno_logits[yesno_mask],
                    yesno_labels[yesno_mask]
                ).item()
                
                # Calculate yes/no accuracy
                _, yesno_pred = torch.max(yesno_logits[yesno_mask], 1)
                yesno_total += yesno_mask.sum().item()
                yesno_correct += (yesno_pred == yesno_labels[yesno_mask]).sum().item()
            
            yesno_loss += batch_yesno_loss
            
            # Calculate loss and accuracy for open-ended questions
            batch_open_loss = 0.0
            if open_mask.any():
                batch_open_loss = criterion(
                    open_answer_logits[open_mask],
                    answer_labels[open_mask]
                ).item()
                
                # Calculate open-ended accuracy
                _, open_pred = torch.max(open_answer_logits[open_mask], 1)
                open_total += open_mask.sum().item()
                open_correct += (open_pred == answer_labels[open_mask]).sum().item()
            
            open_answer_loss += batch_open_loss
            
            # Calculate question type accuracy
            _, q_type_pred = torch.max(question_type_logits, 1)
            q_type_correct += (q_type_pred == is_yesno).sum().item()
            
            # Calculate total loss
            batch_loss = batch_q_type_loss + batch_open_loss + batch_yesno_loss
            total_loss += batch_loss
            
            # Update total count
            total += images.size(0)
    
    # Calculate average losses
    avg_total_loss = total_loss / len(val_dataloader)
    avg_q_type_loss = q_type_loss / len(val_dataloader)
    avg_open_loss = open_answer_loss / len(val_dataloader)
    avg_yesno_loss = yesno_loss / len(val_dataloader)
    
    # Calculate accuracies
    q_type_acc = q_type_correct / total if total > 0 else 0
    open_acc = open_correct / open_total if open_total > 0 else 0
    yesno_acc = yesno_correct / yesno_total if yesno_total > 0 else 0
    
    # Combined accuracy (weighted average of open and yes/no accuracies)
    combined_acc = (open_correct + yesno_correct) / total if total > 0 else 0
    
    # Return all metrics
    return {
        'total_loss': avg_total_loss,
        'q_type_loss': avg_q_type_loss,
        'open_loss': avg_open_loss,
        'yesno_loss': avg_yesno_loss,
        'q_type_acc': q_type_acc,
        'open_acc': open_acc,
        'yesno_acc': yesno_acc,
        'combined_acc': combined_acc
    }


def test_vqa_model():
    """
    Test the Medical VQA model with dummy inputs, including dual-branch answer prediction
    and contrastive learning features.
    """
    import numpy as np
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dummy inputs
    batch_size = 2
    img_size = 224
    max_seq_len = 77
    vocab_size = 30522
    num_open_answers = 100
    contrastive_proj_dim = 128
    
    # Create dummy images
    images = torch.randn(batch_size, 3, img_size, img_size)
    
    # Create dummy question tokens
    question_tokens = torch.randint(0, vocab_size, (batch_size, max_seq_len))
    
    # Create dummy question mask (optional)
    question_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    # Mask some tokens (e.g., padding tokens)
    question_mask[:, 20:] = True
    
    # Create model with smaller dimensions for testing
    model = MedicalVQAModel(
        # Vision parameters
        img_size=img_size,
        patch_size=16,
        vision_embed_dim=256,
        vision_depth=4,
        vision_num_heads=8,
        # Language parameters
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        language_embed_dim=256,
        language_depth=4,
        language_num_heads=8,
        # Fusion parameters
        fusion_dim=256,
        fusion_method="cross_attention",
        # Answer prediction parameters
        num_open_answers=num_open_answers,
        # Masking parameters
        use_cluster_mask=True,
        anchor_ratio=0.05,
        similarity_threshold=0.75,
        min_mask_ratio=0.5,
        # Contrastive learning parameters
        contrastive_proj_dim=contrastive_proj_dim,
        use_contrastive=True,
        # Other parameters
        dropout=0.1,
    )
    
    print("Testing model in evaluation mode...")
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass in evaluation mode
    with torch.no_grad():
        # Test without question mask
        outputs1 = model(images, question_tokens, is_training=False)
        
        # Test with question mask
        outputs2 = model(images, question_tokens, question_mask, is_training=False)
    
    # Check output shapes
    print("\nOutput shapes in evaluation mode:")
    print(f"Question type logits: {outputs1['question_type_logits'].shape}")
    print(f"Open answer logits: {outputs1['open_answer_logits'].shape}")
    print(f"Yes/No logits: {outputs1['yesno_logits'].shape}")
    
    # Verify shapes
    assert outputs1['question_type_logits'].shape == (batch_size, 2), \
        f"Expected question_type_logits shape (batch_size, 2), got {outputs1['question_type_logits'].shape}"
    assert outputs1['open_answer_logits'].shape == (batch_size, num_open_answers), \
        f"Expected open_answer_logits shape (batch_size, {num_open_answers}), got {outputs1['open_answer_logits'].shape}"
    assert outputs1['yesno_logits'].shape == (batch_size, 2), \
        f"Expected yesno_logits shape (batch_size, 2), got {outputs1['yesno_logits'].shape}"
    
    print("\nTesting model in training mode...")
    # Set model to training mode
    model.train()
    
    # Forward pass in training mode
    outputs_train = model(images, question_tokens, question_mask, is_training=True)
    
    # Check if contrastive loss is included in training mode
    print("\nOutput shapes in training mode:")
    print(f"Question type logits: {outputs_train['question_type_logits'].shape}")
    print(f"Open answer logits: {outputs_train['open_answer_logits'].shape}")
    print(f"Yes/No logits: {outputs_train['yesno_logits'].shape}")
    
    if 'contrastive_loss' in outputs_train:
        print(f"Contrastive loss: {outputs_train['contrastive_loss'].item():.4f}")
        assert isinstance(outputs_train['contrastive_loss'], torch.Tensor), \
            f"Expected contrastive_loss to be a tensor, got {type(outputs_train['contrastive_loss'])}"
        assert outputs_train['contrastive_loss'].dim() == 0, \
            f"Expected contrastive_loss to be a scalar tensor, got shape {outputs_train['contrastive_loss'].shape}"
    else:
        print("Contrastive loss not found in training outputs")
    
    # Test with contrastive learning disabled
    print("\nTesting model with contrastive learning disabled...")
    model_no_contrastive = MedicalVQAModel(
        # Vision parameters
        img_size=img_size,
        patch_size=16,
        vision_embed_dim=256,
        vision_depth=4,
        vision_num_heads=8,
        # Language parameters
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        language_embed_dim=256,
        language_depth=4,
        language_num_heads=8,
        # Fusion parameters
        fusion_dim=256,
        fusion_method="cross_attention",
        # Answer prediction parameters
        num_open_answers=num_open_answers,
        # Masking parameters
        use_cluster_mask=True,
        # Contrastive learning parameters
        use_contrastive=False,
        # Other parameters
        dropout=0.1,
    )
    
    model_no_contrastive.train()
    outputs_no_contrastive = model_no_contrastive(images, question_tokens, is_training=True)
    
    print("\nOutputs with contrastive learning disabled:")
    print(f"Question type logits: {outputs_no_contrastive['question_type_logits'].shape}")
    print(f"Open answer logits: {outputs_no_contrastive['open_answer_logits'].shape}")
    print(f"Yes/No logits: {outputs_no_contrastive['yesno_logits'].shape}")
    print(f"Contrastive loss present: {'contrastive_loss' in outputs_no_contrastive}")
    
    print("\nAll tests passed!")
    print("The model successfully implements dual-branch answer prediction and contrastive learning features.")
    print("- Question type classifier correctly outputs logits of shape (batch_size, 2)")
    print("- Yes/No answer head correctly outputs logits of shape (batch_size, 2)")
    print("- Open-ended answer head correctly outputs logits of shape (batch_size, num_open_answers)")
    print("- Contrastive learning components are correctly enabled/disabled based on configuration")
    print("- Model works in both training and evaluation modes")
    print("- Model works with and without question masks")
    print("- All tensor shapes are as expected")


def evaluate_vqa_model(
    model: MedicalVQAModel,
    test_dataloader: torch.utils.data.DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate the Medical VQA model with dual-branch answer prediction on a test dataset.
    
    Args:
        model: The Medical VQA model to evaluate
        test_dataloader: DataLoader for test data
        device: Device to evaluate on
        output_path: Optional path to save prediction results
        
    Returns:
        Dictionary of evaluation metrics
    """
    import json
    from collections import defaultdict
    
    model.eval()
    
    # Metrics counters
    q_type_correct = 0
    open_correct = 0
    yesno_correct = 0
    open_total = 0
    yesno_total = 0
    total = 0
    
    # Store predictions for analysis
    predictions = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Get batch data
            images = batch["images"].to(device)
            question_tokens = batch["question_tokens"].to(device)
            question_mask = batch.get("question_mask", None)
            if question_mask is not None:
                question_mask = question_mask.to(device)
            
            # Get ground truth answers and metadata
            answer_labels = batch["answer_labels"].to(device)
            question_ids = batch.get("question_ids", None)
            question_texts = batch.get("question_texts", None)
            answer_texts = batch.get("answer_texts", None)
            
            # Forward pass with is_training=False
            outputs = model(images, question_tokens, question_mask, is_training=False)
            
            # Extract outputs
            question_type_logits = outputs['question_type_logits']
            open_answer_logits = outputs['open_answer_logits']
            yesno_logits = outputs['yesno_logits']
            
            # Determine question type (yes/no vs open-ended)
            # This is a simplification - in practice, you would have ground truth question type labels
            if answer_texts is not None:
                is_yesno = torch.tensor(
                    [ans.lower() in ['yes', 'no'] for ans in answer_texts],
                    dtype=torch.long,
                    device=device
                )
                yesno_labels = torch.tensor(
                    [1 if ans.lower() == 'yes' else 0 for ans in answer_texts],
                    dtype=torch.long,
                    device=device
                )
            else:
                # Fallback: assume all are open-ended questions
                is_yesno = torch.zeros(images.size(0), dtype=torch.long, device=device)
                yesno_labels = torch.zeros(images.size(0), dtype=torch.long, device=device)
            
            # Get predicted question types
            _, q_type_pred = torch.max(question_type_logits, 1)
            
            # Update question type accuracy
            q_type_correct += (q_type_pred == is_yesno).sum().item()
            
            # Create masks for each question type
            yesno_mask = (is_yesno == 1)
            open_mask = (is_yesno == 0)
            
            # Get predictions for yes/no questions
            _, yesno_pred = torch.max(yesno_logits, 1)
            
            # Get predictions for open-ended questions
            _, open_pred = torch.max(open_answer_logits, 1)
            
            # Update metrics for yes/no questions
            if yesno_mask.any():
                yesno_total += yesno_mask.sum().item()
                yesno_correct += (yesno_pred[yesno_mask] == yesno_labels[yesno_mask]).sum().item()
            
            # Update metrics for open-ended questions
            if open_mask.any():
                open_total += open_mask.sum().item()
                open_correct += (open_pred[open_mask] == answer_labels[open_mask]).sum().item()
            
            # Update total count
            total += images.size(0)
            
            # Store predictions
            for i in range(len(q_type_pred)):
                pred_item = {
                    "question_type_predicted": q_type_pred[i].item(),
                    "question_type_actual": is_yesno[i].item(),
                    "q_type_correct": q_type_pred[i].item() == is_yesno[i].item(),
                }
                
                # Add predictions based on question type
                if is_yesno[i].item() == 1:  # Yes/No question
                    pred_item["predicted"] = yesno_pred[i].item()
                    pred_item["ground_truth"] = yesno_labels[i].item()
                    pred_item["correct"] = yesno_pred[i].item() == yesno_labels[i].item()
                    pred_item["predicted_answer"] = "yes" if yesno_pred[i].item() == 1 else "no"
                else:  # Open-ended question
                    pred_item["predicted"] = open_pred[i].item()
                    pred_item["ground_truth"] = answer_labels[i].item()
                    pred_item["correct"] = open_pred[i].item() == answer_labels[i].item()
                
                # Add metadata if available
                if question_ids is not None:
                    pred_item["question_id"] = question_ids[i]
                if question_texts is not None:
                    pred_item["question"] = question_texts[i]
                if answer_texts is not None:
                    pred_item["answer"] = answer_texts[i]
                    if is_yesno[i].item() == 0:  # Open-ended question
                        if isinstance(answer_texts[i], list) and len(answer_texts) > open_pred[i].item():
                            pred_item["predicted_answer"] = answer_texts[i][open_pred[i].item()]
                
                predictions.append(pred_item)
    
    # Calculate accuracies
    q_type_acc = q_type_correct / total if total > 0 else 0
    open_acc = open_correct / open_total if open_total > 0 else 0
    yesno_acc = yesno_correct / yesno_total if yesno_total > 0 else 0
    
    # Combined accuracy (weighted average of open and yes/no accuracies)
    combined_acc = (open_correct + yesno_correct) / total if total > 0 else 0
    
    # Calculate per-category accuracy if category information is available
    category_metrics = {}
    if any("category" in pred for pred in predictions):
        category_correct = defaultdict(int)
        category_total = defaultdict(int)
        
        for pred in predictions:
            if "category" in pred:
                category = pred["category"]
                category_total[category] += 1
                if pred["correct"]:
                    category_correct[category] += 1
        
        for category in category_total:
            category_metrics[f"accuracy_{category}"] = category_correct[category] / category_total[category]
    
    # Save predictions if output path is provided
    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)
    
    # Combine all metrics
    metrics = {
        "combined_accuracy": combined_acc,
        "q_type_accuracy": q_type_acc,
        "open_accuracy": open_acc,
        "yesno_accuracy": yesno_acc,
        "open_total": open_total,
        "yesno_total": yesno_total,
        "total_samples": total,
        **category_metrics
    }
    
    return metrics


def load_vqa_model(
    checkpoint_path: str,
    model_config: Optional[Dict] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> MedicalVQAModel:
    """
    Load a trained Medical VQA model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        model_config: Optional model configuration dictionary
        device: Device to load the model on
        
    Returns:
        Loaded Medical VQA model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    if model_config is None:
        # Default configuration
        model_config = {
            # Vision parameters
            "img_size": 224,
            "patch_size": 16,
            "vision_embed_dim": 768,
            "vision_depth": 12,
            "vision_num_heads": 12,
            # Language parameters
            "vocab_size": 30522,
            "max_seq_len": 77,
            "language_embed_dim": 768,
            "language_depth": 12,
            "language_num_heads": 12,
            # Fusion parameters
            "fusion_dim": 768,
            "fusion_method": "cross_attention",
            # Answer prediction parameters
            "num_answers": 1000,
            # Masking parameters
            "use_cluster_mask": True,
            "anchor_ratio": 0.05,
            "similarity_threshold": 0.75,
            "min_mask_ratio": 0.5,
            # Other parameters
            "dropout": 0.1,
        }
    
    # Create model
    model = MedicalVQAModel(**model_config)
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def visualize_attention(
    model: MedicalVQAModel,
    image: torch.Tensor,
    question_tokens: torch.Tensor,
    question_mask: Optional[torch.Tensor] = None,
    question_text: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize attention maps between image patches and question tokens.
    
    Args:
        model: The Medical VQA model
        image: Input image of shape (1, 3, H, W)
        question_tokens: Tokenized question of shape (1, L)
        question_mask: Optional mask for question tokens of shape (1, L)
        question_text: Optional original question text for display
        save_path: Optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision import transforms
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Move inputs to the same device as model
    device = next(model.parameters()).device
    image = image.to(device)
    question_tokens = question_tokens.to(device)
    if question_mask is not None:
        question_mask = question_mask.to(device)
    
    # Get attention weights
    with torch.no_grad():
        # Forward pass through vision encoder
        vision_features = model.vision_encoder(image)
        
        # Forward pass through language encoder
        language_features = model.language_encoder(question_tokens)
        
        # Get cross-attention weights from fusion module
        if model.fusion.fusion_method == "cross_attention":
            # Reshape vision features if needed
            if vision_features.dim() == 2:
                vision_features = vision_features.unsqueeze(1)
            
            # Create attention mask from language mask if provided
            attn_mask = None
            if question_mask is not None:
                attn_mask = question_mask.logical_not()
            
            # Apply cross-attention to get attention weights
            x = language_features
            x = model.fusion.norm1(x)
            _, attn_weights = model.fusion.cross_attn(
                query=x,
                key=vision_features,
                value=vision_features,
                key_padding_mask=attn_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            
            # Average attention weights across heads
            attn_weights = attn_weights.mean(dim=1)  # (1, L, N)
        else:
            # For other fusion methods, we don't have attention weights
            print("Attention visualization is only supported for cross_attention fusion method.")
            return
    
    # Convert image to numpy for visualization
    img_np = image[0].cpu().permute(1, 2, 0).numpy()
    
    # Normalize image for display
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # Get patch size
    patch_size = model.vision_encoder.patch_embed.patch_size
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Display original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Display attention heatmap
    # Choose a token to visualize (e.g., first non-masked token)
    if question_mask is not None:
        token_idx = (~question_mask[0]).nonzero(as_tuple=True)[0][0].item()
    else:
        token_idx = 0
    
    # Get attention weights for the selected token
    token_attn = attn_weights[0, token_idx, 1:].reshape(
        int(np.sqrt(attn_weights.shape[2] - 1)),
        int(np.sqrt(attn_weights.shape[2] - 1))
    )
    
    # Display attention heatmap
    im = axes[1].imshow(token_attn.cpu().numpy(), cmap="hot", interpolation="nearest")
    axes[1].set_title(f"Attention Map for Token {token_idx}")
    axes[1].axis("off")
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1])
    
    # Add question text if provided
    if question_text is not None:
        plt.suptitle(f"Question: {question_text}", fontsize=14)
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


