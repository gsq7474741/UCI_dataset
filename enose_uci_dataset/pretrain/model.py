"""VQ-VAE model for e-nose time series pretraining.

This module implements a Vector Quantized Variational Autoencoder (VQ-VAE)
for self-supervised pretraining of gas sensor time series data.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import MeanMetric, MinMetric


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time series."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SampleRateEncoding(nn.Module):
    """Continuous sample rate encoding using log-scale Fourier features.
    
    Encodes the physical time duration of each patch (patch_size / sample_rate)
    using sinusoidal features, enabling generalization to unseen sample rates.
    """
    
    def __init__(self, d_model: int, base_patch_size: int = 16):
        super().__init__()
        self.d_model = d_model
        self.base_patch_size = base_patch_size
        
        # Precompute div_term for Fourier encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
    
    def forward(self, sample_rate: torch.Tensor, patch_size: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            sample_rate: [B] tensor of sample rates in Hz
            patch_size: Optional override for patch size (default: base_patch_size)
            
        Returns:
            encoding: [B, d_model] sample rate encoding
        """
        if patch_size is None:
            patch_size = self.base_patch_size
        
        # Compute physical duration of each patch in seconds
        # e.g., patch_size=16, rate=100Hz -> 0.16s; rate=1Hz -> 16s
        physical_duration = patch_size / sample_rate.float()  # [B]
        
        # Log transform for perceptually uniform scaling
        # +0.01 to avoid log(0) for edge cases
        log_duration = torch.log(physical_duration + 0.01)  # [B]
        
        # Fourier encoding (same style as positional encoding)
        log_duration = log_duration.unsqueeze(-1)  # [B, 1]
        pe = torch.zeros(sample_rate.size(0), self.d_model, device=sample_rate.device)
        pe[:, 0::2] = torch.sin(log_duration * self.div_term)
        pe[:, 1::2] = torch.cos(log_duration * self.div_term)
        
        return pe


class SensorEmbedding(nn.Module):
    """Embedding layer for sensor metadata.
    
    Creates learnable embeddings for sensor models to inject prior knowledge.
    """
    
    SENSOR_MODELS = [
        "TGS2600", "TGS2602", "TGS2610", "TGS2611", "TGS2612", "TGS2620",
        "TGS2603", "TGS2630", "TGS3870-A04", "SB-500-12",
        "MQ-135", "MQ-137", "MQ-138", "MQ-7", "MQ-8",
        "QCM1", "QCM2", "QCM3", "QCM4", "QCM5",
        "UNKNOWN",
    ]
    
    def __init__(self, d_model: int):
        super().__init__()
        self.model_to_idx = {m: i for i, m in enumerate(self.SENSOR_MODELS)}
        self.embedding = nn.Embedding(len(self.SENSOR_MODELS), d_model)
    
    def get_sensor_idx(self, sensor_model: str) -> int:
        return self.model_to_idx.get(sensor_model, self.model_to_idx["UNKNOWN"])
    
    def forward(self, sensor_indices: torch.Tensor) -> torch.Tensor:
        # sensor_indices: [B, C] or [C]
        return self.embedding(sensor_indices)


class VectorQuantizer(nn.Module):
    """Vector Quantizer for VQ-VAE with EMA codebook updates.
    
    Uses Exponential Moving Average to update codebook embeddings,
    which is more stable than gradient-based updates.
    """
    
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # EMA updates - cluster size and embedding sum
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: [B, T, D] -> flatten to [B*T, D]
        z_flat = z.reshape(-1, self.embedding_dim)
        
        # Compute distances
        distances = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        
        # Get nearest embedding
        encoding_indices = distances.argmin(1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        z_q = self.embedding(encoding_indices).view(z.shape)
        
        # EMA update (only during training)
        if self.training:
            with torch.no_grad():
                # Update cluster size with EMA
                encodings_sum = encodings.sum(0)
                self.ema_cluster_size.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
                
                # Update embedding sum with EMA
                dw = z_flat.t() @ encodings  # [D, K]
                self.ema_w.mul_(self.decay).add_(dw.t(), alpha=1 - self.decay)
                
                # Normalize to get new embeddings
                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
                # Clamp to avoid division by very small values (numerical stability for FP16)
                cluster_size = cluster_size.clamp(min=1e-3)
                self.embedding.weight.copy_(self.ema_w / cluster_size.unsqueeze(1))
        
        # Compute loss (only commitment loss with EMA, codebook is updated via EMA)
        commitment_loss = F.mse_loss(z_q.detach(), z)
        vq_loss = self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # Perplexity for monitoring codebook usage
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-5)))
        
        return z_q, vq_loss, perplexity


class PatchEmbedding(nn.Module):
    """Patch embedding for time series.
    
    Splits time series into patches and projects to embedding space.
    """
    
    def __init__(self, patch_size: int = 16, in_channels: int = 1, d_model: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> patches: [B, C, num_patches, D]
        B, C, T = x.shape
        # Process each channel
        x = x.reshape(B * C, 1, T)  # [B*C, 1, T]
        patches = self.proj(x)  # [B*C, D, num_patches]
        patches = patches.transpose(1, 2)  # [B*C, num_patches, D]
        patches = patches.reshape(B, C, -1, patches.size(-1))  # [B, C, num_patches, D]
        return patches


class EnoseEncoder(nn.Module):
    """Encoder for e-nose time series.
    
    Uses Transformer encoder with channel-aware attention.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        patch_size: int = 16,
        max_channels: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        
        self.patch_embed = PatchEmbedding(patch_size, 1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=10000, dropout=dropout)
        self.sensor_embed = SensorEmbedding(d_model)
        self.sample_rate_encoding = SampleRateEncoding(d_model, base_patch_size=patch_size)
        
        # Channel embedding
        self.channel_embed = nn.Embedding(max_channels, d_model)
        
        # Learnable mask token (like BERT's [MASK]) - crucial for cross-channel learning
        # When a channel is masked (zeroed), we replace its patch embeddings with this token
        # This gives the model a meaningful Query vector for attending to visible channels
        # 
        # CRITICAL: Initialize with same scale as patch embeddings (~0.5 std, ~6.5 norm)
        # Previous init (0.02 std) was ~30x smaller, making mask token invisible in attention!
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.5)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        
        # Project to VQ dimension
        self.to_vq = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        sensor_indices: Optional[torch.Tensor] = None,
        sample_rate: Optional[torch.Tensor] = None,
        training_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input time series
            channel_mask: [B, C] boolean mask for PADDING (True = valid, False = padding)
            sensor_indices: [B, C] sensor model indices
            sample_rate: [B] sample rate in Hz for each sample
            training_mask: [B, C] boolean mask for TRAINING (True = visible, False = masked for learning)
            
        Returns:
            z: [B, C, num_patches, D] encoded representations
        """
        B, C, T = x.shape
        
        # Patch embedding: [B, C, num_patches, D]
        patches = self.patch_embed(x)
        num_patches = patches.size(2)
        
        # CRITICAL: Replace TRAINING-masked channel patches with learnable mask token
        # This gives masked channels a meaningful Query vector for cross-channel attention
        # Use torch.where to maintain gradient flow (direct assignment breaks gradients!)
        # 
        # IMPORTANT: Only use mask_token for channels that are:
        #   1. Valid (not padding) - indicated by channel_mask = True
        #   2. Masked for training - indicated by training_mask = False
        # Do NOT use mask_token for padding channels!
        if training_mask is not None:
            # training_mask: [B, C] where True = visible, False = masked for training
            # Expand mask_token to match patches shape: [B, C, num_patches, D]
            mask_token_expanded = self.mask_token.expand(B, C, num_patches, -1)
            
            # Expand training_mask to patches shape
            training_mask_expanded = training_mask[:, :C].unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            
            # Use torch.where: where training_mask is True (visible), keep patches; else use mask_token
            patches = torch.where(training_mask_expanded, patches, mask_token_expanded)
        
        # Flatten channels and patches for transformer
        # [B, C * num_patches, D]
        patches = patches.reshape(B, C * num_patches, self.d_model)
        
        # Add positional encoding (repeat for each channel)
        pos_enc = self.pos_encoding.pe[:, :num_patches, :].repeat(1, C, 1)
        patches = patches + pos_enc
        
        # Add channel embedding
        channel_ids = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)  # [B, C]
        channel_emb = self.channel_embed(channel_ids)  # [B, C, D]
        channel_emb = channel_emb.unsqueeze(2).expand(-1, -1, num_patches, -1)  # [B, C, num_patches, D]
        channel_emb = channel_emb.reshape(B, C * num_patches, self.d_model)
        patches = patches + channel_emb
        
        # Add sensor embedding if provided
        if sensor_indices is not None:
            sensor_emb = self.sensor_embed(sensor_indices)  # [B, C, D]
            sensor_emb = sensor_emb.unsqueeze(2).expand(-1, -1, num_patches, -1)
            sensor_emb = sensor_emb.reshape(B, C * num_patches, self.d_model)
            patches = patches + sensor_emb
        
        # Add sample rate encoding if provided
        if sample_rate is not None:
            rate_enc = self.sample_rate_encoding(sample_rate)  # [B, D]
            rate_enc = rate_enc.unsqueeze(1).expand(-1, C * num_patches, -1)  # [B, C*num_patches, D]
            patches = patches + rate_enc
        
        # Create attention mask ONLY for TRUE PADDING channels (not training-masked channels)
        # IMPORTANT: channel_mask here indicates which channels are VALID (True = valid, not padding)
        # Training-masked channels (zeroed for learning) should still attend to other channels!
        # Only TRUE padding channels (from pad_channels in collate) should be ignored in attention.
        attn_mask = None
        if channel_mask is not None:
            # channel_mask: True = valid channel, False = padding OR training-masked
            # We need to distinguish padding from training-masked channels
            # Padding channels have ALL zeros and should be ignored
            # Training-masked channels have zeros but should participate in attention
            # 
            # For now, we allow ALL channels to attend (no src_key_padding_mask)
            # The model should learn to reconstruct masked channels from visible ones
            # through the standard self-attention mechanism
            pass  # No attention masking - all channels can see each other
        
        # Transformer encoding (no src_key_padding_mask so masked channels can attend to visible ones)
        z = self.transformer(patches)
        
        # Reshape back: [B, C, num_patches, D]
        z = z.reshape(B, C, num_patches, self.d_model)
        z = self.to_vq(z)
        
        return z


class EnoseDecoder(nn.Module):
    """MAE-style Decoder for e-nose time series reconstruction.
    
    Key insight: Use cross-attention where:
    - Memory (Key/Value): encoder outputs from ALL channels (visible have real info, masked have mask_token)
    - Query: learnable decoder queries + channel/sensor metadata
    
    This allows the decoder to use channel position metadata to determine WHAT to reconstruct,
    while attending to visible channels to learn HOW to reconstruct it.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        patch_size: int = 16,
        max_channels: int = 20,
        max_patches: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.max_patches = max_patches
        
        # Metadata embeddings (same as before)
        self.channel_embed = nn.Embedding(max_channels, d_model)
        self.sensor_embed = SensorEmbedding(d_model)
        self.sample_rate_encoding = SampleRateEncoding(d_model, base_patch_size=patch_size)
        
        # Learnable positional encoding for decoder queries (use large max_len for flexibility)
        self.pos_encoding = PositionalEncoding(d_model, max_len=10000, dropout=dropout)
        
        # Learnable decoder query tokens - these ask "what should channel X, patch Y be?"
        # Initialized with proper scale to match encoder outputs
        self.decoder_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.5)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Reconstruct patches
        self.to_patch = nn.Linear(d_model, patch_size)
    
    def forward(
        self,
        z: torch.Tensor,
        sensor_indices: Optional[torch.Tensor] = None,
        sample_rate: Optional[torch.Tensor] = None,
        training_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MAE-style decoding: use cross-attention from decoder queries to encoder outputs.
        
        Args:
            z: [B, C, num_patches, D] quantized encoder representations
            sensor_indices: [B, C] sensor model indices
            sample_rate: [B] sample rate in Hz for each sample
            training_mask: [B, C] boolean mask (True = visible, False = masked) - used for memory masking
            
        Returns:
            x_recon: [B, C, T] reconstructed time series
        """
        B, C, num_patches, D = z.shape
        
        # === Build Memory (Key/Value) from encoder outputs ===
        # Flatten encoder outputs: [B, C * num_patches, D]
        memory = z.reshape(B, C * num_patches, D)
        
        # Add channel embedding to memory
        channel_ids = torch.arange(C, device=z.device).unsqueeze(0).expand(B, -1)
        channel_emb = self.channel_embed(channel_ids)  # [B, C, D]
        channel_emb_expanded = channel_emb.unsqueeze(2).expand(-1, -1, num_patches, -1)
        channel_emb_flat = channel_emb_expanded.reshape(B, C * num_patches, D)
        memory = memory + channel_emb_flat
        
        # Add sensor embedding to memory
        if sensor_indices is not None:
            sensor_emb = self.sensor_embed(sensor_indices)
            sensor_emb_expanded = sensor_emb.unsqueeze(2).expand(-1, -1, num_patches, -1)
            sensor_emb_flat = sensor_emb_expanded.reshape(B, C * num_patches, D)
            memory = memory + sensor_emb_flat
        
        # Add sample rate encoding to memory
        if sample_rate is not None:
            rate_enc = self.sample_rate_encoding(sample_rate)  # [B, D]
            rate_enc_expanded = rate_enc.unsqueeze(1).expand(-1, C * num_patches, -1)
            memory = memory + rate_enc_expanded
        
        # === Build Query for decoder ===
        # Query = learnable token + positional encoding + channel embedding + sensor embedding
        # This tells the decoder "reconstruct channel X, patch Y"
        
        # Start with learnable query token expanded to all positions
        query = self.decoder_query.expand(B, C * num_patches, -1).clone()
        
        # Add positional encoding (repeat for each channel)
        pos_enc = self.pos_encoding.pe[:, :num_patches, :].repeat(1, C, 1)  # [1, C*num_patches, D]
        query = query + pos_enc
        
        # Add channel embedding (tells decoder WHICH channel to reconstruct)
        query = query + channel_emb_flat
        
        # Add sensor embedding (tells decoder the sensor type for this channel)
        if sensor_indices is not None:
            query = query + sensor_emb_flat
        
        # Add sample rate encoding
        if sample_rate is not None:
            query = query + rate_enc_expanded
        
        # === Cross-attention decoding ===
        # Query attends to Memory to gather information for reconstruction
        # The key insight: masked channels' queries can attend to visible channels' memory
        decoded = self.transformer(query, memory)
        
        # Reconstruct patches
        patches = self.to_patch(decoded)  # [B, C * num_patches, patch_size]
        patches = patches.reshape(B, C, num_patches, self.patch_size)
        
        # Concatenate patches to get time series
        x_recon = patches.reshape(B, C, -1)  # [B, C, T]
        
        return x_recon


class EnoseVQVAE(L.LightningModule):
    """VQ-VAE Lightning Module for e-nose pretraining.
    
    Self-supervised pretraining that learns:
    1. Time series patch representations via VQ-VAE
    2. Cross-channel dependencies via channel masking
    3. Sensor-specific patterns via metadata conditioning
    """
    
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        patch_size: int = 16,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        max_channels: int = 20,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        mask_ratio: float = 0.25,
        lambda_visible: float = 1.0,
        lambda_masked: float = 1.0,
        # VQ control
        disable_vq: bool = False,
        # Loss type
        loss_type: str = "mse",  # mse, mae, huber, cosine, correlation, mse_mmd
        huber_delta: float = 1.0,
        # Optimizer params
        optimizer_type: str = "adamw",  # adamw, muon, soap
        # LR scheduler params
        lr_scheduler: str = "cosine_warmup",  # cosine_warmup, cosine, constant
        lr_warmup_steps: int = 1000,
        lr_T_mult: int = 2,
        lr_min: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = EnoseEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            patch_size=patch_size,
            max_channels=max_channels,
        )
        
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=d_model,
            commitment_cost=commitment_cost,
        )
        
        self.decoder = EnoseDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            patch_size=patch_size,
            max_channels=max_channels,
        )
        
        # Metrics
        self.train_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        self.train_recon_visible = MeanMetric()
        self.train_recon_masked = MeanMetric()
        self.train_vq_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_recon_visible = MeanMetric()
        self.val_recon_masked = MeanMetric()
        self.val_best_loss = MinMetric()
    
    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        sensor_indices: Optional[torch.Tensor] = None,
        sample_rate: Optional[torch.Tensor] = None,
        training_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, T] input time series
            channel_mask: [B, C] boolean mask for PADDING (True = valid, False = padding)
            sensor_indices: [B, C] sensor model indices
            sample_rate: [B] sample rate in Hz for each sample
            training_mask: [B, C] boolean mask for TRAINING (True = visible, False = masked for learning)
        """
        # Encode - pass training_mask to use mask_token for masked channels
        z = self.encoder(x, channel_mask, sensor_indices, sample_rate, training_mask)
        
        # Vector quantize (or bypass if disabled)
        B, C, num_patches, D = z.shape
        if self.hparams.disable_vq:
            z_q = z
            vq_loss = torch.tensor(0.0, device=z.device)
            perplexity = torch.tensor(0.0, device=z.device)
        else:
            z_flat = z.reshape(B * C, num_patches, D)
            z_q, vq_loss, perplexity = self.vq(z_flat)
            z_q = z_q.reshape(B, C, num_patches, D)
        
        # Decode - pass training_mask for MAE-style cross-attention
        x_recon = self.decoder(z_q, sensor_indices, sample_rate, training_mask)
        
        return {
            "x_recon": x_recon,
            "z": z,
            "z_q": z_q,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
        }
    
    def _apply_random_mask(
        self,
        x: torch.Tensor,
        valid_channel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random channel masking only to valid (non-padded) channels.
        
        Args:
            x: [B, C, T] input tensor
            valid_channel_mask: [B, C] boolean mask where True = valid channel (not padding)
            
        Returns:
            x_masked: [B, C, T] with some channels zeroed
            mask: [B, C] where True = visible, False = masked for training
        """
        B, C, T = x.shape
        mask = torch.ones(B, C, dtype=torch.bool, device=x.device)
        
        for b in range(B):
            if valid_channel_mask is not None:
                # Only mask among valid channels
                valid_indices = torch.where(valid_channel_mask[b])[0]
                num_valid = len(valid_indices)
                num_mask = int(num_valid * self.hparams.mask_ratio)
                if num_mask > 0 and num_valid > 0:
                    perm = torch.randperm(num_valid, device=x.device)[:num_mask]
                    mask_indices = valid_indices[perm]
                    mask[b, mask_indices] = False
            else:
                # No valid mask provided, mask among all channels
                num_mask = int(C * self.hparams.mask_ratio)
                if num_mask > 0:
                    mask_indices = torch.randperm(C, device=x.device)[:num_mask]
                    mask[b, mask_indices] = False
        
        # Apply mask
        x_masked = x.clone()
        x_masked[~mask.unsqueeze(-1).expand_as(x)] = 0
        
        return x_masked, mask
    
    def _compute_recon_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss based on configured loss type.
        
        Args:
            pred: Predicted values (flattened)
            target: Target values (flattened)
            
        Returns:
            Scalar loss value
        """
        loss_type = self.hparams.loss_type
        
        if loss_type == "mse":
            return F.mse_loss(pred, target)
        
        elif loss_type == "mae":
            return F.l1_loss(pred, target)
        
        elif loss_type == "huber":
            return F.huber_loss(pred, target, delta=self.hparams.huber_delta)
        
        elif loss_type == "cosine":
            # Cosine similarity loss: 1 - cos_sim
            # Reshape to [N, D] for cosine similarity
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
                target = target.unsqueeze(0)
            cos_sim = F.cosine_similarity(pred, target, dim=-1)
            return 1 - cos_sim.mean()
        
        elif loss_type == "correlation":
            # Pearson correlation loss: 1 - correlation
            pred_centered = pred - pred.mean()
            target_centered = target - target.mean()
            pred_std = pred_centered.std() + 1e-8
            target_std = target_centered.std() + 1e-8
            correlation = (pred_centered * target_centered).mean() / (pred_std * target_std)
            return 1 - correlation
        
        elif loss_type == "mse_corr":
            # Combined MSE + Correlation loss
            mse = F.mse_loss(pred, target)
            pred_centered = pred - pred.mean()
            target_centered = target - target.mean()
            pred_std = pred_centered.std() + 1e-8
            target_std = target_centered.std() + 1e-8
            correlation = (pred_centered * target_centered).mean() / (pred_std * target_std)
            return mse + 0.5 * (1 - correlation)
        
        elif loss_type == "mse_mmd":
            # Combined MSE + MMD (Maximum Mean Discrepancy) loss
            mse = F.mse_loss(pred, target)
            mmd = self._compute_mmd(pred, target)
            return mse + 0.5 * mmd
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _compute_mmd(self, x: torch.Tensor, y: torch.Tensor, kernel: str = "rbf") -> torch.Tensor:
        """Compute Maximum Mean Discrepancy (MMD) between two distributions.
        
        Args:
            x: Samples from distribution P (flattened)
            y: Samples from distribution Q (flattened)
            kernel: Kernel type ('rbf' or 'linear')
            
        Returns:
            MMD distance (scalar)
        """
        # Reshape to [N, 1] for kernel computation
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        
        # Sample subset for efficiency (MMD is O(n²))
        max_samples = 1000
        if x.shape[0] > max_samples:
            idx = torch.randperm(x.shape[0], device=x.device)[:max_samples]
            x = x[idx]
        if y.shape[0] > max_samples:
            idx = torch.randperm(y.shape[0], device=y.device)[:max_samples]
            y = y[idx]
        
        if kernel == "rbf":
            # RBF kernel with automatic bandwidth (median heuristic)
            xx = torch.cdist(x, x) ** 2
            yy = torch.cdist(y, y) ** 2
            xy = torch.cdist(x, y) ** 2
            
            # Median heuristic for bandwidth
            median_dist = torch.median(torch.cat([xx.view(-1), yy.view(-1), xy.view(-1)]))
            sigma = median_dist / (2 * torch.log(torch.tensor(x.shape[0] + 1.0, device=x.device)))
            sigma = torch.clamp(sigma, min=1e-5)
            
            k_xx = torch.exp(-xx / (2 * sigma))
            k_yy = torch.exp(-yy / (2 * sigma))
            k_xy = torch.exp(-xy / (2 * sigma))
        else:  # linear kernel
            k_xx = x @ x.T
            k_yy = y @ y.T
            k_xy = x @ y.T
        
        # MMD² = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
        return torch.clamp(mmd, min=0)  # MMD should be non-negative
    
    def _compute_loss(
        self,
        batch: Dict[str, Any],
        stage: str = "train",
    ) -> Dict[str, torch.Tensor]:
        x = batch["data"]  # [B, C, T]
        sensor_indices = batch.get("sensor_indices")  # [B, C]
        sample_rate = batch.get("sample_rate")  # [B]
        valid_channel_mask = batch.get("channel_mask")  # [B, C] True = valid (not padding)
        
        # Apply channel masking during training (only to valid channels)
        x_masked, train_mask = self._apply_random_mask(x, valid_channel_mask)
        
        # Forward pass - pass BOTH valid_channel_mask (for padding) and train_mask (for mask_token)
        outputs = self(x_masked, valid_channel_mask, sensor_indices, sample_rate, training_mask=train_mask)
        
        # Separate reconstruction losses for visible and masked channels
        # But only compute loss on VALID channels (exclude padding)
        x_recon = outputs["x_recon"]
        
        # Expand masks to time dimension: [B, C] -> [B, C, T]
        train_mask_expanded = train_mask.unsqueeze(-1).expand_as(x)  # True = visible in training
        
        if valid_channel_mask is not None:
            valid_expanded = valid_channel_mask.unsqueeze(-1).expand_as(x)  # True = valid channel
            # Only compute loss on valid channels
            visible_mask = train_mask_expanded & valid_expanded  # visible AND valid
            masked_mask = (~train_mask_expanded) & valid_expanded  # masked AND valid
        else:
            visible_mask = train_mask_expanded
            masked_mask = ~train_mask_expanded
        
        # Visible channel reconstruction loss (only valid channels)
        if visible_mask.any():
            loss_visible = self._compute_recon_loss(x_recon[visible_mask], x[visible_mask])
        else:
            loss_visible = torch.tensor(0.0, device=x.device)
        
        # Masked channel prediction loss (only valid channels)
        if masked_mask.any():
            loss_masked = self._compute_recon_loss(x_recon[masked_mask], x[masked_mask])
        else:
            loss_masked = torch.tensor(0.0, device=x.device)
        
        # Combined reconstruction loss with separate weights
        recon_loss = (self.hparams.lambda_visible * loss_visible + 
                      self.hparams.lambda_masked * loss_masked)
        
        # Total loss
        total_loss = recon_loss + outputs["vq_loss"]
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "loss_visible": loss_visible,
            "loss_masked": loss_masked,
            "vq_loss": outputs["vq_loss"],
            "perplexity": outputs["perplexity"],
        }
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        losses = self._compute_loss(batch, stage="train")
        
        self.train_loss(losses["loss"].detach())
        self.train_recon_loss(losses["recon_loss"].detach())
        self.train_recon_visible(losses["loss_visible"].detach())
        self.train_recon_masked(losses["loss_masked"].detach())
        self.train_vq_loss(losses["vq_loss"].detach())
        
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", self.train_recon_loss, on_step=False, on_epoch=True)
        self.log("train/recon_visible", self.train_recon_visible, on_step=False, on_epoch=True)
        self.log("train/recon_masked", self.train_recon_masked, on_step=False, on_epoch=True)
        self.log("train/vq_loss", self.train_vq_loss, on_step=False, on_epoch=True)
        self.log("train/perplexity", losses["perplexity"].detach(), on_step=False, on_epoch=True)
        
        return losses["loss"]
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        losses = self._compute_loss(batch, stage="val")
        
        self.val_loss(losses["loss"].detach())
        self.val_recon_visible(losses["loss_visible"].detach())
        self.val_recon_masked(losses["loss_masked"].detach())
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon_loss", losses["recon_loss"].detach(), on_step=False, on_epoch=True)
        self.log("val/recon_visible", self.val_recon_visible, on_step=False, on_epoch=True)
        self.log("val/recon_masked", self.val_recon_masked, on_step=False, on_epoch=True)
        self.log("val/vq_loss", losses["vq_loss"].detach(), on_step=False, on_epoch=True)
        self.log("val/perplexity", losses["perplexity"].detach(), on_step=False, on_epoch=True)
        
        return losses["loss"]
    
    def on_validation_epoch_end(self) -> None:
        val_loss = self.val_loss.compute()
        self.val_best_loss(val_loss)
        self.log("val/best_loss", self.val_best_loss.compute(), prog_bar=True)
    
    def configure_optimizers(self):
        # Select optimizer
        optimizer_type = self.hparams.optimizer_type
        
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif optimizer_type == "muon":
            # Muon optimizer (PyTorch 2.x built-in)
            try:
                from torch.optim import Muon
                optimizer = Muon(
                    self.parameters(),
                    lr=self.hparams.learning_rate,
                    momentum=0.95,
                )
            except ImportError:
                print("Warning: Muon not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.hparams.learning_rate,
                    weight_decay=self.hparams.weight_decay,
                )
        elif optimizer_type == "soap":
            from enose_uci_dataset.pretrain.optimizers import SOAP
            optimizer = SOAP(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                precondition_frequency=10,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        lr_scheduler_type = self.hparams.lr_scheduler
        
        if lr_scheduler_type == "cosine_warmup":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.hparams.lr_warmup_steps,
                T_mult=self.hparams.lr_T_mult,
                eta_min=self.hparams.lr_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 500,
                eta_min=self.hparams.lr_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        elif lr_scheduler_type == "onecycle":
            # OneCycleLR: single cycle, no restart, smooth warmup and decay
            # Estimate total steps
            if self.trainer and self.trainer.estimated_stepping_batches:
                total_steps = self.trainer.estimated_stepping_batches
            else:
                total_steps = 10000  # fallback
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,  # 10% warmup
                anneal_strategy='cos',
                final_div_factor=self.hparams.learning_rate / self.hparams.lr_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif lr_scheduler_type == "plateau":
            # ReduceLROnPlateau: reduce LR when val loss plateaus
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=20,
                min_lr=self.hparams.lr_min,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            }
        elif lr_scheduler_type == "warmup_cosine":
            # Linear warmup + cosine decay (no restart)
            from torch.optim.lr_scheduler import LambdaLR
            import math
            
            warmup_steps = self.hparams.lr_warmup_steps
            if self.trainer and self.trainer.estimated_stepping_batches:
                total_steps = self.trainer.estimated_stepping_batches
            else:
                total_steps = 10000
            
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, warmup_steps))
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(self.hparams.lr_min / self.hparams.learning_rate, 
                          0.5 * (1.0 + math.cos(math.pi * progress)))
            
            scheduler = LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif lr_scheduler_type == "cosine_restart_decay":
            # Cosine annealing with warm restarts AND peak decay
            # Each restart, the peak LR is multiplied by lr_decay_factor
            from torch.optim.lr_scheduler import LambdaLR
            import math
            
            T_0 = self.hparams.lr_warmup_steps  # Initial period
            T_mult = self.hparams.lr_T_mult
            lr_decay = 0.8  # Peak decay factor per restart
            eta_min_ratio = self.hparams.lr_min / self.hparams.learning_rate
            
            def lr_lambda(current_step):
                # Find which cycle we're in and position within cycle
                step = current_step
                T_cur = T_0
                cycle = 0
                
                while step >= T_cur:
                    step -= T_cur
                    T_cur = int(T_cur * T_mult)
                    cycle += 1
                
                # Peak LR for this cycle (decays each restart)
                peak_mult = lr_decay ** cycle
                
                # Cosine annealing within cycle
                cos_value = 0.5 * (1 + math.cos(math.pi * step / T_cur))
                
                # Scale between eta_min and peak
                return max(eta_min_ratio, eta_min_ratio + (peak_mult - eta_min_ratio) * cos_value)
            
            scheduler = LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:  # constant
            return optimizer
    
    def get_latent(
        self,
        x: torch.Tensor,
        sensor_indices: Optional[torch.Tensor] = None,
        sample_rate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get quantized latent representations for downstream tasks."""
        self.eval()
        with torch.no_grad():
            z = self.encoder(x, None, sensor_indices, sample_rate)
            B, C, num_patches, D = z.shape
            z_flat = z.reshape(B * C, num_patches, D)
            z_q, _, _ = self.vq(z_flat)
            z_q = z_q.reshape(B, C, num_patches, D)
        return z_q
