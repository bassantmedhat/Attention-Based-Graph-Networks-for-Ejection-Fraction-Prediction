import torch.nn as nn
import numpy as np


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        """
        MultiHeadAttentionWrapperV2 using PyTorch's nn.MultiheadAttention.

        Args:
            embed_dim (int): Input feature size (channels_in).
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super(MultiHeadAttentionWrapper, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # PyTorch's built-in MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Post-attention processing
        self.layer_norm = nn.LayerNorm(
            embed_dim
        )  # Normalization over the feature dimension
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, query=None, key=None, value=None, attn_mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_samples, embed_dim).
            query (torch.Tensor, optional): Query tensor (default: x).
            key (torch.Tensor, optional): Key tensor (default: x).
            value (torch.Tensor, optional): Value tensor (default: x).
            attn_mask (torch.Tensor, optional): Attention mask (default: None).

        Returns:
            output (torch.Tensor): Processed tensor of shape (batch_size, num_samples, embed_dim).
            attn_weights (torch.Tensor): Attention weights of shape (batch_size, num_heads, num_samples, num_samples).
        """
        batch_size, num_samples, embed_dim = x.shape
        assert (
            embed_dim == self.embed_dim
        ), f"Expected input with {self.embed_dim} features, got {embed_dim}."

        # Default query, key, and value to x if not provided
        query = query if query is not None else x
        key = key if key is not None else x
        value = value if value is not None else x

        # Pass through MultiheadAttention (batch_first=True ensures input shape consistency)
        attn_output, attn_weights = self.attention(
            query, key, value, attn_mask=attn_mask
        )

        # Residual connection, normalization, and dropout
        output = self.layer_norm(x + attn_output)  # Add residual connection
        output = self.dropout(output)

        return output, attn_weights


