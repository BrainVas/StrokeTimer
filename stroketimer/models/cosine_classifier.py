# -*- coding: utf-8 -*-
# Cosine classifier with optional learnable scale.
# Compatible with your runner: forward(features, centroids, phase, labels)
#
# z_j = scale * cos(theta_j) = scale * (w_j^T f) / (||w_j|| * ||f||)
# Reference: "CosFace", "ArcFace", "Balanced Softmax for Long-Tailed Recognition"

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, scale=30.0, learnable_scale=True):
        """
        Args:
            feat_dim (int): Feature dimension (e.g., 256)
            num_classes (int): Number of classes
            scale (float): Scaling factor applied to cosine logits
            learnable_scale (bool): Whether 'scale' is a learnable parameter
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # Class weight vectors (each row corresponds to a class direction)
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(scale))
        else:
            self.register_buffer("scale", torch.tensor(scale))

    def forward(self, features, centroids=None, phase='train', labels=None):
        """
        Args:
            features: Tensor of shape [B, feat_dim]
            centroids, phase, labels: Ignored (kept for runner API compatibility)
        Returns:
            logits: [B, num_classes]
            aux: None
        """
        # Normalize both features and class weights
        f_norm = F.normalize(features, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = torch.matmul(f_norm, w_norm.t())  # [B, C]

        # Scale logits
        logits = self.scale * cosine
        return logits, None


def create_model(feat_dim, num_classes, scale=30.0, learnable_scale=True, **kwargs):
    """
    Factory entrypoint (for YAML-based runner).
    """
    print(f"Loading Cosine Classifier (feat_dim={feat_dim}, num_classes={num_classes}, scale={scale})")
    return CosineClassifier(
        feat_dim=feat_dim,
        num_classes=num_classes,
        scale=scale,
        learnable_scale=learnable_scale,
    )
