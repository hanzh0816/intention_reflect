"""
SNN version of MapEncoder module

This module encodes map information including road polygons,
traffic lights, speed limits, and route information using SNN components.
"""

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

from ..layers.snn_embedding import SNNPointsEncoder


class SNNMapEncoder(nn.Module):
    """
    SNN version of MapEncoder

    Encodes map polygon features using SNN components.
    """

    def __init__(
        self,
        polygon_channel=6,
        dim=128,
        tau=2.0,
        backend='torch',
    ):
        super().__init__()

        self.dim = dim

        # Polygon geometry encoder
        self.polygon_encoder = SNNPointsEncoder(
            feat_channel=polygon_channel,
            encoder_channel=dim,
            tau=tau,
            backend=backend,
        )

        # Speed limit embedding
        self.speed_limit_fc1 = nn.Linear(1, dim, bias=False)
        self.speed_limit_bn1 = nn.BatchNorm1d(dim)
        self.speed_limit_lif1 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

        self.speed_limit_fc2 = nn.Linear(dim, dim, bias=False)
        self.speed_limit_bn2 = nn.BatchNorm1d(dim)
        self.speed_limit_lif2 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

        # Categorical embeddings (not time-dependent)
        self.type_emb = nn.Embedding(3, dim)
        self.on_route_emb = nn.Embedding(2, dim)
        self.traffic_light_emb = nn.Embedding(4, dim)
        self.unknown_speed_emb = nn.Embedding(1, dim)

    def forward(self, data, time_steps=4):
        """
        Args:
            data: Input feature dict
            time_steps: Number of SNN time steps (T)

        Returns:
            [T, B, M, C] - encoded map polygon features
        """
        T = time_steps

        polygon_center = data["map"]["polygon_center"]
        polygon_type = data["map"]["polygon_type"].long()
        polygon_on_route = data["map"]["polygon_on_route"].long()
        polygon_tl_status = data["map"]["polygon_tl_status"].long()
        polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
        polygon_speed_limit = data["map"]["polygon_speed_limit"]
        point_position = data["map"]["point_position"]
        point_vector = data["map"]["point_vector"]
        point_orientation = data["map"]["point_orientation"]
        valid_mask = data["map"]["valid_mask"]

        # Prepare polygon features
        polygon_feature = torch.cat([
            point_position[:, :, 0] - polygon_center[..., None, :2],
            point_vector[:, :, 0],
            torch.stack([
                point_orientation[:, :, 0].cos(),
                point_orientation[:, :, 0].sin(),
            ], dim=-1),
        ], dim=-1)

        bs, M, P, C = polygon_feature.shape
        valid_mask_reshaped = valid_mask[:, :, 0, :]  # Take first lane
        valid_mask_flat = valid_mask_reshaped.view(bs * M, P)
        polygon_feature_flat = polygon_feature.reshape(bs * M, P, C)

        # Encode polygon geometry
        # Note: SNNPointsEncoder processes without time dimension, we expand later
        x_polygon = self.polygon_encoder(
            polygon_feature_flat.view(bs, M, P, C),
            valid_mask_reshaped.view(bs, M, P)
        )  # [B, M, dim]

        # Expand to time dimension
        x_polygon = x_polygon.unsqueeze(0).repeat(T, 1, 1, 1)  # [T, B, M, dim]

        # Process categorical embeddings (expanded to time dimension)
        x_type = self.type_emb(polygon_type).unsqueeze(0).repeat(T, 1, 1, 1)
        x_on_route = self.on_route_emb(polygon_on_route).unsqueeze(0).repeat(T, 1, 1, 1)
        x_tl_status = self.traffic_light_emb(polygon_tl_status).unsqueeze(0).repeat(T, 1, 1, 1)

        # Process speed limit with SNN
        x_speed_limit = torch.zeros(T, bs, M, self.dim, device=x_polygon.device)

        # For polygons with speed limit
        if polygon_has_speed_limit.any():
            speed_values = polygon_speed_limit[polygon_has_speed_limit].unsqueeze(-1)  # [N, 1]
            N = speed_values.shape[0]

            # Expand to time dimension
            speed_values_t = speed_values.unsqueeze(0).repeat(T, 1, 1)  # [T, N, 1]
            speed_values_flat = speed_values_t.reshape(T * N, 1)

            # First layer
            x_speed = self.speed_limit_fc1(speed_values_flat)
            x_speed = self.speed_limit_bn1(x_speed)
            x_speed = x_speed.reshape(T, N, -1)
            x_speed = self.speed_limit_lif1(x_speed)

            # Second layer
            x_speed_flat = x_speed.reshape(T * N, self.dim)
            x_speed = self.speed_limit_fc2(x_speed_flat)
            x_speed = self.speed_limit_bn2(x_speed)
            x_speed = x_speed.reshape(T, N, self.dim)
            x_speed = self.speed_limit_lif2(x_speed)

            # Scatter back to full shape
            x_speed_limit[:, polygon_has_speed_limit] = x_speed

        # For polygons without speed limit
        if (~polygon_has_speed_limit).any():
            # unknown_speed_emb.weight: [1, dim]
            x_speed_limit[:, ~polygon_has_speed_limit] = (
                self.unknown_speed_emb.weight.unsqueeze(0).repeat(T, (~polygon_has_speed_limit).sum(), 1)
            )

        # Combine all embeddings
        x_polygon = x_polygon + x_type + x_on_route + x_tl_status + x_speed_limit

        return x_polygon
