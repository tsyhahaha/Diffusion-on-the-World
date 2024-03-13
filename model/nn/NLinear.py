import torch.nn as nn
import torch
import numpy as np
from einops import rearrange

class NLinear(nn.Module):
    def __init__(self, position_dim=6, embedding_dim=16):
        # generally, what is score network needed as input?
        self.position_proj = nn.Linear(position_dim, embedding_dim)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, position_dim)
        self.time_step_embedder = TimestepEmbedder(embedding_dim=32, output_dim=embedding_dim)

    def forward(self, batch):
        x = self.relu(self.position_proj(batch['position']))
        time_emb = self.time_step_embedder(batch['t'])
        x = self.relu(self.fc1(x + time_emb))
        x = self.fc2(x)
        return x

class TimestepEmbedder(nn.Module):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    def __init__(self, embedding_dim, output_dim, max_positions=1000):
        super().__init__()
        self.max_positions = max_positions

        half_dim = embedding_dim // 2
        emb = np.log(max_positions) / (half_dim - 1)
        self.emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,) * -emb)

        # no lora config for timestep embeddings
        self.proj_out = nn.Linear(embedding_dim, output_dim, init='final', bias=False)

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1

        # (batch_size, )
        timesteps = rearrange(timesteps * self.max_positions, 'b -> b ()')
        # (embed_dim,)
        emb = rearrange(self.emb.to(device=timesteps.device), 'c -> () c')

        # (batch_size, embed_dim)
        emb = timesteps * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        # (batch_size, seq_channel)
        emb = self.proj_out(emb)

        return emb
