import torch
import torch.nn.functional as F
from torch import Tensor, nn
import random
import numpy as np
import math

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ff = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        self.residual_proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor):
        return self.ff(x) + self.residual_proj(x)

# ---------------- FFN MoE ----------------
class FFNMoE(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.w_gate = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([
            FeedForward(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.register_buffer("eps", torch.tensor(1e-9))

    def forward(self, x: Tensor):
        B, N, D = x.shape
        logits = self.w_gate(x)  # [B,N,E]
        probs = F.softmax(logits, dim=-1)
        if self.top_k < self.num_experts:
            topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
            mask = torch.zeros_like(probs).scatter(-1, topk_idx, topk_vals)
            gate_scores = mask / (mask.sum(dim=-1, keepdim=True) + self.eps)
        else:
            gate_scores = probs
        expert_outputs = [expert(x) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=-1)  # [B,N,D,E]
        moe_output = torch.sum(gate_scores.unsqueeze(-2) * stacked, dim=-1)  # [B,N,D]
        return moe_output, gate_scores


# ---------------- Low-Rank Temporal Interaction ----------------
class LowRankTemporalInteraction(nn.Module):
    def __init__(self, hidden_dim, rank=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.U = nn.Linear(hidden_dim, rank, bias=False)
        self.V = nn.Linear(hidden_dim, rank, bias=False)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, H):
        # x: [B,N,D], H: [B,N,T,D]
        B, N, D = x.shape
        _, _, T, _ = H.shape

        x_r = self.U(x).unsqueeze(2)      # [B,N,1,rank]
        H_r = self.V(H)                    # [B,N,T,rank]
        interaction = (x_r * H_r).sum(-1)  # [B,N,T]
        attn_w = F.softmax(interaction, dim=-1).unsqueeze(-1)  # [B,N,T,1]
        temporal_fused = (H * attn_w).sum(dim=2)               # [B,N,D]
        return x + self.W(temporal_fused)                      # [B,N,D]

# ---------------- S_MLPMixerBlock ----------------
class S_MLPMixerBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim, top_k_moe=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Token mixer
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mixer = nn.Sequential(
            nn.Linear(num_patches, num_patches*4),
            nn.GELU(),
            nn.Linear(num_patches*4, num_patches)
        )
        self.token_dropout = nn.Dropout(0.1)

        self.temporal_interaction = LowRankTemporalInteraction(hidden_dim, rank=min(32, hidden_dim//2))

        # Channel mixer: MoE
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mixer = FFNMoE(hidden_dim, num_experts=4, top_k=top_k_moe)

    def forward(self, x, adj, history_emb):
        # node -> group
        xg = torch.einsum('ng,bnd->bgd', adj, x)
        xg = self.ln_token(xg)
        # token level
        y = xg.transpose(1,2)
        y = self.token_mixer(y)
        y = self.token_dropout(y)
        y = y.transpose(1,2)
        yn = torch.einsum('gn,bgd->bnd', adj.mT, y)
        x = x + self.alpha * yn

        # temporal low-rank interaction
        x = self.temporal_interaction(x, history_emb)

        # channel mixing: MoE
        x_norm = self.ln_channel(x)
        y_c, _ = self.channel_mixer(x_norm)
        x = x + self.alpha * y_c
        return x

class GLTMoE(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.num_nodes = model_args["num_nodes"]
        self.num_group = model_args["num_group"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["input_embedding_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]

        # embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.group = nn.Parameter(torch.randn(self.num_nodes, self.num_group))
        self.group_proj = nn.Linear(self.embed_dim + self.node_dim*int(self.if_spatial)
                                    + self.temp_dim_tid*int(self.if_time_in_day)
                                    + self.temp_dim_diw*int(self.if_day_in_week),
                                    self.num_group)
        self.hidden_dim = (self.embed_dim
                           + self.node_dim*int(self.if_spatial)
                           + self.temp_dim_tid*int(self.if_time_in_day)
                           + self.temp_dim_diw*int(self.if_day_in_week))
        self.time_series_emb_layer = nn.Linear(self.input_dim*self.input_len, self.embed_dim)
        self.time_step_proj = nn.Linear(self.input_dim, self.hidden_dim)

        self.encoder = nn.ModuleList([
            S_MLPMixerBlock(self.num_group, self.hidden_dim, top_k_moe=3)
            for _ in range(self.num_layer)
        ])
        self.regression_layer = nn.Linear(self.hidden_dim, self.output_len)

    def forward(self, history_data: Tensor):
        device = history_data.device
        B,T,N,C = history_data.shape
        input_data = history_data[..., :self.input_dim]  # [B,T,N,input_dim]
        input_flat = input_data.transpose(1, 2).contiguous().view(B, N, -1)
        time_series_emb = self.time_series_emb_layer(input_flat)

        per_step_proj = self.time_step_proj(input_data.permute(0, 2, 1, 3))  # [B,N,T,D]

        emb_list = [time_series_emb]
        if self.if_spatial:
            emb_list.append(self.node_emb.unsqueeze(0).expand(B,-1,-1))
        if self.if_time_in_day:
            t_vals = history_data[...,1][:,-1,:]
            idx_tid = torch.floor(t_vals*self.time_of_day_size).long().to(device)
            emb_list.append(self.time_in_day_emb[idx_tid])
        if self.if_day_in_week:
            d_vals = history_data[...,2][:,-1,:]
            idx_diw = torch.floor(d_vals).long().to(device)
            emb_list.append(self.day_in_week_emb[idx_diw])

        hidden = torch.cat(emb_list, dim=2)

        node2group = self.group_proj(hidden)
        group_adj = F.softmax(self.group + node2group.mean(dim=0), dim=1)

        x = hidden
        for block in self.encoder:
            x = block(x, group_adj, history_emb=per_step_proj)

        pred = self.regression_layer(x).transpose(1, 2).unsqueeze(-1)
        return pred


if __name__ == "__main__":
    args = {
        "num_nodes": 170,
        "num_group": 10,
        "node_dim": 32,
        "input_len": 12,
        "input_dim": 3,
        "input_embedding_dim": 32,
        "output_len": 12,
        "num_layer": 3,
        "temp_dim_tid": 32,
        "temp_dim_diw": 32,
        "time_of_day_size": 288,
        "day_of_week_size": 7,
        "if_T_i_D": True,
        "if_D_i_W": True,
        "if_node": True,
    }

    model = GLTMoE(**args).cuda()
    x = torch.rand(16, 12, 170, 3).cuda()
    y = model(x)
    print("Output:", y.shape)  # [16, 12, 170, 1]