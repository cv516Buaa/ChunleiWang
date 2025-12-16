import math
from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.pscan import pscan
from einops import rearrange
import random
import numpy as np
from models.transformer.attention import ImplicitPhysicsAttention


@dataclass
class MambaConfig:
    d_model: int = 512  # D
    n_layers: int = 3
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    res_drop = 0.1

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False  # apply layernorms to internal activations

    pscan: bool = True  # use parallel scan mode or sequential mode when training
    use_cuda: bool = False  # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class MutiHeadSelfAttention(nn.Module):
    def __init__(self, dim, attention_dim, d_state, h=1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MutiHeadSelfAttention, self).__init__()
        attention_dim = attention_dim//16
        self.decoder_att = nn.Linear(dim * d_state, attention_dim)
        d_k = attention_dim // h
        d_v = d_k
        self.fc_q = nn.Linear(attention_dim, h * d_k)
        self.fc_k = nn.Linear(attention_dim, h * d_k)
        self.fc_v = nn.Linear(attention_dim, h * d_v)
        self.fc_o = nn.Linear(h * d_v, dim * d_state)
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()


    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, hidden):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq, decoder_dim, state = hidden.shape

        queries = self.decoder_att(hidden.contiguous().view(b_s, nq, -1))
        keys = queries
        values = queries
        
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)


        att = torch.softmax(att, -1)
        #from feature_visualization import draw_feature_map
        #draw_feature_map(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out).view(b_s, nq, decoder_dim, state)  # (b_s, nq, d_model)

        return out



class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config
        self.layers = nn.Sequential(ResidualBlock(config,dilation=2),
                                    ResidualBlock(config,dilation=4),
                                    ResidualBlock(config, dilation=8))
        
        self.coord_att = CoordAttention(in_channels=49, h=8, w=64)
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=512 // 2, h=8, w=64)
        self.proj = nn.Conv1d(512, 49, 1)
        self.env_phy_attn = ImplicitPhysicsAttention(d_model = 512,h = 8)
    def forward(self, x):
        # x : (B, L, D)
        # y : (B, L, D)
        outs = []
        #for layer in self.layers:
        #    x = layer(x)
        #    outs.append(x.unsqueeze(1))

        for l in range(len(self.layers)):
            x = 0.8 * x + 0.2 * self.env_phy_attn(x)
            x = self.layers[l](x)
            if l==0:
                x = 0.9*x + 0.1*self.coord_att(x)
                #plot_feature_heatmaps(x)
            elif l==1:
                x = x
            elif l==2:
                x = 0.9*x + 0.1*self.proj(self.pos_enc(x))
            outs.append(x.unsqueeze(1))
        outs = torch.cat(outs, 1)

        return outs

class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, h=8, w=64):
        super().__init__()
        self.h = h
        self.w = w
        self.reduction = reduction

        self.x_pool = nn.AdaptiveAvgPool1d(1)
        self.y_pool = nn.AdaptiveAvgPool1d(1)

        self.channel_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, L = x.size()

        x_2d = x.view(B, C, self.h, self.w)


        x_h = self.x_pool(x_2d.mean(dim=2).view(B, C, -1))  # [B,C,W] → [B,C,1]
        x_w = self.y_pool(x_2d.mean(dim=3).view(B, C, -1))  # [B,C,H] → [B,C,1]
        channel_att = self.channel_conv(x_h + x_w)  # [B,C,1]


        avg_pool = x_2d.mean(dim=1, keepdim=True)  # [B,1,H,W]
        max_pool, _ = x_2d.max(dim=1, keepdim=True)
        spatial_feat = torch.cat([avg_pool, max_pool], dim=1)  # [B,2,H,W]
        spatial_att = self.spatial_conv(
            spatial_feat.view(B, 2, -1)  # [B,2,H*W]
        ).view(B, 1, self.h, self.w)  # [B,1,H,W]


        return (x * channel_att.view(B, C, 1) * spatial_att.view(B, 1, -1)).view(B, C, L)


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, h=8, w=64):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.h = h
        self.w = w

    def forward(self, x):
        B, C, L = x.size()
        device = x.device


        y_embed, x_embed = torch.meshgrid(
            torch.arange(self.h, device=device),
            torch.arange(self.w, device=device)
        )


        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed / (self.h - 1 + eps)) * 2 - 1
            x_embed = (x_embed / (self.w - 1 + eps)) * 2 - 1


        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed.unsqueeze(-1) / dim_t  # [H,W,D]
        pos_y = y_embed.unsqueeze(-1) / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)


        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1)  # [C,H,W]
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)  # [B,C,H,W]
        return pos.flatten(2)  # [B,C,H*W]

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig,dilation=None):
        super().__init__()
        self.mixer = MambaBlock(config,dilation=dilation)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

        self.mlp_channels = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.res_drop)
        )
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x):
        # x : (B, L, D)
        # output : (B, L, D)
        output = self.mixer(self.norm(x)) + x
        output = output + self.mlp_channels(self.norm2(output))

        return output



class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig,dilation=None):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=dilation,
                      dilation=dilation), nn.BatchNorm2d(512, affine=False), nn.ReLU(inplace=True))

        self.attention = MutiHeadSelfAttention(dim=config.expand_factor * config.d_model, attention_dim=config.d_model,
                                               d_state=config.d_state)
        #self.attention_b = MutiHeadSelfAttention(dim=config.expand_factor * config.d_model, attention_dim=config.d_model,
        #                                       d_state=config.d_state)


        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                kernel_size=config.d_conv, bias=config.conv_bias,
                                groups=config.d_inner,
                                padding=config.d_conv - 1)

        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)


        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False
        self.mask_ratio = 0.1


    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):

        x_d = x.permute(0, 2, 1)
        x_d = x_d.reshape(x_d.shape[0], x_d.shape[1], 7, 7)
        x_d = self.net(x_d)
        x = x_d.reshape(x_d.shape[0], x_d.shape[1], x_d.shape[2] * x_d.shape[3]).permute(0, 2, 1) + x

        
        _, L, _ = x.shape
        xz = self.in_proj(x)  # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, L, ED), (B, L, ED)
        x_b=x

        # x branch  crossover
        x = x.transpose(1, 2)  # (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  # (B, L, ED)
        x = F.silu(x)
        z = F.silu(z)
        x_batch_1 = torch.flip(x,dims=[1])
        z_batch_1 = torch.flip(z,dims=[1])
        y = self.ssm(x, z)
        y_batch_1 = self.ssm(x_batch_1, z_batch_1)
        output_0 = y * z
        output_1 = y_batch_1 * z_batch_1
        output_0 = self.out_proj(output_0)  # (B, L, D)
        output_1 = self.out_proj(output_1)
        output_1 = torch.flip(output_1,dims=[1])
        output = (output_0 + output_1)/2
        
        #y = self.ssm(x,z)
        # z branch
        #z = F.silu(z)
        #output = y * z
        #output = y_b * z + y * z
        #output = self.out_proj(output)  # (B, L, D)

        return output

    def ssm(self, x, z):
        # x : (B, L, ED)
        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x)  # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2)  # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)

        if self.config.use_cuda:

            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)


            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True,
                                         delta_bias=self.dt_proj.bias.float())
            y = y.transpose(1, 2)  # (B, L, ED)

        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D, isdecoder=False):


        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)
        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N) 50,49,1024,16
        hs = pscan(deltaA, BX)
        hs = self.attention(hs)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @(B, L, N, 1) -> (B, L, ED, 1)
        y = y + D * x

        return y
    

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)
        # y : (B, L, ED)

        _, L, _ = x.shape
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)
        BX = deltaB * (x.unsqueeze(-1))  # (B, L, ED, N)
        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  # (B, ED, N)
        hs = []
        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  # (B, L, ED, N)
        y = (hs @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED, N) @(B, L, N, 1) -> (B, L, ED, 1)
        y = y + D * x

        return y





class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

class ImplicitPhysicsAttention(nn.Module):
    def __init__(self, d_model, h, latent_phy_dim=32, dilation=None):
        super().__init__()
        self.heads = h
        self.scale = (d_model // h) ** -0.5
        
        self.phy_net = nn.Sequential(
            nn.Conv2d(d_model, latent_phy_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(latent_phy_dim, h * 2)
        )
        self.to_out = nn.Linear(d_model, d_model)

        self.embed_dim = d_model
        self.num_heads = h
        self.head_dim = d_model // h
        env_embeddings = torch.from_numpy(np.load('UWSeg_IC_keywords.npz')['embeddings']).to(torch.float32)

        self.env_keys = nn.Parameter(env_embeddings)  # [n_env, d]
        self.env_proj = nn.Linear(d_model, d_model)
        self.lambda_ = nn.Parameter(torch.tensor(0.5))  # 初始平衡系数

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)


        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x, attention_mask=None, attention_weights=None, group_mask=None,input_gl=None, memory=None, isencoder=None, dilation=None):

        b, n, _ = x.shape
        #h = w = int(n ** 0.5)
        queries, keys, values = x, x, x
        h_dim = int(math.sqrt(n))
        w_dim = n // h_dim
        queries, keys, values = queries.contiguous(), keys.contiguous(), values.contiguous()

        K_2d = keys.view(b, h_dim, w_dim, -1).permute(0, 3, 1, 2)

        phy_params = self.phy_net(K_2d)
        attenuation = phy_params[:, :self.heads].unsqueeze(-1).unsqueeze(-1)  # [b, 1, heads]
        scattering = phy_params[:, self.heads:].unsqueeze(-1).unsqueeze(-1)  # [b, 1, heads]

        q = rearrange(queries, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(keys, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(values, 'b n (h d) -> b h n d', h=self.heads)


        k = k * (1.0 / (attenuation + 1e-6))  # [b,h,n,d]


        dots = (q @ k.transpose(-2, -1)) * self.scale
        dots = dots * scattering # [b,h,n,n]

        if attention_weights is not None:
            dots = dots * attention_weights
        if attention_mask is not None:

            dots = dots.masked_fill(attention_mask, -np.inf)
        if group_mask is not None:
            group_mask_mat=group_mask.masked_fill(group_mask.bool(), torch.tensor(-1e9))
            dots=dots+group_mask_mat

        attn = dots.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        tgt_len_env, bsz_env, _ = queries.shape
        src_len_env = keys.size(0)
        n_env_env = self.env_keys.size(0)

        q_env = self.q_proj(queries)  # [tgt_len, b, d]
        k_env = self.k_proj(keys)  # [src_len, b, d]
        v_env = self.v_proj(values)  # [src_len, b, d]

        env_k_env = self.env_proj(self.env_keys)  # [n_env, d]
        env_k_env = env_k_env.unsqueeze(1).expand(-1, bsz_env, -1)  # [n_env, b, d]

        q_env = q_env.view(tgt_len_env, bsz_env * self.num_heads, self.head_dim).transpose(0, 1)
        k_env = k_env.view(src_len_env, bsz_env * self.num_heads, self.head_dim).transpose(0, 1)
        v_env = v_env.view(src_len_env, bsz_env * self.num_heads, self.head_dim).transpose(0, 1)
        env_k_env = env_k_env.contiguous().view(n_env_env, bsz_env * self.num_heads, self.head_dim).transpose(0,1)

        attn_score_env = torch.bmm(q_env, k_env.transpose(1, 2))  # [b*h, tgt, src]
        env_score_env = torch.bmm(q_env, env_k_env.transpose(1, 2))  # [b*h, tgt, n_env]

        combined_score_env = torch.cat([attn_score_env, self.lambda_ * env_score_env], dim=-1)
        combined_score_env = F.softmax(combined_score_env / (self.head_dim ** 0.5), dim=-1)

        attn_weight_env = combined_score_env[..., :src_len_env]
        env_weight_env = combined_score_env[..., src_len_env:]

        env_v_env = self.env_proj(self.env_keys)  # [n_env, embed_dim]
        env_v_env = env_v_env.view(n_env_env,self.num_heads,self.head_dim)
        env_v_env = env_v_env.unsqueeze(1)  # [n_env, 1, num_heads, head_dim]
        env_v_env = env_v_env.expand(-1, bsz_env, -1, -1)  # [n_env, bsz, num_heads, head_dim]
        env_v_env = env_v_env.permute(2, 1, 0, 3)  # [num_heads, bsz, n_env, head_dim]
        env_v_env = env_v_env.reshape(bsz_env * self.num_heads,n_env_env,self.head_dim)
        output_env = torch.bmm(attn_weight_env, v_env) + torch.bmm(env_weight_env, env_v_env)

        output_env = output_env.transpose(0, 1).contiguous().view(tgt_len_env, bsz_env, self.embed_dim)
        output_env = self.out_proj(output_env)

        fusion_gate = self.gate(torch.cat([out, output_env], dim=-1))
        out_all = fusion_gate * out + (1 - fusion_gate) * output_env
        return out_all
