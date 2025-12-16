import numpy as np
import torch
from torch import nn
from models.containers import Module
import math

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
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

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            # att = att.masked_fill(attention_mask, -np.inf)
            att = att.masked_fill(attention_mask.bool(), -np.inf)

        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out



class MultiHeadAttention(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out
    
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
        env_embeddings = torch.from_numpy(np.load('UIC_keywords.npz')['embeddings']).to(torch.float32)

        self.env_keys = nn.Parameter(env_embeddings)  # [n_env, d]
        self.env_proj = nn.Linear(d_model, d_model)
        self.lambda_ = nn.Parameter(torch.tensor(0.5)) 

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)


        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, group_mask=None,input_gl=None, memory=None, isencoder=None, dilation=None):
        """
        x: [batch, seq_len, dim]
        """
        b, n, _ = queries.shape
        #h = w = int(n ** 0.5)
        h_dim = int(math.sqrt(n))
        w_dim = n // h_dim
        queries, keys, values = queries.contiguous(), keys.contiguous(), values.contiguous()

        K_2d = keys.view(b, h_dim, w_dim, -1).permute(0, 3, 1, 2)

        phy_params = self.phy_net(K_2d)  # [b, heads*2]
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
            #dots += attention_mask.unsqueeze(1)
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
