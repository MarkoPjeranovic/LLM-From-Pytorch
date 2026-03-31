from typing import Optional, TYPE_CHECKING
from collections.abc import Callable
import torch
import torch.nn as nn
if TYPE_CHECKING:
    from config import Config


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        RMSNorm module.
        Args:
            hidden_size: Size of the hidden layer of the MLP.
            eps (`float`): Number to add for numerical stability, 1e-6 by default.
        
        Returns a tensor of the same shape and dtype as the input.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32) # Upcasting for numerical stability
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)

def create_causal_mask(
    batch_size: int,
    q_len: int,
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """
    Creates a standard causal mask.

    Output shape:
    (batch, 1, q_len, kv_len)
    """

    mask = torch.full(
        (q_len, kv_len),
        float("-inf"),
        dtype=dtype,
        device=device,
    )

    mask = torch.triu(mask, diagonal=1) # does not change shape, but zeroes out values on and under the diagonal

    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, q_len, kv_len)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups) # (bsz, n_kv_h * n_rep, sqlen, h_dim) -> (bsz, n_heads, sqlen, h_dim)
    value_states = repeat_kv(value, module.num_key_value_groups) # (bsz, n_kv_h * n_rep, sqlen, h_dim) -> (bsz, n_heads, sqlen, h_dim)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling # (bsz, n_heads, sqlen, h_dim) * (bsz, n_heads, h_dim, seqlen) -> (bsz, n_heads, seq_len_q, seq_len_k), seq_len_q = seq_len_k
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype) # Computes softmax over the seq_len_k
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states) # (bsz, n_heads, seq_len_q, seq_len_k) * (bsz, n_heads, seq_len, h_dim) -> (bsz, n_heads, seq_len, h_dim)
    attn_output = attn_output.transpose(1, 2).contiguous() # (bsz, seq_len, n_heads, h_dim)

    return attn_output, attn_weights

def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class GQA(nn.Module):
    """Grouped-Query Attention"""

    def __init__(self, config: "Config", layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = RMSNorm(config.num_attention_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.num_key_value_heads * self.head_dim, config.rms_norm_eps)
#        self.attention_type = config.layer_types[layer_idx]
#        self.sliding_window = config.sliding_window if self.attention_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor, # (bsz, seq_len, hid_size)
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        start_pos: int,
        cache_k: torch.Tensor | None,
        cache_v: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = hidden_states.shape

        input_shape = hidden_states.shape[:-1] # (bsz, seq_len, hid_size) -> (bsz, seq_len)
#        hidden_shape = (*input_shape, -1, self.head_dim) # (bsz, seq_len) -> (bsz, seq_len, -1, h_dim)

        query_states = self.q_norm(self.q_proj(hidden_states)) # (bsz, seq_len, hid_size) * (hid_size, num_att_h * h_dim) -> (bsz, seq_len, num_att_h * h_dim)
        key_states = self.k_norm(self.k_proj(hidden_states)) # (bsz, seq_len, hid_size) * (hid_size, num_kv_h * h_dim) -> (bsz, seq_len, num_kv_h * h_dim)
        value_states = self.v_proj(hidden_states) # (bsz, seq_len, hid_size) * (hid_size, num_kv_h * h_dim) -> (bsz, seq_len, num_kv_h * h_dim)

        query_states = query_states.view(batch_size, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2) # (bsz, seq_len, num_att_h * h_dim) -> (bsz, seq_len, num_att_h, h_dim) -> (bsz, num_att_h, seq_len, h_dim)
        key_states = key_states.view(batch_size, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2) # (bsz, seq_len, num_kv_h * h_dim) -> (bsz, seq_len, num_kv_h, h_dim) -> (bsz, num_kv_h, seq_len, h_dim)
        value_states = value_states.view(batch_size, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2) # (bsz, seq_len, num_kv_h * h_dim) -> (bsz, seq_len, num_kv_h, h_dim) -> (bsz, num_kv_h, seq_len, h_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if cache_k is not None and cache_v is not None:
            cache_k[:, :, start_pos:start_pos + seq_len] = key_states
            cache_v[:, :, start_pos:start_pos + seq_len] = value_states
            key_states = cache_k[:, :, :start_pos + seq_len]
            value_states = cache_v[:, :, :start_pos + seq_len]

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
#            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous() # (bsz, seq_len, n_heads, h_dim) -> (bsz, seq_len, n_heads * h_dim)
        attn_output = self.o_proj(attn_output) # (bsz, seq_len, n_heads * h_dim) * (n_heads*h_dim, hid_size) -> (bsz, seq_len, hid_size)
        return attn_output, attn_weights


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) # (bsz, seq_len, hid_size) @ (hid_size, int_size) -> (SiLU(bsz, seq_len, int_size)) * (bsz, seq_len, int_size) -> (bsz, seq_len, int_size) @ (int_size, hid_size) -> (bsz, seq_len, hid_size)
        return down_proj
    
class DecoderLayer(nn.Module):
    def __init__(self, config: "Config", layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GQA(config=config, layer_idx=layer_idx)
        self.mlp = MLP(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_pos: int,
        cache_k: torch.Tensor | None,
        cache_v: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            start_pos=start_pos,
            cache_k=cache_k,
            cache_v=cache_v,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class RotaryEmbedding(nn.Module):

    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        t = torch.arange(max_seq_len)

        freqs = torch.outer(t, inv_freq) # (t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1) # (t, 2 * inv_freq)

        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, seq_len):

        return (
            self.cos[:seq_len],
            self.sin[:seq_len]
        )

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        start_pos=0,
        cache_k=None,
        cache_v=None,
    ):
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len, _ = hidden_states.shape

        kv_len = seq_len if cache_k is None else start_pos + seq_len

        causal_mask = create_causal_mask(
            batch_size=batch_size,
            q_len=seq_len,
            kv_len=kv_len,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        if attention_mask is not None:
            causal_mask = causal_mask + attention_mask

        attention_mask = causal_mask
        position_embeddings = self.rotary_emb(start_pos + seq_len) # Compute the rotary embeddings for the longest sequence
        # Slice to only the positions we care about
        cos = position_embeddings[0][start_pos:start_pos + seq_len]
        sin = position_embeddings[1][start_pos:start_pos + seq_len]
        position_embeddings = (cos, sin)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                start_pos=start_pos,
                cache_k=cache_k[i] if cache_k is not None else None,
                cache_v=cache_v[i] if cache_v is not None else None,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states
    
class CausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = Model(config)

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        # weight tying
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids,
        attention_mask,
        start_pos,
        cache_k,
        cache_v,
        labels=None
    ):

        hidden_states = self.model(
            input_ids,
            attention_mask,
            start_pos,
            cache_k,
            cache_v
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        return logits, loss
    
    def init_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config.initializer_range
                )

                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config.initializer_range
                )