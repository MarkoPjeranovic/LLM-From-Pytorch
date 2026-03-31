from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 50304
    hidden_size: int = 4096
    intermediate_size: int = 11008

    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8

    max_position_embeddings: int = 2048

    attention_bias: bool = False
    attention_dropout: float = 0.0
    rms_norm_eps: float = 1e-5

    rope_theta: float = 10000.0

    pad_token_id: int = 1
    bos_token_id: int | None = None
    eos_token_id: int = 50279

    initializer_range: float = 0.02

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0