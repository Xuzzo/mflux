import mlx.core as mx
from mlx import nn

from flux_1.models.transformer.feed_forward import LoRALinearLayer


class JointAttention(nn.Module):
    head_dimension = 128
    batch_size = 1
    num_heads = 24

    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(3072, 3072)
        self.to_k = nn.Linear(3072, 3072)
        self.to_v = nn.Linear(3072, 3072)
        self.to_out = [nn.Linear(3072, 3072)]
        self.add_q_proj = nn.Linear(3072, 3072)
        self.add_k_proj = nn.Linear(3072, 3072)
        self.add_v_proj = nn.Linear(3072, 3072)
        self.to_add_out = nn.Linear(3072, 3072)
        self.norm_q = nn.RMSNorm(128)
        self.norm_k = nn.RMSNorm(128)
        self.norm_added_q = nn.RMSNorm(128)
        self.norm_added_k = nn.RMSNorm(128)
        

    def forward(
            self,
            hidden_states: mx.array,
            encoder_hidden_states: mx.array,
            image_rotary_emb: mx.array
    ) -> (mx.array, mx.array):

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = mx.transpose(mx.reshape(query, (1, -1, 24, 128)), (0, 2, 1, 3))
        key = mx.transpose(mx.reshape(key, (1, -1, 24, 128)), (0, 2, 1, 3))
        value = mx.transpose(mx.reshape(value, (1, -1, 24, 128)), (0, 2, 1, 3))

        query = self.norm_q(query)
        key = self.norm_k(key)

        encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = mx.transpose(mx.reshape(encoder_hidden_states_query_proj, (1, -1, 24, 128)), (0, 2, 1, 3))
        encoder_hidden_states_key_proj = mx.transpose(mx.reshape(encoder_hidden_states_key_proj, (1, -1, 24, 128)), (0, 2, 1, 3))
        encoder_hidden_states_value_proj = mx.transpose(mx.reshape(encoder_hidden_states_value_proj, (1, -1, 24, 128)),   (0, 2, 1, 3))

        encoder_hidden_states_query_proj = self.norm_added_q(encoder_hidden_states_query_proj)
        encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)

        query = mx.concatenate([encoder_hidden_states_query_proj, query], axis=2)
        key = mx.concatenate([encoder_hidden_states_key_proj, key], axis=2)
        value = mx.concatenate([encoder_hidden_states_value_proj, value], axis=2)

        query, key = JointAttention.apply_rope(query, key, image_rotary_emb)

        hidden_states = JointAttention.attention(query, key, value)
        hidden_states = mx.transpose(hidden_states, (0, 2, 1, 3))
        hidden_states = mx.reshape(hidden_states, (self.batch_size, -1, self.num_heads * self.head_dimension))
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1]:],
        )

        hidden_states = self.to_out[0](hidden_states)
        encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states

    @staticmethod
    def attention(query, key, value):
        scale = 1 / mx.sqrt(query.shape[-1])
        scores = (query * scale) @ key.transpose(0, 1, 3, 2)
        attn = mx.softmax(scores, axis=-1)
        hidden_states = (attn @ value)
        return hidden_states

    @staticmethod
    def apply_rope(xq: mx.array, xk: mx.array, freqs_cis: mx.array):
        xq_ = xq.astype(mx.float32).reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.astype(mx.float32).reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).astype(mx.float32), xk_out.reshape(*xk.shape).astype(mx.float32)
    

class JointLoraAttention(JointAttention):
    head_dimension = 128
    batch_size = 1
    num_heads = 24

    def __init__(self, lora_strength=0.0):
        super().__init__()
        self.to_q = nn.Linear(3072, 3072)
        self.to_k = nn.Linear(3072, 3072)
        self.to_v = nn.Linear(3072, 3072)
        self.to_out = [nn.Linear(3072, 3072)]
        self.add_q_proj = nn.Linear(3072, 3072)
        self.add_k_proj = nn.Linear(3072, 3072)
        self.add_v_proj = nn.Linear(3072, 3072)
        self.to_add_out = nn.Linear(3072, 3072)
        self.norm_q = nn.RMSNorm(128)
        self.norm_k = nn.RMSNorm(128)
        self.norm_added_q = nn.RMSNorm(128)
        self.norm_added_k = nn.RMSNorm(128)
        self.qkv_lora1 = LoRALinearLayer(3072, 3072 * 3)
        self.proj_lora1 = LoRALinearLayer(3072, 3072)
        self.qkv_lora2 = LoRALinearLayer(3072, 3072 * 3)
        self.proj_lora2 = LoRALinearLayer(3072, 3072)
        self.lora_strength = lora_strength
    
    def forward(
            self,
            hidden_states: mx.array,
            encoder_hidden_states: mx.array,
            image_rotary_emb: mx.array
    ) -> (mx.array, mx.array):

        qkv_lora1 = self.qkv_lora1.forward(hidden_states)
        query = self.to_q(hidden_states) + qkv_lora1[:, :, :3072] * self.lora_strength
        key = self.to_k(hidden_states) + qkv_lora1[:, :, 3072: 3072 * 2] * self.lora_strength
        value = self.to_v(hidden_states) + qkv_lora1[:, :, 3072 * 2:] * self.lora_strength

        query = mx.transpose(mx.reshape(query, (1, -1, 24, 128)), (0, 2, 1, 3))
        key = mx.transpose(mx.reshape(key, (1, -1, 24, 128)), (0, 2, 1, 3))
        value = mx.transpose(mx.reshape(value, (1, -1, 24, 128)), (0, 2, 1, 3))

        query = self.norm_q(query)
        key = self.norm_k(key)

        qkv_lora2 = self.qkv_lora2.forward(encoder_hidden_states)
        encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states) + qkv_lora2[:, :, :3072] * self.lora_strength
        encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states) + qkv_lora2[:, :, 3072: 3072 * 2] * self.lora_strength
        encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states) + qkv_lora2[:, :, 3072 * 2:] * self.lora_strength

        encoder_hidden_states_query_proj = mx.transpose(mx.reshape(encoder_hidden_states_query_proj, (1, -1, 24, 128)), (0, 2, 1, 3))
        encoder_hidden_states_key_proj = mx.transpose(mx.reshape(encoder_hidden_states_key_proj, (1, -1, 24, 128)), (0, 2, 1, 3))
        encoder_hidden_states_value_proj = mx.transpose(mx.reshape(encoder_hidden_states_value_proj, (1, -1, 24, 128)),   (0, 2, 1, 3))

        encoder_hidden_states_query_proj = self.norm_added_q(encoder_hidden_states_query_proj)
        encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)

        query = mx.concatenate([encoder_hidden_states_query_proj, query], axis=2)
        key = mx.concatenate([encoder_hidden_states_key_proj, key], axis=2)
        value = mx.concatenate([encoder_hidden_states_value_proj, value], axis=2)

        query, key = JointAttention.apply_rope(query, key, image_rotary_emb)

        hidden_states = JointAttention.attention(query, key, value)
        hidden_states = mx.transpose(hidden_states, (0, 2, 1, 3))
        hidden_states = mx.reshape(hidden_states, (self.batch_size, -1, self.num_heads * self.head_dimension))
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1]:],
        )

        hidden_states = self.to_out[0](hidden_states) + self.proj_lora1.forward(hidden_states) * self.lora_strength
        encoder_hidden_states = self.to_add_out(encoder_hidden_states) + self.proj_lora2.forward(encoder_hidden_states) * self.lora_strength

        return hidden_states, encoder_hidden_states
