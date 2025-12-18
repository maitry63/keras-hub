import keras
from keras import layers
from keras import ops
from keras_hub.src.utils.keras_utils import gelu_approximate

class ModernBertMLP(layers.Layer):
    """ModernBERT MLP block using Gated Linear Units (GeGLU)."""
    def __init__(self, hidden_dim, intermediate_dim, activation=gelu_approximate, **kwargs):
        super().__init__(**kwargs)
        self.wi_0 = layers.Dense(intermediate_dim, name="wi_0")
        self.wi_1 = layers.Dense(intermediate_dim, name="wi_1")
        self.wo = layers.Dense(hidden_dim, name="wo")
        self.activation = activation

    def call(self, x):
        return self.wo(self.activation(self.wi_0(x)) * self.wi_1(x))

class ModernBertAttention(layers.Layer):
    """ModernBERT Attention with Rotary Positional Embeddings (RoPE)."""
    def __init__(self, hidden_dim, num_heads, rotary_embedding=None, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.rotary_embedding = rotary_embedding
        self.qkv = layers.Dense(hidden_dim * 3, name="qkv")
        self.out_dense = layers.Dense(hidden_dim, name="out_dense")

    def call(self, x, padding_mask=None):
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]
        
        qkv = self.qkv(x) # (batch, seq, hidden*3)
        # Reshape to (batch, seq, 3, heads, head_dim)
        qkv = ops.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        q, k, v = ops.unstack(qkv, axis=2)

        if self.rotary_embedding is not None:
            q, k = self.rotary_embedding(q), self.rotary_embedding(k)

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 3, 1)) # Ready for matmul
        v = ops.transpose(v, (0, 2, 1, 3))

        scale = ops.cast(ops.sqrt(ops.cast(self.key_dim, x.dtype)), x.dtype)
        scores = ops.matmul(q, k) / scale
    
        if padding_mask is not None:
            m = ops.cast(padding_mask[:, None, None, :], scores.dtype)
            scores = scores + (1.0 - m) * -1e9
        
        
        attn = ops.softmax(scores, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        # Flatten heads and head_dim back into hidden_dim
        out = ops.reshape(out, (batch_size, seq_len, self.hidden_dim))
    
        return self.out_dense(out)

class ModernBertEncoderLayer(layers.Layer):
    """ModernBERT Encoder Layer implementation.

    This layer implements a modernized Transformer block featuring:
    1. Pre-Normalization (Norm-First architecture).
    2. RMSNorm scaling (LayerNorm without additive bias).
    3. Gated Linear Unit (GeGLU) activation in the MLP.

    Args:
        hidden_dim: int. Dimensionality of the encoder layer.
        intermediate_dim: int. Dimensionality of the MLP intermediate layer.
        num_heads: int. Number of attention heads.
        rotary_embedding: `RotaryEmbedding` layer. Optional rotary positional encoding.
        activation: function. Activation function for the MLP.
        layer_norm_epsilon: float. Epsilon for the LayerNorm layers.
    """
    def __init__(self, hidden_dim, intermediate_dim, num_heads, 
                 rotary_embedding=None,
                 dropout=0.0, 
                 activation=gelu_approximate, layer_norm_epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.attn_norm = layers.LayerNormalization(epsilon=layer_norm_epsilon, rms_scaling=True)
        self.attn = ModernBertAttention(hidden_dim, num_heads, rotary_embedding)
        self.mlp_norm = layers.LayerNormalization(epsilon=layer_norm_epsilon, rms_scaling=True)
        self.mlp = ModernBertMLP(hidden_dim, intermediate_dim, activation=activation)
        self.dropout_layer = layers.Dropout(dropout)
    
    def call(self, x, padding_mask=None):
        # Attention Residual path
        x = x + self.dropout_layer(self.attn(self.attn_norm(x), padding_mask=padding_mask))
        # MLP Residual path
        x = x + self.dropout_layer(self.mlp(self.mlp_norm(x)))
        return x

    def compute_output_shape(self, input_shape):
        return input_shape