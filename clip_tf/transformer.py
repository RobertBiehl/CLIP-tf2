import sys

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


class LayerNorm(layers.LayerNormalization):
    """Subclass torch's LayerNorm to handle fp16."""

    def __init__(self, name="LayerNorm"):
        super(LayerNorm, self).__init__(epsilon=1e-05, name=name)

    # TODO: but check dimension, might be different betwen torch and keras
    def call(self, x: tf.Tensor):
        return super().call(x)


class QuickGELU(keras.layers.Layer):
    def __init__(self, name="QuickGELU"):
        super(QuickGELU, self).__init__(name=name)
        
    def call(self, x: tf.Tensor):
        return x * tf.sigmoid(1.702 * x)


class ResidualAttentionBlock(layers.Layer):
    def __init__(self, d_model: int, n_head: int, attn_mask: tf.Tensor = None, name="ResidualAttentionBlock", idx=0):
        super().__init__(name=name)
        self.idx = idx
        self.attn = layers.MultiHeadAttention(num_heads=n_head, key_dim=d_model // n_head, name="attn")
        self.ln_1 = LayerNorm(name="ln_1")
        self.mlp = keras.Sequential([
            layers.Dense(d_model * 4, name=name+"/mlp/c_fc"),
            QuickGELU(name=name+"/mlp/gelu"),
            layers.Dense(d_model, name=name+"/mlp/c_proj")
        ], name="mlp")
        self.ln_2 = LayerNorm(name="ln_2")
        self.attn_mask = attn_mask

    def attention(self, x: tf.Tensor):
        return self.attn(x, x, x, attention_mask=self.attn_mask)

    def call(self, x: tf.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(layers.Layer):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: tf.Tensor = None, name="transformer"):
        super().__init__(name=name)
        self.width = width
        self.layers = layers
        self.resblocks = keras.Sequential([
            ResidualAttentionBlock(width, heads, attn_mask, name=name + f".resblocks.{i}", idx=i)
            for i in range(layers)
        ], name=name + ".resblocks")

    def call(self, x: tf.Tensor):
        return self.resblocks(x)
