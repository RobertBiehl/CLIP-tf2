import tensorflow as tf
from tensorflow.keras import layers as klayers
from .layers import LayerNorm
from .transformer import Transformer


class VisualTransformer(klayers.Layer):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, name="VisualTransformer"):
        super().__init__(name=name)
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.output_dim = output_dim

        self.conv1 = klayers.Conv2D(width, patch_size, strides=patch_size, use_bias=False, name="conv1")

        scale = width ** -0.5

        self.transformer = Transformer(width, layers, heads, name=name + "/transformer")

        with tf.name_scope(name):
            self.class_embedding = tf.Variable(scale * tf.random.normal((width,)), name="class_embedding")
            self.positional_embedding = tf.Variable(scale * tf.random.normal(((input_resolution // patch_size) ** 2 + 1, width)), name="positional_embedding")
            self.ln_pre = LayerNorm(name="ln_pre")

            self.ln_post = LayerNorm(name="ln_post")
            self.proj = tf.Variable(scale * tf.random.normal((width, output_dim)), name="proj")

    def get_config(self):
        return {
            "input_resolution": self.input_resolution,
            "patch_size": self.patch_size,
            "width": self.width,
            "layers": self.layers,
            "heads": self.heads,
            "output_dim": self.output_dim,
            "name": self.name
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x: tf.Tensor):
        x = self.conv1(x)  # shape = [*, grid, grid, width]

        x_shape = tf.shape(x)
        x = tf.reshape(x, (x_shape[0], x_shape[1]*x_shape[2], x_shape[3]))  # shape = [*, grid ** 2, width] # TODO: check if correct for tensorflow
        #x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x_shape = tf.shape(x)
        print(f"ViT 1 x.shape={x.shape}")
        x = tf.concat([tf.broadcast_to(tf.cast(self.class_embedding, x.dtype), (x_shape[0], 1, x_shape[-1])), x], axis=1)  # shape = [*, grid ** 2 + 1, width]
        print(f"ViT 2 x.shape={x.shape}")
        x = x + tf.cast(self.positional_embedding, x.dtype)
        x = self.ln_pre(x)
        print(f"ViT 3 x.shape={x.shape}")
        #tf.print(f"ViT 3 x.shape=", x.shape, "x=", x)

        #x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        #x = x.permute(1, 0, 2)  # LND -> NLD
        print(f"ViT 4 x.shape={x.shape}")
        #tf.print(f"ViT 4 x.shape=", x.shape, "x=", x)

        x = self.ln_post(x[:, 0, :])
        print(f"ViT 4 x.shape={x.shape}")

        if self.proj is not None:
            x = x @ self.proj

        print(f"ViT 5 x.shape={x.shape}")

        return x

