import keras
from keras import layers, ops, Model


def MLP(out_features, activation, hidden_size, num_layers, dropout):
    def _apply(x):
        for i in range(num_layers):
            x = layers.Dense(hidden_size, activation=activation)(x)
            x = layers.Dropout(dropout)(x)
        return layers.Dense(out_features)(x)

    return _apply


def Head(num_classes, head_dim, dropout=0.1):
    return MLP(num_classes, "relu", head_dim, 1, dropout)


def Classifier(
    backbone, num_classes, input_shape, head_dim, dropout=0.0, task="classification"
):
    inputs = layers.Input(shape=input_shape)
    x = backbone(inputs)
    if task == "classification":
        x = layers.GlobalAveragePooling1D()(x)
    outputs = Head(num_classes, head_dim, dropout=dropout)(x)
    return Model(inputs, outputs)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_1d(patches, temperature=10000, dtype="float32"):
    n, dim = ops.shape(patches)[1], ops.shape(patches)[2]
    n = ops.arange(n)
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = ops.arange(dim // 2) / (dim // 2 - 1)
    omega = ops.cast(omega, patches.dtype)
    omega = 1.0 / (temperature**omega)
    n = ops.expand_dims(ops.reshape(n, [-1]), 1)
    n = ops.cast(n, patches.dtype)
    n = n * ops.expand_dims(omega, 0)
    pe = ops.concatenate((ops.sin(n), ops.cos(n)), 1)
    return ops.cast(pe, dtype)


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = ops.arange(start=0, stop=self.num_patch, step=1)
        return self.proj(patch) + self.pos_embed(pos)
