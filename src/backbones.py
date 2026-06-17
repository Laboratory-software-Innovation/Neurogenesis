import keras
from keras import layers, ops
from _layers import *


def RNNBackbone(encoder_hidden_size, encoder_n_layers, bidirectional=True, **kwargs):
    def _apply(input):
        x = input
        if bidirectional:
            for i in range(encoder_n_layers):
                x = layers.Bidirectional(
                    layers.SimpleRNN(
                        encoder_hidden_size, return_sequences=True, **kwargs
                    )
                )(x)
        else:
            for i in range(encoder_n_layers):
                x = layers.SimpleRNN(
                    encoder_hidden_size, return_sequences=True, **kwargs
                )(x)
        return x

    return _apply


def LSTMBackbone(encoder_hidden_size, encoder_n_layers, bidirectional=True, **kwargs):
    def _apply(input):
        x = input
        if bidirectional:
            for i in range(encoder_n_layers):
                x = layers.Bidirectional(
                    layers.LSTM(encoder_hidden_size, return_sequences=True, **kwargs)
                )(x)
        else:
            for i in range(encoder_n_layers):
                x = layers.LSTM(encoder_hidden_size, return_sequences=True, **kwargs)(x)
        return x

    return _apply


def GRUBackbone(encoder_hidden_size, encoder_n_layers, bidirectional=True, **kwargs):
    def _apply(input):
        x = input
        if bidirectional:
            for i in range(encoder_n_layers):
                x = layers.Bidirectional(
                    layers.GRU(encoder_hidden_size, return_sequences=True, **kwargs)
                )(x)
        else:
            for i in range(encoder_n_layers):
                x = layers.GRU(encoder_hidden_size, return_sequences=True, **kwargs)(x)
        return x

    return _apply


def MLPBackbone(encoder_hidden_size, encoder_n_layers, **kwargs):
    def _apply(input):
        x = input
        for i in range(encoder_n_layers):
            x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        return x

    return _apply


"""
Replacing the self-attention sublayer in a Transformer encoder with a standard, unparameterized Fourier Transform
https://arxiv.org/abs/2105.03824
"""


def FNetBackbone(encoder_hidden_size, encoder_n_layers, **kwargs):
    def _apply(input):
        x = input
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        for i in range(encoder_n_layers):
            x = FNetLayer(encoder_hidden_size, **kwargs)(x)
        return x

    return _apply


def gMLPBackbone(encoder_hidden_size, encoder_n_layers, **kwargs):
    def _apply(input):
        x = input
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        num_patches = ops.shape(x)[1]
        for i in range(encoder_n_layers):
            x = gMLPLayer(num_patches, encoder_hidden_size, **kwargs)(x)
        return x

    return _apply


def MixerBackbone(encoder_hidden_size, encoder_n_layers, **kwargs):
    def _apply(input):
        x = input
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        num_patches = ops.shape(x)[1]
        for i in range(encoder_n_layers):
            x = MLPMixerLayer(num_patches, encoder_hidden_size, **kwargs)(x)
        return x

    return _apply


def FeedForward(dim, hidden_dim, dropout=0.0):
    return keras.Sequential(
        [
            layers.LayerNormalization(),
            layers.Dense(hidden_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout),
        ]
    )


def TransformerBackbone(
    encoder_hidden_size, encoder_n_layers, heads, dim_head, mlp_dim, **kwargs
):
    def _apply(x):
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        for _ in range(encoder_hidden_size):
            x += layers.MultiHeadAttention(heads, dim_head)(x, x)
            x += FeedForward(encoder_hidden_size, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def FeedForward(dim, hidden_dim, dropout=0.0):
    return keras.Sequential(
        [
            layers.LayerNormalization(),
            layers.Dense(hidden_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout),
        ]
    )


def ExternalTransformerBackbone(
    encoder_hidden_size,
    encoder_n_layers,
    heads,
    mlp_dim,
    dim_coefficient=4,
    projection_dropout=0.0,
    attention_dropout=0,
    **kwargs
):
    def _apply(x):
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        for _ in range(encoder_n_layers):
            x += ExternalAttention(
                encoder_hidden_size,
                heads,
                dim_coefficient=dim_coefficient,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )(x)
            x += FeedForward(encoder_hidden_size, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(filters: int, patch_size: int):
    def _apply(x):
        x = layers.Conv1D(filters, kernel_size=patch_size, strides=patch_size)(x)
        return activation_block(x)

    return _apply


def ConvMixer(filters: int, kernel_size: int):
    def _apply(x):
        # Depthwise convolution.
        x0 = x
        x = layers.DepthwiseConv1D(kernel_size=kernel_size, padding="same")(x)
        x = layers.Add()([activation_block(x), x0])  # Residual.

        # Pointwise convolution.
        x = layers.Conv1D(filters, kernel_size=1)(x)
        x = activation_block(x)

        return x

    return _apply


def ConvMixerBackbone(depth, filters, patch_size, kernel_size):
    def _apply(x):
        # Extract patch embeddings.
        x = conv_stem(filters, patch_size)(x)

        # ConvMixer blocks.
        for _ in range(depth):
            x = ConvMixer(filters, kernel_size)(x)
        return x

    return _apply


def AFTFullBackbone(
    encoder_hidden_size, encoder_n_layers, mlp_dim, position_bias=True, **kwargs
):
    def _apply(x):
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        for _ in range(encoder_n_layers):
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x += AFTFull(encoder_hidden_size, position_bias=position_bias)(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x += FeedForward(encoder_hidden_size, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def ResidualAttentionBackbone(encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs):
    def _apply(x):
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        for _ in range(encoder_n_layers):
            x += layers.LayerNormalization(epsilon=1e-6)(x)
            x += ResidualAttention(encoder_hidden_size)(x)
            x += FeedForward(encoder_hidden_size, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def SEAttentionBackbone(
    encoder_hidden_size, encoder_n_layers, mlp_dim, reduction=16, **kwargs
):
    def _apply(x):
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        for _ in range(encoder_n_layers):
            x += layers.LayerNormalization(epsilon=1e-6)(x)
            x += SEAttention(reduction=reduction)(x)
            x += FeedForward(encoder_hidden_size, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def SimAMBackbone(encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs):
    def _apply(x):
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        for _ in range(encoder_n_layers):
            x += layers.LayerNormalization(epsilon=1e-6)(x)
            x += SimAM()(x)
            x += FeedForward(encoder_hidden_size, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def DoubleAttentionBackbone(encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs):
    def _apply(x):
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        for _ in range(encoder_n_layers):
            x += layers.LayerNormalization(epsilon=1e-6)(x)
            x += DoubleAttention(encoder_hidden_size)(x)
            x += FeedForward(encoder_hidden_size, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def PerformerAttentionBackbone(
    encoder_hidden_size, num_heads, encoder_n_layers, mlp_dim, **kwargs
):
    def _apply(x):
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        for _ in range(encoder_n_layers):
            x += layers.LayerNormalization(epsilon=1e-6)(x)
            x += PerformerAttention(encoder_hidden_size, num_heads, **kwargs)(x)
            x += FeedForward(encoder_hidden_size, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def ParNetAttentionBackbone(encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs):
    def _apply(x):
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        for _ in range(encoder_n_layers):
            x += layers.LayerNormalization(epsilon=1e-6)(x)
            x += ParNetAttention()(x)
            x += FeedForward(encoder_hidden_size, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def UFOAttentionBackbone(
    encoder_hidden_size, encoder_n_layers, mlp_dim, num_heads, dropout=0.1, **kwargs
):
    def _apply(x):
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        for _ in range(encoder_n_layers):
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x += UFOAttention(
                encoder_hidden_size,
                encoder_hidden_size,
                encoder_hidden_size,
                num_heads,
                dropout=dropout,
            )(x, x, x)
            x += FeedForward(encoder_hidden_size, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def ECAAttentionBackbone(
    encoder_hidden_size, encoder_n_layers, mlp_dim, kernel_size, dropout=0.1, **kwargs
):
    def _apply(x):
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        for _ in range(encoder_n_layers):
            x += layers.LayerNormalization(epsilon=1e-6)(x)
            x += ECAAttention(kernel_size)(x)
            x += FeedForward(encoder_hidden_size, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def SwitchTransformerBlock(num_experts, embed_dim, ff_dim, num_heads):

    def _apply(x):
        num_tokens_per_batch = ops.shape(x)[1]
        switch = Switch(num_experts, embed_dim, ff_dim, num_tokens_per_batch)
        transformer_block = TransformerBlock(embed_dim // num_heads, num_heads, switch)
        return transformer_block(x)

    return _apply


def SwitchTransformerBackbone(
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    num_experts,
    num_heads,
    dropout=0.1,
    **kwargs
):
    def _apply(x):
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        for _ in range(encoder_n_layers):
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x += SwitchTransformerBlock(
                num_experts, encoder_hidden_size, mlp_dim, num_heads
            )(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply


def CBAMBackbone(
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    kernel_size,
    dropout=0.1,
    reduction=16,
    **kwargs
):
    def _apply(x):
        x = layers.Dense(encoder_hidden_size, **kwargs)(x)
        for _ in range(encoder_n_layers):
            x += layers.LayerNormalization(epsilon=1e-6)(x)
            x += CBAMBlock(reduction, kernel_size)(x)
            x += FeedForward(encoder_hidden_size, mlp_dim)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x)

    return _apply
