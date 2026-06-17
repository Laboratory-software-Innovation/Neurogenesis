from backbones import *
from commons import *


def AudioClassifier(
    backbone,
    head,
    seq_len: int,
    patch_size: int,
    dim: int,
    channels: int = 1,
):
    # print(seq_len)
    # assert seq_len % patch_size == 0
    patch_dim = channels * patch_size
    i_p = layers.Input((seq_len, channels))
    patches = layers.Reshape((-1, patch_dim))(i_p)
    patches = layers.LayerNormalization()(patches)
    patches = layers.Dense(dim)(patches)
    patches = layers.LayerNormalization()(patches)
    pos_embedding = posemb_sincos_1d(patches)
    patches += pos_embedding
    dim = ops.shape(patches)[-1]
    x = backbone(patches)
    x = layers.GlobalAveragePooling1D()(x)
    out = head(x)
    return keras.Model(inputs=i_p, outputs=out)


# audio classifiers
def rnn_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    bidirectional=True,
    **kwargs
):
    backbone = RNNBackbone(
        encoder_hidden_size, encoder_n_layers, bidirectional=bidirectional, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def gru_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    bidirectional=True,
    **kwargs
):
    backbone = GRUBackbone(
        encoder_hidden_size, encoder_n_layers, bidirectional=bidirectional, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def mlp_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    **kwargs
):
    backbone = MLPBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def lstm_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    bidirectional=True,
    **kwargs
):
    backbone = LSTMBackbone(
        encoder_hidden_size, encoder_n_layers, bidirectional=bidirectional, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def fnet_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    **kwargs
):
    backbone = FNetBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def gmlp_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    **kwargs
):
    backbone = gMLPBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def mixer_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    **kwargs
):
    backbone = MixerBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def transformer_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    heads,
    dim_head,
    mlp_dim,
    **kwargs
):
    backbone = TransformerBackbone(
        encoder_hidden_size, encoder_n_layers, heads, dim_head, mlp_dim, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def exattention_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    heads,
    dim_head,
    mlp_dim,
    **kwargs
):
    backbone = ExternalTransformerBackbone(
        encoder_hidden_size, encoder_n_layers, heads, mlp_dim, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def convmixer_audio_cls(
    num_classes, seq_len, patch_size, channels, depth, filters, kernel_size, **kwargs
):
    input_shape = (seq_len, channels)
    backbone = ConvMixerBackbone(depth, filters, patch_size, kernel_size, **kwargs)
    return Classifier(backbone, num_classes, input_shape, filters, **kwargs)


def aftfull_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    position_bias=True,
    **kwargs
):
    backbone = AFTFullBackbone(
        encoder_hidden_size,
        encoder_n_layers,
        mlp_dim,
        position_bias=position_bias,
        **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def residualattention_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    **kwargs
):
    backbone = ResidualAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def simam_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    **kwargs
):
    backbone = SimAMBackbone(encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def seattention_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    **kwargs
):
    backbone = SEAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, reduction=16, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def doubleattention_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    **kwargs
):
    backbone = DoubleAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def parnet_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    **kwargs
):
    backbone = ParNetAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
    )

    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def ufoattention_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    num_heads,
    **kwargs
):
    backbone = UFOAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, num_heads, dropout=0.1, **kwargs
    )

    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def ecaattention_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    kernel_size,
    **kwargs
):
    backbone = ECAAttentionBackbone(
        encoder_hidden_size,
        encoder_n_layers,
        mlp_dim,
        kernel_size,
        dropout=0.1,
        **kwargs
    )

    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )


def cbam_audio_cls(
    num_classes,
    seq_len,
    patch_size,
    channels,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    kernel_size,
    reduction=16,
    dropout=0.1,
    **kwargs
):
    backbone = CBAMBackbone(
        encoder_hidden_size,
        encoder_n_layers,
        mlp_dim,
        kernel_size,
        dropout=dropout,
        reduction=reduction,
        **kwargs
    )

    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return AudioClassifier(
        backbone, head, seq_len, patch_size, encoder_hidden_size, channels
    )
