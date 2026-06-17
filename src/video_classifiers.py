from backbones import *
from commons import *


def VideoClassifier(
    backbone,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    num_classes,
    dim,
    channels=3,
):
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(image_patch_size)

    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."
    assert (
        frames % frame_patch_size == 0
    ), "Frames must be divisible by the frame patch size"

    nf, nh, nw = (
        frames // frame_patch_size,
        image_height // patch_height,
        image_width // patch_width,
    )
    patch_dim = channels * patch_height * patch_width * frame_patch_size

    i_p = layers.Input((frames, image_height, image_width, channels))
    tubelets = layers.Reshape(
        (frame_patch_size, nf, patch_height, nh, patch_width, nw, channels)
    )(i_p)
    tubelets = ops.transpose(tubelets, (0, 2, 4, 6, 1, 3, 5, 7))
    tubelets = layers.Reshape((nf, nh, nw, -1))(tubelets)
    tubelets = layers.LayerNormalization()(tubelets)
    tubelets = layers.Dense(dim)(tubelets)
    tubelets = layers.LayerNormalization()(tubelets)
    tubelets = layers.Reshape((-1, dim))(tubelets)
    tubelets = backbone(tubelets)

    tubelets = layers.GlobalAveragePooling1D(name="avg_pool")(tubelets)
    o_p = layers.Dense(num_classes)(tubelets)

    return keras.Model(inputs=i_p, outputs=o_p)


def rnn_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    bidirectional=True,
    channels=3,
    **kwargs
):
    backbone = RNNBackbone(
        encoder_hidden_size, encoder_n_layers, bidirectional=bidirectional, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def gru_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    bidirectional=True,
    channels=3,
    **kwargs
):
    backbone = GRUBackbone(
        encoder_hidden_size, encoder_n_layers, bidirectional=bidirectional, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def mlp_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    channels=3,
    **kwargs
):
    backbone = MLPBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def lstm_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    bidirectional=True,
    channels=3,
    **kwargs
):
    backbone = LSTMBackbone(
        encoder_hidden_size, encoder_n_layers, bidirectional=bidirectional, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def fnet_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    channels=3,
    **kwargs
):
    backbone = FNetBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def gmlp_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    channels=3,
    **kwargs
):
    backbone = gMLPBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def mixer_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    channels=3,
    **kwargs
):
    backbone = MixerBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def transformer_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    heads,
    dim_head,
    mlp_dim,
    channels=3,
    **kwargs
):
    backbone = TransformerBackbone(
        encoder_hidden_size, encoder_n_layers, heads, dim_head, mlp_dim, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def exattention_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    heads,
    dim_head,
    mlp_dim,
    channels=3,
    **kwargs
):
    backbone = ExternalTransformerBackbone(
        encoder_hidden_size, encoder_n_layers, heads, mlp_dim, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def aftfull_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    position_bias=True,
    channels=3,
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
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def residualattention_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    channels=3,
    **kwargs
):
    backbone = ResidualAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def simam_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    channels=3,
    **kwargs
):
    backbone = SimAMBackbone(encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def seattention_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    channels=3,
    **kwargs
):
    backbone = SEAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, reduction=16, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def doubleattention_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    channels=3,
    **kwargs
):
    backbone = DoubleAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def parnet_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    channels=3,
    **kwargs
):
    backbone = ParNetAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
    )

    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def ecaattention_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    kernel_size,
    channels=3,
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
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )


def cbam_video_cls(
    num_classes,
    image_size,
    image_patch_size,
    frames,
    frame_patch_size,
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
    return VideoClassifier(
        backbone,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        encoder_hidden_size,
        channels=channels,
    )
