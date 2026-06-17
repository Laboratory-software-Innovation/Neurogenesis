from backbones import *
from commons import *


def ImageClassifier(
    backbone,
    head,
    input_shape,
    patch_size,
    embedding_dim,
    aug=None,
):
    inputs = layers.Input(shape=input_shape)
    if aug is not None:
        img = aug(inputs)
    else:
        img = inputs
    num_patches = (input_shape[0] // patch_size) ** 2
    x = Patches(patch_size)(img)
    # B, C = ops.shape(img)[0], ops.shape(img)[-1]
    # x = ops.image.extract_patches(x, patch_size)
    # x = ops.reshape(x, (B, -1, patch_size * patch_size * C))
    x = PatchEmbedding(num_patches, embedding_dim)(x)
    x = backbone(x)
    x = layers.GlobalMaxPool1D()(x)
    outputs = head(x)
    return keras.Model(inputs=inputs, outputs=outputs)


# image classifiers
def rnn_image_cls(
    num_classes,
    image_shape,
    patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    bidirectional=True,
    **kwargs
):
    backbone = RNNBackbone(
        encoder_hidden_size, encoder_n_layers, bidirectional=bidirectional, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def gru_image_cls(
    num_classes,
    image_shape,
    patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    bidirectional=True,
    **kwargs
):
    backbone = GRUBackbone(
        encoder_hidden_size, encoder_n_layers, bidirectional=bidirectional, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def mlp_image_cls(
    num_classes,
    image_shape,
    patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    **kwargs
):
    backbone = MLPBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def lstm_image_cls(
    num_classes,
    image_shape,
    patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    bidirectional=True,
    **kwargs
):
    backbone = LSTMBackbone(
        encoder_hidden_size, encoder_n_layers, bidirectional=bidirectional, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def fnet_image_cls(
    num_classes,
    image_shape,
    patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    **kwargs
):
    backbone = FNetBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def gmlp_image_cls(
    num_classes,
    image_shape,
    patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    **kwargs
):
    backbone = gMLPBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def mixer_image_cls(
    num_classes,
    image_shape,
    patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    **kwargs
):
    backbone = MixerBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def transformer_image_cls(
    num_classes,
    image_shape,
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
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def exattention_image_cls(
    num_classes,
    image_shape,
    patch_size,
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
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def aftfull_image_cls(
    num_classes,
    image_shape,
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
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def residualattention_image_cls(
    num_classes,
    image_shape,
    patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    **kwargs
):
    backbone = ResidualAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def simam_image_cls(
    num_classes,
    image_shape,
    patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    **kwargs
):
    backbone = SimAMBackbone(encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs)
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def seattention_image_cls(
    num_classes,
    image_shape,
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
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def doubleattention_image_cls(
    num_classes,
    image_shape,
    patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    **kwargs
):
    backbone = DoubleAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
    )
    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def parnet_image_cls(
    num_classes,
    image_shape,
    patch_size,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    **kwargs
):
    backbone = ParNetAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
    )

    head = Head(num_classes, head_dim=int(encoder_hidden_size / 2), dropout=0.0)
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def ecaattention_image_cls(
    num_classes,
    image_shape,
    patch_size,
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
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )


def cbam_image_cls(
    num_classes,
    image_shape,
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
    return ImageClassifier(
        backbone,
        head,
        image_shape,
        patch_size,
        encoder_hidden_size,
        aug=None,
    )
