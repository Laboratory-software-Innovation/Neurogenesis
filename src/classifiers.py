import keras as keras
from backbones import *
from commons import *


def RNNClassifier(
    num_classes,
    input_shape,
    encoder_hidden_size,
    encoder_n_layers,
    bidirectional=True,
    **kwargs
):
    backbone = RNNBackbone(
        encoder_hidden_size, encoder_n_layers, bidirectional=bidirectional, **kwargs
    )
    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def GRUClassifier(
    num_classes,
    input_shape,
    encoder_hidden_size,
    encoder_n_layers,
    bidirectional=True,
    **kwargs
):
    backbone = GRUBackbone(
        encoder_hidden_size, encoder_n_layers, bidirectional=bidirectional, **kwargs
    )
    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def MLPClassifier(
    num_classes, input_shape, encoder_hidden_size, encoder_n_layers, **kwargs
):
    backbone = MLPBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def LSTMClassifier(
    num_classes,
    input_shape,
    encoder_hidden_size,
    encoder_n_layers,
    bidirectional=True,
    **kwargs
):
    backbone = LSTMBackbone(
        encoder_hidden_size, encoder_n_layers, bidirectional=bidirectional, **kwargs
    )
    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def FNetClassifier(
    num_classes, input_shape, encoder_hidden_size, encoder_n_layers, **kwargs
):
    backbone = FNetBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def gMLPClassifier(
    num_classes, input_shape, encoder_hidden_size, encoder_n_layers, **kwargs
):
    backbone = gMLPBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def MixerClassifier(
    num_classes, input_shape, encoder_hidden_size, encoder_n_layers, **kwargs
):
    backbone = MixerBackbone(encoder_hidden_size, encoder_n_layers, **kwargs)
    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def TransformerClassifier(
    num_classes,
    input_shape,
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
    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def ExternalTransformerClassifier(
    num_classes,
    input_shape,
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
    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def ConvMixerClassifier(
    num_classes, input_shape, depth, filters, patch_size, kernel_size, **kwargs
):
    backbone = ConvMixerBackbone(depth, filters, patch_size, kernel_size, **kwargs)
    return Classifier(backbone, num_classes, input_shape, filters, **kwargs)


def AFTFullClassifier(
    num_classes,
    input_shape,
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
    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def ResidualAttentionClassifier(
    num_classes, input_shape, encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
):
    backbone = ResidualAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
    )
    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def SimAMClassifier(
    num_classes, input_shape, encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
):
    backbone = SimAMBackbone(encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs)
    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def SEAttentionClassifier(
    num_classes, input_shape, encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
):
    backbone = SEAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, reduction=16, **kwargs
    )
    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def DoubleAttentionClassifier(
    num_classes, input_shape, encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
):
    backbone = DoubleAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
    )
    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def PerformerAttentionClassifier(
    num_classes,
    input_shape,
    encoder_hidden_size,
    num_heads,
    encoder_n_layers,
    mlp_dim,
    **kwargs
):
    backbone = PerformerAttentionBackbone(
        encoder_hidden_size, num_heads, encoder_n_layers, mlp_dim, **kwargs
    )

    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def ParNetAttentionClassifier(
    num_classes, input_shape, encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
):
    backbone = ParNetAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, **kwargs
    )

    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def UFOAttentionClassifier(
    num_classes,
    input_shape,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    num_heads,
    **kwargs
):
    backbone = UFOAttentionBackbone(
        encoder_hidden_size, encoder_n_layers, mlp_dim, num_heads, dropout=0.1, **kwargs
    )

    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def ECAAttentionClassifier(
    num_classes,
    input_shape,
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

    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def CBAMClassifier(
    num_classes,
    input_shape,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    kernel_size,
    **kwargs
):
    backbone = CBAMBackbone(
        encoder_hidden_size,
        encoder_n_layers,
        mlp_dim,
        kernel_size,
        dropout=0.1,
        reduction=16,
        **kwargs
    )

    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def SwitchTransformerClassifier(
    num_classes,
    input_shape,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    num_experts,
    num_heads,
    **kwargs
):
    backbone = SwitchTransformerBackbone(
        encoder_hidden_size,
        encoder_n_layers,
        mlp_dim,
        num_experts,
        num_heads,
        dropout=0.1,
        **kwargs
    )

    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def PerformerAttentionClassifier(
    num_classes,
    input_shape,
    encoder_hidden_size,
    num_heads,
    encoder_n_layers,
    mlp_dim,
    **kwargs
):
    backbone = PerformerAttentionBackbone(
        encoder_hidden_size, num_heads, encoder_n_layers, mlp_dim, **kwargs
    )

    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)


def SwitchTransformerClassifier(
    num_classes,
    input_shape,
    encoder_hidden_size,
    encoder_n_layers,
    mlp_dim,
    num_experts,
    num_heads,
    **kwargs
):
    backbone = SwitchTransformerBackbone(
        encoder_hidden_size,
        encoder_n_layers,
        mlp_dim,
        num_experts,
        num_heads,
        dropout=0.1,
        **kwargs
    )

    return Classifier(backbone, num_classes, input_shape, encoder_hidden_size, **kwargs)
