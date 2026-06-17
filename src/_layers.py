import keras
from keras import layers, ops, Sequential, backend

from keras import ops, layers, backend, Sequential
from keras.src.ops.operation_utils import compute_pooling_output_shape

from functools import partial


class FNetLayer(layers.Layer):
    """
    https://keras.io/examples/vision/mlp_image_classification/
    """

    def __init__(self, embedding_dim, dropout_rate=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ffn = keras.Sequential(
            [
                layers.Dense(units=embedding_dim, activation="gelu"),
                layers.Dropout(rate=dropout_rate),
                layers.Dense(units=embedding_dim),
            ]
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Apply fourier transformations.
        real_part = inputs
        im_part = keras.ops.zeros_like(inputs)
        x = keras.ops.fft2((real_part, im_part))[0]
        # Add skip connection.
        x = x + inputs
        # Apply layer normalization.
        x = self.normalize1(x)
        # Apply Feedfowrad network.
        x_ffn = self.ffn(x)
        # Add skip connection.
        x = x + x_ffn
        # Apply layer normalization.
        return self.normalize2(x)


class gMLPLayer(layers.Layer):
    def __init__(self, num_patches, embedding_dim, dropout_rate=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channel_projection1 = keras.Sequential(
            [
                layers.Dense(units=embedding_dim * 2, activation="gelu"),
                layers.Dropout(rate=dropout_rate),
            ]
        )

        self.channel_projection2 = layers.Dense(units=embedding_dim)

        self.spatial_projection = layers.Dense(
            units=num_patches, bias_initializer="Ones"
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def spatial_gating_unit(self, x):
        # Split x along the channel dimensions.
        # Tensors u and v will in the shape of [batch_size, num_patchs, embedding_dim].
        u, v = keras.ops.split(x, indices_or_sections=2, axis=2)
        # Apply layer normalization.
        v = self.normalize2(v)
        # Apply spatial projection.
        v_channels = keras.ops.transpose(v, axes=(0, 2, 1))
        v_projected = self.spatial_projection(v_channels)
        v_projected = keras.ops.transpose(v_projected, axes=(0, 2, 1))
        # Apply element-wise multiplication.
        return u * v_projected

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize1(inputs)
        # Apply the first channel projection. x_projected shape: [batch_size, num_patches, embedding_dim * 2].
        x_projected = self.channel_projection1(x)
        # Apply the spatial gating unit. x_spatial shape: [batch_size, num_patches, embedding_dim].
        x_spatial = self.spatial_gating_unit(x_projected)
        # Apply the second channel projection. x_projected shape: [batch_size, num_patches, embedding_dim].
        x_projected = self.channel_projection2(x_spatial)
        # Add skip connection.
        return x + x_projected


class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches, activation="gelu"),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches, activation="gelu"),
                layers.Dense(units=hidden_units),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = keras.ops.transpose(x, axes=(0, 2, 1))
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = keras.ops.transpose(mlp1_outputs, axes=(0, 2, 1))
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x


def ExternalAttention(
    dim,
    num_heads,
    dim_coefficient=4,
    attention_dropout=0,
    projection_dropout=0,
):
    assert dim % num_heads == 0

    def _apply(x):
        nonlocal num_heads
        _, num_patch, channel = x.shape
        num_heads = num_heads * dim_coefficient
        x = layers.Dense(int(dim * dim_coefficient))(x)
        # create tensor [batch_size, num_patches, num_heads, dim*dim_coefficient//num_heads]
        x = ops.reshape(
            x, (-1, num_patch, num_heads, dim * dim_coefficient // num_heads)
        )
        x = ops.transpose(x, axes=[0, 2, 1, 3])
        # a linear layer M_k
        attn = layers.Dense(dim // dim_coefficient)(x)
        # normalize attention map
        attn = layers.Softmax(axis=2)(attn)
        # dobule-normalization
        attn = layers.Lambda(
            lambda attn: ops.divide(
                attn,
                ops.convert_to_tensor(1e-9) + ops.sum(attn, axis=-1, keepdims=True),
            )
        )(attn)
        attn = layers.Dropout(attention_dropout)(attn)
        # a linear layer M_v
        x = layers.Dense(dim * dim_coefficient // num_heads)(attn)
        x = ops.transpose(x, axes=[0, 2, 1, 3])
        x = ops.reshape(x, [-1, num_patch, dim * dim_coefficient])
        # a linear layer to project original dim
        x = layers.Dense(dim)(x)
        x = layers.Dropout(projection_dropout)(x)
        return x

    return _apply


class AFTFull(layers.Layer):
    """An Attention Free Transformer [https://arxiv.org/pdf/2105.14103v1.pdf]"""

    def __init__(self, projection_dim, position_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.position_bias = position_bias
        self.projection_dim = projection_dim
        self.to_q = layers.Dense(projection_dim)
        self.to_k = layers.Dense(projection_dim)
        self.to_v = layers.Dense(projection_dim)

    def build(self, input_shape):
        seq_len = input_shape[1]
        if self.position_bias:
            self.position_biases = self.add_weight(
                (1, seq_len, self.projection_dim), "zeros"
            )

    def call(self, inputs):
        b, n, d = ops.shape(inputs)
        q = self.to_q(inputs)  # bs,n,dim
        k = self.to_k(inputs)
        k = ops.expand_dims(k, 0)  # 1,bs,n,dim
        v = self.to_v(inputs)
        v = ops.expand_dims(v, 0)  # 1,bs,n,dim
        if self.position_bias:
            k = k + self.position_biases
            v = v + self.position_biases
        numerator = ops.sum(ops.exp(k) * v, axis=2)  # n,bs,dim
        denominator = ops.sum(ops.exp(k), axis=2)  # n,bs,dim
        out = numerator / denominator  # n,bs,dim
        out = ops.sigmoid(q) * (ops.transpose(out, (1, 0, 2)))  # bs,n,dim
        return out


class ResidualAttention(layers.Layer):
    """Residual Attention: A Simple but Effective Method for Multi-Label Recognition [https://arxiv.org/abs/2108.02456]"""

    def __init__(self, dim, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.dim = dim
        self.fc = layers.Dense(self.dim, use_bias=False)

    def call(self, x):
        b, n, c = ops.shape(x)
        x = self.fc(x)
        x_raw = x

        x_avg = ops.mean(x_raw, axis=1)  # b,num_class
        x_max = ops.max(x_raw, axis=1)[0]  # b,num_class

        out = x_avg + self.alpha * x_max
        return out


def n_tuple(inputs, n):
    if isinstance(inputs, int):
        return tuple([inputs] * n)
    elif isinstance(inputs, (list, tuple)):
        assert len(inputs) == n, f"Pool size must have length {n}"
        return tuple(inputs)
    else:
        raise ValueError(
            f"inputs must be an integer or a {tuple, list} found: {inputs}"
        )


two_tuple = partial(n_tuple, n=2)
three_tuple = partial(n_tuple, n=3)


def standardize_pool_size(dims, output_size, data_format=None):
    if dims == 1:
        return output_size
    elif dims == 2:
        return two_tuple(output_size)
    elif dims == 3:
        return three_tuple(output_size)
    else:
        raise ValueError(
            f"Invalid Output Size received, Expected Int or Tuple of size {dims}"
        )


def get_input_dim(data_format, dims):
    if data_format == "channels_last":
        if dims == 1:
            return 1
        elif dims == 2:
            return slice(1, 3)
        elif dims == 3:
            return slice(1, 4)
        else:
            raise ValueError(
                f"Invalid Pool Dimensions received, Expected Int or Tuple of size {dims}"
            )
    if data_format == "channels_first":
        if dims == 1:
            return 2
        elif dims == 2:
            return slice(2, 4)
        elif dims == 3:
            return slice(2, 4)
        else:
            raise ValueError(
                f"Invalid Pool Dimensions received, Expected Int or Tuple of size {dims}"
            )


class BaseAdaptivePool(layers.Layer):
    def __init__(
        self,
        output_size,
        pool_dimensions,
        data_format=None,
        padding="valid",
        pool_mode="average",
        **kwargs,
    ):
        super(BaseAdaptivePool, self).__init__(**kwargs)
        assert pool_mode in [
            "average",
            "max",
        ], "Adaptive Pooling mode must be either average or max"
        self.pool_mode = pool_mode
        self.pool_dimensions = pool_dimensions
        self.output_size = standardize_pool_size(pool_dimensions, output_size)
        if data_format is None:
            self.data_format = backend.image_data_format()
        else:
            self.data_format = data_format
        assert self.data_format in [
            "channels_last",
            "channels_first",
        ], "Data format must be either channels_last or channels_first"
        self.input_dim = get_input_dim(self.data_format, pool_dimensions)
        self.padding = padding

    def build(self, input_shape):
        input_size = input_shape[self.input_dim]
        if self.pool_dimensions == 1:
            self.strides = input_size // self.output_size
            self.kernel_size = input_size - (self.output_size - 1) * self.strides
        elif self.pool_dimensions == 2:
            self.strides = (
                input_size[0] // self.output_size[0],
                input_size[1] // self.output_size[1],
            )
            self.kernel_size = (
                input_size[0] - (self.output_size[0] - 1) * self.strides[0],
                input_size[1] - (self.output_size[1] - 1) * self.strides[1],
            )
        elif self.pool_dimensions == 3:
            self.strides = (
                input_size[0] // self.output_size[0],
                input_size[1] // self.output_size[1],
                input_size[2] // self.output_size[2],
            )
            self.kernel_size = (
                input_size[0] - (self.output_size[0] - 1) * self.strides[0],
                input_size[1] - (self.output_size[1] - 1) * self.strides[1],
                (self.output_size[2] - 1) * self.strides[2],
            )

    def call(self, inputs):
        if self.pool_mode == "max":
            return ops.max_pool(
                inputs,
                pool_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
            )
        elif self.pool_mode == "average":
            return ops.average_pool(
                inputs,
                pool_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
            )
        else:
            raise ValueError(
                "`pool_mode` must be either 'max' or 'average'. Received: "
                f"{self.pool_mode}."
            )

    def compute_output_shape(self, input_shape):
        return compute_pooling_output_shape(
            input_shape,
            self.kernel_size,
            self.strides,
            self.padding,
            self.data_format,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_size": self.output_size,
                "pool_dimensions": self.pool_dimensions,
                "data_format": self.data_format,
                "padding": self.padding,
                "pool_mode": self.pool_mode,
            }
        )
        return config


class AdaptiveAveragePool1D(BaseAdaptivePool):
    """Adaptive Pooling like torch.nn.AdaptiveAvgPool1d"""

    def __init__(self, output_size, data_format=None, padding="valid", **kwargs):
        super(AdaptiveAveragePool1D, self).__init__(
            output_size,
            pool_dimensions=1,
            pool_mode="average",
            data_format=data_format,
            padding=padding,
            **kwargs,
        )


class SEAttention(layers.Layer):
    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.pooling = AdaptiveAveragePool1D(1)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.fc = Sequential(
            [
                layers.Dense(
                    input_dim // self.reduction, use_bias=False, activation="relu"
                ),
                layers.Dense(input_dim, use_bias=False, activation="sigmoid"),
            ]
        )

    def call(self, x):
        x_skip = x
        b, n, c = ops.shape(x)
        x = self.pooling(x)
        x = ops.reshape(x, (b, c))
        x = self.fc(x)
        x = ops.expand_dims(x, axis=1)
        return x_skip * ops.broadcast_to(x, ops.shape(x_skip))


class SimAM(layers.Layer):
    def __init__(self, e_lambda=1e-4, activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.e_lambda = e_lambda
        self.activation = layers.Activation(activation)

    def call(self, x):
        b, n, c = ops.shape(x)  # b n c
        x_mean = ops.mean(x, axis=1, keepdims=True)  # b 1 c
        x_minus_mu_square = ops.square(x - x_mean)  # b n c
        denom = ops.sum(x_minus_mu_square, axis=1, keepdims=True) / ops.cast(
            n, "float32"
        )  # b n c
        weights = x_minus_mu_square / (4 * (denom + self.e_lambda)) + 0.5
        return x * self.activation(weights)


class DoubleAttention(layers.Layer):
    """A2-Nets: Double Attention Networks [https://arxiv.org/pdf/1810.11579.pdf]"""

    def __init__(self, dim, value_dim=None, reconstruct=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.value_dim = value_dim or dim
        self.reconstruct = reconstruct

    def build(self, input_shape):
        self.in_dim = input_shape[-1]
        self.q = layers.Dense(self.dim)
        self.k = layers.Dense(self.value_dim)
        self.v = layers.Dense(self.value_dim)
        if self.reconstruct:
            self.reconstruct = layers.Dense(self.in_dim)

    def call(self, x):
        b, n, c = ops.shape(x)
        assert c == self.in_dim, "input dim not equal to in_dim"
        q = self.q(x)  # b,n,dim
        # q = ops.reshape(q, (b, h * w, self.dim))  # bnd
        k = self.k(x)  # b,h,w,vd
        # k = ops.reshape(k, (b, h * w, self.value_dim))
        attention_maps = ops.softmax(k, axis=1)  # bnv
        v = self.v(x)  # b,n,vd
        # v = ops.reshape(v, (b, h * w, self.value_dim))
        attention_vectors = ops.softmax(v, axis=1)  # b n vd
        global_descriptors = ops.einsum("bnd,bnv->bvd", q, attention_maps)  # b,d,vd
        out = ops.einsum("bvd,bnv->bnd", global_descriptors, attention_vectors)
        # out = ops.reshape(out, (b, h, w, self.dim))  # b,h,w,c_n
        if self.reconstruct:
            out = self.reconstruct(out)  # b,h,w,c
        return out


def _orthogonal_matrix(dim: int, seed: int = None):
    # Random matrix from normal distribution
    mat = keras.random.normal((dim, dim), seed=seed)
    # QR decomposition to two orthogonal matrices
    q, _ = ops.qr(mat, mode="reduced")
    return ops.transpose(q, [1, 0])


def orthogonal_matrix(num_rows: int, num_cols: int, seed=None):
    num_full_blocks = int(num_rows / num_cols)
    blocks = []
    for _ in range(num_full_blocks):
        q = _orthogonal_matrix(num_cols)
        blocks.append(q)
    remain_rows = num_rows - num_full_blocks * num_cols
    if remain_rows > 0:
        q = _orthogonal_matrix(num_cols)
        blocks.append(q[:remain_rows])
    mat = ops.concatenate(blocks)
    return mat


def linear_attention(q, k, v):
    _k = ops.expand_dims(ops.sum(k, axis=-2), axis=-1)
    D_inv = 1.0 / (q @ _k)
    kv = ops.transpose(k, axes=[0, 1, 3, 2]) @ v
    qkv = q @ kv
    out = ops.einsum("...L,...Ld->...Ld", ops.squeeze(D_inv, axis=-1), qkv)
    return out


def generalized_kernel(x, mat, kernel=ops.relu, epsilon=0.001):
    batch_size, num_heads = ops.shape(x)[:2]
    projection = ops.transpose(mat, axes=[1, 0])  # Transpose along correct axes
    projection = ops.tile(projection, [1, num_heads, 1, 1])  # Expand dimensions
    x = ops.matmul(x, projection)
    out = kernel(x) + epsilon
    return out


class PerformerProjection(layers.Layer):
    def __init__(self, num_cols, kernel=ops.relu):
        super().__init__()
        self.num_rows = int(num_cols * ops.log(num_cols))
        self.num_cols = num_cols

        # Generate an orthogonal projection matrix
        self.projection_matrix = orthogonal_matrix(self.num_rows, self.num_cols)
        self.kernel = kernel

    def call(self, q, k, v):
        q = generalized_kernel(q, self.projection_matrix, self.kernel)
        k = generalized_kernel(k, self.projection_matrix, self.kernel)
        out = linear_attention(q, k, v)
        return out


class PerformerAttention(layers.Layer):

    def __init__(
        self,
        channels,
        heads,
        kernel=ops.relu,
        qkv_bias=False,
        attn_out_bias=True,
        dropout=0.0,
    ):
        super().__init__()
        assert channels % heads == 0
        head_channels = channels // heads

        self.heads = heads
        self.head_channels = head_channels
        self.kernel = kernel
        self.fast_attn = PerformerProjection(head_channels, kernel)

        inner_channels = head_channels * heads
        self.q = layers.Dense(inner_channels, use_bias=qkv_bias)
        self.k = layers.Dense(inner_channels, use_bias=qkv_bias)
        self.v = layers.Dense(inner_channels, use_bias=qkv_bias)
        self.attn_out = layers.Dense(channels, use_bias=attn_out_bias)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, mask=None):
        B, N, *_ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)

        q = ops.transpose(
            ops.reshape(q, (-1, N, self.heads, self.head_channels)), axes=(0, 2, 1, 3)
        )
        k = ops.transpose(
            ops.reshape(k, (-1, N, self.heads, self.head_channels)), axes=(0, 2, 1, 3)
        )
        v = ops.transpose(
            ops.reshape(v, (-1, N, self.heads, self.head_channels)), axes=(0, 2, 1, 3)
        )

        if mask is not None:
            mask = mask[:, None, :, None]
            v = ops.where(mask, v, ops.zeros_like(v))

        out = self.fast_attn(q, k, v)
        out_dim = ops.shape(out)[-1]
        out = ops.transpose(out, axes=(0, 2, 1, 3))  # Transpose back
        out = ops.reshape(
            out,
            (
                -1,
                N,
            ),
        )
        out = self.attn_out(out)
        out = self.dropout(out)
        return out


class ParNetAttention(layers.Layer):
    def __init__(self, activation="selu", **kwargs):
        super().__init__(**kwargs)
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.sse = Sequential(
            [
                AdaptiveAveragePool1D(1),
                layers.Conv1D(input_dim, kernel_size=1, activation="sigmoid"),
            ]
        )

        self.conv1 = Sequential(
            [layers.Conv1D(input_dim, kernel_size=1), layers.BatchNormalization()]
        )
        self.conv3 = Sequential(
            [
                layers.Conv1D(input_dim, kernel_size=3, padding="same"),
                layers.BatchNormalization(),
            ]
        )
        self.activation = layers.Activation(self.activation)

    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.sse(x) * x
        out = self.activation(x1 + x2 + x3)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation": self.activation,
            }
        )
        return config


def XNorm(x, gamma):
    norm_tensor = ops.norm(x, ord=2, axis=-1, keepdims=True)
    return x * gamma / norm_tensor


class UFOAttention(layers.Layer):
    def __init__(self, d_model, d_k, d_v, h, dropout=0.1):
        """
        Args:
            d_model: Output dimensionality of the model
            d_k: Dimensionality of queries and keys
            d_v: Dimensionality of values
            h: Number of heads
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.dropout = dropout

    def build(self, input_shape):
        # Initialize layers
        self.fc_q = layers.Dense(self.h * self.d_k, kernel_initializer="glorot_normal")
        self.fc_k = layers.Dense(self.h * self.d_k, kernel_initializer="glorot_normal")
        self.fc_v = layers.Dense(self.h * self.d_v, kernel_initializer="glorot_normal")
        self.fc_o = layers.Dense(self.d_model, kernel_initializer="glorot_normal")

        self.dropout = layers.Dropout(self.dropout)

        # Initialize gamma parameter
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1, self.h, 1, 1),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, queries, keys, values):
        b_s = ops.shape(queries)[0]
        nq = ops.shape(queries)[1]
        nk = ops.shape(keys)[1]

        # Linear transformations and reshape
        q = self.fc_q(queries)  # (b_s, nq, h*d_k)
        k = self.fc_k(keys)  # (b_s, nk, h*d_k)
        v = self.fc_v(values)  # (b_s, nk, h*d_v)

        # Reshape and transpose
        q = ops.reshape(q, [b_s, nq, self.h, self.d_k])
        q = ops.transpose(q, [0, 2, 1, 3])  # (b_s, h, nq, d_k)

        k = ops.reshape(k, [b_s, nk, self.h, self.d_k])
        k = ops.transpose(k, [0, 2, 3, 1])  # (b_s, h, d_k, nk)

        v = ops.reshape(v, [b_s, nk, self.h, self.d_v])
        v = ops.transpose(v, [0, 2, 1, 3])  # (b_s, h, nk, d_v)

        # Compute attention
        kv = ops.matmul(k, v)  # (b_s, h, d_k, d_v)
        kv_norm = XNorm(kv, self.gamma)  # (b_s, h, d_k, d_v)
        q_norm = XNorm(q, self.gamma)  # (b_s, h, nq, d_k)

        # Final computations
        out = ops.matmul(q_norm, kv_norm)  # (b_s, h, nq, d_v)
        out = ops.transpose(out, [0, 2, 1, 3])  # (b_s, nq, h, d_v)
        out = ops.reshape(out, [b_s, nq, self.h * self.d_v])  # (b_s, nq, h*d_v)

        # Apply final linear transformation
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out


class ECAAttention(layers.Layer):
    """
    ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks [https://arxiv.org/pdf/1910.03151.pdf]
    """

    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.pooling = AdaptiveAveragePool1D(1)
        self.conv = layers.Conv1D(1, kernel_size=kernel_size, padding="same")

    def call(self, x):
        y = self.pooling(x)  # b 1, c
        # y = ops.squeeze(y, axis=2)  # b, 1, c
        y = ops.transpose(y, axes=[0, 2, 1])  # b, c, 1
        y = self.conv(y)  # b, c, 1
        y = ops.sigmoid(y)  # b, c, 1
        y = ops.transpose(y, axes=[0, 2, 1])  # b, 1, c
        # y = ops.expand_dims(y, axis=2)  # bs, 1, 1, c
        return x * ops.broadcast_to(y, ops.shape(x))


def create_feedforward_network(ff_dim, embed_dim, name=None):
    return keras.Sequential(
        [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)], name=name
    )


def load_balanced_loss(router_probs, expert_mask):
    # router_probs [tokens_per_batch, num_experts] is the probability assigned for
    # each expert per token. expert_mask [tokens_per_batch, num_experts] contains
    # the expert with the highest router probability in one−hot format.

    num_experts = ops.shape(expert_mask)[-1]
    # Get the fraction of tokens routed to each expert.
    # density is a vector of length num experts that sums to 1.
    density = ops.mean(expert_mask, axis=0)
    # Get fraction of probability mass assigned to each expert from the router
    # across all tokens. density_proxy is a vector of length num experts that sums to 1.
    density_proxy = ops.mean(router_probs, axis=0)
    # Want both vectors to have uniform allocation (1/num experts) across all
    # num_expert elements. The two vectors will be pushed towards uniform allocation
    # when the dot product is minimized.
    loss = ops.mean(density_proxy * density) * ops.cast((num_experts**2), "float32")
    return loss


class Router(layers.Layer):
    def __init__(self, num_experts, expert_capacity):
        self.num_experts = num_experts
        self.route = layers.Dense(units=num_experts)
        self.expert_capacity = expert_capacity
        super().__init__()

    def call(self, inputs, training=False):
        # inputs shape: [tokens_per_batch, embed_dim]
        # router_logits shape: [tokens_per_batch, num_experts]
        router_logits = self.route(inputs)

        if training:
            # Add noise for exploration across experts.
            router_logits += keras.random.uniform(
                shape=router_logits.shape, minval=0.9, maxval=1.1
            )
        # Probabilities for each token of what expert it should be sent to.
        router_probs = keras.activations.softmax(router_logits, axis=-1)
        # Get the top−1 expert for each token. expert_gate is the top−1 probability
        # from the router for each token. expert_index is what expert each token
        # is going to be routed to.
        expert_gate, expert_index = ops.top_k(router_probs, k=1)
        # expert_mask shape: [tokens_per_batch, num_experts]
        expert_mask = ops.one_hot(expert_index, self.num_experts)
        # Compute load balancing loss.
        aux_loss = load_balanced_loss(router_probs, expert_mask)
        self.add_loss(aux_loss)
        # Experts have a fixed capacity, ensure we do not exceed it. Construct
        # the batch indices, to each expert, with position in expert make sure that
        # not more that expert capacity examples can be routed to each expert.
        position_in_expert = ops.cast(
            ops.cumsum(expert_mask, axis=0) * expert_mask, "int32"
        )
        # Keep only tokens that fit within expert capacity.
        expert_mask *= ops.cast(
            ops.less(ops.cast(position_in_expert, "int32"), self.expert_capacity),
            "float32",
        )
        expert_mask_flat = ops.sum(expert_mask, axis=-1)
        # Mask out the experts that have overflowed the expert capacity.
        expert_gate *= expert_mask_flat
        # Combine expert outputs and scaling with router probability.
        # combine_tensor shape: [tokens_per_batch, num_experts, expert_capacity]
        combined_tensor = ops.expand_dims(
            expert_gate
            * expert_mask_flat
            * ops.squeeze(ops.one_hot(expert_index, self.num_experts), 1),
            -1,
        ) * ops.squeeze(ops.one_hot(position_in_expert, self.expert_capacity), 1)
        # Create binary dispatch_tensor [tokens_per_batch, num_experts, expert_capacity]
        # that is 1 if the token gets routed to the corresponding expert.
        dispatch_tensor = ops.cast(combined_tensor, "float32")

        return dispatch_tensor, combined_tensor


class Switch(layers.Layer):
    def __init__(
        self, num_experts, embed_dim, ff_dim, num_tokens_per_batch, capacity_factor=1
    ):
        self.num_tokens_per_batch = num_tokens_per_batch
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.experts = [
            create_feedforward_network(ff_dim, embed_dim) for _ in range(num_experts)
        ]

        self.expert_capacity = num_tokens_per_batch // self.num_experts
        self.router = Router(self.num_experts, self.expert_capacity)
        super().__init__()

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        num_tokens_per_example = ops.shape(inputs)[1]

        # inputs shape: [num_tokens_per_batch, embed_dim]
        inputs = ops.reshape(inputs, [self.num_tokens_per_batch, self.embed_dim])
        # dispatch_tensor shape: [expert_capacity, num_experts, tokens_per_batch]
        # combine_tensor shape: [tokens_per_batch, num_experts, expert_capacity]
        dispatch_tensor, combine_tensor = self.router(inputs)
        # expert_inputs shape: [num_experts, expert_capacity, embed_dim]
        expert_inputs = ops.einsum("ab,acd->cdb", inputs, dispatch_tensor)
        expert_inputs = ops.reshape(
            expert_inputs, [self.num_experts, self.expert_capacity, self.embed_dim]
        )
        # Dispatch to experts
        expert_input_list = ops.unstack(expert_inputs, axis=0)
        expert_output_list = [
            self.experts[idx](expert_input)
            for idx, expert_input in enumerate(expert_input_list)
        ]
        # expert_outputs shape: [expert_capacity, num_experts, embed_dim]
        expert_outputs = ops.stack(expert_output_list, axis=1)
        # expert_outputs_combined shape: [tokens_per_batch, embed_dim]
        expert_outputs_combined = ops.einsum(
            "abc,xba->xc", expert_outputs, combine_tensor
        )
        # output shape: [batch_size, num_tokens_per_example, embed_dim]
        outputs = ops.reshape(
            expert_outputs_combined,
            [batch_size, num_tokens_per_example, self.embed_dim],
        )
        return outputs


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn, dropout_rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # The ffn can be either a standard feedforward network or a switch
        # layer with a Mixture of Experts.
        self.ffn = ffn
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class AdaptiveMaxPool1D(BaseAdaptivePool):
    """Adaptive Pooling like torch.nn.AdaptiveMaxPool2d"""

    def __init__(self, output_size, data_format=None, padding="valid", **kwargs):
        super(AdaptiveMaxPool1D, self).__init__(
            output_size,
            pool_dimensions=1,
            pool_mode="max",
            data_format=data_format,
            padding=padding,
            **kwargs,
        )


class ChannelAttention(layers.Layer):
    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.maxpool = AdaptiveMaxPool1D(1)
        self.avgpool = AdaptiveAveragePool1D(1)

    def build(self, input_shape):

        input_dim = input_shape[-1]
        self.se = Sequential(
            [
                layers.Conv1D(input_dim // self.reduction, 1, use_bias=False),
                layers.Activation("relu"),
                layers.Conv1D(input_dim, 1, use_bias=False),
            ]
        )

    def call(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = ops.sigmoid(max_out + avg_out)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "reduction": self.reduction,
            }
        )
        return config


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv1D(1, kernel_size=kernel_size, padding="same")

    def call(self, x):

        axis = 2
        max_result = ops.max(x, axis=axis, keepdims=True)
        avg_result = ops.mean(x, axis=axis, keepdims=True)
        result = ops.concatenate([max_result, avg_result], axis)
        output = self.conv(result)
        output = ops.sigmoid(output)
        return output


class CBAMBlock(layers.Layer):
    """
    CBAM: Convolutional Block Attention Module [https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf]
    """

    def __init__(self, reduction=16, kernel_size=49, **kwargs):
        super().__init__(**kwargs)
        self.channel_attention = ChannelAttention(reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def call(self, x):
        residual = x
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out + residual
