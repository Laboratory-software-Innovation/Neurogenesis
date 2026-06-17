import argparse
import numpy as np

from dotenv import load_dotenv

load_dotenv()

from keras.optimizers import Adam
from audio_classifiers import *
from image_classifiers import *
from video_classifiers import *
from classifiers import *
from commons import *

MODEL_MAP = {
    "audio": {
        "rnn": rnn_audio_cls,
        "gru": gru_audio_cls,
        "mlp": mlp_audio_cls,
        "lstm": lstm_audio_cls,
        "fnet": fnet_audio_cls,
        "gmlp": gmlp_audio_cls,
        "mixer": mixer_audio_cls,
        "transformer": transformer_audio_cls,
        "external_attention": exattention_audio_cls,
        "convmixer": convmixer_audio_cls,
        "aftfull": aftfull_audio_cls,
        "residual_attention": residualattention_audio_cls,
        "simam": simam_audio_cls,
        "se_attention": seattention_audio_cls,
        "double_attention": doubleattention_audio_cls,
        "parnet": parnet_audio_cls,
        "ufo_attention": ufoattention_audio_cls,
        "eca_attention": ecaattention_audio_cls,
        "cbam": cbam_audio_cls,
    },
    "image": {
        "rnn": rnn_image_cls,
        "gru": gru_image_cls,
        "mlp": mlp_image_cls,
        "lstm": lstm_image_cls,
        "fnet": fnet_image_cls,
        "gmlp": gmlp_image_cls,
        "mixer": mixer_image_cls,
        "transformer": transformer_image_cls,
        "external_attention": exattention_image_cls,
        "aftfull": aftfull_image_cls,
        "residual_attention": residualattention_image_cls,
        "simam": simam_image_cls,
        "se_attention": seattention_image_cls,
        "double_attention": doubleattention_image_cls,
        "parnet": parnet_image_cls,
        "eca_attention": ecaattention_image_cls,
        "cbam": cbam_image_cls,
    },
    "video": {
        "rnn": rnn_video_cls,
        "gru": gru_video_cls,
        "mlp": mlp_video_cls,
        "lstm": lstm_video_cls,
        "fnet": fnet_video_cls,
        "gmlp": gmlp_video_cls,
        "mixer": mixer_video_cls,
        "transformer": transformer_video_cls,
        "external_attention": exattention_video_cls,
        "aftfull": aftfull_video_cls,
        "residual_attention": residualattention_video_cls,
        "simam": simam_video_cls,
        "se_attention": seattention_video_cls,
        "double_attention": doubleattention_video_cls,
        "parnet": parnet_video_cls,
        "eca_attention": ecaattention_video_cls,
        "cbam": cbam_video_cls,
    },
    "timeseries": {  # Time-series classification & regression models
        "rnn": RNNClassifier,
        "gru": GRUClassifier,
        "mlp": MLPClassifier,
        "lstm": LSTMClassifier,
        "fnet": FNetClassifier,
        "gmlp": gMLPClassifier,
        "mixer": MixerClassifier,
        "transformer": TransformerClassifier,
        "external_transformer": ExternalTransformerClassifier,
        "conv_mixer": ConvMixerClassifier,
        "aft_full": AFTFullClassifier,
        "residual_attention": ResidualAttentionClassifier,
        "simam": SimAMClassifier,
        "se_attention": SEAttentionClassifier,
        "double_attention": DoubleAttentionClassifier,
        "performer_attention": PerformerAttentionClassifier,
        "par_net_attention": ParNetAttentionClassifier,
        "ufo_attention": UFOAttentionClassifier,
        "eca_attention": ECAAttentionClassifier,
        "cbam": CBAMClassifier,
        "switch_transformer": SwitchTransformerClassifier,
    },
}

UNIQUE_MODELS = set([model for modality in MODEL_MAP for model in MODEL_MAP[modality]])


def load_data(x_path, y_path):
    """Load dataset from .npy files"""
    return np.load(x_path), np.load(y_path)


def train_model(args):
    """Train the specified model with user-defined hyperparameters"""
    print(f"Loading dataset: {args.dataset_x}, {args.dataset_y}")
    X_train, y_train = load_data(args.dataset_x, args.dataset_y)
    X_test, y_test = load_data(args.dataset_test, args.data_test_y)

    # Get the classifier function based on modality and model type
    model_func = MODEL_MAP.get(args.modality, {}).get(args.model)
    if not model_func:
        raise ValueError(
            f"Unsupported model '{args.model}' for modality '{args.modality}'"
        )

    print(f"Initializing {args.model} model for {args.modality} with task {args.task}")
    input_shape = X_train.shape[1:]

    # Define the number of classes or output dimension based on the task
    if args.task == "classification":
        num_classes = len(np.unique(y_train))
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [
            "accuracy",
            # keras.metrics.Precision(name="precision"),
            # keras.metrics.Recall(name="recall"),
            # keras.metrics.AUC(name="auc"),  # AUC-ROC score
        ]

    else:  # Regression
        num_classes = 1  # Regression outputs a single continuous value
        loss_fn = "mse"
        metrics = ["mae"]

    # Model initialization with user-defined hyperparameters
    if args.modality == "timeseries":
        model = model_func(
            num_classes=num_classes,
            input_shape=input_shape,
            encoder_hidden_size=args.hidden_size,
            encoder_n_layers=args.n_layers,
        )
    elif args.modality == "image":
        model = model_func(
            num_classes=num_classes,
            image_shape=input_shape,
            patch_size=args.patch_size,
            encoder_hidden_size=args.hidden_size,
            encoder_n_layers=args.n_layers,
        )
    else:
        model = model_func(
            num_classes=num_classes,
            seq_len=input_shape[0],
            patch_size=args.patch_size,
            channels=args.channels,
            encoder_hidden_size=args.hidden_size,
            encoder_n_layers=args.n_layers,
        )

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate), loss=loss_fn, metrics=metrics
    )

    # Train the model
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Save trained model
    model_filename = f"{args.model}_{args.modality}_{args.task}_model.h5"
    model.save(model_filename)
    print(f"Model saved as {model_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a classification or regression model with user-defined hyperparameters."
    )

    # Required Arguments
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["audio", "image", "video", "timeseries"],
        help="Choose the data modality",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=UNIQUE_MODELS,
        help="Model type to train",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "regression"],
        help="Specify the task type",
    )
    parser.add_argument(
        "--dataset-x",
        type=str,
        required=True,
        help="Path to training dataset (features)",
    )
    parser.add_argument(
        "--dataset-y", type=str, required=True, help="Path to training dataset (labels)"
    )
    parser.add_argument(
        "--dataset-test",
        type=str,
        required=True,
        help="Path to test dataset (features)",
    )
    parser.add_argument(
        "--data-test-y", type=str, required=True, help="Path to test dataset (labels)"
    )

    # Hyperparameters
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=128, help="Hidden size for model layers"
    )
    parser.add_argument(
        "--n-layers", type=int, default=2, help="Number of layers in the model"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=4,
        help="Patch size for model processing (if applicable)",
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Number of channels in the input data"
    )

    args = parser.parse_args()

    train_model(args)
