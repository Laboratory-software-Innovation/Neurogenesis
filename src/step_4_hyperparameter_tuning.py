import argparse
import json
import numpy as np
import keras_tuner as kt
from keras.optimizers import Adam
from keras import losses
from audio_classifiers import *
from image_classifiers import *
from video_classifiers import *
from classifiers import *
from commons import *

# Model mapping dictionary
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


def get_all_models():
    """Retrieve all model names for argparse choices."""
    return list(
        set([model for models in MODEL_MAP.values() for model in models.keys()])
    )


def load_data(x_path, y_path):
    """Load dataset from .npy files"""
    return np.load(x_path), np.load(y_path)


def load_hyperparameters(file_path):
    """Load hyperparameters from a JSON file"""
    with open(file_path, "r") as f:
        return json.load(f)


def build_model(hp, input_shape, num_classes, model_func, args):
    """Build model with hyperparameter tuning"""
    if args.modality == "timeseries":
        model = model_func(
            num_classes=num_classes,
            input_shape=input_shape,
            encoder_hidden_size=hp["hidden_size"],
            encoder_n_layers=hp["n_layers"],
        )
    elif args.modality == "image":
        model = model_func(
            num_classes=num_classes,
            image_shape=input_shape,
            patch_size=hp["patch_size"],
            encoder_hidden_size=hp["hidden_size"],
            encoder_n_layers=hp["n_layers"],
        )
    else:
        model = model_func(
            num_classes=num_classes,
            seq_len=input_shape[0],
            patch_size=hp["patch_size"],
            channels=hp["channels"],
            encoder_hidden_size=hp["hidden_size"],
            encoder_n_layers=hp["n_layers"],
        )

    loss_function = (
        losses.SparseCategoricalCrossentropy(from_logits=True)
        if args.task == "classification"
        else losses.MeanSquaredError()
    )
    metrics = ["accuracy"] if args.task == "classification" else ["mae"]

    model.compile(
        optimizer=Adam(learning_rate=hp["learning_rate"]),
        loss=loss_function,
        metrics=metrics,
    )
    return model


def tune_hyperparameters(args, hyperparams):
    """Perform Bayesian Optimization from provided hyperparameters"""
    X_train, y_train = load_data(args.dataset_x, args.dataset_y)
    X_test, y_test = load_data(args.dataset_test, args.data_test_y)

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train)) if args.task == "classification" else 1

    model_func = MODEL_MAP.get(args.modality, {}).get(args.model)
    if not model_func:
        raise ValueError(
            f"Unsupported model '{args.model}' for modality '{args.modality}'"
        )

    def model_builder(hp):
        return build_model(hp, input_shape, num_classes, model_func)

    tuner = kt.BayesianOptimization(
        model_builder,
        objective="val_accuracy" if num_classes > 1 else "val_mae",
        max_trials=10,
        directory="tuner_results",
        project_name=f"{args.model}_{args.modality}_{args.task}_tuning",
    )

    tuner.search(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters found:", best_hps.values)
    return best_hps


def train_model(args, hyperparams):
    """Train model with user-defined hyperparameters"""
    X_train, y_train = load_data(args.dataset_x, args.dataset_y)
    X_test, y_test = load_data(args.dataset_test, args.data_test_y)

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train)) if args.task == "classification" else 1

    model_func = MODEL_MAP.get(args.modality, {}).get(args.model)
    if not model_func:
        raise ValueError(
            f"Unsupported model '{args.model}' for modality '{args.modality}'"
        )

    model = build_model(hyperparams, input_shape, num_classes, model_func, args)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    model_filename = f"best_{args.model}_{args.modality}_{args.task}_model.h5"
    model.save(model_filename)
    print(f"Optimized model saved as {model_filename}")


def test_model(args):
    """Test the trained model"""
    from keras.models import load_model

    X_test, y_test = load_data(args.dataset_test, args.data_test_y)
    model = load_model(f"best_{args.model}_{args.modality}_{args.task}_model.h5")
    loss, metric = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy/MAE: {metric}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization and Model Training"
    )
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
        choices=get_all_models(),
        help="Model type",
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
    parser.add_argument(
        "--hyperparams",
        type=str,
        required=True,
        help="Path to JSON file containing hyperparameters",
    )
    parser.add_argument("--test", action="store_true", help="Run model testing")

    args = parser.parse_args()

    hyperparams = load_hyperparameters(args.hyperparams)
    best_hps = tune_hyperparameters(args, hyperparams)
    train_model(args, best_hps)
