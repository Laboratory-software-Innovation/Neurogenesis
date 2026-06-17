import argparse
import numpy as np
import os
import random
import librosa as lb
from PIL import Image, ImageDraw, ImageFilter
from datetime import datetime
import sys

# ======================
# IMAGE DATA GENERATION
# ======================


def generate_image_dataset(
    task="classification",
    image_size=(64, 64),
    n_samples=100,
    n_classes=3,
    color_mode="L",
    noise_level=0.1,
    blur_range=(0, 1),
    occlusion_prob=0.3,
    max_shapes=3,
):
    """Generate synthetic image dataset for classification or regression."""
    X, y = [], []
    for _ in range(n_samples):
        img = Image.new(color_mode, image_size, color=0)
        draw = ImageDraw.Draw(img)
        label_info = {"shapes": 0, "size": 0, "rotation": 0}
        num_shapes = random.randint(1, max_shapes)

        for _ in range(num_shapes):
            shape_type = random.choice(["rectangle", "ellipse", "line"])
            position = (
                random.randint(0, image_size[0] - 10),
                random.randint(0, image_size[1] - 10),
            )
            size = random.randint(5, min(image_size) // 3)
            color = (
                random.randint(50, 255)
                if color_mode == "L"
                else tuple(random.randint(50, 255) for _ in range(3))
            )

            if shape_type == "rectangle":
                draw.rectangle(
                    [position, (position[0] + size, position[1] + size)], fill=color
                )
            elif shape_type == "ellipse":
                draw.ellipse(
                    [position, (position[0] + size, position[1] + size)], fill=color
                )
            elif shape_type == "line":
                draw.line(
                    [position, (position[0] + size, position[1] + size)],
                    fill=color,
                    width=2,
                )

            label_info["shapes"] += 1
            label_info["size"] += size

        rotation = random.randint(0, 360)
        img = img.rotate(rotation, resample=Image.BILINEAR)
        label_info["rotation"] = rotation

        if noise_level > 0:
            arr = np.array(img).astype(float)
            arr += np.random.normal(0, noise_level * 255, arr.shape)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        if random.random() < blur_range[1]:
            img = img.filter(
                ImageFilter.GaussianBlur(radius=random.uniform(*blur_range))
            )

        if random.random() < occlusion_prob:
            draw = ImageDraw.Draw(img)
            draw.rectangle(
                [
                    random.randint(0, image_size[0] // 2),
                    random.randint(0, image_size[1] // 2),
                    random.randint(image_size[0] // 2, image_size[0]),
                    random.randint(image_size[1] // 2, image_size[1]),
                ],
                fill=0,
            )

        label = (
            random.randint(0, n_classes - 1)
            if task == "classification"
            else np.array(
                [
                    label_info["shapes"] / max_shapes,
                    label_info["size"] / (size * num_shapes),
                    rotation / 360,
                    np.mean(np.array(img)) / 255,
                ]
            )
        )

        X.append(np.array(img))
        y.append(label)

    return np.stack(X), np.array(y)


# =====================
# AUDIO DATA GENERATION
# =====================


def generate_audio_dataset(
    task="classification",
    duration=1.0,
    sr=22050,
    n_samples=100,
    n_classes=3,
    noise_type="gaussian",
    pitch_shift_range=(-3, 3),
):
    """Generate synthetic audio dataset."""
    X, y = [], []
    for _ in range(n_samples):
        t = np.linspace(0, duration, int(sr * duration))
        freq = random.choice([220, 440, 880])
        audio = np.sin(2 * np.pi * freq * t)

        if pitch_shift_range:
            n_steps = random.randint(*pitch_shift_range)
            audio = lb.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

        if noise_type == "gaussian":
            audio += np.random.normal(0, 0.1, len(audio))

        label = (
            random.randint(0, n_classes - 1)
            if task == "classification"
            else np.array(
                [
                    freq / 1000,
                    n_steps / 10 if pitch_shift_range else 0,
                    np.mean(audio),
                    np.std(audio),
                ]
            )
        )

        X.append(audio)
        y.append(label)

    return np.array(X), np.array(y)


# ========================
# TIME SERIES GENERATION
# ========================


def generate_timeseries_dataset(
    task="classification",
    seq_length=100,
    n_samples=100,
    n_features=3,
    n_classes=3,
    noise_level=0.1,
):
    """Generate synthetic time-series dataset."""
    X, y = [], []
    for _ in range(n_samples):
        data = np.zeros((seq_length, n_features))
        t = np.linspace(0, 1, seq_length)

        for f in range(n_features):
            data[:, f] = np.sin(2 * np.pi * (5 + f) * t) + np.random.normal(
                0, noise_level, seq_length
            )

        label = (
            random.randint(0, n_classes - 1)
            if task == "classification"
            else np.array([np.mean(data[:, 0]), np.std(data[:, 1])])
        )

        X.append(data)
        y.append(label)

    return np.array(X), np.array(y)


# ====================
# MAIN FUNCTION
# ====================


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets")

    parser.add_argument(
        "--type",
        required=True,
        choices=["image", "audio", "timeseries"],
        help="Type of dataset to generate",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["classification", "regression"],
        help="Task type",
    )
    parser.add_argument("--path", required=True, help="Output directory for NPY files")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        dest="n_samples",
        help="Number of samples to generate (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--n-classes", type=int, default=3, help="Number of classes")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    os.makedirs(args.path, exist_ok=True)

    generators = {
        "image": generate_image_dataset,
        "audio": generate_audio_dataset,
        "timeseries": generate_timeseries_dataset,
    }

    generator_args = {
        "task": args.task,
        "n_classes": args.n_classes if args.task == "classification" else None,
    }

    try:
        X, y = generators[args.type](**generator_args)
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{args.task}_{args.type}_{timestamp}"

    np.save(os.path.join(args.path, f"{fname}_X.npy"), X)
    np.save(os.path.join(args.path, f"{fname}_y.npy"), y)

    print(
        f"✅ Successfully generated {args.task} dataset:\n- Type: {args.type}\n- Samples: {len(X)}\n- Saved to: {args.path}"
    )


if __name__ == "__main__":
    main()
