import numpy as np
from PIL import Image, ImageDraw
import random


def generate_image_classification_dataset(
    image_size=(28, 28), n_samples=100, n_classes=3
):
    X, y = [], []
    for _ in range(n_samples):
        img = Image.new("L", image_size, color=0)
        draw = ImageDraw.Draw(img)
        label = random.randint(0, n_classes - 1)
        rotation = random.randint(0, 360)
        shape_size = random.randint(5, min(image_size) // 2)
        if label == 0:
            draw.rectangle([(5, 5), (5 + shape_size, 5 + shape_size)], fill=255)
        elif label == 1:
            draw.ellipse([(5, 5), (5 + shape_size, 5 + shape_size)], fill=255)
        elif label == 2:
            draw.line([(5, 5), (5 + shape_size, 5 + shape_size)], fill=255, width=2)
        img = img.rotate(rotation)
        X.append(np.array(img))
        y.append(label)
    return np.array(X), np.array(y)


def generate_image_regression_dataset(image_size=(28, 28), n_samples=100):
    X, y = [], []
    for _ in range(n_samples):
        img = Image.new("L", image_size, color=0)
        draw = ImageDraw.Draw(img)
        shape_size = random.randint(5, min(image_size) // 2)
        draw.rectangle([(5, 5), (5 + shape_size, 5 + shape_size)], fill=255)
        rotation = random.randint(0, 360)
        img = img.rotate(rotation)
        intensity = np.mean(np.array(img))
        X.append(np.array(img))
        y.append(intensity)
    return np.array(X), np.array(y)


def generate_audio_classification_dataset(
    duration=1.0, sr=22050, n_samples=100, n_classes=3
):
    X, y = [], []
    t = np.linspace(0, duration, int(sr * duration))
    for _ in range(n_samples):
        label = random.randint(0, n_classes - 1)
        if label == 0:
            audio = np.sin(2 * np.pi * 220 * t)  # Sine wave
        elif label == 1:
            audio = np.sin(2 * np.pi * 440 * t)  # Higher frequency sine wave
        elif label == 2:
            audio = np.sin(2 * np.pi * 220 * t) + np.sin(
                2 * np.pi * 440 * t
            )  # Combination
        random_amplitude = random.uniform(0.5, 1.5)
        audio = audio * random_amplitude
        X.append(audio)
        y.append(label)
    return np.array(X), np.array(y)


def generate_audio_regression_dataset(duration=1.0, sr=22050, n_samples=100):
    X, y = [], []
    t = np.linspace(0, duration, int(sr * duration))
    for _ in range(n_samples):
        audio = np.sin(2 * np.pi * 220 * t)  # Sine wave
        random_amplitude = random.uniform(0.5, 1.5)
        audio = audio * random_amplitude
        target = np.mean(audio)  # Mean value
        X.append(audio)
        y.append(target)
    return np.array(X), np.array(y)


def generate_time_series_classification_dataset(length=100, n_samples=100, n_classes=3):
    X, y = [], []
    t = np.linspace(0, 1, length)
    for _ in range(n_samples):
        label = random.randint(0, n_classes - 1)
        if label == 0:
            ts = np.sin(2 * np.pi * 5 * t)  # Sine wave
        elif label == 1:
            ts = np.cos(2 * np.pi * 5 * t)  # Cosine wave
        elif label == 2:
            ts = np.sin(2 * np.pi * 5 * t) + np.cos(2 * np.pi * 5 * t)  # Combination
        random_scale = random.uniform(0.5, 1.5)
        ts = ts * random_scale
        X.append(ts)
        y.append(label)
    return np.array(X), np.array(y)


def generate_time_series_regression_dataset(length=100, n_samples=100):
    X, y = [], []
    t = np.linspace(0, 1, length)
    for _ in range(n_samples):
        ts = np.sin(2 * np.pi * 5 * t)  # Sine wave
        random_scale = random.uniform(0.5, 1.5)
        ts = ts * random_scale
        target = np.mean(ts)  # Mean value
        X.append(ts)
        y.append(target)
    return np.array(X), np.array(y)


def generate_video_classification_dataset(
    frame_size=(28, 28), n_frames=10, n_samples=100, n_classes=3
):
    X, y = [], []
    for _ in range(n_samples):
        video = []
        label = random.randint(0, n_classes - 1)
        for frame in range(n_frames):
            img = Image.new("L", frame_size, color=0)
            draw = ImageDraw.Draw(img)
            shape_size = random.randint(5, min(frame_size) // 2)
            if label == 0:
                draw.rectangle(
                    [(5, 5 + frame), (5 + shape_size, 5 + shape_size + frame)], fill=255
                )
            elif label == 1:
                draw.ellipse(
                    [(5 + frame, 5), (5 + shape_size + frame, 5 + shape_size)], fill=255
                )
            elif label == 2:
                draw.line(
                    [(5, 5 + frame), (5 + shape_size, 5 + shape_size + frame)],
                    fill=255,
                    width=2,
                )
            rotation = random.randint(0, 360)
            img = img.rotate(rotation)
            video.append(np.array(img))
        X.append(np.array(video))
        y.append(label)
    return np.array(X), np.array(y)


def generate_video_regression_dataset(frame_size=(28, 28), n_frames=10, n_samples=100):
    X, y = [], []
    for _ in range(n_samples):
        video = []
        for frame in range(n_frames):
            img = Image.new("L", frame_size, color=0)
            draw = ImageDraw.Draw(img)
            shape_size = random.randint(5, min(frame_size) // 2)
            draw.rectangle(
                [(5, 5 + frame), (5 + shape_size, 5 + shape_size + frame)], fill=255
            )
            rotation = random.randint(0, 360)
            img = img.rotate(rotation)
            video.append(np.array(img))
        target = np.mean(video)  # Mean intensity
        X.append(np.array(video))
        y.append(target)
    return np.array(X), np.array(y)


# generate mock data for multivariate timeseries forecasting
X, y = generate_time_series_regression_dataset(100, 10000)

y[0]

# generate mock data for video regression
X, y = generate_video_regression_dataset((30, 30), 15, 1500)

X.shape

X[0].shape

y[0]

# generate mock data for audio classification
X, y = generate_audio_classification_dataset(10, sr=22050, n_samples=100, n_classes=4)

X[0]

y.shape

y[0]

# Commented out IPython magic to ensure Python compatibility.
# %%writefile dataset_generation.py
# import numpy as np
# import pandas as pd
# from PIL import Image, ImageDraw, ImageFilter
# import random
# import librosa as lb
# import soundfile as sf
# from scipy import signal
# import warnings
#
#
# import argparse
# import os
# from datetime import datetime
# import inspect
# import sys
# import random
#
# warnings.filterwarnings("ignore", category=FutureWarning)
#
# # ======================
# # IMAGE DATA GENERATION
# # ======================
#
# def generate_image_dataset(
#     task='classification',
#     image_size=(64, 64),
#     n_samples=100,
#     n_classes=3,
#     color_mode='L',
#     noise_level=0.1,
#     blur_range=(0, 1),
#     occlusion_prob=0.3,
#     max_shapes=3,
#     **kwargs
# ):
#     """Unified image generator for both classification and regression"""
#     channels = 1 if color_mode == 'L' else 3
#     X, y = [], []
#
#     for _ in range(n_samples):
#         img = Image.new(color_mode, image_size, color=0)
#         draw = ImageDraw.Draw(img)
#         label_info = {'shapes': 0, 'size': 0, 'rotation': 0}
#
#         # Generate random shapes
#         num_shapes = random.randint(1, max_shapes)
#         for _ in range(num_shapes):
#             shape_type = random.choice(['rectangle', 'ellipse', 'line'])
#             position = (
#                 random.randint(0, image_size[0]-10),
#                 random.randint(0, image_size[1]-10)
#             )
#             size = random.randint(5, min(image_size)//3)
#             color = random.randint(50, 255) if color_mode == 'L' else tuple(
#                 random.randint(50, 255) for _ in range(3))
#
#             # Draw shapes
#             if shape_type == 'rectangle':
#                 draw.rectangle([position, (position[0]+size, position[1]+size)], fill=color)
#             elif shape_type == 'ellipse':
#                 draw.ellipse([position, (position[0]+size, position[1]+size)], fill=color)
#             elif shape_type == 'line':
#                 draw.line([position, (position[0]+size, position[1]+size)], fill=color, width=2)
#
#             label_info['shapes'] += 1
#             label_info['size'] += size
#
#         # Random transformations
#         rotation = random.randint(0, 360)
#         img = img.rotate(rotation, resample=Image.BILINEAR)
#         label_info['rotation'] = rotation
#
#         # Add noise
#         if noise_level > 0:
#             arr = np.array(img).astype(float)
#             arr += np.random.normal(0, noise_level*255, arr.shape)
#             arr = np.clip(arr, 0, 255).astype(np.uint8)
#             img = Image.fromarray(arr)
#
#         # Add blur
#         if random.random() < blur_range[1]:
#             img = img.filter(ImageFilter.GaussianBlur(
#                 radius=random.uniform(*blur_range)))
#
#         # Add occlusions
#         if random.random() < occlusion_prob:
#             draw = ImageDraw.Draw(img)
#             draw.rectangle([
#                 random.randint(0, image_size[0]//2),
#                 random.randint(0, image_size[1]//2),
#                 random.randint(image_size[0]//2, image_size[0]),
#                 random.randint(image_size[1]//2, image_size[1])
#             ], fill=0)
#
#         # Create labels
#         if task == 'classification':
#             label = random.randint(0, n_classes - 1)
#         else:  # Regression
#             label = np.array([
#                 label_info['shapes']/max_shapes,
#                 label_info['size']/(size*num_shapes),
#                 rotation/360,
#                 np.mean(np.array(img))/255
#             ])
#
#         X.append(np.array(img))
#         y.append(label)
#
#     return np.stack(X), np.array(y)
#
# # =====================
# # AUDIO DATA GENERATION
# # =====================
#
# def generate_audio_dataset(
#     task='classification',
#     duration=1.0,
#     sr=22050,
#     n_samples=100,
#     n_classes=3,
#     noise_type='gaussian',
#     pitch_shift_range=(-3, 3),
#     **kwargs
# ):
#     X, y = [], []
#
#     for _ in range(n_samples):
#         t = np.linspace(0, duration, int(sr * duration))
#         features = {}
#
#         # Base signal generation
#         wave_type = random.choice(['sine', 'square', 'sawtooth'])
#         freq = random.choice([220, 440, 880])
#         audio = {
#             'sine': np.sin(2 * np.pi * freq * t),
#             'square': signal.square(2 * np.pi * freq * t),
#             'sawtooth': signal.sawtooth(2 * np.pi * freq * t)
#         }[wave_type]
#
#         # Add harmonics
#         audio += 0.3 * np.sin(4 * np.pi * freq * t)
#
#         # Pitch shifting
#         if pitch_shift_range:
#             n_steps = random.randint(*pitch_shift_range)
#             audio = lb.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
#
#         # Add noise
#         if noise_type == 'gaussian':
#             audio += np.random.normal(0, 0.1, len(audio))
#         elif noise_type == 'impulse':
#             positions = np.random.choice(len(audio), size=10, replace=False)
#             audio[positions] += np.random.uniform(-0.5, 0.5, 10)
#
#         # Add room effects
#         audio = np.convolve(audio, [1, 0.6, 0.3, 0.1], mode='same')
#
#         # Create labels
#         if task == 'classification':
#             label = random.randint(0, n_classes - 1)
#         else:
#             label = np.array([
#                 freq/1000,
#                 n_steps/10 if pitch_shift_range else 0,
#                 np.mean(audio),
#                 np.std(audio)
#             ])
#
#         X.append(audio)
#         y.append(label)
#
#     return np.array(X), np.array(y)
#
# # ========================
# # TIME SERIES GENERATION
# # ========================
#
# def generate_timeseries_dataset(
#     task='classification',
#     seq_length=100,
#     n_samples=100,
#     n_features=3,
#     n_classes=3,
#     noise_level=0.1,
#     trend_strength=0.5,
#     missing_prob=0.05,
#     **kwargs
# ):
#     X, y = [], []
#
#     for _ in range(n_samples):
#         data = np.zeros((seq_length, n_features))
#         features = {}
#
#         # Base pattern
#         t = np.linspace(0, 1, seq_length)
#         trend = trend_strength * np.arange(seq_length)
#
#         for f in range(n_features):
#             components = []
#             if random.random() < 0.7:
#                 components.append(np.sin(2 * np.pi * (5 + f) * t))
#             if random.random() < 0.7:
#                 components.append(0.5 * np.cos(2 * np.pi * (3 + f) * t))
#             if random.random() < 0.3:
#                 components.append(trend)
#
#             data[:, f] = sum(components) if components else np.zeros(seq_length)
#
#             # Add noise
#             data[:, f] += np.random.normal(0, noise_level, seq_length)
#
#             # Add missing values
#             mask = np.random.rand(seq_length) < missing_prob
#             data[mask, f] = np.nan
#
#         # Create labels
#         if task == 'classification':
#             label = random.randint(0, n_classes - 1)
#         else:
#             label = np.array([
#                 np.nanmean(data[:, 0]),
#                 np.nanstd(data[:, 1]),
#                 np.sum(~np.isnan(data[:, 2]))/seq_length
#             ])
#
#         X.append(data)
#         y.append(label)
#
#     return np.array(X), np.array(y)
#
# # ====================
# # VIDEO DATA GENERATION
# # ====================
#
# def generate_video_dataset(
#     task='classification',
#     frame_size=(64, 64),
#     n_frames=16,
#     n_samples=100,
#     n_classes=3,
#     motion_types=['linear', 'circular'],
#     compression_quality=75,
#     **kwargs
# ):
#     X, y = [], []
#
#     for _ in range(n_samples):
#         video = []
#         features = {
#             'motion_type': random.choice(motion_types),
#             'color_changes': 0,
#             'size_variation': 0
#         }
#
#         # Initialize object parameters
#         obj_size = random.randint(5, min(frame_size)//4)
#         x, y_pos = random.randint(0, frame_size[0]), random.randint(0, frame_size[1])
#
#         for frame in range(n_frames):
#             img = Image.new('RGB', frame_size, color=(0, 0, 0))
#             draw = ImageDraw.Draw(img)
#
#             # Update object parameters
#             if features['motion_type'] == 'linear':
#                 x += random.randint(-2, 2)
#                 y_pos += random.randint(-2, 2)
#             elif features['motion_type'] == 'circular':
#                 angle = 2 * np.pi * frame / n_frames
#                 x = frame_size[0]//2 + int(frame_size[0]//4 * np.cos(angle))
#                 y_pos = frame_size[1]//2 + int(frame_size[1]//4 * np.sin(angle))
#
#             # Random color changes
#             if random.random() < 0.2:
#                 color = (random.randint(100,255), random.randint(100,255), random.randint(100,255))
#                 features['color_changes'] += 1
#
#             # Size variation
#             if random.random() < 0.1:
#                 obj_size = random.randint(5, min(frame_size)//4)
#                 features['size_variation'] += 1
#
#             draw.ellipse([(x, y_pos), (x + obj_size, y_pos + obj_size)], fill=color)
#
#             # Add compression artifacts
#             img.save('temp.jpg', quality=compression_quality)
#             img = Image.open('temp.jpg')
#
#             video.append(np.array(img))
#
#         # Create labels
#         if task == 'classification':
#             label = random.randint(0, n_classes - 1)
#         else:
#             label = np.array([
#                 features['color_changes']/n_frames,
#                 features['size_variation']/n_frames,
#                 obj_size/min(frame_size)
#             ])
#
#         X.append(np.stack(video))
#         y.append(label)
#
#
#     return np.array(X), np.array(y)
#
#
# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Generate synthetic datasets with classification/regression support')
#
#     # Required arguments
#     parser.add_argument('--type', required=True,
#                         choices=['image', 'audio', 'timeseries', 'video'],
#                         help='Type of dataset to generate')
#     parser.add_argument('--task', required=True,
#                         choices=['classification', 'regression'],
#                         help='Type of machine learning task')
#     parser.add_argument('--path', required=True,
#                         help='Output directory for NPY files')
#
#     # Common parameters
#     parser.add_argument('--num-samples', type=int, default=1000, dest='n_samples',
#                         help='Number of samples to generate (default: 1000)')
#     parser.add_argument('--seed', type=int, default=None,
#                         help='Random seed for reproducibility')
#     parser.add_argument('--n-classes', type=int, default=3,
#                         help='Number of classes (classification only)')
#
#     # Image/Video parameters
#     img_vid_group = parser.add_argument_group('Image/Video options')
#     img_vid_group.add_argument('--image-size', type=int, nargs=2, default=(64, 64),
#                               dest='image_size', help='Frame dimensions (H W)')
#     img_vid_group.add_argument('--color-mode', default='L', choices=['L', 'RGB'],
#                               dest='color_mode', help='Color space')
#     img_vid_group.add_argument('--max-shapes', type=int, default=3,
#                               help='Maximum number of shapes per image')
#     img_vid_group.add_argument('--noise-level', type=float, default=0.1,
#                               help='Noise intensity (0-1 scale)')
#     img_vid_group.add_argument('--blur-range', type=float, nargs=2, default=[0.0, 1.0],
#                               dest='blur_range', help='Blur radius range')
#     img_vid_group.add_argument('--occlusion-prob', type=float, default=0.3,
#                               dest='occlusion_prob', help='Occlusion probability')
#
#     # Audio parameters
#     audio_group = parser.add_argument_group('Audio options')
#     audio_group.add_argument('--duration', type=float, default=1.0,
#                             help='Audio duration in seconds')
#     audio_group.add_argument('--sr', type=int, default=22050,
#                             help='Sampling rate')
#     audio_group.add_argument('--pitch-shift', type=int, nargs=2, default=[-3, 3],
#                             dest='pitch_shift_range', help='Pitch shift range')
#     audio_group.add_argument('--noise-type', default='gaussian',
#                             choices=['gaussian', 'impulse'], dest='noise_type',
#                             help='Type of noise to add')
#
#     # Time Series parameters
#     ts_group = parser.add_argument_group('Time Series options')
#     ts_group.add_argument('--seq-length', type=int, default=100,
#                          dest='seq_length', help='Sequence length')
#     ts_group.add_argument('--n-features', type=int, default=3,
#                          dest='n_features', help='Number of features')
#     ts_group.add_argument('--trend-strength', type=float, default=0.5,
#                          dest='trend_strength', help='Trend intensity')
#     ts_group.add_argument('--missing-prob', type=float, default=0.05,
#                          dest='missing_prob', help='Missing value probability')
#
#     # Video parameters
#     video_group = parser.add_argument_group('Video options')
#     video_group.add_argument('--n-frames', type=int, default=16,
#                             dest='n_frames', help='Frames per video')
#     video_group.add_argument('--motion-types', nargs='+', default=['linear', 'circular'],
#                             dest='motion_types', help='Allowed motion patterns')
#     video_group.add_argument('--compression', type=int, default=75,
#                             dest='compression_quality', help='JPEG quality (1-100)')
#
#     return parser.parse_args()
#
# def main():
#     args = parse_arguments()
#
#     # Set random seeds
#     if args.seed is not None:
#         np.random.seed(args.seed)
#         random.seed(args.seed)
#
#     # Create output directory
#     os.makedirs(args.path, exist_ok=True)
#
#
#     # Map dataset types to generators
#     generators = {
#         'image': generate_image_dataset,
#         'audio': generate_audio_dataset,
#         'timeseries': generate_timeseries_dataset,
#         'video': generate_video_dataset
#     }
#
#     # Prepare arguments for generator
#     generator_args = {
#         'task': args.task,
#         'n_classes': args.n_classes if args.task == 'classification' else None
#     }
#
#     # Add modality-specific parameters
#     if args.type in ['image', 'video']:
#         generator_args.update({
#             'image_size': args.image_size,
#             'color_mode': args.color_mode,
#             'max_shapes': args.max_shapes,
#             'noise_level': args.noise_level,
#             'blur_range': args.blur_range,
#             'occlusion_prob': args.occlusion_prob
#         })
#
#     if args.type == 'audio':
#         generator_args.update({
#             'duration': args.duration,
#             'sr': args.sr,
#             'pitch_shift_range': args.pitch_shift_range,
#             'noise_type': args.noise_type
#         })
#
#     if args.type == 'timeseries':
#         generator_args.update({
#             'seq_length': args.seq_length,
#             'n_features': args.n_features,
#             'trend_strength': args.trend_strength,
#             'missing_prob': args.missing_prob
#         })
#
#     if args.type == 'video':
#         generator_args.update({
#             'n_frames': args.n_frames,
#             'motion_types': args.motion_types,
#             'compression_quality': args.compression_quality
#         })
#
#     # Generate dataset
#     try:
#         X, y = generators[args.type](**generator_args)
#     except Exception as e:
#         print(f"Generation failed: {str(e)}")
#         sys.exit(1)
#
#     # Save results
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     fname = f"{args.task}_{args.type}_{timestamp}"
#
#     np.save(os.path.join(args.path, f"{fname}_X.npy"), X)
#     np.save(os.path.join(args.path, f"{fname}_y.npy"), y)
#
#     print(f"Successfully generated {args.task} dataset:")
#     print(f"- Type: {args.type}")
#     print(f"- Samples: {len(X)}")
#     print(f"- Features shape: {X[0].shape if hasattr(X, 'shape') else X[0].shape}")
#     print(f"- Saved to: {os.path.join(args.path, fname)}_[X|y].npy")
#
# if __name__ == "__main__":
#     main()
