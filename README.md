# NeuroGenesis: Leveraging Large Language Models for Next-Gen AutoML

For references to backbone networks, please check [Backbones.md](./Backbones.md)

For prompt structure, please check [prompts directory](./prompts) 

## Overview
NeuroGenesis is an end-to-end automated system for model selection and training, designed to streamline the machine learning workflow. It supports dataset selection, synthetic data generation, model training, and hyperparameter tuning, ensuring optimal performance for various machine learning tasks.

### **Pipeline Workflow**
1. **Dataset Selection:** Users provide a dataset and define the problem type.
2. **Toy Dataset Selection or Synthetic Data Generation:** If necessary, the system selects an appropriate toy dataset or generates synthetic data.
3. **Model Training:** Multiple models and their compact variants are trained on the dataset.
4. **Model Selection and Scaling:** The best model is chosen based on predefined performance metrics, and scaling recommendations are provided.
5. **Hyperparameter Tuning:** The selected model undergoes fine-tuning using Bayesian Optimization for performance enhancement.

## Repository Structure

```
src/
├── step_1_dataset_selection.py     # Handles dataset selection and synthetic data generation
├── step_2_model_training.py       # Manages training of various models
├── step_3_model_selection.py     # Evaluates and selects the best model
├── step_4_hyperparameter_tuning.py # Fine-tunes the selected model using optimization techniques
├── requirements.txt               # Lists all required dependencies
└── README.md                      # Documentation
```

## Installation

To install all necessary dependencies, execute:

```bash
pip install -r requirements.txt
```

## API Key Configuration

Before running the scripts, create a `.env` file in the root directory to store your API keys:

```
OPENAI_API_KEY=your_openai_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
KERAS_BACKEND=torch
```

Replace `your_openai_api_key` and `your_deepseek_api_key` with your actual API keys.

## Execution Steps

Run the following commands in sequence to complete the pipeline:

```bash
python src/step_1_dataset_selection.py --dataset_file records/datasets.xlsx  --dataset_type image_classification --dataset_description "Find the best dataset for training an image classification model." --dataset_features dataset_features.json  --model gpt-4o

python src/step_2_model_training.py --modality timeseries --model gmlp --task classification \
    --dataset-x dummy_data/X_train.npy --dataset-y dummy_data/y_train.npy \
    --dataset-test dummy_data/X_test.npy --data-test-y dummy_data/y_test.npy \
    --learning-rate 0.001 --batch-size 32 --epochs 10 --hidden-size 128 --n-layers 2 --patch-size 4 --channels 7

python src/step_3_model_selection.py --csv_dir records/logs/ --pattern "*audio_cls*" \
    --task_description "Evaluate models for image classification." \
    --dataset_description "Dataset contains 10,000 labeled images." \
    --criteria ./model_selection.json --model gpt-4o

python src/step_4_hyperparameter_tuning.py --modality timeseries --model gmlp --task classification --dataset-x dummy_data/X_train.npy --dataset-y dummy_data/y_train.npy --dataset-test dummy_data/X_test.npy --data-test-y dummy_data/y_test.npy --hyperparams ./hyperparams.json
```

Each script requires relevant arguments to configure dataset selection, model training, evaluation, and tuning.

## Configuration

- Modify the script parameters to fit specific dataset and problem requirements.
- Customize the `criteria.json` file to define model selection preferences.

## Example Usage

To evaluate models for a given dataset with a predefined selection criteria:

```bash
python src/step_3_model_selection.py --csv_dir logs/ --pattern "*.csv" \
    --task_description "Image Classification Task" \
    --dataset_description "Dataset of 50,000 labeled images" \
    --criteria ./model_selection.json --model gpt-4o
```

## Authors

Developed for automated machine learning model selection and optimization, leveraging synthetic and real-world datasets.
