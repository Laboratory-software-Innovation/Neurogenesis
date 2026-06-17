import argparse
import os
import pandas as pd
import glob
import json
import scipy.stats as stats
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


def parse_arguments():
    """
    Parses command-line arguments for processing model statistics and querying LLM.
    """
    parser = argparse.ArgumentParser(description="Model Selection & Evaluation")

    parser.add_argument(
        "--csv_pattern",
        type=str,
        required=True,
        help="Pattern for CSV files (e.g., *audio_reg*.csv)",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        required=True,
        help="Description of the machine learning task",
    )
    parser.add_argument(
        "--dataset_description",
        type=str,
        required=True,
        help="Dataset used in the project",
    )
    parser.add_argument(
        "--criteria",
        type=str,
        required=True,
        help="Path to JSON file containing selection criteria",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM to use (gpt-4o or deepseek/deepseek-chat)",
    )

    return parser.parse_args()


def generate_statistical_significance_csv(csv_pattern):
    """
    Generate statistical significance results and save them as a CSV file.
    """
    csv_files = glob.glob(csv_pattern)
    results = []
    alpha = 0.05

    if not csv_files:
        print("❌ No CSV files found matching the given pattern.")
        return ""

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            model_name = os.path.basename(file).split("_")[0]

            t_stat, p_val = stats.ttest_rel(df["loss"], df["val_loss"])
            results.append(
                {
                    "CSV": file,
                    "model_name": model_name,
                    "Comparison": "Loss vs Val_Loss",
                    "p-value": p_val,
                    "Significant": p_val < alpha,
                }
            )

            t_stat_loss, p_val_loss = stats.ttest_1samp(df["loss"], 0)
            results.append(
                {
                    "CSV": file,
                    "model_name": model_name,
                    "Comparison": "All Losses Across Epochs",
                    "p-value": p_val_loss,
                    "Significant": p_val_loss < alpha,
                }
            )

            t_stat_val_loss, p_val_val_loss = stats.ttest_1samp(df["val_loss"], 0)
            results.append(
                {
                    "CSV": file,
                    "model_name": model_name,
                    "Comparison": "All Val_Losses Across Epochs",
                    "p-value": p_val_val_loss,
                    "Significant": p_val_val_loss < alpha,
                }
            )

        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")

    results_df = pd.DataFrame(results)
    csv_filename = "statistical_significance_results.csv"
    results_df.to_csv(csv_filename, index=False)
    return csv_filename


def query_llm(prompt, model="gpt-4o"):
    """
    Queries the selected LLM (OpenAI or DeepSeek) for model selection.
    """
    if "gpt" in model.lower():
        api_key = OPENAI_API_KEY
        base_url = None
    elif "deepseek" in model.lower():
        api_key = DEEPSEEK_API_KEY
        base_url = "https://api.deepseek.com"
        model = "deepseek-reasoner"
    else:
        return "❌ Unsupported model."

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"🔹 Querying {model}...")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model
        )
        return (
            response.choices[0].message.content
            if response.choices
            else "No response received."
        )
    except Exception as e:
        return f"❌ Error querying {model}: {e}"


def read_statistical_significance_with_losses(csv_file):
    df = pd.read_csv(csv_file)

    models = {}
    model_summaries = {
        "rnn": "A recurrent neural network (RNN) designed for sequential data processing, handling time-series or text data for classification tasks.",
        "gru": "A Gated Recurrent Unit (GRU) model, an improved version of RNN, designed to capture long-term dependencies with fewer parameters.",
        "mlp": "A Multi-Layer Perceptron (MLP) model, which consists of fully connected layers for processing non-sequential data.",
        "lstm": "A Long Short-Term Memory (LSTM) model, a type of RNN capable of learning long-term dependencies and overcoming the vanishing gradient problem.",
        "fnet": "An FNet model that replaces traditional attention mechanisms with Fourier transforms for efficient large-scale data processing.",
        "gmlp": "A gMLP model (Gated MLP), a type of MLP that uses gated activation functions to better capture dependencies in data.",
        "mixer": "A Mixer model, which combines MLP layers for both spatial and channel mixing to process input data efficiently.",
        "transformer": "A Transformer model, leveraging multi-head attention mechanisms for sequence-to-sequence tasks like text classification or machine translation.",
        "externaltransformer": "An External Transformer model, optimized for handling external data or large-scale inputs with transformer-based architectures.",
        "convmixer": "A ConvMixer model, combining convolutional operations with mixing layers to handle high-dimensional data such as images.",
        "aftfull": "An Attention-Free Transformer (AFTFull) model, designed for efficient processing without relying on traditional attention mechanisms.",
        "residualattention": "A Residual Attention model, which incorporates residual connections within attention layers for better feature propagation in deep networks.",
        "simam": "A SimAM (Simple Attention Mechanism) model, designed to reduce complexity while maintaining effective attention mechanisms.",
        "seattention": "A Squeeze-and-Excitation Attention (SEAttention) model, which recalibrates channel-wise feature responses for improved performance.",
        "doubleattention": "A Double Attention model, combining two attention mechanisms for both spatial and temporal contexts.",
        "performerattention": "A Performer model using linear attention mechanisms (FAVOR+) for scalable, efficient sequence processing.",
        "parnetattention": "A ParNet (Position-Aware Relation Network) model, designed for image-text matching using both semantic and spatial relationships.",
        "ufoattention": "A UFO Attention model, utilizing flexible attention heads and regularization for better generalization in classification tasks.",
        "ecaattention": "An Efficient Channel Attention (ECA) model, which uses 1D convolution to model cross-channel relationships while minimizing computational cost.",
        "cbam": "A CBAM (Convolutional Block Attention Module) model, which combines both channel and spatial attention to enhance feature extraction for image classification.",
        "switchtransformer": "A Switch Transformer model, incorporating a mixture of experts approach for scalable computation, using only a subset of parameters during inference.",
    }

    for _, row in df.iterrows():
        model_name = row["CSV"].split("/")[-1].split(".")[0]
        comparison = row["Comparison"]
        p_value = row["p-value"]

        if model_name not in models:
            model_key = model_name.split("/")[-1].split("_")[0]
            summary = model_summaries.get(
                model_key, "Summary not found"
            )  # Default if no match
            models[model_name] = {
                "loss": None,
                "val_loss": None,
                "loss_significance": None,
                "val_loss_significance": None,
                "summary": model_summaries.get(
                    model_name.split("/")[-1].split("_")[0], "No summary available."
                ),
            }

        if comparison == "All Losses Across Epochs" and isinstance(
            p_value, (int, float)
        ):
            models[model_name]["loss_significance"] = (
                p_value if not pd.isna(p_value) else None
            )

        elif comparison == "All Val_Losses Across Epochs" and isinstance(
            p_value, (int, float)
        ):
            models[model_name]["val_loss_significance"] = (
                p_value if not pd.isna(p_value) else None
            )
    for model_name, model_data in models.items():
        loss_file = [
            row["CSV"]
            for _, row in df.iterrows()
            if row["CSV"].split("/")[-1].split(".")[0] == model_name
        ][0]
        try:
            loss_df = pd.read_csv(loss_file)

            # Function to filter out non-numeric values
            def filter_numeric(series):
                return [
                    f"e{idx + 1}:{value}"
                    for idx, value in enumerate(series)
                    if isinstance(value, (int, float)) and not pd.isna(value)
                ]

            # Format losses and val_losses ensuring only numeric values are included
            loss_with_epochs = "; ".join(filter_numeric(loss_df["loss"]))
            val_loss_with_epochs = "; ".join(filter_numeric(loss_df["val_loss"]))
            mae_with_epochs = "; ".join(filter_numeric(loss_df["mae"]))
            mape_with_epochs = "; ".join(filter_numeric(loss_df["mape"]))
            val_mae_with_epochs = "; ".join(filter_numeric(loss_df["val_mae"]))
            val_mape_with_epochs = "; ".join(filter_numeric(loss_df["val_mape"]))

            # Assign values to model_data
            model_data["loss"] = loss_with_epochs
            model_data["val_loss"] = val_loss_with_epochs
            model_data["mae"] = mae_with_epochs
            model_data["mape"] = mape_with_epochs
            model_data["val_mae"] = val_mae_with_epochs
            model_data["val_mape"] = val_mape_with_epochs

        except Exception as e:
            print(f"Error reading {loss_file}: {e}")

    return models


def model_selection_prompt(models, task_description, original_dataset, criteria):
    """
    Generate a detailed and comprehensive prompt for selecting the best model
    based on provided statistics, criteria, and model summaries.

    Args:
        models (dict): Dictionary containing model statistics, including loss details.
        task_description (str): A description of the task for which the model is being selected.
        original_dataset (str): Description of the dataset used in the project.
        criteria (dict): A dictionary of criteria to guide model selection.

    Returns:
        str: The generated, detailed prompt.
    """
    prompt = (
        f"You are a highly skilled and detail-oriented machine learning engineer. "
        f"Your expertise is required to select the most suitable model for a specific task. "
        f"You will evaluate several candidate models based on their performance metrics, behavior, and alignment "
        f"with the given criteria to make an informed and well-justified decision.\n\n"
        f"### Task Description\n"
        f"The model selection is for the following task:\n"
        f"{task_description}\n\n"
        f"### Dataset Description\n"
        f"The task uses the dataset described as follows:\n"
        f"{original_dataset}\n\n"
        f"### Available Models and Their Performance Metrics\n"
        f"Below are details of the candidate models. Each model has been evaluated based on its performance metrics "
        f"across training and validation phases. Carefully review the details to identify patterns, strengths, and weaknesses.\n\n"
    )

    for idx, (model_name, stats) in enumerate(models.items(), 1):
        model_summary = stats.get("summary", "No summary available.")
        prompt += (
            f"#### Model {idx}: {model_name}\n"
            f"- **Summary:** {model_summary}\n"
            f"- **Performance Metrics:**\n"
            f"  - Loss Significance: {stats.get('loss_significance', 'N/A')}\n"
            f"  - Validation Loss Significance: {stats.get('val_loss_significance', 'N/A')}\n"
            f"  - Epoch-wise Loss: {stats.get('loss', 'N/A')}\n"
            f"  - Epoch-wise Validation Loss: {stats.get('val_loss', 'N/A')}\n"
            f"  - Mean Absolute Error (MAE): {stats.get('mae', 'N/A')}\n"
            f"  - Mean Absolute Percentage Error (MAPE): {stats.get('mape', 'N/A')}\n"
            f"  - Validation MAE: {stats.get('val_mae', 'N/A')}\n"
            f"  - Validation MAPE: {stats.get('val_mape', 'N/A')}\n\n"
        )

    prompt += (
        f"### Selection Criteria\n"
        f"The ideal model should align with the following criteria:\n"
    )
    if criteria:
        for criterion, value in criteria.items():
            prompt += f"- **{criterion}:** {value}\n"
        prompt += "\n"
    else:
        prompt += "- No specific criteria provided. Use your expert judgment based on the task description and metrics.\n\n"

    prompt += (
        "### Instructions for Model Evaluation and Selection\n"
        "To determine the best model, follow these steps:\n\n"
        "1. **Understand the Context:** Analyze the task and dataset description to understand the project's requirements.\n"
        "2. **Examine Model Details:** Review the performance metrics and summaries for each model. Identify strengths, weaknesses, "
        "and any signs of overfitting, underfitting, or instability.\n"
        "3. **Compare Against Criteria:** Cross-reference the model metrics with the provided selection criteria. Focus on metrics "
        "such as validation loss, MAE, MAPE, and other indicators of generalization.\n"
        "4. **Evaluate Trade-offs:** Consider trade-offs such as higher accuracy versus computational efficiency, robustness, "
        "or scalability.\n"
        "5. **Provide Recommendations for Scaling:** Suggest parameter adjustments, such as embedding dimensions, number of layers, "
        "dropout rates, or other hyperparameters, to further optimize the selected model.\n"
        "6. **Justify Your Selection:** Articulate why the chosen model is the most suitable for the task. Include a detailed rationale "
        "that references metrics and task requirements. Acknowledge any limitations and propose strategies to mitigate them.\n\n"
        "### Example Evaluation Approach\n"
        "For example, if one model has lower validation loss but higher MAPE compared to another, prioritize the model that "
        "strikes the right balance for the task (e.g., achieving better trend predictions for time-series forecasting). "
        "Provide clear reasoning based on the project's objectives and data nuances.\n\n"
        "### Deliverables\n"
        "Provide the following information:\n"
        "- **Selected Model Name:** The name of the chosen model.\n"
        "- **Detailed Rationale:** An explanation for your choice, referencing metrics, criteria, and task requirements.\n"
        "- **Recommended Adjustments:** Suggestions for scaling the model in the form of a clear list. Include parameter details such as:\n"
        "  - List of Embedding dimensions\n"
        "  - List of Number of layers\n"
        "  - Dropout rates\n"
        "  - List of Learning rate\n"
        "  - Other relevant hyperparameters\n"
        "  Example: ['embedding_dims: [64, 128, 196]', 'num_layers: [2, 3, 4]', 'learning_rates':[0.01, 0.1]]\n"
        "- **Trade-offs and Mitigation Strategies:** Highlight trade-offs of the selection and how they might be addressed.\n\n"
        "Your analysis and recommendations will be instrumental in ensuring the project's success. Proceed thoughtfully and confidently."
    )
    return prompt


def main():
    args = parse_arguments()

    if os.path.exists(args.criteria):
        with open(args.criteria, "r") as f:
            criteria = json.load(f)
    else:
        criteria = json.loads(args.criteria)

    print("\n🔹 Generating statistical significance CSV...")
    csv_filename = generate_statistical_significance_csv(args.csv_pattern)

    if not csv_filename:
        print("❌ No valid data available for model selection.")
        return

    print("\n🔹 Reading statistical significance results...")
    models = read_statistical_significance_with_losses(csv_filename)

    print("\n🔹 Generating model selection prompt...")
    selection_prompt = model_selection_prompt(
        models, args.task_description, args.dataset_description, criteria
    )

    print("\n🔹 Querying LLM for model evaluation...")
    llm_response = query_llm(selection_prompt, args.model)
    print("\n🔹 **LLM Response:**\n", llm_response)

    print("\n🔹 Removing temporary CSV file...")
    os.remove(csv_filename)


if __name__ == "__main__":
    main()
