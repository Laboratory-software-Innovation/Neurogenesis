import argparse
import json
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


def selection_prompt(ds_list, dataset_description, dataset_features):
    """
    Generates a detailed and comprehensive prompt for selecting the most suitable dataset.
    """
    prompt = (
        f"You are a highly skilled data scientist tasked with selecting the most appropriate dataset "
        f"for a specific project. Your task requires careful consideration of the dataset's properties, "
        f"its compatibility with the intended application, and its ability to meet predefined criteria.\n\n"
        f"**Objective of the Dataset Selection Task:**\n{dataset_description}.\n\n"
        "### Available Dataset Options\n"
    )

    for idx, ds in enumerate(ds_list, 1):
        prompt += f"**Dataset {idx}: {ds['name']}**\n- **Description:** {ds['description']}\n- **Key Features:**\n"
        for feature, value in ds["features"].items():
            prompt += f"  - {feature}: {value}\n"
        prompt += "\n"

    prompt += "### Criteria for Dataset Selection\n"
    for feature, value in dataset_features.items():
        prompt += f"- {feature}: {value}\n"

    prompt += (
        "\n### Instructions for Selection\n"
        "1. **Review the Dataset Options:** Examine each dataset’s description and features.\n"
        "2. **Compare with Desired Features:** Match the dataset’s attributes against the listed criteria.\n"
        "3. **Evaluate Additional Factors:** Consider dataset size, diversity, and metadata.\n"
        "4. **Make a Justified Decision:** Select the dataset that best aligns with the project objectives.\n\n"
        "### Summary\nAnalyze datasets carefully and justify your selection using clear reasoning."
    )

    return prompt


def load_datasets(file_path):
    """
    Loads dataset information from an Excel file and structures it.
    """
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    df.fillna(method="ffill", inplace=True)
    all_datasets = {}

    for _, row in df.iterrows():
        category_key = (
            f"{row['Modality'].strip()}_{row['Problem'].strip()}".lower().replace(
                " ", "_"
            )
        )

        dataset_entry = {
            "name": row["Dataset"],
            "description": row["Description"],
            "features": {
                "Size": row["Download/Dataset Size"],
                "Annotations": row["Features"],
                "Samples": row["Number of Samples"],
            },
        }

        if category_key not in all_datasets:
            all_datasets[category_key] = []
        all_datasets[category_key].append(dataset_entry)

    return all_datasets


def query_llm(prompt, model="gpt-4o"):
    """
    Queries the selected LLM (OpenAI or DeepSeek) for dataset selection.
    """
    if "gpt" in model.lower():
        api_key = OPENAI_API_KEY
        base_url = None
    elif "deepseek" in model.lower():
        api_key = DEEPSEEK_API_KEY
        base_url = "https://openrouter.ai/api/v1"
    else:
        return "❌ Unsupported model. Use 'gpt-4o' or 'deepseek/deepseek-chat'."

    if not api_key:
        return f"❌ Missing API key for {model}. Please check your .env file."

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        return (
            response.choices[0].message.content
            if response.choices
            else "No response received."
        )

    except Exception as e:
        return f"❌ Error querying {model}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Dataset Selection Script")
    parser.add_argument(
        "--dataset_file", type=str, required=True, help="Path to dataset Excel file"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        help="Type of dataset (e.g., image_classification)",
    )
    parser.add_argument(
        "--dataset_description", type=str, required=True, help="Purpose of the dataset"
    )
    parser.add_argument(
        "--dataset_features",
        type=str,
        default="{}",
        help="Desired dataset features in JSON format",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM to use (gpt-4o or deepseek/deepseek-chat)",
    )

    args = parser.parse_args()

    if args.dataset_features.endswith(".json") and os.path.exists(
        args.dataset_features
    ):
        with open(args.dataset_features, "r", encoding="utf-8") as f:
            dataset_features = json.load(f)
    else:
        dataset_features = json.loads(args.dataset_features)

    if not os.path.exists(args.dataset_file):
        print(f"❌ Error: File '{args.dataset_file}' not found.")
        return

    all_datasets = load_datasets(args.dataset_file)

    if args.dataset_type not in all_datasets:
        print(f"❌ Error: Dataset type '{args.dataset_type}' not found.")
        return

    ds_list = all_datasets[args.dataset_type]
    prompt = selection_prompt(ds_list, args.dataset_description, dataset_features)

    print("\n🔹 **Generated Selection Prompt:**\n")
    print(prompt)

    print("\n🔹 **Querying LLM...**\n")
    llm_response = query_llm(prompt, args.model)
    print("\n🔹 **LLM Response:**\n", llm_response)


if __name__ == "__main__":
    main()
