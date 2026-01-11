from datasets import load_dataset, Dataset, Features, Value
from huggingface_hub import HfApi
import json

def load_jsonl_dataset(file_path):
    """Load dataset from JSONL file"""
    ds = load_dataset("json", data_files=file_path)
    return ds["train"]

def delete_repo(repo_id, token=None):
    """Delete a repository from HuggingFace Hub"""
    api = HfApi()
    try:
        api.delete_repo(repo_id=repo_id, token=token, repo_type="dataset")
        print(f"Deleted dataset repo: {repo_id}")
    except Exception as e:
        print(f"Could not delete repo: {e}")

def upload_to_hub(dataset, new_repo_id, token=None):
    # Create a new repo if it does not exist
    api = HfApi()
    try:
        api.create_repo(repo_id=new_repo_id, token=token, public=True, repo_type="dataset")
        print(f"Created new dataset repo: {new_repo_id}")
    except Exception as e:
        print(f"Failed to upload dataset to hub: {e}")

    # Push dataset to hub
    dataset.push_to_hub(repo_id=new_repo_id, token=token)
    print(f"Dataset uploaded to hub at: https://huggingface.co/datasets/{new_repo_id}")

def process_extract_extra(ds):
    ds = ds.filter(lambda x: x["score"] > 0)
    print(f"Extracted {len(ds)} examples")
    # Get all original column names to remove them
    original_columns = ds.column_names
    # Map to extract only x["extra"] and remove all original columns
    return ds.map(lambda x: x["extra"], remove_columns=original_columns)

def process_split_conversation(ds):
    new_data = []
    for example in ds:
        conversation = example['conversation']
        
        # Clean up conversation: remove reasoning_content field entirely
        cleaned_conversation = []
        for msg in conversation:
            cleaned_msg = dict(msg)
            if 'reasoning_content' in cleaned_msg:
                del cleaned_msg['reasoning_content']
            cleaned_conversation.append(cleaned_msg)
        
        system_prompt = cleaned_conversation[0]
        for i in range(1, len(cleaned_conversation)):
            if cleaned_conversation[i]['role'] == 'assistant':
                new_data.append({
                    **example,
                    "conversation": conversation,
                    "prompt": json.dumps([system_prompt, cleaned_conversation[i - 1]]),
                    "response": json.dumps(conversation[i]),
                    "answer": cleaned_conversation[i]['content'].split("\n")[-1]
                })

    # Let HuggingFace auto-infer the schema or define it properly
    processed_ds = Dataset.from_list(new_data)

    return processed_ds

def upload_single_dataset(data_file, new_repo_id, token):
    delete_repo(new_repo_id, token=token)

    ds = load_jsonl_dataset(data_file)
    ds = process_extract_extra(ds)
    # ds = process_split_conversation(ds)

    upload_to_hub(ds, new_repo_id, token=token)

if __name__ == "__main__":
    # data_files=["dataset/goofspiel_8.jsonl", "dataset/goofspiel_10.jsonl", "dataset/goofspiel_12.jsonl", "dataset/goofspiel_14.jsonl", "dataset/goofspiel_16.jsonl"]
    # repo_ids = ["top-50000/goof_8", "top-50000/goof_10", "top-50000/goof_12", "top-50000/goof_14", "top-50000/goof_16"]    
    data_files = ["affinetes/environments/openspiel/cheatdb_liars_dice.jsonl"]
    repo_ids = ["top-50000/fit"]
    token = "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"  # Place your HuggingFace User Access Token here if needed

    for data_file, repo_id in zip(data_files, repo_ids):
        upload_single_dataset(data_file, repo_id, token)
