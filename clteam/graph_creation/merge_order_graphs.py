import os
import pandas as pd
import json
from typing import List, Dict
from collections import defaultdict
import argparse

# Load order from CSV
def extract_order(file_path: str) -> List[str]:
    """Extracts the unique document order from a CSV file."""
    data = pd.read_csv(file_path)
    order = data['doc_id'].unique().tolist()
    return order

def load_json(file_path: str) -> List[Dict]:
    """Loads JSON data from a file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data: List[Dict], file_path: str):
    """Saves data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# Function to merge conversations
def merge_conversations(data: List[Dict]) -> List[Dict]:
    """Merges conversation dialogues by their ID."""
    merged_conversations = defaultdict(list)
    
    # Group dialogues by conversation ID
    for entry in data:
        if "Conversation ID" not in entry:
            print(f"Skipping entry with missing 'Conversation ID': {entry}")
            continue
        conversation_id = entry["Conversation ID"]
        merged_conversations[conversation_id].append(entry)
    
    # Merge dialogues, ensuring the longest parts are first
    merged_data = []
    for conversation_id, entries in merged_conversations.items():
        # Sort entries by the length of their dialogue in descending order
        entries = sorted(entries, key=lambda x: len(x['dialogue']), reverse=True)
        merged_dialogue = []
        for entry in entries:
            merged_dialogue.extend(entry['dialogue'])
        merged_data.append({"Conversation ID": conversation_id, "dialogue": merged_dialogue})
    
    return merged_data

# Function to reorder data
def reorder_data(processed_data: List[Dict], order: List[str]) -> List[Dict]:
    """Reorders data according to the extracted order."""
    ordered_data = []
    data_dict = {entry["Conversation ID"]: entry for entry in processed_data}
    for doc_id in order:
        if doc_id in data_dict:
            ordered_data.append(data_dict[doc_id])
    return ordered_data

# Main processing function
def process_files(languages: List[str], datasets: List[str], base_path: str, merge: bool, reorder: bool):
    """Processes and optionally merges or reorders files."""
    for lang in languages:
        for dataset in datasets:
            json_file_path = os.path.join(base_path, "clteam/graphs", dataset, f"{lang}.json")
            csv_file_path = os.path.join(base_path, dataset, f"{lang}.csv")
            final_output_path = os.path.join(base_path, "clteam/graphs", dataset, f"{lang}_final.json")
            
            if os.path.exists(json_file_path) and (not reorder or os.path.exists(csv_file_path)):
                # Load JSON data
                data = load_json(json_file_path)

                # Optionally merge conversations
                if merge:
                    data = merge_conversations(data)
                    print(f"Merged data for {lang} - {dataset}")

                # Optionally reorder merged data
                if reorder:
                    order = extract_order(csv_file_path)
                    data = reorder_data(data, order)
                    print(f"Reordered data for {lang} - {dataset}")

                # Save the processed data to a JSON file
                save_json(data, final_output_path)
                
                print(f"Processed {json_file_path} and saved to {final_output_path}")
            else:
                print(f"Skipping {lang} - {dataset} as required files do not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge and reorder conversation data files.')
    parser.add_argument('--merge', action='store_true', help='Perform merging of conversation data.')
    parser.add_argument('--reorder', action='store_true', help='Perform reordering of conversation data.')
    parser.add_argument('--base_path', type=str, default='../../', help='Base path for data files.')

    args = parser.parse_args()

    languages = ['en-fr', 'en-de', 'en-nl', 'en-pt']
    datasets = ['train', 'valid']
    
    # Perform both actions by default if no specific action is given
    perform_merge = args.merge or not (args.merge or args.reorder)
    perform_reorder = args.reorder or not (args.merge or args.reorder)

    process_files(languages, datasets, args.base_path, perform_merge, perform_reorder)
