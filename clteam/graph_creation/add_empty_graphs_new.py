import os
import pandas as pd
import json
from typing import List, Dict, Tuple
from fuzzywuzzy import fuzz
import warnings

# Suppress the warning message
warnings.filterwarnings("ignore", category=UserWarning, message="Using slow pure-python SequenceMatcher")

# Extract English texts from the CSV
def extract_order_and_english_messages(file_path: str) -> List[Tuple[str, List[str]]]:
    data = pd.read_csv(file_path)
    grouped = data.groupby('doc_id')
    order_and_messages = []
    for doc_id, group in grouped:
        english_texts = []
        for _, row in group.iterrows():
            if row['target_language'] == 'en':
                english_texts.append(row['reference'])
            elif row['source_language'] == 'en':
                english_texts.append(row['source'])
        order_and_messages.append((doc_id, english_texts))
    return order_and_messages

# Function to load JSON data
def load_json(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Function to save JSON data
def save_json(data: List[Dict], file_path: str):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# Function to check if two texts are approximately equal
def texts_approximately_equal(text1: str, text2: str, threshold: int = 5) -> bool:
    # Ensure both texts are strings
    if not isinstance(text1, str) or not isinstance(text2, str):
        return False
    return fuzz.ratio(text1, text2) >= threshold

# Function to add empty JSONs for missing messages in correct positions
def add_empty_jsons(processed_data: Dict[str, Dict], order_and_messages: List[Tuple[str, List[str]]]) -> List[Dict]:
    total_csv_messages = 0
    total_json_messages_before = 0
    total_json_messages_after = 0

    for doc_id, messages in order_and_messages:
        total_csv_messages += len(messages)
        if doc_id in processed_data:
            dialogue = processed_data[doc_id]["dialogue"]
            total_json_messages_before += len(dialogue)

            new_dialogue = []
            csv_idx = 0
            json_idx = 0

            while csv_idx < len(messages) and json_idx < len(dialogue):
                csv_msg = messages[csv_idx]
                json_msg = dialogue[json_idx]["text"]

                if texts_approximately_equal(csv_msg, json_msg):
                    new_dialogue.append(dialogue[json_idx])
                    json_idx += 1
                else:
                    new_dialogue.append({
                        "sender": "",
                        "text": csv_msg,
                        "triples": []
                    })
                csv_idx += 1

            # Add any remaining CSV messages as empty JSON entries
            while csv_idx < len(messages):
                new_dialogue.append({
                    "sender": "",
                    "text": messages[csv_idx],
                    "triples": []
                })
                csv_idx += 1

            # Add any remaining JSON messages
            while json_idx < len(dialogue):
                new_dialogue.append(dialogue[json_idx])
                json_idx += 1

            processed_data[doc_id]["dialogue"] = new_dialogue
            total_json_messages_after += len(new_dialogue)
        else:
            # Add an entry with all messages as empty JSON structures
            empty_dialogue = [{
                "sender": "",
                "text": msg,
                "triples": []
            } for msg in messages]
            processed_data[doc_id] = {
                "Conversation ID": doc_id,
                "dialogue": empty_dialogue
            }
            total_json_messages_after += len(empty_dialogue)

    # Print lengths for verification
    print(f"Total messages in CSV: {total_csv_messages}")
    print(f"Total messages in JSON before processing: {total_json_messages_before}")
    print(f"Total messages in JSON after processing: {total_json_messages_after}")

    return list(processed_data.values())

# Main function to add empty JSONs and reorder data
def add_empty_and_reorder_json(languages: List[str], datasets: List[str], base_path: str):
    for lang in languages:
        for dataset in datasets:
            csv_file_path = os.path.join(base_path, dataset, f"{lang}.csv")
            json_file_path = os.path.join(base_path, "clteam/graphs", dataset, f"{lang}.json")
            updated_json_file_path = os.path.join(base_path, "clteam/graphs", dataset, f"{lang}_updated.json")

            if os.path.exists(csv_file_path) and os.path.exists(json_file_path):
                # Extract order and messages from CSV
                order_and_messages = extract_order_and_english_messages(csv_file_path)

                # Load processed JSON data
                processed_data = load_json(json_file_path)

                # Convert list to dictionary for easy lookup
                processed_data_dict = {entry["Conversation ID"]: entry for entry in processed_data}

                # Add empty JSONs for missing messages and reorder the processed data
                updated_data = add_empty_jsons(processed_data_dict, order_and_messages)

                # Save the updated processed data to a JSON file
                save_json(updated_data, updated_json_file_path)

                print(f"Updated {json_file_path} with empty JSONs for missing messages and saved to {updated_json_file_path}")
            else:
                print(f"Skipping {lang} - {dataset} as either CSV or JSON file does not exist.")

if __name__ == "__main__":
    languages = ['en-de', 'en-fr', 'en-nl', 'en-pt'] 
    datasets = ['train', 'valid'] # , 'test'
    base_path = '/home/lkrause/data/llm-storage/selea/chat-task-2024-data'

    add_empty_and_reorder_json(languages, datasets, base_path)
