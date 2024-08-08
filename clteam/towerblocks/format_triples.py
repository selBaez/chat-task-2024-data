import json
from typing import List

def load_json_data(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_processed_data(data, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def process_conversations(data):
    for conversation in data:
        for message in conversation.get('dialogue', []):
            for triple in message.get('triples', []):
                # Ensure that subject, predicate, and object are not None before replacing spaces with underscores
                if triple['subject'] is not None:
                    triple['subject'] = triple['subject'].replace(' ', '_')
                if triple['predicate'] is not None:
                    triple['predicate'] = triple['predicate'].replace(' ', '_')
                if triple['object'] is not None:
                    triple['object'] = triple['object'].replace(' ', '_')
                
                # Construct the translated triple if it doesn't exist or is empty
                if not triple.get('translated_triple'):
                    triple['translated_triple'] = f"{triple['subject']} {triple['predicate']} {triple['object']}"
                
                # Replace the third and subsequent spaces with underscores in the translated triple
                parts = triple['translated_triple'].split(' ', 2)
                if len(parts) == 3:
                    parts[2] = parts[2].replace(' ', '_')
                triple['translated_triple'] = ' '.join(parts)
    return data

def process_dialogue_files(languages: List[str], datasets: List[str]):
    for lang in languages:
        for dataset in datasets:
            file_path = f'./{dataset}/{lang}.json'
            output_path = f'./{dataset}/{lang}.json'
            
            # Load, process and save the data
            data = load_json_data(file_path)
            processed_conversations = process_conversations(data)
            save_processed_data(processed_conversations, output_path)
            print(f"Processed and saved data for {dataset} - {lang}")

if __name__ == "__main__":
    languages = ['en-de', 'en-fr', 'en-nl', 'en-pt']
    datasets = ['test']
    process_dialogue_files(languages, datasets)
