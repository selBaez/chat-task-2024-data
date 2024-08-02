import csv
import json
import concurrent.futures
from transformers import pipeline
import torch
import time
import argparse
from format_triples import load_json_data, process_conversations, save_processed_data

def create_argument_parser():
    """Create argument parser for command-line options."""
    parser = argparse.ArgumentParser(description="Translation with dialogue history options.")
    parser.add_argument("--use_dialogue_history", type=bool, default=True, help="Use dialogue history in translation.")
    parser.add_argument("--dialogue_history_source", type=str, choices=['json', 'csv'], default='json',
                        help="Source of dialogue history: 'json' or 'csv'.")
    return parser

def create_dialogue_overview(dialogue, up_to_text=None, from_json=True):
    """Create dialogue overview from JSON or CSV data."""
    overview = []
    for entry in dialogue:
        if from_json:
            triples = entry.get("triples", [])
            for triple in triples:
                translated_triple = triple.get("translated_triple", "")
                if translated_triple:
                    overview.append(translated_triple)
            if entry.get("text") == up_to_text:
                break
        else:
            overview.append(entry['source'])
    return ", ".join(overview)

def process_translation(index, row, dialogue_data, rows, use_history, history_source):
    """Process a single row for translation."""
    print(f"Translating instance {index + 1}")
    source_lang = row['source_language']
    target_lang = row['target_language']
    source_text = row['source']

    # Prepare dialogue content if history is used
    if use_history:
        dialogue_content = ""
        if history_source == 'json':
            dialogue = next((d['dialogue'] for d in dialogue_data if d['Conversation ID'] == row['doc_id']), None)
            if dialogue:
                dialogue_content = f"Dialogue Overview: {create_dialogue_overview(dialogue, up_to_text=source_text)}"
        elif history_source == 'csv':
            dialogue_content = f"Dialogue Overview: {create_dialogue_overview(rows[:index+1], from_json=False)}"
    else:
        dialogue_content = ""

    # Create the prompt
    messages = [{"role": "user", "content": dialogue_content}] if dialogue_content else []
    messages.append({"role": "user", "content": f"Translate the following text from {source_lang} into {target_lang}.\n{source_lang.capitalize()}: {source_text}\n{target_lang.capitalize()}:"})

    # Format the messages using the tokenizer's chat template
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate the output
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False)

    # Extract the generated translation
    translated_text = outputs[0]["generated_text"].split(f"{target_lang.capitalize()}:")[-1].strip()

    return index, translated_text

def process_language_dataset(lang, dataset, use_history, history_source):
    """Process all translations for a given language and dataset."""
    csv_file_path = f'/home/lkrause/data/llm-storage/selea/chat-task-2024-data/{dataset}/{lang}.csv'
    json_file_path = f'/home/lkrause/data/llm-storage/selea/chat-task-2024-data/clteam/towerblocks/test/{lang}.json'
    output_path = f'/home/lkrause/data/llm-storage/selea/chat-task-2024-data/clteam/towerblocks/predictions/{lang}_csv.txt'

    # Load and process the dialogue history JSON file if using JSON
    dialogue_data = None
    if use_history and history_source == 'json':
        dialogue_data = load_json_data(json_file_path)
        dialogue_data = process_conversations(dialogue_data)
        save_processed_data(dialogue_data, json_file_path)

    # Load the CSV file
    with open(csv_file_path, 'r') as f:
        reader = list(csv.DictReader(f))

    translations = [None] * len(reader)

    # Use ThreadPoolExecutor to process translations concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(process_translation, idx, row, dialogue_data, reader, use_history, history_source): idx
            for idx, row in enumerate(reader)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                _, translated_text = future.result()
                translations[idx] = translated_text
            except Exception as e:
                print(f"Error processing row {idx}: {e}")

    # Save translations to a plaintext output file
    with open(output_path, 'w') as f:
        for translation in translations:
            f.write(translation + '\n')

    print(f"Translations for {lang} saved to {output_path}")

def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # Initialize the pipeline
    global pipe
    pipe = pipeline("text-generation", model="Unbabel/TowerInstruct-7B-v0.2", torch_dtype=torch.bfloat16, device_map="auto")

    # Measure the execution time
    start_time = time.time()

    # Define languages and datasets
    languages = ['en-de', 'en-fr', 'en-nl', 'en-pt']
    datasets = ['test']

    # Process each language and dataset
    for lang in languages:
        print(f"Translating language pair: {lang}")
        for dataset in datasets:
            process_language_dataset(lang, dataset, args.use_dialogue_history, args.dialogue_history_source)

    # Measure and print the execution time
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
