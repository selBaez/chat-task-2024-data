import csv
import json
import concurrent.futures
from transformers import pipeline
import torch

# Define languages and datasets
languages = ['en-de', 'en-fr', 'en-nl', 'en-pt']
datasets = ['test']

# Option to use dialogue history
use_dialogue_history = True

# Initialize the pipeline
pipe = pipeline("text-generation", model="Unbabel/TowerInstruct-7B-v0.2", torch_dtype=torch.bfloat16, device_map="auto")

# Function to create dialogue overview from the translated_triple in the JSON data
def create_dialogue_overview(dialogue):
    overview = []
    for entry in dialogue:
        triples = entry["triples"]
        for triple in triples:
            translated_triple = triple.get("translated_triple", "")
            if translated_triple:
                overview.append(translated_triple)
    return ", ".join(overview)

# Function to process a single row for translation
def process_translation(index, row, dialogue_data):
    source_language = row['source_language']
    target_language = row['target_language']
    source_text = row['source']
    doc_id = row['doc_id']
    client_id = row['client_id']
    sender = row['sender']

    # Find the corresponding dialogue in the JSON data
    if use_dialogue_history:
        dialogue = next((d['dialogue'] for d in dialogue_data if d['Conversation ID'] == doc_id), None)
        if dialogue:
            dialogue_overview = create_dialogue_overview(dialogue)
        else:
            dialogue_overview = ""
        dialogue_content = f"Dialogue Overview: {dialogue_overview}"
    else:
        dialogue_content = ""

    # Create the messages list
    messages = []
    if use_dialogue_history and dialogue_content:
        messages.append({"role": "user", "content": dialogue_content})
    messages.append({"role": "user", "content": f"Translate the following text from {source_language} into {target_language}.\n{source_language.capitalize()}: {source_text}\n{target_language.capitalize()}:"})

    # Format the messages using the tokenizer's chat template
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate the output
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False)

    # Extract the generated translation
    translated_text = outputs[0]["generated_text"].split(f"{target_language.capitalize()}:")[-1].strip()

    return index, translated_text

# Process each language and dataset
for lang in languages:
    print(f"Translating language pair: {lang}")
    for dataset in datasets:
        # Construct file paths
        csv_file_path = f'/home/lkrause/data/llm-storage/selea/chat-task-2024-data/{dataset}/{lang}.csv'
        json_file_path = f'/home/lkrause/data/llm-storage/selea/chat-task-2024-data/clteam/towerblocks/test/{lang}.json'
        output_path = f'/home/lkrause/data/llm-storage/selea/chat-task-2024-data/clteam/towerblocks/predictions/{lang}.txt'

        # Load the dialogue history JSON file
        with open(json_file_path, 'r') as f:
            dialogue_data = json.load(f)

        # Load the CSV file
        with open(csv_file_path, 'r') as f:
            reader = list(csv.DictReader(f))

        translations = [None] * len(reader)  # Initialize a list to store translations in the correct order

        # Use ThreadPoolExecutor to process translations concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_index = {executor.submit(process_translation, index, row, dialogue_data): index for index, row in enumerate(reader)}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    index, translated_text = future.result()
                    translations[index] = translated_text
                except Exception as e:
                    print(f"Error processing row {index}: {e}")

        # Save translations to a plaintext output file
        with open(output_path, 'w') as f:
            for translation in translations:
                f.write(translation + '\n')

        print(f"Translations for {lang} saved to {output_path}")
