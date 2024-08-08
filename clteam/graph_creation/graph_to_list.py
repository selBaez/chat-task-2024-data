import json
import os

def format_triple(triple):
    if 'translated_triple' in triple:
        translated_triple = triple['translated_triple']
        parts = translated_triple.split(' ', 2)
        if len(parts) == 3:
            formatted_triple = f"{parts[0]}</s><s>{parts[1]}</s><s>{parts[2].replace(' ', '_')}"
        else:
            formatted_triple = '</s><s>'.join(parts)
        return formatted_triple
    
    subject = triple.get('subject', '')
    predicate = triple.get('predicate', '')
    obj = triple.get('object', '')
    return f"{subject}</s><s>{predicate}</s><s>{obj}"

def format_dialogue_triples(dialogue):
    formatted_triples = []
    for entry in dialogue:
        for triple in entry['triples']:
            formatted_triples.append(format_triple(triple))
    # Join the formatted triples into one continuous list
    formatted_dialogue = "</s><s>".join(formatted_triples)
    return [formatted_dialogue]

def process_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    formatted_dialogues = []
    for conversation in data:
        dialogue = conversation['dialogue']
        formatted_triples = format_dialogue_triples(dialogue)
        formatted_dialogues.append(formatted_triples)
    
    return formatted_dialogues

def write_to_file(output_path, formatted_dialogues):
    with open(output_path, 'w') as file:
        json.dump(formatted_dialogues, file, indent=4)

# Define languages and datasets
languages = ['en-de', 'en-fr', 'en-nl', 'en-pt']
datasets = ['train', 'valid'] # TODO add test

# Process each file and write the output
for lang in languages:
    for dataset in datasets:
        file_path = f'../graphs/{dataset}/{lang}.json'
        output_path = f'../graphs/{dataset}/{lang}.txt'
        
        if os.path.exists(file_path):
            print(f"Processing file: {file_path}")
            formatted_dialogues = process_json_file(file_path)
            if formatted_dialogues:
                print(f"Writing to file: {output_path}")
                write_to_file(output_path, formatted_dialogues)
                print(f"Successfully wrote to {output_path}")
            else:
                print(f"No data to write for {file_path}")
        else:
            print(f"File not found: {file_path}")
