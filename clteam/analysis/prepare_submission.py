import json
import argparse
import os

def process_alpaca(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    lines = data.get('preds', [])
    with open(output_path, 'w', encoding='utf-8') as file:
        for line in lines:
            if line.strip():
                file.write(line + '\n')

def process_towerblocks(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_path, 'w', encoding='utf-8') as file:
        for line in lines:
            if line.startswith("Translation in English:"):
                line = line.replace("Translation in English:", "").strip()
            elif line.startswith("Translation in en:"):
                line = line.replace("Translation in en:", "").strip()
            elif line.startswith("As per your request the translation in English:"):
                line = line.replace("As per your request the translation in English:", "").strip()
            elif line.startswith("Translation in English as listed:"):
                line = line.replace("Translation in English as listed:", "").strip()
            elif line.startswith("EN: "):
                line = line.replace("EN: ", "").strip()
            elif line.startswith("<|im_end|>"):
                line = line.replace("<|im_end|>", "").strip()
            elif line.startswith("<|im_start|>assistant"):
                line = line.replace("<|im_start|>assistant", "").strip()
            file.write(line + '\n')

def main():
    parser = argparse.ArgumentParser(description="Process and format files for submission.")
    # parser.add_argument('base_output_folder', default='/home/lkrause/data/llm-storage/selea/chat-task-2024-data/clteam/submission_clteam', help='The base output folder path')
    parser.add_argument('--type', choices=['alpaca', 'towerblocks'], default='towerblocks', help='The type of the input file: alpaca or towerblocks')
    parser.add_argument('--model', choices=['graph_flan-alpaca_w-history', 'graph_flan-alpaca_wo-history', 'towerblocks_w-history', 'towerblocks_wo-history'], default='towerblocks_wo-history', help='The model name used for naming output files')

    args = parser.parse_args()

    languages = ['en-de', 'en-fr', 'en-nl', 'en-pt']
    base_input_path = "/home/lkrause/data/llm-storage/selea/chat-task-2024-data/clteam/"
    output_path = '/home/lkrause/data/llm-storage/selea/chat-task-2024-data/clteam/submission_clteam/'

    for lang in languages:
        if args.type == 'alpaca':
            file_path = os.path.join(base_input_path, f"experiments/with_dialogue_history/test/{lang}_declare-lab-flan-alpaca-base_ep50/predictions_ans_eval.json")
        elif args.type == 'towerblocks':
            file_path = os.path.join(base_input_path, f"towerblocks/predictions/w_history/{lang}.txt")
        
        output_folder = os.path.join(output_path, lang)
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{args.model}.txt")

        if args.type == 'alpaca':
            process_alpaca(file_path, output_file)
        elif args.type == 'towerblocks':
            process_towerblocks(file_path, output_file)

if __name__ == "__main__":
    main()
