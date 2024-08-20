import json
import argparse
import os
import pandas as pd

def process_flan_t5(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data.get('preds', [])

def process_towerblocks(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    processed_lines = []
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
        elif line.startswith(""):
            line = line.replace("", "").strip()
        elif line.startswith("assistant"):
            line = line.replace("assistant", "").strip()
        processed_lines.append(line)
    
    return processed_lines

def update_csv_with_predictions(csv_path, lang, flan_t5_predictions=None, towerblocks_predictions=None):
    df = pd.read_csv(csv_path)
    
    if flan_t5_predictions is not None:
        df[f'submission_clteam_graphflant5'] = flan_t5_predictions
    if towerblocks_predictions is not None:
        df[f'submission_clteam_towerblocks_wohistory'] = towerblocks_predictions
    
    df.to_csv(csv_path, index=False)
    print(f"CSV file updated successfully for {lang}.")

def main():
    parser = argparse.ArgumentParser(description="Process and format files for submission.")
    parser.add_argument('--type', choices=['flan-t5', 'towerblocks'], required=True, help='The type of the input file: flan-t5 or towerblocks')
    parser.add_argument('--notebook_eval', action='store_true', help='Flag to update CSV file with predictions for notebook evaluation')
    parser.add_argument('--base_csv_path', default='/home/lkrause/data/baran_storage_hpc/chat-task-2024-data/clteam/submission_clteam/', help='Base path to the CSV files')

    args = parser.parse_args()

    languages = ['en-de', 'en-fr', 'en-nl', 'en-pt'] 

    for lang in languages:
        csv_path = os.path.join(args.base_csv_path, lang, f'{lang}.csv')

        if args.notebook_eval:
            if args.type == 'flan-t5':
                flan_t5_file = os.path.join('submission_clteam', lang, f'{lang}_predictions_ans_test.json')
                flan_t5_predictions = process_flan_t5(flan_t5_file)
                update_csv_with_predictions(csv_path, lang, flan_t5_predictions=flan_t5_predictions)
            elif args.type == 'towerblocks':
                towerblocks_file = os.path.join('submission_clteam', lang, 'towerblocks_wo-history.txt')
                towerblocks_predictions = process_towerblocks(towerblocks_file)
                update_csv_with_predictions(csv_path, lang, towerblocks_predictions=towerblocks_predictions)
        else:
            # If not notebook_eval, perform standard file processing and output
            output_path = os.path.join('submission_clteam', lang)

            if args.type == 'flan-t5':
                flan_t5_file = os.path.join('submission_clteam', lang, f'{lang}_predictions_ans_test.json')
                predictions = process_flan_t5(flan_t5_file)
                output_file = os.path.join(output_path, 'flan_t5_predictions.txt')
            elif args.type == 'towerblocks':
                towerblocks_file = os.path.join('submission_clteam', lang, 'towerblocks_wo-history.txt')
                predictions = process_towerblocks(towerblocks_file)
                output_file = os.path.join(output_path, 'towerblocks_predictions.txt')

            os.makedirs(output_path, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as output:
                for line in predictions:
                    if line.strip():
                        output.write(line + '\n')

if __name__ == "__main__":
    main()
