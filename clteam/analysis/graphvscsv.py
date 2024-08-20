import csv
import json

def count_csv_entries(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        csv_entries = sum(1 for row in reader)  # Count the number of rows
    return csv_entries

def count_json_entries(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        json_entries = sum(len(conversation['dialogue']) for conversation in data)  # Count all dialogue entries
    return json_entries

def compare_lengths(csv_file, json_file):
    csv_count = count_csv_entries(csv_file)
    json_count = count_json_entries(json_file)

    print(f"Number of entries in CSV: {csv_count}")
    print(f"Number of dialogue entries in JSON: {json_count}")

    if csv_count == json_count:
        print("The number of entries in both files is the same.")
    else:
        print("The number of entries in both files is different.")

json_file = '/home/lkrause/data/baran_storage_hpc/chat-task-2024-data/clteam/graphs/test/en-de_new.json'
csv_file = '/home/lkrause/data/baran_storage_hpc/chat-task-2024-data/test/en-de.csv'

compare_lengths(csv_file, json_file)