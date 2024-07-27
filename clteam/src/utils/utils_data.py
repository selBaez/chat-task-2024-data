import json
from datetime import datetime
from itertools import groupby
from pathlib import Path

import pandas as pd


def make_save_directory(args):
    model_name = args.model.replace("/", "-")
    save_dir = f"{args.output_dir}/{model_name}_ep{args.epoch}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    mk_dir(save_dir)
    print(save_dir)

    return save_dir


def mk_dir(dir):
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)


def load_all_data(args, split):
    data_per_language = []

    for language in args.languages:
        datapath = Path(args.data_root) / f"{split}" / f"{language}.json"
        with open(datapath, 'r', encoding='utf-8') as file:
            dialogues = json.load(file)
        data_per_language.append(dialogues)

    return data_per_language


def load_language_data(args, split, language):
    datapath = Path(args.raw_root) / f"{split}" / f"{language}.csv"  # / "mc_coref_clusters.json"
    dialogues = pd.read_csv(datapath).to_dict('records')
    dialogues = [list(v) for k, v in groupby(dialogues, key=lambda x: x['doc_id'])]

    return dialogues


def load_raw_data(args, split, dry_run=False):
    print(f"""[Model]: Loading {args.model}...\n""")
    print(f"[Data]: Reading data...\n")

    if dry_run:
        problems = load_language_data(args, 'mini-valid', args.language)
    else:
        problems = load_language_data(args, split, args.language)

    print(f"number of {split} problems: {len(problems)}\n")

    return problems
