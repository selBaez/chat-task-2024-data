import os
import pickle

import torch
from torch.utils.data import Dataset

from utils.utils_data import load_raw_data, load_triple_data
from utils.utils_prompt import build_train_pair, get_en_history, match_utterances


class ChatDatasetWithGraph(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(self, split, tokenizer, args, dry_run=False):
        self.tokenizer = tokenizer
        self.raw_data = load_raw_data(args, split, dry_run=dry_run)
        self.triple_data = load_triple_data(args, split, dry_run=dry_run)
        self.source_len = args.input_len
        self.summ_len = args.output_len
        self.target_text = []
        self.source_text = []
        self.input_sep = []
        self.input_mat = []

        with open(os.path.join(args.data_root, split, args.language, 'mc_input_text.pkl'), 'rb') as f:
            self.got_input_text_list = pickle.load(f)
        with open(os.path.join(args.data_root, split, args.language, 'mc_adj_matrix.pkl'), 'rb') as f:
            self.got_adj_matrix_list = pickle.load(f)

        self.data = match_utterances(self.raw_data, self.triple_data,
                                     self.got_input_text_list, self.got_adj_matrix_list)

        if split == "test":
            self.raw_data = get_en_history(self.raw_data, self.triple_data)

        for og, post_prompt in zip(self.data, self.triple_data):
            prompt, target, separator, matrix = build_train_pair(og, post_prompt, args.exclude_context)
            self.target_text.extend(target)
            self.source_text.extend(prompt)
            self.input_sep.extend(separator)
            self.input_mat.extend(matrix)

        print(f"Dataset ({split}) loaded")
        print(f"\tDialogues in raw data: {len(self.raw_data)}, and validation data:{len(self.data)}")
        print(f"\tUtterances in original graphs: {len(self.got_input_text_list)}, "
              f"matched text: {len(self.target_text)}, and matched graphs: {len(self.input_sep)}")
        print("\n")

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        got_input_text = self.input_sep[index]
        got_adj_matrix = self.input_mat[index]
        got_adj_matrix = torch.tensor(got_adj_matrix)

        source = self.tokenizer.batch_encode_plus([source_text],
                                                  max_length=self.source_len,
                                                  pad_to_max_length=True,
                                                  truncation=True,
                                                  padding="max_length",
                                                  return_tensors="pt",
                                                  )
        target = self.tokenizer.batch_encode_plus([target_text],
                                                  max_length=self.summ_len,
                                                  pad_to_max_length=True,
                                                  truncation=True,
                                                  padding="max_length",
                                                  return_tensors="pt",
                                                  )

        encoded_got_input_text = self.tokenizer.batch_encode_plus(got_input_text,
                                                                  max_length=self.source_len,
                                                                  pad_to_max_length=True,
                                                                  truncation=True,
                                                                  padding="max_length",
                                                                  return_tensors="pt",
                                                                  )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        encoded_got_input_text_ids = encoded_got_input_text["input_ids"].squeeze()
        encoded_got_input_text_mask = encoded_got_input_text["attention_mask"].squeeze()

        return {"input_ids": source_ids,
                "attention_mask": source_mask,
                "labels": target_ids,
                "got_adj_matrix": got_adj_matrix,
                "got_input_ids": encoded_got_input_text_ids,
                "got_mask": encoded_got_input_text_mask,
                }
