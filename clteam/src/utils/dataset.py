import os
import pickle

import torch
from torch.utils.data import Dataset
from utils.utils_data import load_raw_data
from utils.utils_prompt import build_train_pair


class ChatDatasetWithGraph(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(self, split, tokenizer, args, dry_run=False):
        self.tokenizer = tokenizer
        self.data = load_raw_data(args, split, dry_run=dry_run)  # {qid : problems[qid] for qid in qids}
        self.source_len = args.input_len
        self.summ_len = args.output_len
        self.target_text = []
        self.source_text = []

        with open(os.path.join(args.data_root, split, args.language, 'mc_input_text.pkl'), 'rb') as f:
            self.got_input_text_list = pickle.load(f)
        with open(os.path.join(args.data_root, split, args.language, 'mc_adj_matrix.pkl'), 'rb') as f:
            self.got_adj_matrix_list = pickle.load(f)

        for qid, prob in enumerate(self.data):
            prompt, target = build_train_pair(prob, args.exclude_context)
            self.target_text.append(target)
            self.source_text.append(prompt)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        got_input_text = self.got_input_text_list[index]
        got_adj_matrix = self.got_adj_matrix_list[index]
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
