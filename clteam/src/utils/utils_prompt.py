'''
Adapted from https://github.com/lupantech/ScienceQA and https://github.com/amazon-science/mm-cot
'''

from dataclasses import dataclass
from typing import List, Optional

import nltk


def build_train_pair(problems, exclude_context=False):
    # Format text to translate
    to_translate = problems[-1]["source"]
    if "reference" in problems[-1].keys():
        target_translation = problems[-1]["reference"]
    else:
        target_translation = ""

    # create the prompt input
    if exclude_context:
        prompt_input = f"Source segment:\n{to_translate}\n\n" \
                       f"Translation:\n"
    else:
        # Build dialogue history
        dialogue_history = []
        for utt in problems[:-1]:
            if utt["source_language"] == "en":
                dialogue_history.append(f'{utt["sender"]}: {utt["source"]}')
            else:
                if "reference" in utt.keys():
                    dialogue_history.append(f'{utt["sender"]}: {utt["reference"]}')

        dialogue_history = "\n".join(dialogue_history)

        prompt_input = f"Dialogue History:\n{dialogue_history}\n\n" \
                       f"Source segment:\n{to_translate}\n\n" \
                       f"Translation:\n"


    return prompt_input, target_translation


def postprocess_text(preds, labels):
    processed_preds = []
    for pred in preds:
        pred = pred.strip()
        try:
            # use nltk to split the text into sentences
            processed_pred = "\n".join(nltk.sent_tokenize(pred))
        except IndexError:
            # if the text is too long, it may cause an IndexError
            print(f"IndexError occurred with text: {pred}")
            processed_pred = pred
        processed_preds.append(processed_pred)

    processed_labels = []
    for label in labels:
        label = label.strip()
        try:
            # use nltk to split the text into sentences
            processed_label = "\n".join(nltk.sent_tokenize(label))
        except IndexError:
            print(f"IndexError occurred with text: {label}")
            processed_label = label
        processed_labels.append(processed_label)

    return processed_preds, processed_labels


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    le_input_ids: List[List[int]]
    le_attention_mask: Optional[List[List[int]]]
    le_token_type_ids: Optional[List[List[int]]]
    label: Optional[int]
