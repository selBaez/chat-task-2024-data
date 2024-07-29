'''
Adapted from https://github.com/lupantech/ScienceQA and https://github.com/amazon-science/mm-cot
'''

from dataclasses import dataclass
from typing import List, Optional, Tuple

import nltk
import pandas as pd
from fuzzywuzzy import fuzz


def texts_approximately_equal(text1: str, text2: str, threshold: int = 90, look_ahead: str = "") -> Tuple[bool, bool]:
    # Single utterance, fuzzy matching
    if fuzz.ratio(text1, text2) >= threshold:
        return True, False

    # Try two utterances
    elif fuzz.ratio(text1 + look_ahead, text2) >= threshold:
        return True, True

    # elif len(text1) > 10:
    #     if text2.startswith(text1):
    #         return True, False

    return False, False


def clean_text_for_comparison(og_utt):
    text_to_compare = og_utt["source"] if og_utt["source_language"] == "en" else og_utt["reference"]
    if text_to_compare.startswith("NAME-M_TEXT:"):
        text_to_compare = text_to_compare[12:]

    return text_to_compare

def match_utterances_2(raw_dialogues, tripled_dialogues, input_txt, input_matrix):
    flat_triple_dialogues = []
    for td in tripled_dialogues:
        flat_triple_dialogues.extend(td["dialogue"])

    fixed_og = []
    rolling_tripled_idx = 0
    for dialogue in raw_dialogues:
        for utt in dialogue:
            text_to_compare = clean_text_for_comparison(utt)

            if texts_approximately_equal(text_to_compare, flat_triple_dialogues[rolling_tripled_idx]):
                pass





def match_utterances(raw_dialogues, tripled_dialogues, input_txt, input_matrix):
    super_idx = 0
    fixed_og = []
    for og, post_prompt in zip(raw_dialogues, tripled_dialogues):
        # This dialogue is the right length, leave it alone
        if len(og) == len(post_prompt["dialogue"]):
            for og_utt in og:
                og_utt["matched"] = True
                og_utt["input_txt"] = input_txt[super_idx]
                og_utt["input_matrix"] = input_matrix[super_idx]
                super_idx += 1

        # These lengths do not match, try to match
        else:
            skipped = 0
            double_match = False
            for i, og_utt in enumerate(og):
                if double_match:
                    this_match = True
                else:
                    # Prepare texts to compare
                    current_prompted_utt = post_prompt["dialogue"][i - skipped]
                    text_to_compare = clean_text_for_comparison(og_utt)
                    next_text = clean_text_for_comparison(og[i + 1]) if ((i + 1) < len(og)) else ""
                    # Compare
                    this_match, double_match = texts_approximately_equal(text_to_compare, current_prompted_utt["text"],
                                                                         look_ahead=next_text)

                # Assign accordingly
                if this_match:
                    # This is a match
                    og_utt["matched"] = True
                    og_utt["input_txt"] = input_txt[super_idx]
                    og_utt["input_matrix"] = input_matrix[super_idx]
                    super_idx += 1
                else:
                    # Move index by 1
                    skipped += 1
                    og_utt["matched"] = False
                    # if len(text_to_compare) > 5:
                    #     print(f"Utterances not matched: \n\t{text_to_compare}\n\t{current_prompted_utt['text']}")

                og_utt["post_prompt"] = current_prompted_utt["text"]

            if skipped >= 2:
                print(f"Big oopsie: skipped {skipped}")

        temp = pd.DataFrame.from_dict(og)
        fixed_og.append(og)

    return fixed_og


def get_en_history(raw_dialogues, tripled_dialogues):
    for og, post_prompt in zip(raw_dialogues, tripled_dialogues):
        skipped = 0

        for i, og_utt in enumerate(og):
            en_history = []
            current_prompted_utt = post_prompt["dialogue"][i + skipped]

            if og_utt["source"].split(":")[-1] == current_prompted_utt["text"]:
                # This is a match
                for trip in current_prompted_utt["triples"]:
                    if "translated_triple" in trip.keys():
                        en_history.append(trip["translated_triple"].replace("_", " "))
                    else:
                        en_history.append(trip["source"])
                og_utt["reference"] = " ".join(en_history)


            else:
                # Move index by 1
                skipped += 1
                og_utt["reference"] = current_prompted_utt["text"]

    return raw_dialogues


def build_train_pair(og, post_prompt, exclude_context=False):
    all_inputs, all_targets = [], []
    for i in range(len(og)):
        current_dialogue = og[:i + 1]
        if current_dialogue[-1]["matched"]:
            # Format text to translate
            to_translate = current_dialogue[-1]["source"]
            if "reference" in current_dialogue[-1].keys():
                target_translation = current_dialogue[-1]["reference"]
            else:
                target_translation = ""

            # create the prompt input
            if exclude_context:
                prompt_input = f"Source segment:\n{to_translate}\n\n" \
                               f"Translation:\n"
            else:
                # Build dialogue history
                dialogue_history = []
                for utt in current_dialogue[:-1]:
                    if utt["source_language"] == "en":
                        dialogue_history.append(f'{utt["sender"]}: {utt["source"]}')
                    else:
                        if "reference" in utt.keys():
                            dialogue_history.append(f'{utt["sender"]}: {utt["reference"]}')

                dialogue_history = "\n".join(dialogue_history)

                prompt_input = f"Dialogue History:\n{dialogue_history}\n\n" \
                               f"Source segment:\n{to_translate}\n\n" \
                               f"Translation:\n"

            all_inputs.append(prompt_input)
            all_targets.append(target_translation)

    return all_inputs, all_targets


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
