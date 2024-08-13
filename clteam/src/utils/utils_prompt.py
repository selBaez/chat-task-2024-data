from typing import Tuple

import nltk
import pandas as pd
from fuzzywuzzy import fuzz


def texts_approximately_equal(text1: str, text2: str, threshold: int = 95, look_ahead: str = "") -> Tuple[bool, bool]:
    # Single utterance, fuzzy matching
    if fuzz.ratio(text1, text2) >= threshold:
        return True, False

    # Try two utterances
    elif fuzz.ratio(text1 + look_ahead, text2) >= threshold:
        return True, True

    return False, False


def clean_text_for_comparison(og_utt, potential_triples):
    # Check which is the text we need to clean
    if og_utt["source_language"] == "en":
        text_to_compare = og_utt["source"]
    else:
        if "reference" in og_utt.keys():
            text_to_compare = og_utt["reference"]
        else:
            text_to_compare = [trip["translated_triple"].replace("_", " ") for trip in potential_triples["triples"]]
            text_to_compare = " ".join(text_to_compare)

    # Clean
    if text_to_compare.startswith("NAME-M_TEXT:"):
        text_to_compare = text_to_compare[12:]

    return text_to_compare


def match_utterances(raw_dialogues, tripled_dialogues, input_txt, input_matrix):
    super_idx = 0
    fixed_og = []
    for og, post_prompt in zip(raw_dialogues, tripled_dialogues):
        # Initialize 'matched' key for all utterances
        for og_utt in og:
            og_utt["matched"] = False  # Default to False

        # This dialogue is the right length, leave it alone
        if len(og) == len(post_prompt["dialogue"]):
            for i, og_utt in enumerate(og):
                og_utt["matched"] = True
                og_utt["input_txt"] = input_txt[super_idx]
                og_utt["input_matrix"] = input_matrix[super_idx]
                og_utt["post_prompt"] = post_prompt["dialogue"][i]["text"]
                super_idx += 1

        # These lengths do not match, try to match
        else:
            skipped, count_doubles = 0, 0
            double_match = False
            for i, og_utt in enumerate(og):
                try:
                    if double_match:
                        # This was already a match
                        count_doubles += 1
                        og_utt["matched"] = True
                        og_utt["input_txt"] = input_txt[super_idx - 1]
                        og_utt["input_matrix"] = input_matrix[super_idx - 1]
                        og_utt["post_prompt"] = post_prompt["dialogue"][i - count_doubles - skipped]["text"]
                        double_match = False
                    else:
                        # Prepare texts to compare
                        current_prompted_utt = post_prompt["dialogue"][i - count_doubles - skipped]
                        text_to_compare = clean_text_for_comparison(og_utt, current_prompted_utt)
                        next_text = ""
                        if (i + 1) < len(og):
                            next_text = clean_text_for_comparison(og[i + 1],
                                                                  post_prompt["dialogue"][
                                                                      i + 1 - count_doubles - skipped])

                        # Compare
                        this_match, double_match = texts_approximately_equal(text_to_compare,
                                                                             current_prompted_utt["text"],
                                                                             look_ahead=next_text)

                        # Assign accordingly
                        if this_match:
                            # This is a match
                            og_utt["matched"] = True
                            og_utt["input_txt"] = input_txt[super_idx]
                            og_utt["input_matrix"] = input_matrix[super_idx]
                            og_utt["reference"] = " ".join([trip["translated_triple"].replace("_", " ")
                                                            for trip in current_prompted_utt["triples"]])
                            og_utt["post_prompt"] = post_prompt["dialogue"][i - count_doubles - skipped]["text"]
                            super_idx += 1
                        else:
                            # Move index by 1
                            og_utt["matched"] = False
                            og_utt["post_prompt"] = post_prompt["dialogue"][i - count_doubles - skipped]["text"]
                            skipped += 1

                except KeyError as e:
                    print(f"KeyError: {e}")
                except Exception as e:
                    print(f"Unhandled exception: {e}")
                    continue
        temp = pd.DataFrame.from_dict(og)
        fixed_og.append(og)

    return fixed_og


def build_train_pair(og, exclude_context=False):
    all_inputs, all_targets, all_separator, all_matrix = [], [], [], []
    for i in range(len(og)):
        current_dialogue = og[:i + 1]
        if current_dialogue[-1].get("matched", False):  # Check if 'matched' key exists and is True
            # Format text to translate
            to_translate = current_dialogue[-1]["source"]
            if "reference" in current_dialogue[-1].keys():
                target_translation = current_dialogue[-1]["reference"]
            else:
                target_translation = ""

            # Create the prompt input
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
            all_separator.append(current_dialogue[-1]["input_txt"])
            all_matrix.append(current_dialogue[-1]["input_matrix"])
        else:
            print(f"No match found for dialogue at index {i}.")  # Log unmatched dialogues for debugging

    return all_inputs, all_targets, all_separator, all_matrix


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
