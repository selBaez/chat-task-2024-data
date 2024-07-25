import argparse
import itertools
import pickle
import string
from pathlib import Path

import neuralcoref
import numpy as np
import spacy
from tqdm import tqdm

from utils.utils_data import load_data

# stanza.install_corenlp()
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)  # Add neural coref to SpaCy's pipe
punc = string.punctuation
alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"
max_nodes = 100


def coreference(s):
    doc = nlp(s)
    return doc._.coref_clusters


def compress_triple(dialogue, coref):
    triples = [utt["triples"] for utt in dialogue["dialogue"]]
    triples = list(itertools.chain.from_iterable(triples))

    temp_set = []
    for cur in triples:
        cur_subject = cur['subject'].lower() if cur['subject'] else ""
        cur_relation = cur['predicate'].lower() if cur['predicate'] else ""
        cur_object = cur['object'].lower() if cur['object'] else ""

        for cluster in coref:
            span = [w.text.lower() for w in cluster.mentions]
            if cur_subject in span:
                cur_subject = cluster.main.text.lower()
            if cur_object in span:
                cur_object = cluster.main.text.lower()

        if len(temp_set) == 0:
            temp_set.append([cur_subject, cur_relation, cur_object])
        else:
            flag = 0
            # print(temp_set)
            for j in range(0, len(temp_set)):
                ###save the longest when have two same entities
                if temp_set[j][0] == cur_subject and temp_set[j][1] == cur_relation:

                    if len(cur_object) > len(temp_set[j][2]):
                        temp_set[j][2] = cur_object
                    flag = 1

                elif temp_set[j][0] == cur_subject and temp_set[j][2] == cur_object:
                    if len(cur_relation) > len(temp_set[j][1]):
                        temp_set[j][1] = cur_relation
                    flag = 1

                elif temp_set[j][2] == cur_object and temp_set[j][1] == cur_relation:
                    if len(cur_subject) > len(temp_set[j][0]):
                        temp_set[j][0] = cur_subject
                    flag = 1

            if flag == 0:
                ##if no editing, then it is a new triplet, add to temp
                temp_set.append([cur_subject, cur_relation, cur_object])

    return temp_set


def get_mind_chart(mc_context, annotate_result, max_nodes):
    """get mind chart

    Args:
        mc_context (string): the context to construct mind chart (question+" "+context+" "+lecture+" "+solution+" "+choice)

    Returns:
        triples(list of triplets list): [[I, love, NLP],[NLP,is,fun]]
        action_input(list):["I</s><s>love</s><s>NLP</s><s>NLP</s><s>is</s><s>fun"]
        action_adj(list): [adjecent matrix] 
    """
    mc_context = mc_context.replace("\n", " ")
    coref = coreference(mc_context)
    triples = compress_triple(annotate_result, coref)

    action_input = []
    coref_clusters = []

    id2node = {}
    node2id = {}
    adj_temp = np.zeros([max_nodes, max_nodes])
    index = 0
    if len(triples) == 0:
        action_input.append('<pad>')
    else:
        temp_text = ' <s> '
        for u in triples:
            if u[0] not in node2id:
                node2id[u[0]] = index
                id2node[index] = u[0]
                if index < max_nodes:
                    if temp_text == ' <s> ':
                        temp_text = temp_text + u[0]
                    else:
                        temp_text = temp_text + ' </s> <s> ' + u[0]

                    index = index + 1
                else:
                    break
            if u[1] not in node2id:
                node2id[u[1]] = index
                id2node[index] = u[1]

                if index < max_nodes:
                    if temp_text == ' <s> ':
                        temp_text = temp_text + u[1]
                    else:
                        temp_text = temp_text + ' </s> <s> ' + u[1]
                    index = index + 1
                else:
                    break

            if u[2] not in node2id:
                node2id[u[2]] = index
                id2node[index] = u[2]

                if index < max_nodes:
                    if temp_text == ' <s> ':
                        temp_text = temp_text + u[2]
                    else:
                        temp_text = temp_text + ' </s> <s> ' + u[2]
                    index = index + 1
                else:
                    break

            adj_temp[node2id[u[0]]][node2id[u[0]]] = 1
            adj_temp[node2id[u[1]]][node2id[u[1]]] = 1
            adj_temp[node2id[u[2]]][node2id[u[2]]] = 1

            adj_temp[node2id[u[0]]][node2id[u[1]]] = 1
            adj_temp[node2id[u[1]]][node2id[u[0]]] = 1

            adj_temp[node2id[u[1]]][node2id[u[2]]] = 1
            adj_temp[node2id[u[2]]][node2id[u[1]]] = 1

        action_input.append(temp_text)
        coref =[[el.string for el in cluster.mentions] for cluster in coref]
        coref_clusters.append(coref)
    # action_adj.append(adj_temp)

    return action_input, adj_temp, coref_clusters


def make_output_directory(args, split):
    # Make path for output
    outpath = Path(args.output_dir) / f"{split}/"
    outpath.mkdir(parents=True, exist_ok=True)

    # Check if the files exist already?
    # if os.path.isfile(args.input_text_path) or os.path.isfile(args.adj_matrix_path):
    #     assert False

    return outpath


def save_data(mc_input_text_list, mc_adj_matrix_list, mc_coref_clusters_list, outpath, args):
    mc_input_text_path = outpath / args.input_text_file
    with open(mc_input_text_path, 'wb') as f:
        pickle.dump(mc_input_text_list, f)

    mc_adj_matrix_path = outpath / args.adj_matrix_file
    with open(mc_adj_matrix_path, 'wb') as f:
        pickle.dump(mc_adj_matrix_list, f)

    mc_coref_clusters_path = outpath / args.coref_clusters_file
    with open(mc_coref_clusters_path, 'wb') as f:
        pickle.dump(mc_coref_clusters_list, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../graphs')
    parser.add_argument('--splits', nargs="+", default=["mini-valid"])  # "train", "valid", "test"])
    parser.add_argument('--languages', nargs="+", default=["en-de_short"])
    parser.add_argument('--output_dir', type=str, default='./../preprocessed/')
    parser.add_argument('--input_text_file', type=str, default='mc_input_text.pkl')
    parser.add_argument('--adj_matrix_file', type=str, default='mc_adj_matrix.pkl')
    parser.add_argument('--coref_clusters_file', type=str, default='mc_coref_clusters.pkl')
    parser.add_argument('--exclude_context', action='store_true', help='remove dialogue history from the prompt')

    args = parser.parse_args()
    return args


def main(args):
    for split in args.splits:
        # Create directories
        outpath = make_output_directory(args, split)

        # Read data
        data_per_language = load_data(args, split)

        # Loop through languages
        for dialogues in data_per_language:

            # Analyze
            mc_input_text_list, mc_adj_matrix_list, mc_coref_clusters_list = [], [], []
            for dialogue in tqdm(dialogues):
                dialogue_history = "\n".join([f'{utt["sender"]}: {utt["text"]}' for utt in dialogue["dialogue"]])
                to_translate = dialogue["dialogue"][-1]["text"]

                if args.exclude_context:
                    mc_context_text = f"{to_translate}"
                    dialogue = dialogue["dialogue"][-1]
                else:
                    mc_context_text = f"{dialogue_history}\n{to_translate}"

                mc_input_text, mc_adj_matrix, mc_coref_clusters = get_mind_chart(mc_context_text, dialogue, max_nodes)
                mc_input_text_list.append(mc_input_text)
                mc_adj_matrix_list.append(mc_adj_matrix)
                mc_coref_clusters_list.append(mc_coref_clusters)

            # Save data
            save_data(mc_input_text_list, mc_adj_matrix_list, mc_coref_clusters_list, outpath, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
