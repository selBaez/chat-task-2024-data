import argparse
import json
import os
import random

import evaluate
import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from utils.dataset import ChatDatasetWithGraph
from utils.model import T5GenerationWithGraph
from utils.utils_data import mk_dir, make_save_directory
from utils.utils_prompt import postprocess_text

os.environ["WANDB_PROJECT"] = "WMT_24"


def T5Trainer(args):
    set_random_seeds(args)

    # make directories for output
    print('====Make directories====')
    save_dir = make_save_directory(args)
    mk_dir(args.output_dir)

    # Create tokenizer
    print(f'====Create tokenizer====')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<s>']})
    vocab = tokenizer.get_vocab()
    s_token_id = vocab["<s>"]
    datacollator = DataCollatorForSeq2Seq(tokenizer)

    # Load data as dataset
    print('====Load dataset====')
    if args.eval_dir == "":
        eval_set = ChatDatasetWithGraph("train", tokenizer, args)
        train_set = ChatDatasetWithGraph("valid", tokenizer, args)

        # TODO uncomment for testing script
        # train_set = ChatDatasetWithGraph("mini-valid", tokenizer, args, dry_run=True)
        # eval_set = ChatDatasetWithGraph("mini-valid", tokenizer, args, dry_run=True)
    else:
        train_set = None
        eval_set = ChatDatasetWithGraph("test", tokenizer, args)
        args.model = args.eval_dir

    # Load model
    print(f'====Load model: {args.model} ====')
    model = T5GenerationWithGraph.from_pretrained(args.model, s_token_id=s_token_id)
    model.resize_token_embeddings(len(tokenizer))
    print("model parameters: ", model.num_parameters())

    # chrf for cn generation
    metric = evaluate.load("chrf")

    def compute_metrics_rougel(eval_preds):
        preds, targets = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        pred_result = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(pred_result, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_preds, decoded_labels = postprocess_text(preds, targets)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # evaluate at each epoch
    print('====Load training arguments====')
    training_args = Seq2SeqTrainingArguments(save_dir,
                                             do_train=True,
                                             do_eval=True,
                                             evaluation_strategy="steps",
                                             logging_strategy="steps",
                                             logging_steps=10,
                                             save_strategy="steps",
                                             eval_steps=10000,
                                             save_steps=500,
                                             save_total_limit=2,
                                             learning_rate=args.lr,
                                             eval_accumulation_steps=args.eval_acc,
                                             per_device_train_batch_size=args.bs,
                                             per_device_eval_batch_size=args.eval_bs,
                                             weight_decay=args.weight_decay,
                                             num_train_epochs=args.epoch,
                                             metric_for_best_model="score",
                                             predict_with_generate=True,
                                             generation_max_length=args.output_len,
                                             load_best_model_at_end=True,
                                             report_to="wandb",
                                             bf16=args.bf16
                                             )

    print('====Load trainer====')
    trainer = Seq2SeqTrainer(model=model,
                             args=training_args,
                             train_dataset=train_set,
                             eval_dataset=eval_set,
                             data_collator=datacollator,
                             tokenizer=tokenizer,
                             compute_metrics=compute_metrics_rougel,
                             preprocess_logits_for_metrics=None
                             )

    # Train
    if args.eval_dir == "":
        print('====Train====')
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(save_dir)

    # # Evaluate TODO: This makes it slower, not sure why but we might not need it now
    # print('====Evaluate with HF====')
    # metrics = trainer.evaluate(eval_dataset=eval_set, max_length=args.output_len)
    # print("Evaluation metrics:", metrics)
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)

    def generate_predictions(dataset):
        predict_results = trainer.predict(test_dataset=dataset, max_length=args.output_len)
        preds, targets = predict_results.predictions, predict_results.label_ids
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        preds = [pred.strip() for pred in preds]

        return preds, targets

    # Generate predictions for eval set
    print('====Generate predictions====')
    torch.cuda.empty_cache()
    if trainer.is_world_process_zero():
        preds, targets = generate_predictions(eval_set)
        output_data = {"preds": preds,
                       "labels": targets}

        # Save predictions
        if args.eval_dir == "":
            output_prediction_file = os.path.join(save_dir, "predictions_ans_eval.json")
        else:
            output_prediction_file = os.path.join(save_dir, "predictions_ans_test.json")

        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(output_data, indent=4))


def set_random_seeds(args):
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_root', type=str, default='./../..')
    parser.add_argument('--triple_data_root', type=str, default='./../graphs')
    parser.add_argument('--data_root', type=str, default='./../preprocessed/without_dialogue_history_exploded')
    parser.add_argument('--output_dir', type=str, default='./../experiments/without_dialogue_history/test')
    parser.add_argument('--exclude_context', action='store_true', help='remove dialogue history from the prompt')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--eval_strategy', type=str, default="steps", help='evaluation strategy')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--bf16', action='store_true', help='use bf16 dtype')

    parser.add_argument('--language', default='en-de', help='language pair for data loader')
    parser.add_argument('--model', type=str, default='declare-lab/flan-alpaca-base')
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--eval_bs', type=int, default=8)
    parser.add_argument('--eval_dir', type=str, default="", help='the directory of model for evaluation')


    # TODO uncomment for testing script
    # parser.add_argument('--language', default='en-de_short', help='language pair for data loader')
    # parser.add_argument('--model', type=str, default='declare-lab/flan-alpaca-base')
    # parser.add_argument('--epoch', type=int, default=2)
    # parser.add_argument('--bs', type=int, default=4)
    # parser.add_argument('--eval_bs', type=int, default=4)

    args = parser.parse_args()

    print("args", args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    return args


def main():
    print(f"\n\n\nCUDA AVAILABLE? {torch.cuda.is_available()}\n\n\n")

    args = parse_args()

    T5Trainer(args)


if __name__ == '__main__':
    main()
