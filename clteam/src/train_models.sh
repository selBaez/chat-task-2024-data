#!/bin/bash

languages=("en-de" "en-fr" "en-nl" "en-pt")

for lang in "${languages[@]}"
do
  echo "Training model for language: $lang"
  python ./train_model.py --language "$lang" --eval_dir /home/lkrause/data/llm-storage/selea/chat-task-2024-data/clteam/experiments/with_dialogue_history/${lang}_declare-lab-flan-alpaca-base_ep50
done

echo "All training processes completed."
