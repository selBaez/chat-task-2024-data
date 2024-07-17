# CLTeam submission for WMT 2024 Chat Shared Task

This repository contains the code from CLTeam for the chat shared task organized with WMT 2024.

## Getting started

In order to run the code, follow these steps:

1) Create a virtual environment for the project (conda, venv, etc)

```bash
conda create --name chat-task-2024-data python=3.8
conda activate chat-task-2024-data
```

1) Install the required dependencies in `requirements.txt`

```bash
pip install -r requirements.txt --no-cache
```

## Tasks

- [ ] Script to extract graphs (first discuss which models)
  - [ ] Content graphs 
  - [ ] Perspective values
- [ ] Run dataset preprocessing to create graphs
  - [ ] Create ```action_input(list):["I</s><s>love</s><s>NLP</s><s>NLP</s><s>is</s><s>fun"]```
  - [ ] Create ```action_adj(list): [adjecency matrix] ```
- [ ] Adapt dataset class 
- [ ] Adapt model to take in several graphs
  - [ ] Dialogue history
  - [ ] Previous utterance
  - [ ] Source segment
- [ ] Prepare and run slurm scripts for experiments
  - [ ] main
  - [ ] ablation only some graphs
  - [ ] ablation with text 
- [ ] Make submissions 