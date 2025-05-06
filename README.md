# SemEval2024-Task2

A repository that contains public code which was used to submit runs to SemEval2024, specifically in the context of [Task 2: Safe Biomedical Natural Language Inference for Clinical Trials](https://sites.google.com/view/nli4ct/). This is the full implementation of the system we submitted to the [Task Leaderboards of the post competion task]([https://codalab.lisn.upsaclay.fr/competitions/16190#results](https://codalab.lisn.upsaclay.fr/competitions/16190?secret_key=4863f655-9dd6-43f0-b710-f17cb67af607#results)) under the username "nabais".


In order to fully understand the scope of our work, we recommend reading my System Paper which will be available soon

## Repository Setup

```bash
├── corpus/  # Contains the CT corpus
├── prompts/ # Contains all used prompts
├── qrels/   # Contains all qrels files
├── queries/ # Contains all query files
├── eval_prompt.py # Contains all functions to generate text and evaluate a given prompt
├── finetune_Mistral.py # Training functions for Mistral-7b
├── label_prompt_funcs.py # Contains all functions that format queries, outputted labels and prompts
├── parsel_qrels2queries # Script to parse queries into the intended qrel form
├── README.md
├── run_inference.py # Script to use the Mistral model to run inference
├── utils.py # General purpose util functions
└── .gitignore
```


