# SemEval2024-Task2

A repository that contains public code which was used to submit runs to SemEval2024, specifically in the context of [Task 2: Safe Biomedical Natural Language Inference for Clinical Trials](https://sites.google.com/view/nli4ct/). This is the full implementation of the system we submitted to the [Task Leaderboards](https://codalab.lisn.upsaclay.fr/competitions/16190#results) under the username "araag2".


In order to fully understand the scope of our work, we recommend reading our System Paper (TO:DO insert link)

## Repository Setup

TO:DO

## Task Description

TO:DO

### Available Data

TO:DO

## Experimental Results

TO:DO

### Evaluation Criteria

TO:DO

### Mistral Results

The model we chose to conduct most experiments on was the [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) model, using the pubicly available huggingface weights, and the python libraries torch, transformers and peft. 

TO:DO

### T5 Base Results

We also obtained baseline results using [Flan-T5 from huggingface](https://huggingface.co/google/flan-t5-base), with the following generation prompt: `$premise \n Question: Does this imply that $hypothesis? $options`, checking the outputs for "Entailment" or "Contradiction".

#### Train Set (0-shot)

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| flanT5-small | - | - | - | Always Contradiction |
| flanT5-base | 0.32 | 0.50 | 0.23 | - |
| flanT5-large | 0.53 | 0.56 | 0.49 | - |
| flanT5-xl | 0.67 | 0.59 | 0.77 | - |
| flanT5-xxl | 0.69 | 0.61 | 0.79 | - |

#### Dev Set (0-shot)

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| flanT5-small | - | - | - | Always Contradiction |
| flanT5-base | 0.34 | 0.55 | 0.25 | - |
| flanT5-large | 0.57 | 0.61 | 0.53 | - |
| flanT5-xl | 0.69 | 0.61 | 0.79 | - |
| flanT5-xxl | 0.71 | 0.59 | 0.88 | - |

#### Dev Set (fine-tuned)

| **Metrics**    | F1-score | Precision | Recall | Notes |
|:-------------- |:--:|:--:|:--:|:--:|
| flanT5-xl | 0.754 | 0.59 | 0.831 | - |