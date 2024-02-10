# SemEval2024-Task2

A repository that contains public code which was used to submit runs to SemEval2024, specifically in the context of [Task 2: Safe Biomedical Natural Language Inference for Clinical Trials](https://sites.google.com/view/nli4ct/). This is the full implementation of the system we submitted to the [Task Leaderboards](https://codalab.lisn.upsaclay.fr/competitions/16190#results) under the username "araag2".


In order to fully understand the scope of our work, we recommend reading our System Paper (TO:DO insert link)

## Repository Setup

```bash
├── corpus/  # Contains the CT corpus
├── prompts/ # Contains all used prompts
├── qrels/   # Contains all qrels files
├── queries/ # Contains all query files
├── finetune_Mistral.py
├── TODO.py
├── parsel_qrels2queries # Script to parse queries into the intended qrel form
├── README.md
├── run_inference.py # Script to use the Mistral model to run inference
├── utils.py # General purpose util functions
└── .gitignore
```

## Task Description

TO:DO

### Available Data

The Dev and Train sets are balanced in labels, having 50% of Entailment and Contradiction samples.

| **Set**        | #Samples | Single | Comparison |
|:-------------- |:--:|:--:|:--:|
| Train          | 1700     | 1035   | 665        |
| Dev            | 200      | 140    | 60         |

The Pratice-test and Test set are not balanced, being heavily (65.92% and 66.53% respectively) slated towards Contradictions.

| **Set**        | #Samples | Single | Comparison  | Entailment | Contradiction |
|:-------------- |:--:|:--:|:--:|:--:|:--:|
| Pratice-Test   | 2142     | 1526   | 616         | 730        | 1412          |
| Test           | 5500     | 2553   | 2947        | 1841       | 3659          |

Both the Pratice-test and the test set include data augmentation of their original queries, by using textual alteration techniques, some of which preserve and others that alter the intended label for the query.

| **Alteration**   | Paraphrase | Contradiction | Text_App  | Num_contra | Num_para |
|:--------------   |:--:|:--:|:--:|:--:|:--:|
| Pratice-Test     | 600        | 600           | 600       | 78         | 64       |
| Test             | 1500       | 1500          | 1500      | 276        | 224      |

| **Type of Alteration** | #Total Number | Preserving    | Altering  |
|:-------------- |:--:|:--:|:--:|
| Pratice-Test           | 1942          | 1606          | 336       | 
| Test                   | 5000          | 4136          | 864       | 

The Pratice-test also includes the dev-set. whilst the Test set includes several rephrasing of the same queries, in order to test Faithfullness and Consistency.


We also expanded sets in order to train the model on additional data:

-
-
-

| **Set**                | #Samples | Single | Comparison  |
|:-------------- |:--:|:--:|:--:|
| TREC-synthetic         | 1630     | 1542   | 88          |
| Train-manual-expand    | TODO     | TODO   | TODO        | 
| Train-synthetic-expand | TODO     | TODO   | TODO        |




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