import os
import json
import torch
import typing
import random
import re
import argparse

from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForCausalLM

ENTAILMENT_LABELS = {"entailment", "yes", "y"}
CONTRADICTION_LABELS = {"contradiction", "no", "not", "n"}

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def extract_info_from_query(query : dict) -> dict:
    relevant_info = {}
    relevant_info["hypothesis"] = query["Statement"]
    relevant_info["primary_evidence"] = query["Primary_id_txt"]
    relevant_info["secondary_evidence"] = query["Secondary_id_txt"] if "Secondary_id_txt" in query else ""
    return relevant_info

def generate_query_from_prompt(text_to_replace: dict, prompt: str) -> str:
    prompt = prompt.replace("$primary_evidence", text_to_replace["primary_evidence"])
    prompt = prompt.replace("$secondary_evidence", text_to_replace["secondary_evidence"])
    prompt = prompt.replace("$hypothesis", text_to_replace["hypothesis"])
    return prompt

ENTAILMENT_LABELS = {"entailment", "yes", "y"}
CONTRADICTION_LABELS = {"contradiction", "no", "not", "n"}

def textlabel_2_binarylabel(text_label: list[str]) -> int:
    for label in text_label:
        if label.lower() in ENTAILMENT_LABELS:
            return 1
        elif label.lower() in CONTRADICTION_LABELS:
            return 0
    #print(f'Text label: [{text_label=}.] This label output a random option because the label was not found.')
    #return random.randint(0,1)
    return 1

def label_2_SemEval2024(labels : dict) -> dict:
    res = {}
    for q_id in labels:
        pred = "None" # random.choice(["Entailment", "Contradiction"])
        if labels[q_id] == 1:
            pred = "Entailment"
        elif labels[q_id] == 0:
            pred = "Contradiction"
        res[q_id] = {"Prediction" : pred}
    return res

def create_qid_prompt_label_dict(queries : dict, qrels : dict, prompt : str) -> dict:
    queries_dict = {}
    for q_id in queries:
        queries_dict[q_id] = { 
            "text" : generate_query_from_prompt(extract_info_from_query(queries[q_id]), prompt), 
            "gold_label" : textlabel_2_binarylabel([qrels[q_id]["Label"].strip()])
        }
    return queries_dict

def create_qdid_prompt(queries : dict, prompt : str) -> dict:
    queries_dict = {}
    for q_id in queries:
        queries_dict[q_id] = {"text" : generate_query_from_prompt(extract_info_from_query(queries[q_id]), prompt)}
    return queries_dict

def query_inference(model : object, tokenizer : object, queries : dict) -> dict:
    res_labels = {}
    with torch.inference_mode():
        for q_id in tqdm(queries):
            tokenized = tokenizer(queries[q_id]["text"], return_tensors="pt")
            tokenized["input_ids"] = tokenized.input_ids.to(device="cuda")
            tokenized["attention_mask"] = tokenized.attention_mask.to(device="cuda")
            outputs =  model.generate(**tokenized, max_new_tokens=50, top_k = 5, do_sample=True, pad_token_id=tokenizer.eos_token_id)

            decoded_output = tokenizer.decode(outputs[0][tokenized["input_ids"].shape[1]:]).strip()
            decoded_output_sub = re.sub("[,!\.]+", " ", decoded_output)
            decoded_output_sub = re.sub("(\\n)+", " ", decoded_output_sub)
            decoded_output_sub = re.sub("(<\/s>)+", " ", decoded_output_sub)
            #print(f'The postprocessed decoded output was {decoded_output_sub.split(" ")[:30]=}')
            res_labels[q_id] = textlabel_2_binarylabel(decoded_output_sub.split(" ")[:30])
    return res_labels

def query_inference_no_rand(model : object, tokenizer : object, queries : dict) -> dict:
    res_labels = {}
    with torch.inference_mode():
        for q_id in tqdm(queries):
            tokenized = tokenizer(queries[q_id]["text"], return_tensors="pt")
            tokenized["input_ids"] = tokenized.input_ids.to(device="cuda")
            tokenized["attention_mask"] = tokenized.attention_mask.to(device="cuda")
            outputs =  model.generate(**tokenized, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)

            decoded_output = tokenizer.decode(outputs[0][tokenized["input_ids"].shape[1]:]).strip()
            decoded_output_sub = re.sub("[,!\.]+", " ", decoded_output)
            decoded_output_sub = re.sub("(\\n)+", " ", decoded_output_sub)
            decoded_output_sub = re.sub("(<\/s>)+", " ", decoded_output_sub)
            #print(f'The postprocessed decoded output was {decoded_output_sub.split(" ")[:30]=}')
            res_labels[q_id] = textlabel_2_binarylabel(decoded_output_sub.split(" ")[:10])
    return res_labels


def calculate_metrics(pred_labels : dict, gold_labels : dict) -> dict:
    res_labels = [[],[]]
    mistakes = []
    for q_id in pred_labels:
        #print(f'{gold_labels[q_id]["gold_label"]=} {pred_labels[q_id]=}')
        res_labels[0].append(gold_labels[q_id]["gold_label"])
        res_labels[1].append(pred_labels[q_id])
        if res_labels[0][-1] != res_labels[1][-1]:
            mistakes.append({"q_id" : q_id, "gold_label" : res_labels[0][-1], "pred_label" : res_labels[1][-1]})

    precison_bin = precision_score(res_labels[0], res_labels[1])
    precision_micro = precision_score(res_labels[0], res_labels[1], average="micro")
    precision_macro = precision_score(res_labels[0], res_labels[1], average="macro")
    recall_bin = recall_score(res_labels[0], res_labels[1])
    recall_micro = recall_score(res_labels[0], res_labels[1], average="micro")
    recall_macro = recall_score(res_labels[0], res_labels[1], average="macro")
    f1_bin = f1_score(res_labels[0], res_labels[1])
    f1_micro = f1_score(res_labels[0], res_labels[1], average="micro")
    f1_macro = f1_score(res_labels[0], res_labels[1], average="macro")

    return {"precison_bin" :precison_bin, "precison_micro" : precision_micro, "precision_macro" : precision_macro, "recall_bin" : recall_bin,"recall_micro" : recall_micro, "recall_macro" : recall_macro, "f1_bin" : f1_bin, "f1_micro" : f1_micro, "f1_macro" : f1_macro}, mistakes

def output_mistakes(args : dict, mistakes : list, prompt : str, queries : dict, qrels : dict, used_set : str):
    # Output Mistakes
    mistakes = {
        "prompt" : prompt,
        "used_set" : used_set, 
        "mistake_stats" : {"Total" : len(mistakes), "Single" : 0, "Comparison" : 0, "Entailment" : 0, "Contradiction" : 0}, 
        "mistakes" : mistakes, 
        "og_queries" : {}
    }

    for dict_q_id in mistakes["mistakes"]:
        q_id = dict_q_id["q_id"]
        mistakes["og_queries"][q_id] = queries[q_id]
        mistakes["og_queries"][q_id]["gold_label"] = qrels[q_id]["Label"]
        mistakes["mistake_stats"][mistakes["og_queries"][q_id]["Type"]] += 1
        mistakes["mistake_stats"][qrels[q_id]["Label"]] += 1

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    with safe_open_w(f'{args.output_dir}mistakes/{timestamp}_{args.model.split("/")[-1]}_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(mistakes, ensure_ascii=False, indent=4))

def output_full_metrics(args : dict, prompt_id : str, full_prompt : str, used_set : str, metrics : dict):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    results = {"timestamp" : timestamp}
    for arg in vars(args):
        results[arg] = getattr(args, arg)
    results["prompt"] = full_prompt
    results["set"] = used_set
    results["metrics"] = metrics
    results["formated_metrics"] =f'| {args.model.split("/")[-1]}-{prompt_id}   | {metrics["f1_macro"]} | {metrics["precision_macro"]} | {metrics["recall_macro"]} | - |'

    with safe_open_w(f'{args.output_dir}combination_output/{timestamp}_{args.model.split("/")[-1]}_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(results, ensure_ascii=False, indent=4))

def full_evaluate_prompt(model: object, tokenizer: object, queries: dict, qrels: dict, prompt_id : str, prompt: str, args : object, used_set : str) -> dict:
    # Replace prompt with query info
    queries_dict = create_qid_prompt_label_dict(queries, qrels, prompt)

    # 0-shot inference from queries TODO
    pred_labels = query_inference(model, tokenizer, queries_dict)

    # Compute metrics
    metrics, mistakes = calculate_metrics(pred_labels, queries_dict)
    output_mistakes(args, mistakes, prompt, queries, qrels, used_set)
    
    output_full_metrics(args, prompt_id, prompt, used_set, metrics)
    return metrics

def output_prompt_labels(model : object, tokenizer : object, queries : dict, prompt : str, args : object, used_set : str):
    # Replace prompt with query info
    queries_dict = create_qdid_prompt(queries, prompt)

    # 0-shot inference from queries
    pred_labels = query_inference_no_rand(model, tokenizer, queries_dict)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Output results
    with safe_open_w(f'{args.output_dir}{timestamp}_{args.checkpoint.split("/")[-3]}_{args.checkpoint.split("/")[-2]}_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(label_2_SemEval2024(pred_labels), ensure_ascii=False, indent=4))

def generate_pos_prompts(mistral_prompts : dict):
    prompt_combinations = { "base_mistral_prompts" : {field : mistral_prompts[field] for field in mistral_prompts}, "combination_prompts" : {}}

    for task_id, task in mistral_prompts["task_descriptions"].items():
        for ctr_id, ctr in mistral_prompts["ctr_descriptions"].items():
            for statement_id, statement in mistral_prompts["statement_descriptions"].items():
                for option_id, option in mistral_prompts["option_descriptions"].items():
                    combination = mistral_prompts["task_template_prompt_comparison"].replace("$task_description", task).replace("$ctr_description", ctr).replace("$statement_description", statement).replace("$option_description", option)

                    prompt_combinations["combination_prompts"][f'<s>[INST]{task_id}_{ctr_id}_{statement_id}_{option_id}[/INST]'] = combination

    with safe_open_w(f'prompts/MistralPromptsCombination_V2.json') as output_file:
        output_file.write(json.dumps(prompt_combinations, ensure_ascii=False, indent=4))

    return prompt_combinations

def main():
    parser = argparse.ArgumentParser()

    #TheBloke/Llama-2-70B-Chat-GPTQ
    #mistralai/Mistral-7B-Instruct-v0.2
    #Upstage/SOLAR-10.7B-Instruct-v1.0
    parser.add_argument('--model', type=str, help='name of the model used to fine-tune prompts for', default='mistralai/Mistral-7B-Instruct-v0.2')

    used_set = "train" # train | dev | test

    # Path to corpus file
    parser.add_argument('--dataset_path', type=str, help='path to corpus file', default="../../datasets/SemEval2024/CT_corpus.json")

    # Path to queries, qrels and prompt files
    parser.add_argument('--queries', type=str, help='path to queries file', default=f'queries/queries2024_{used_set}.json')
    parser.add_argument('--qrels', type=str, help='path to qrels file', default=f'qrels/qrels2024_{used_set}.json')
    # "prompts/T5prompts.json"
    parser.add_argument('--prompts', type=str, help='path to prompts file', default="prompts/MistralPrompts.json")

    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="outputs/")

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    #model.quantize_config = GPTQConfig(bits=4, exllama_config={"version":2}, desc_act=True)
    #model = exllama_set_max_input_length(model, 4096)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset, queries, qrels and prompts
    queries = json.load(open(args.queries))
    qrels = json.load(open(args.qrels))
    prompts = json.load(open(args.prompts))

    #prompt = "<s>[INST]Evaluate the semantic entailment between individual sections of Clinical Trial Reports (CTRs) and statements issued by clinical domain experts. CTRs expound on the methodology and outcomes of clinical trials, appraising the efficacy and safety of new treatments. The statements, on the other hand, assert claims about the information within specific sections of CTRs, for a single CTR or comparative analysis of two. For entailment validation, the statement's claim should align with clinical trial information, find support in the CTR, and refrain from contradicting provided descriptions.\n\nThe provided descriptions coincide with the content in a specific section of Clinical Trial Reports (CTRs), detailing relevant information to the trial.\n\nPrimary Trial:\n$primary_evidence\n\nSecondary Trial:\n$secondary_evidence\n\nConsider also the following statement generated by a clinical domain expert, a clinical trial organizer, or a medical researcher.\n\n$hypothesis\n\nAnswer YES or NO to the question of whether one can conclude the validity of the statement with basis on the clinical trial report information.[/INST]" 

    #full_evaluate_prompt(model, tokenizer, queries, qrels, "base_prompt", prompt, args, used_set)

    combination_prompts = generate_pos_prompts(prompts)

    for prompt_id, prompt in tqdm(combination_prompts["combination_prompts"].items()):
        full_evaluate_prompt(model, tokenizer, queries, qrels, prompt_id, prompt, args, used_set)
    
if __name__ == '__main__':
    main()