import json
import re
import collections
from typing import Tuple

# Local Files
from utils import safe_open_w

ENTAILMENT_LABELS = {"entailment", "yes", "y"}
CONTRADICTION_LABELS = {"contradiction", "no", "not", "n"}

def textlabel_2_binarylabel(text_label: list[str]) -> int:
    for label in text_label:
        if label.lower() in ENTAILMENT_LABELS:
            return 1
        elif label.lower() in CONTRADICTION_LABELS:
            return 0
    return 1 # In case of no label, default to Entailment

def cotlabel_2_binarylabel(cot_label: list[str]) -> int:
    cot_label = ''.join(cot_label)
    match = re.search(r'THE\s*FINAL\s*ANSWER\s*IS\s*:*\s*(YES|NO)', cot_label.upper())
    if match:
        if match.group().endswith("YES"):
            return 1
        elif match.group().endswith("NO"):
            return 0
    return 1 # In case of no label, default to Entailment

def get_label_from_cot(cot_label: list[str]) -> int:
    predicted_labels = []
    
    for part in cot_label:
        entail = re.search(r'entailment|yes', part, re.IGNORECASE)
        contra = re.search(r'contradiction|not\s+entailed|no', part, re.IGNORECASE)
        neutral = re.search(r'not\s+enough\s+information|inconclusive|undetermined|neutral|undefined|indeterminate', part, re.IGNORECASE) 
        if entail:
            predicted_labels.append('Entailment')
        if contra:
            predicted_labels.append('Contradiction')
        if neutral:
            predicted_labels.append('Neutral')
            
    return predicted_labels

def most_common(lst):
    return max(set(lst), key=lst.count)

def simple_majority_voting_sc(cot_labels: list[int]) -> int:
    predicted_labels = get_label_from_cot(cot_labels)
    predicted_label_major = most_common(predicted_labels)  
    return 1 if predicted_label_major == 'Entailment' else 0

"""
Complex Majority Voting for Self-Consistency:
Checks if the majority label is at least 1 more than the second most common label and
guarantee that at least 80% of the reasoning paths are not neutral
"""
def complex_majority_voting_sc(cot_labels: list[str], reasoning_paths: int) -> Tuple[bool, int]:
    decided = False
    
    predicted_labels = get_label_from_cot(cot_labels)
    
    if len(predicted_labels) >= int(reasoning_paths * 0.8):
        counter = collections.Counter(predicted_labels)
        d = dict(counter)
        predicted_label_major = most_common(predicted_labels)
        labels = ['Entailment', 'Contradiction']
        labels.pop(labels.index(predicted_label_major))
        other_label = labels[0]
        if other_label in d and d[predicted_label_major] - d[other_label] > 1:
            decided = True
        elif other_label not in d:
            decided = True
        
        return (decided,1) if predicted_label_major == 'Entailment' else (decided,0)

    return (decided, 0)

def label_2_SemEval2024(labels : dict) -> dict:
    return {q_id : {"Prediction" : "Entailment" if labels[q_id] == 1 else "Contradiction"} for q_id in labels}

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
    try:
        prompt = prompt.replace("$output_format", text_to_replace["output_format"])
    except Exception:
        pass
    
    return prompt

def create_qid_prompt_label_dict(queries : dict, qrels : dict, prompt : str) -> dict:
    queries_dict = {}
    for q_id in queries:
        queries_dict[q_id] = { 
            "text" : generate_query_from_prompt(extract_info_from_query(queries[q_id]), prompt), 
            "gold_label" : textlabel_2_binarylabel([qrels[q_id]["Label"].strip()])
        }
    return queries_dict

def create_majority_eval_prompt(outputs : list[str], prompt : str) -> dict:
    concatenated_outputs = ""
    for i in range(len(outputs)):
        concatenated_outputs += f"\n{i+1}- " + outputs[i]
    prompt = prompt.replace("$outputs", concatenated_outputs)
    return concatenated_outputs

def create_qdid_prompt(queries : dict, prompt : str) -> dict:
    queries_dict = {}
    for q_id in queries:
        queries_dict[q_id] = {"text" : generate_query_from_prompt(extract_info_from_query(queries[q_id]), prompt)}
    return queries_dict

def generate_pos_prompts(mistral_prompts : dict):
    prompt_combinations = { "base_mistral_prompts" : {field : mistral_prompts[field] for field in mistral_prompts}, "combination_prompts" : {}}

    for task_id, task in mistral_prompts["task_descriptions"].items():
        for ctr_id, ctr in mistral_prompts["ctr_descriptions"].items():
            for statement_id, statement in mistral_prompts["statement_descriptions"].items():
                for option_id, option in mistral_prompts["option_descriptions"].items():
                    combination = mistral_prompts["task_template_prompt_comparison"].replace("$task_description", task).replace("$ctr_description", ctr).replace("$statement_description", statement).replace("$option_description", option)

                    prompt_combinations["combination_prompts"][f'<s>[INST]{task_id}_{ctr_id}_{statement_id}_{option_id}[/INST]'] = combination

    with safe_open_w('prompts/MistralPromptsCombination_V2.json') as output_file:
        output_file.write(json.dumps(prompt_combinations, ensure_ascii=False, indent=4))

    return prompt_combinations

def init_mistral_prompt(prompt_file: str, prompt_name: str, output_format: str) -> str:
    prompt = json.load(open(prompt_file))[prompt_name]
    try:
        prompt = prompt.replace("$output_format", output_format)
    except Exception:
        pass
    return prompt

def init_llama_prompt(prompt_file: str, prompt_name: str, tokenizer: object) -> str:
    try:
        system_prompt = json.load(open(prompt_file))["system"][prompt_name]
        user_prompt = json.load(open(prompt_file))["user"][prompt_name]
    except Exception:
        raise Exception(f"Prompt {prompt_name} not found in {prompt_file}")
    
    final_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    final_prompt = tokenizer.apply_chat_template(final_prompt)
    final_prompt = tokenizer.decode(final_prompt)
    
    if prompt_name == "best_combination":    
        final_prompt = final_prompt.rsplit("<|eot_id|>", 1)
        final_prompt = final_prompt[0] + "\n\nAnswer:<|eot_id|>" + final_prompt[1]
    
    return final_prompt