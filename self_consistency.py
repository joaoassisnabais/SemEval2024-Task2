import json
import torch

# Local Files
from utils import safe_open_w
from label_prompt_funcs import create_majority_eval_prompt, label_2_SemEval2024, create_qid_prompt_label_dict, cotlabel_2_binarylabel
from eval_prompt import calculate_metrics, output_mistakes, output_full_metrics

# Util libs
from datetime import datetime
from tqdm import tqdm

def evaluate_final_answer(model : object, tokenizer : object, decoded_outputs : list[str], majority_eval_prompt_skeleton, terminators) -> int:
    majority_eval_prompt = create_majority_eval_prompt(decoded_outputs, majority_eval_prompt_skeleton)
    
    tokenized = tokenizer(majority_eval_prompt, return_tensors="pt")
    tokenized["input_ids"] = tokenized.input_ids.to(device="cuda")
    tokenized["attention_mask"] = tokenized.attention_mask.to(device="cuda")
    
    final_outputs = model.generate(**tokenized, max_new_tokens=20, temperature = 0.5, top_k = 40,
                                do_sample=True, eos_token_id=terminators, num_return_sequences=1)
    
    decoded_final_output = tokenizer.decode(final_outputs[0][tokenized["input_ids"].shape[1]:]).strip()

    return cotlabel_2_binarylabel([decoded_final_output])    

            
def inference(model : object, tokenizer : object, queries : dict, majority_eval_prompt_skeleton: str, terminators : list[str], k=40, p=1, temp=0.7, reasoning_paths=8) -> dict:
    res_labels = {}
    with torch.inference_mode():
        for q_id in tqdm(queries):
            tokenized = tokenizer(queries[q_id]["text"], return_tensors="pt")
            tokenized["input_ids"] = tokenized.input_ids.to(device="cuda")
            tokenized["attention_mask"] = tokenized.attention_mask.to(device="cuda")

            decoded_output= {}
            current_labels = []
            for i in range(reasoning_paths):
                outputs =  model.generate(**tokenized, max_new_tokens=200, temperature = temp, top_k = k,
                                        do_sample=True, pad_token_id=tokenizer.eos_token_id, eos_token_id=terminators, top_p=p, num_return_sequences=1)

                decoded_output[i] = tokenizer.decode(outputs[0][tokenized["input_ids"].shape[1]:]).strip()
                current_labels.append([decoded_output[i]])
            
            res_labels[q_id] = evaluate_final_answer(model, tokenizer, decoded_output, majority_eval_prompt_skeleton, terminators)
                
    return res_labels


def self_consisntency(model: object, tokenizer: object, queries: dict, qrels: dict, prompt_id : str, prompt: str, majority_eval_prompt: str, args : object, used_set : str) -> dict:
    # Replace prompt with query info
    queries_dict = create_qid_prompt_label_dict(queries, qrels, prompt)
    
    if model == 'meta-llama/Meta-Llama-3-8B-Instruct':
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    else:
        terminators = tokenizer.eos_token_id
    
    pred_labels = inference(model, tokenizer, queries_dict, majority_eval_prompt, terminators, k=args.top_k, p=args.top_p, temp=args.temperature, reasoning_paths=args.reasoning_paths)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Output results
    with safe_open_w(f'{args.output_dir}{timestamp}_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(label_2_SemEval2024(pred_labels), ensure_ascii=False, indent=4))

    # Compute metrics
    metrics, mistakes = calculate_metrics(pred_labels, queries_dict)
    
    output_mistakes(args, mistakes, prompt, queries, qrels, used_set)
    output_full_metrics(args, prompt_id, prompt, used_set, metrics)
    
    return metrics