import json
import torch
import re

# Local Files
from utils import safe_open_w
from label_prompt_funcs import cotlabel_2_binarylabel, label_2_SemEval2024, create_qid_prompt_label_dict
from eval_prompt import calculate_metrics, output_mistakes, output_full_metrics

# Util libs
from datetime import datetime
from tqdm import tqdm

def inference(model : object, tokenizer : object, queries : dict, k=40, p=0, temp=0.5, reasoning_paths=8) -> dict:
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
                                        do_sample=True, pad_token_id=tokenizer.eos_token_id)

                decoded_output[i] = tokenizer.decode(outputs[0][tokenized["input_ids"].shape[1]:]).strip()
                current_labels.append(cotlabel_2_binarylabel([decoded_output[i]]))
            
            # Check if the majority of the reasoning paths agree
            if (sum(current_labels) / reasoning_paths) >= 0.5:
                res_labels[q_id] = 1
            else:
                res_labels[q_id] = 0
                
    return res_labels


def self_consisntency(model: object, tokenizer: object, queries: dict, qrels: dict, prompt_id : str, prompt: str, args : object, used_set : str) -> dict:
    # Replace prompt with query info
    queries_dict = create_qid_prompt_label_dict(queries, qrels, prompt)
    
    pred_labels = inference(model, tokenizer, queries_dict, k=args.top_k, p=args.top_p, temp=args.temperature, reasoning_paths=args.reasoning_paths)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # Output results
    with safe_open_w(f'{args.output_dir}{timestamp}_{used_set}-set.json') as output_file:
        output_file.write(json.dumps(label_2_SemEval2024(pred_labels), ensure_ascii=False, indent=4))

    # Compute metrics
    metrics, mistakes = calculate_metrics(pred_labels, queries_dict)
    
    output_mistakes(args, mistakes, prompt, queries, qrels, used_set)
    output_full_metrics(args, prompt_id, prompt, used_set, metrics)
    
    return metrics
