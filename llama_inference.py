import json

# Local Files
import self_consistency
from eval_prompt import full_evaluate_prompt, output_prompt_labels
from label_prompt_funcs import init_llama_prompt

def llama_tasks(args, model, tokenizer, queries, qrels):
    
    if args.task == "self_consistency":
        prompt = init_llama_prompt(args.prompts, args.prompt_id, tokenizer)
        #majority_eval_prompt = json.load(open(args.prompts))["new_majority_evaluator_prompt"]
        majority_eval_prompt = "Please select the most appropriate answer from the following options: $outputs"
        
        self_consistency.self_consistency(model, tokenizer, queries, qrels, "self_consistency_prompt", prompt,
                                        majority_eval_prompt, args, args.used_set, args.majority_voting)
            
    elif args.task == "output_labels":
        prompt = init_llama_prompt(args.prompts, args.prompt_id, tokenizer)
        output_prompt_labels(model, tokenizer, queries, prompt, args, args.used_set)

    elif args.task == "evaluate":
        prompt = init_llama_prompt(args.prompts, args.prompt_id, tokenizer)
        full_evaluate_prompt(model, tokenizer, queries, qrels, args.prompt_id, prompt, args, args.used_set)
    return