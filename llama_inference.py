import json

# Local Files
import self_consistency
from eval_prompt import full_evaluate_prompt, output_prompt_labels
from label_prompt_funcs import init_llama_prompt

def llama_tasks(args, model, tokenizer, queries, qrels):
    
    if args.task == "self_consistency":
        prompt = init_llama_prompt(args.prompts, "self_consistency")
        majority_eval_prompt = json.load(open(args.prompts))["new_majority_evaluator_prompt"]
        
        self_consistency.self_consistency(model, tokenizer, queries, qrels, "self_consistency_prompt", prompt,
                                        majority_eval_prompt, args, args.used_set)
            
    elif args.task == "output_labels":
        prompt = init_llama_prompt(args.prompts, "best_combination")
        output_prompt_labels(model, tokenizer, queries, prompt, args, args.used_set)

    elif args.task == "evaluate":
        prompt = init_llama_prompt(args.prompts, "best_combination", tokenizer)
        full_evaluate_prompt(model, tokenizer, queries, qrels, "best_combination_prompt", prompt, args, args.used_set)
    return
