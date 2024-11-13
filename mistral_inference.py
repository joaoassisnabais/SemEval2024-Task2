import json

# Local Files
import self_consistency
import eval_prompt
from label_prompt_funcs import init_mistral_prompt

def mistral_tasks(args : object, model : object, tokenizer : object, queries : dict, qrels : dict):
    if args.task == "self_consistency":
            prompt = init_mistral_prompt(args.prompts, "self_consistency_wo_output", "binary_output_format")
            majority_eval_prompt = json.load(open(args.prompts))["new_majority_evaluator_prompt"]
            
            self_consistency.self_consistency(model, tokenizer, queries, qrels, "self_consistency_prompt", prompt,
                                            majority_eval_prompt, args, args.used_set)
            
    elif args.task == "output_labels":
        prompt = init_mistral_prompt(args.prompts, "best_combination_prompt", "binary_output_format")
        eval_prompt.output_prompt_labels(model, tokenizer, queries, prompt, args, args.used_set)

    elif args.task == "evaluate":
        prompt = init_mistral_prompt(args.prompts, "best_combination_prompt", "binary_output_format")
        eval_prompt.full_evaluate_prompt(model, tokenizer, queries, qrels, "best_combination_prompt", prompt, args, args.used_set)
