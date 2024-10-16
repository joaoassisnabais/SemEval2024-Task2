import argparse
import json
import torch
import os

# Local files
import eval_prompt
import self_consistency
from label_prompt_funcs import init_prompt

# Model Libs
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils import cuda_available

def main():
    parser = argparse.ArgumentParser()

    # Model and checkpoint paths, including a merging flag
    parser.add_argument('--model', type=str, help='name of the model used to generate and combine prompts', default='mistral', choices=['mistral', 'llama', 'biomistral']) #'meta-llama/Meta-Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.2', 'BioMistral/BioMistral-7B'
    parser.add_argument('--merge', dest='merge', action='store_true', help='boolean flag to set if model is merging')
    parser.add_argument('--no-merge', dest='merge', action='store_true', help='boolean flag to set if model is merging')
    parser.set_defaults(merge=False)

    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint, used if merging', default="")


    # Path to queries, qrels and prompt files
    parser.add_argument('--used_set', type=str, help='choose which data to use', default="test") # train | dev | test
    args = parser.parse_known_args()
    parser.add_argument('--queries', type=str, help='path to queries file', default=f'queries/queries2024_{args[0].used_set}.json')
    parser.add_argument('--qrels', type=str, help='path to qrels file', default=f'qrels/qrels2024_{args[0].used_set}.json')
    parser.add_argument('--prompts', type=str, help='path to prompts file', default="")
    
    # Metrics only flag
    parser.add_argument('--metrics_only', action='store_true', help='boolean flag to set if we should run the model ot the metrics only', default=False)
    parser.add_argument('--labels', type=str, help='path to predicted labels file', default='')

    # Task to run
    parser.add_argument('--task', type=str, help='task to run', default='self_consistency', choices=['output_labels', 'evaluate', 'self_consistency'])
    
    # Self-Consistency arguments
    parser.add_argument('--top_k', type=int, help='top k to use in inference', default=40)
    parser.add_argument('--top_p', type=float, help='p to use in nucleus sampling', default=0)
    parser.add_argument('--temperature', type=float, help='temperature to use in inference', default=0.7)
    parser.add_argument('--reasoning_paths', type=int, help='number of reasoning paths to use', default=8)

    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="outputs/")
    
    # Attention type
    parser.add_argument("--attention_type", type=str, help="attention type to use", default="normal", choices=["normal", "flash"])

    args = parser.parse_args()

    model = None

    cuda_available()
    if args.model == "mistral":
        args.model = 'mistralai/Mistral-7B-Instruct-v0.2'
        if args.prompts == "":
            args.prompts = "prompts/MistralPrompts.json"
    elif args.model == "llama":
        args.model = 'meta-llama/Meta-Llama-3-8B-Instruct'
        if args.prompts == "":
            args.prompts = "prompts/llamaPrompts.json"
    elif args.model == "biomistral":
        args.model = 'BioMistral/BioMistral-7B'
        if args.prompts == "":
            args.prompts = "prompts/MistralPrompts.json"

    if args.metrics_only:
        queries = json.load(open(args.queries))
        qrels = json.load(open(args.qrels))
        prompt = json.load(open(args.prompts))["best_combination_prompt"]
        pred_labels = json.load(open(args.labels))
        eval_prompt.evaluate_without_query(pred_labels, queries, qrels, "best_combination_prompt", prompt, args, args.used_set)
        return
        
    if args.merge:
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model = PeftModel.from_pretrained(model, args.checkpoint)
        model = model.merge_and_unload()
        model.to("cuda")
    
    else:
       model = AutoModelForCausalLM.from_pretrained(
            args.model, low_cpu_mem_usage=True,
            return_dict=True, torch_dtype=torch.bfloat16,
            device_map= {"": 0}
       )
       model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    #In case of mistral, we need to add the special tokens to the tokenizer
    '''if args.model == "mistralai/Mistral-7B-Instruct-v0.2":
        # Gives correct IDs: {'input_ids': [2], 'attention_mask': [1]}
        tokenizer("</s>", add_special_tokens=False) 

        # Gives correct IDs: {'input_ids': [842], 'attention_mask': [1]}
        tokenizer(".", add_special_tokens=False)

        # Gives incorrect IDs: {'input_ids': [842, 700, 28713, 28767], 'attention_mask': [1, 1, 1, 1]}
        tokenizer(".</s>", add_special_tokens=False)
'''

    #Assert that the files exist
    assert os.path.isfile(args.queries), f"Queries file {args.queries} does not exist"
    assert os.path.isfile(args.qrels), f"Qrels file {args.qrels} does not exist"
    assert os.path.isfile(args.prompts), f"Prompts file {args.prompts} does not exist"

    # Load dataset, queries, qrels and prompts
    queries = json.load(open(args.queries))
    qrels = json.load(open(args.qrels))

    if args.task == "self_consistency":
        prompt = init_prompt(args.prompts, "self_consistency_wo_output", "binary_output_format")
        
        majority_eval_prompt = json.load(open(args.prompts))["new_majority_evaluator_prompt"]
        self_consistency.self_consistency(model, tokenizer, queries, qrels, "self_consistency_prompt", prompt,
                                           majority_eval_prompt, args, args.used_set)
        
    elif args.task == "output_labels":
        prompt = init_prompt(args.prompts, "best_combination_prompt", "binary_output_format")
        eval_prompt.output_prompt_labels(model, tokenizer, queries, prompt, args, args.used_set)

    elif args.task == "evaluate":
        prompt = init_prompt(args.prompts, "best_combination_prompt", "binary_output_format")
        eval_prompt.full_evaluate_prompt(model, tokenizer, queries, qrels, "best_combination_prompt", prompt, args, args.used_set)

if __name__ == '__main__':
    main()