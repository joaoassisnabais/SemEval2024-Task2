# ruff: noqa: E402

import os
import wandb
import json
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # Set the GPU to use
import torch
import argparse
import typing

# Local Files
from eval_prompt import create_qid_prompt_label_dict
from utils import create_path

# Util libs
from datasets.arrow_dataset import Dataset

# Model Libs
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

def preprocess_dataset(args : argparse, prompt : str , split : str):
    # Load JSON
    set_examples = create_qid_prompt_label_dict(json.load(open(f'{args.queries}queries2024_{split}.json')), json.load(open(f'{args.qrels}qrels2024_{split}.json')), prompt)
    
    set_dict = {"id" : [], "text" : []}
    for q_id in set_examples:
        example = set_examples[q_id]
        set_dict["id"].append(q_id)
        label = "YES" if example["gold_label"] == 1 else "NO"
        set_dict["text"].append(f'{example["text"]} Answer: {label}')
    return Dataset.from_dict(set_dict)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help='model to train')
    parser.add_argument('--exp_name', type=str, default="Mistral SemEval Fine-Tune", help='Describes the conducted experiment')
    parser.add_argument('--run', type=int, default=14, help='run number for wandb logging')

    # I/O paths for models, CT, queries and qrels
    parser.add_argument('--save_dir', type=str, default="outputs/models/run_14/", help='path to model save dir')

    parser.add_argument("--used_prompt", default="prompts/MistralPrompts.json", type=str)
    parser.add_argument("--queries", default="queries/", type=str)
    parser.add_argument("--qrels", default="qrels/", type=str)

    #Model Hyperparamenters
    parser.add_argument("--max_length", type=int, default=6000)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--train_epochs", default=5, type=int)
    parser.add_argument("--lr", type=float, default=2e-5)

    # Lora Hyperparameters
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_alpha", type=float, default=16)

    #Speed and memory optimization parameters
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_false", help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.")
    args = parser.parse_args()

    return args

def create_model_and_tokenizer(args : argparse):
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_use_double_quant= False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config= bnb_config,
        device_map= {"": 0}
    )

    #### LLAMA STUFF
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r = args.lora_r,
        lora_alpha= args.lora_alpha,
        lora_dropout= args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj"],
    )

    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    return model, peft_config, tokenizer

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    
    try:
        if torch.cuda.is_available():
            print("Number of available GPUs:", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {name}")
        else:
            raise torch.cuda.CudaError
    except Exception as e:
        print(f"No CUDA devices available: {e}")

    wandb.init(
        project="SemEval_Mistra",
        name = f'{args.model_name}/{args.exp_name}/run-{args.run}',
        group = f'{args.model_name}/{args.exp_name}',
        config = { arg : getattr(args, arg) for arg in vars(args)}
    )

    # Load tokenizer and model
    model, peft_config, tokenizer = create_model_and_tokenizer(args)

    # Load dataset and prompt
    prompt = json.load(open(args.used_prompt))["best_combination_prompt"]
    train_dataset = preprocess_dataset(args, prompt, "train-manual-expand_and_dev")
    eval_dataset = preprocess_dataset(args, prompt, "dev")

    training_arguments = TrainingArguments(
        output_dir = args.save_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit= 5,
        num_train_epochs = args.train_epochs,
        per_device_train_batch_size= args.batch_size,
        optim = "paged_adamw_8bit",
        logging_steps= 25,
        learning_rate= args.lr,
        bf16= False,
        group_by_length= True,
        lr_scheduler_type= "constant",
        #model load
        load_best_model_at_end= True,
        #Speed and memory optimization parameters
        gradient_accumulation_steps= args.gradient_accumulation_steps,
        gradient_checkpointing= args.gradient_checkpointing,
        fp16= args.fp16,
        report_to="wandb"
    )    

    ## Data collator for completing with "Answer: YES" or "Answer: NO"
    collator = DataCollatorForCompletionOnlyLM("Answer:", tokenizer= tokenizer)

    ## Setting sft parameters
    trainer = SFTTrainer(
        model= model,
        data_collator= collator,
        train_dataset= train_dataset,
        eval_dataset= eval_dataset,
        peft_config= peft_config,
        max_seq_length= args.max_length,
        dataset_text_field= "text",
        tokenizer= tokenizer,
        args= training_arguments,
        packing= False,
    )

    ## Training
    trainer.train()

    ## Save model and finish run
    create_path(f'{args.save_dir}end_model/')
    trainer.model.save_pretrained(f'{args.save_dir}end_model/')
    wandb.finish()


if __name__ == '__main__':
    main()
