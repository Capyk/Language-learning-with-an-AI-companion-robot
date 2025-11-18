import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os
import numpy as np

# --- Configuration ---
MODEL_ID = "openai/gpt-oss-20b" 
LOCAL_GEC_DATA_PATH = "/app/data/gec_training_pairs.csv"
OUTPUT_DIR = "./gpt_oss_gec_finetuned"
MAX_SEQ_LENGTH = 512 

# --- 1. Data Preparation and Formatting (Single-String Prompt for GPT-OSS) ---

def format_merlin_for_gpt(row):
    """Formats MERLIN GEC data into the single Instruction -> Correction prompt."""
    full_prompt = (
        f"[Instruction]: Correct the following German sentence: {row['L2_Text']}\n"
        f"[Correction]: {row['TH1_Correction']}"
    )
    return {'text': full_prompt}

def format_wmt_for_gpt(example):
    """Formats WMT Translation data into the single Instruction -> Translation prompt."""
    en_text = example['translation']['en']
    de_text = example['translation']['de']
    
    full_prompt = (
        f"[Instruction]: Translate this English text to German: {en_text}\n"
        f"[Translation]: {de_text}"
    )
    return {'text': full_prompt}

# --- 2. Tokenization and Data Mapping ---

def tokenize_data(examples):
    """Tokenizes the full prompt (Instruction + Completion) for Causal Language Modeling (CLM)."""
    # Global tokenizer must be initialized outside this function
    return tokenizer(
        examples['text'], 
        truncation=True, 
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )

# --- 3. Main Training Execution ---

def run_fine_tuning():
    print("--- Initializing Tokenizer and Model ---")
    global tokenizer
    # Loads the tokenizer configuration (either from HF Hub or local path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Configure tokenizer for CLM (Decoder-Only) models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token}) 

    # --- Data Loading and Combining (MERLIN GEC + WMT Multilingual) ---
    print("\n--- Loading and Combining Datasets (MERLIN + WMT14) ---")
    
    # 1. Load MERLIN GEC Data
    df_gec = pd.read_csv(LOCAL_GEC_DATA_PATH)
    merlin_dataset = Dataset.from_pandas(df_gec).map(format_merlin_for_gpt, remove_columns=df_gec.columns.tolist())
    
    # 2. Load WMT14 Multilingual Data (for general linguistic competence)
    wmt_dataset = load_dataset("wmt14", "de-en", split="train[:10000]").map(
        format_wmt_for_gpt, remove_columns=['translation']
    )
    
    # --- Split and Concatenate Logic ---
    train_ratio = 0.9
    merlin_train_len = int(len(merlin_dataset) * train_ratio)
    merlin_train_split = merlin_dataset.select(range(merlin_train_len))
    merlin_eval_split = merlin_dataset.select(range(merlin_train_len, len(merlin_dataset)))

    # Create the final unified training dataset by concatenating MERLIN and WMT
    all_train_datasets = [
        merlin_train_split,
        wmt_dataset 
    ]
    raw_combined_dataset = concatenate_datasets(all_train_datasets)
    raw_combined_dataset = raw_combined_dataset.shuffle(seed=42) 

    # Tokenize the combined training dataset
    train_dataset = raw_combined_dataset.map(tokenize_data, batched=True, remove_columns=[col for col in raw_combined_dataset.column_names if col != 'text'])

    # Tokenize the evaluation dataset (only MERLIN data for relevant evaluation)
    eval_dataset = merlin_eval_split.map(tokenize_data, batched=True, remove_columns=[col for col in merlin_eval_split.column_names if col not in ['text']])
    
    print(f"Total training samples: {len(train_dataset)}. Total evaluation samples: {len(eval_dataset)}")
    
    # --- Model Loading (QLoRA/PEFT Setup) ---
    print("\n--- Loading Model with QLoRA (Critical for 20B Model) ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        load_in_4bit=True, # Use 4-bit quantization
        torch_dtype=torch.bfloat16 # Use bfloat16 for performance
    )
    model = prepare_model_for_kbit_training(model)
    
    # LoRA Configuration
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 

    # --- Training ---
    print("\n--- Starting Supervised Fine-Tuning (SFT) ---")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3, 
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=16, # Simulates a batch size of 32
        learning_rate=2e-4,
        bf16=True, 
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    
    # --- Final Save ---
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora_weights"))
    print(f"\nTraining COMPLETE. Final weights saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    # Ensure you install required libraries on the remote machine:
    # pip install datasets transformers accelerate peft bitsandbytes pandas torch
    run_fine_tuning()