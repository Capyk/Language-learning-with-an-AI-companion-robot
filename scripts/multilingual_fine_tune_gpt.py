import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os
import numpy as np

# --- Configuration ---
# ⚠️ ACTION: REPLACE with the exact ID of your 20B GPT-OSS model (e.g., Llama-2-20B)
MODEL_ID = "openai/gpt-oss-20b" 
LOCAL_GEC_DATA_PATH = "/app/data/gec_training_pairs.csv"
OUTPUT_DIR = "./gpt_oss_gec_finetuned"
MAX_SEQ_LENGTH = 512 

# --- 1. Data Preparation and Formatting (Single-String Prompt for GPT-OSS) ---

def format_merlin_for_gpt(row):
    """Formats GEC data into the single Instruction -> Correction prompt."""
    full_prompt = (
        f"[Instruction]: Correct the following German sentence: {row['L2_Text']}\n"
        f"[Correction]: {row['TH1_Correction']}"
    )
    return {'text': full_prompt}

def format_wmt_for_gpt(example):
    """Formats Translation data into the single Instruction -> Translation prompt."""
    en_text = example['translation']['en']
    de_text = example['translation']['de']
    
    full_prompt = (
        f"[Instruction]: Translate this English text to German: {en_text}\n"
        f"[Translation]: {de_text}"
    )
    return {'text': full_prompt}

# --- 2. Tokenization and Data Mapping ---

def tokenize_data(examples):
    """Tokenizes the full prompt (Instruction + Completion) for Causal Language Modeling."""
    # Global tokenizer must be initialized outside this function
    # It learns to predict the next token (the completion) in the entire sequence.
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
    # Using AutoTokenizer loads the correct vocabulary for your specific GPT-OSS base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Add the pad token to the model's vocabulary to avoid errors
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token}) 

    # --- Data Loading and Combining ---
    print("\n--- Loading and Combining Datasets ---")
    
    # 1. Load MERLIN GEC Data and apply instruction formatting
    df_gec = pd.read_csv(LOCAL_GEC_DATA_PATH)
    merlin_dataset = Dataset.from_pandas(df_gec).map(format_merlin_for_gpt, remove_columns=df_gec.columns.tolist())
    
    # 2. Load WMT14 Multilingual Data (Translational competence)
    # Using a small subset of the training data for quick multilingual exposure (10,000 samples)
    wmt_dataset = load_dataset("wmt14", "de-en", split="train[:10000]").map(
        format_wmt_for_gpt, remove_columns=['translation']
    )
    
    # Combine datasets for unified training
    # This creates a larger training set mixing GEC and Translation tasks
    train_ratio = 0.9
    
    # --- Corrected Data Combination Logic ---

    # Split MERLIN into train/eval splits
    merlin_train_len = int(len(merlin_dataset) * 0.9)
    merlin_train_split = merlin_dataset.select(range(merlin_train_len))
    merlin_eval_split = merlin_dataset.select(range(merlin_train_len, len(merlin_dataset)))

    # 1. Create a list of all training datasets (MERLIN GEC + WMT Multilingual)
    all_train_datasets = [
        merlin_train_split,
        wmt_dataset # The full WMT dataset we loaded for multilingual exposure
    ]

    # 2. Concatenate them into a single training set
    raw_combined_dataset = concatenate_datasets(all_train_datasets)

    # Shuffle the combined dataset (essential for good training)
    raw_combined_dataset = raw_combined_dataset.shuffle(seed=42) 

    # Tokenize the combined dataset (no longer a DatasetDict)
    train_dataset = raw_combined_dataset.map(tokenize_data, batched=True, remove_columns=[col for col in raw_combined_dataset.column_names if col != 'text'])

    # Set the evaluation dataset (only MERLIN data)
    eval_dataset = merlin_eval_split.map(tokenize_data, batched=True, remove_columns=[col for col in merlin_eval_split.column_names if col not in ['text']])
    # ---------------------------------------------
    
    # Combine WMT and MERLIN training data
    raw_combined_dataset = DatasetDict({
        'train': merlin_train_split.shuffle().add_batch(wmt_dataset).shuffle()
    })
    
    # Tokenize the combined dataset
    train_dataset = raw_combined_dataset['train'].map(tokenize_data, batched=True, remove_columns=[col for col in raw_combined_dataset['train'].column_names if col != 'text'])
    eval_dataset = merlin_eval_split.map(tokenize_data, batched=True, remove_columns=[col for col in merlin_eval_split.column_names if col not in ['text']])
    
    print(f"Total training samples: {len(train_dataset)}. Total evaluation samples: {len(eval_dataset)}")
    
    # --- Model Loading (QLoRA/PEFT Setup) ---
    print("\n--- Loading Model with QLoRA (Critical for 20B Model) ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)
    
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