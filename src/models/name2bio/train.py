import os
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, DatasetDict

# Set up tokenizer and model name constants
PRETRAINED_MODEL = "distilgpt2"
DEFAULT_MAX_LENGTH = 200  # roughly 95th percentile of all texts above 50 tokens, from prior analysis.

def bio_length(bio, tokenizer):
    return len(tokenizer(bio)["input_ids"])

def chunk_text(text, tokenizer, max_length=DEFAULT_MAX_LENGTH, stride=50):
    """Splits text into overlapping chunks while preserving context."""
    tokens = tokenizer(text)["input_ids"]
    chunks = [tokens[i : i + max_length] for i in range(0, len(tokens), max_length - stride)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def format_for_gpt2(name, bio, tokenizer, max_length=DEFAULT_MAX_LENGTH):
    bio_chunks = chunk_text(bio, tokenizer, max_length)
    return [f"[CHARACTER] {name}\n[BIO] {chunk} [END]" for chunk in bio_chunks]

def load_and_preprocess_data(csv_path, tokenizer, token_threshold=50, max_length=DEFAULT_MAX_LENGTH, seed=42):
    # Load data from CSV; ensure there is a 'name' and 'bio' column.
    data = pd.read_csv(csv_path).fillna('')
    data = data[data['bio'] != '']

    bio_lengths = {}
    for bio in tqdm(data["bio"], desc="Calculating bio lengths"):
        bio_lengths[bio] = bio_length(bio, tokenizer)

    # Filter out texts below the token threshold
    data_long = data[~data["bio"].apply(lambda x: bio_lengths[x] < token_threshold)]

    # Create preprocessed texts
    preprocessed_texts = []
    rnd = random.Random(seed)
    for _, row in tqdm(data_long.iterrows(), total=len(data_long), desc="Preprocessing texts"):
        name = row['name']
        bio = row['bio']
        preprocessed_texts += format_for_gpt2(name, bio, tokenizer, max_length)
    
    rnd.shuffle(preprocessed_texts)
    
    # Split into train/eval sets
    split_ratio = 0.9
    split_idx = int(len(preprocessed_texts) * split_ratio)
    train_texts = preprocessed_texts[:split_idx]
    eval_texts = preprocessed_texts[split_idx:]
    
    dataset = DatasetDict({
        "train": Dataset.from_dict({"text": train_texts}),
        "eval": Dataset.from_dict({"text": eval_texts})
    })
    
    return dataset

def tokenize_function(examples, tokenizer, max_length=DEFAULT_MAX_LENGTH):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

def main():
    parser = argparse.ArgumentParser(description="Train DistilGPT2 for bio generation")
    parser.add_argument("--csv_path", type=str, default="data/name_bio.csv", help="Path to CSV file containing name and bio data")
    parser.add_argument("--output_dir", type=str, default="./working/gpt2-bio-generator_checkpoints", help="Directory to store checkpoints and final model")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum token length for each example")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess data
    print("Loading and preprocessing data...")
    dataset = load_and_preprocess_data(args.csv_path, tokenizer, max_length=args.max_length, seed=args.seed)
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer, max_length=args.max_length), batched=True, remove_columns=["text"])

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=True if torch.cuda.is_available() else False,
        logging_strategy='steps',
        logging_steps=50,
        report_to="none",
        dataloader_pin_memory=True,
        dataloader_num_workers=8,
        ddp_find_unused_parameters=False,
    )

    # Load model (either from checkpoint or from scratch)
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = GPT2LMHeadModel.from_pretrained(args.checkpoint)
    else:
        print("Loading model from scratch (pretrained DistilGPT2)...")
        model = GPT2LMHeadModel.from_pretrained(PRETRAINED_MODEL)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = model.to(device)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=data_collator,
    )

    # Train the model
    trainer.train(resume_from_checkpoint=args.checkpoint if args.checkpoint else None)

    # Save the best model
    final_output_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Model and tokenizer saved to {final_output_dir}")

if __name__ == "__main__":
    main()
