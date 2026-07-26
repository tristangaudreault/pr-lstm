import argparse
from pathlib import Path

import lm
from datasets import load_dataset
from rich import print
from rich.pretty import pprint
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument("--model")
parser.add_argument("--hidden-size", type=int)
parser.add_argument("--context", "--ctx", type=int, default=512)
parser.add_argument("--total-tokens", type=int, default=100_000_000)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--learning-rate", "--lr", type=float, default=3e-4)
parser.add_argument("--output-dir", "--output", "-o", type=Path, default="runs")

args = parser.parse_args()

pprint(args)

steps = args.total_tokens // (args.context * args.batch_size)
print("steps:", steps)

model = getattr(lm, args.model)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def tokenize(batch):
    return tokenizer(
        batch["text"],
        add_special_tokens=False,
    )


dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

model = model(vocab_size=len(tokenizer), hidden_size=args.hidden_size)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) * 1e-6
print("param_count (M):", param_count)


def group_texts(examples):
    ids = sum(examples["input_ids"], [])

    ids = ids[: len(ids) // args.context * args.context]

    return {
        "input_ids": [
            ids[i : i + args.context] for i in range(0, len(ids), args.context)
        ],
        "labels": [ids[i : i + args.context] for i in range(0, len(ids), args.context)],
    }


dataset = tokenized.map(
    group_texts,
    batched=True,
    remove_columns=tokenized["train"].column_names,
)


training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    max_steps=steps,
    logging_steps=100,
    save_strategy="no",
    eval_strategy="steps",
    eval_steps=1000,
    # torch_compile=True,
    report_to="tensorboard",
    run_name=f"{args.model}-wt103-{param_count}M-ctx{args.context}",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()
