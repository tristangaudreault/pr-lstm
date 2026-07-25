import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import lm

from tag_utils import branch

CTX = 512
TOKENS_TARGET = 100_000_000
BATCH_SIZE = 16
LR = 3e-4

steps = TOKENS_TARGET // (CTX * BATCH_SIZE)


parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument(
    "--hidden-size",
    type=int,
    help="Hidden size.",
    nargs="+",
)
parser.add_argument(
    "--count-params",
    action="store_true",
    help="Prints parameter count and exits before training",
)
parser.add_argument(
    "--early-exit", "--ee", nargs="*", default=[], help="Early exit breakpoints."
)
parser.add_argument("--model", help="Name of model class")
args = parser.parse_args()


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

hidden_size = branch(args.hidden_size)
model = model(vocab_size=len(tokenizer), hidden_size=hidden_size)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) * 1e-6
if "param_count" in args.early_exit:
    print(args.model, "param_count (M):", param_count)
    exit()


def group_texts(examples):
    ids = sum(examples["input_ids"], [])

    ids = ids[: len(ids) // CTX * CTX]

    return {
        "input_ids": [ids[i : i + CTX] for i in range(0, len(ids), CTX)],
        "labels": [ids[i : i + CTX] for i in range(0, len(ids), CTX)],
    }


dataset = tokenized.map(
    group_texts,
    batched=True,
    remove_columns=tokenized["train"].column_names,
)


training_args = TrainingArguments(
    output_dir="./out",
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    max_steps=steps,
    logging_steps=100,
    save_strategy="no",
    eval_strategy="steps",
    eval_steps=1000,
    torch_compile=True,
    report_to="tensorboard",
    run_name=f"{args.model}-wt103-{param_count}M-ctx{CTX}",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()
