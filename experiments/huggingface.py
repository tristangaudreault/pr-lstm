import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import lm

from tag_utils import branch

SEQ_LEN = 512
TOKENS_TARGET = 100_000_000
BATCH_SIZE = 16

steps = TOKENS_TARGET // (SEQ_LEN * BATCH_SIZE)


parser = argparse.ArgumentParser()
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

if args.count_params:
    print(
        f"Hidden Size: {hidden_size}, Parameter count (M): {sum(p.numel() for p in model.parameters() if p.requires_grad) * 1e-6}"
    )
    exit()


def group_texts(examples):
    ids = sum(examples["input_ids"], [])

    ids = ids[: len(ids) // SEQ_LEN * SEQ_LEN]

    return {
        "input_ids": [ids[i : i + SEQ_LEN] for i in range(0, len(ids), SEQ_LEN)],
        "labels": [ids[i : i + SEQ_LEN] for i in range(0, len(ids), SEQ_LEN)],
    }


dataset = tokenized.map(
    group_texts,
    batched=True,
    remove_columns=tokenized["train"].column_names,
)


training_args = TrainingArguments(
    output_dir="./out",
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=3e-4,
    max_steps=steps,
    logging_steps=100,
    save_strategy="no",
    eval_strategy="steps",
    eval_steps=1000,
    torch_compile=True,
    report_to="tensorboard",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()
