from typing import cast

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from pr_lstm.torch.model import ParallelRecursive


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = cast(DatasetDict, load_dataset("Salesforce/wikitext", "wikitext-103-v1"))


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
    )


dataset = dataset.map(tokenize, batched=True, remove_columns=["text"], load_from_cache_file=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

model = ParallelRecursive(vocab_size=len(tokenizer), hidden_size=256)

training_args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=16,
    learning_rate=3e-4,
    num_train_epochs=10,
    logging_steps=100,
    save_steps=1000,
    eval_strategy="steps",
    eval_steps=1000,
    # torch_compile=True,
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)

trainer.train()
