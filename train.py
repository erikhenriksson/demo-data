import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import numpy as np

from transformers import EarlyStoppingCallback

# Load and prepare data
df = pd.read_csv("fincore_simplified.tsv", sep="\t")
le = LabelEncoder()
df["label"] = le.fit_transform(df["register"])

# Create stratified splits
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

# Create datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Print dataset sizes
print(f"\nDataset sizes:")
print(f"Training set: {len(train_dataset)} samples")
print(f"Validation set: {len(val_dataset)} samples")
print(f"Test set: {len(test_dataset)} samples")

# Print class distribution
print("\nClass distribution:")
for split_name, df in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Test", test_df),
]:
    print(f"\n{split_name} set:")
    class_dist = df["label"].value_counts()
    for class_name, count in class_dist.items():
        print(f"{class_name}: {count} samples ({count/len(df)*100:.1f}%)")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
model = AutoModelForSequenceClassification.from_pretrained(
    "TurkuNLP/bert-base-finnish-cased-v1", num_labels=len(le.classes_)
)


# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512
    )


# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)


# Compute metrics function
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, predictions, average="weighted")}


# Training arguments
training_args = TrainingArguments(
    output_dir="finnish_bert_finetuned",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_strategy="steps",
    save_steps=100,
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Train the model
trainer.train()

# Evaluate on test set
predictions = trainer.predict(tokenized_test)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

# Generate and print classification report
class_names = le.classes_
report = classification_report(labels, preds, target_names=class_names)
print("\nClassification Report on Test Set:")
print(report)

# Save the model
trainer.save_model("finnish_bert_finetuned/final_model")
