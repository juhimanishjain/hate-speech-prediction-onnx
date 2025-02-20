# Import the libraries
import pandas as pd
import torch
import torch.onnx
import os
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

# Load the dataset
url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
df = pd.read_csv(url)

# Preprocess the dataset
df = df[["tweet", "class"]]
label_dict = {0: "hate_speech", 1: "offensive", 2: "neutral"}
df["class"] = df["class"].map(label_dict)
df.rename(columns={"class": "labels"}, inplace=True)
label_map = {"hate_speech": 0, "offensive": 1, "neutral": 2}
df["labels"] = df["labels"].map(label_map)

# Split the dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load pre-trained BERT model & tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)


# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["tweet"], truncation=True, padding="max_length", max_length=128
    )


# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Convert dataset format to PyTorch tensors
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation results:", results)

# Export the trained model to ONNX
onnx_model_path = "hate_speech_model.onnx"

# Move model to CPU before exporting (Avoids MPS/CUDA errors)
device = torch.device("cpu")
model.to(device)
model.eval()  # Ensure model is in evaluation mode

# Use correct dummy input formatting (Explicit dtype)
dummy_input = {
    "input_ids": torch.randint(0, 1000, (1, 128), dtype=torch.long, device=device),
    "attention_mask": torch.ones((1, 128), dtype=torch.long, device=device),
}

# Use torch.no_grad() to prevent unnecessary computation tracking
try:
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            onnx_model_path,
            export_params=True,
            opset_version=13,
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
    print(f"Model successfully exported to ONNX format at: {onnx_model_path}")

except Exception as e:
    print(f"ONNX export failed: {str(e)}")
