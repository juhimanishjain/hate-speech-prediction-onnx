# Import libraries
import argparse
import onnxruntime as ort
import numpy as np
from transformers import BertTokenizer

# Load the ONNX model
onnx_model_path = "hate_speech_model.onnx"
session = ort.InferenceSession(onnx_model_path)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def get_prediction(text):
    # Tokenize input text
    inputs = tokenizer(
        text, return_tensors="np", padding="max_length", truncation=True, max_length=128
    )

    # Prepare input tensors
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    # Run inference
    ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    ort_outs = session.run(None, ort_inputs)

    # Get prediction
    prediction = np.argmax(ort_outs[0], axis=1)
    labels = ["hate_speech", "offensive", "neutral"]
    return labels[prediction[0]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hate Speech Detection CLI")
    parser.add_argument(
        "tweet", type=str, help="Input tweet text for hate speech detection"
    )
    args = parser.parse_args()

    prediction = get_prediction(args.tweet)
    print(f"Prediction: {prediction}")
