import onnxruntime as ort
import numpy as np
from transformers import BertTokenizer
from flask import Flask, jsonify, request  # Import Flask

# Initialize Flask app
app = Flask(__name__)

# Load the ONNX model
onnx_model_path = "hate_speech_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def predict_hate_speech(text):
    # Tokenize the input with max_length=128
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128,  # Ensuring input matches model's expected shape
    )

    # Convert to int64
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    # Run inference
    ort_outputs = ort_session.run(None, ort_inputs)
    prediction = np.argmax(ort_outputs[0], axis=1)[0]

    labels = ["hate_speech", "offensive", "neutral"]
    return labels[prediction]


# Define route after app initialization
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    tweet = data["tweet"]
    prediction = predict_hate_speech(tweet)
    return jsonify({"tweet": tweet, "prediction": prediction})


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5001)
