
import torch
from transformers import BertTokenizer
import joblib
from model import SentimentClassifier

# Load tokenizer and label encoder
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
label_encoder = joblib.load("label_encoder.pkl")

# Load model
num_labels = len(label_encoder.classes_)
model = SentimentClassifier(num_labels=num_labels)
model.load_state_dict(torch.load("bert_sentiment.pt", map_location="cpu"))
model.eval()

def predict_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        pred_id = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([pred_id])[0]

# Test
sample = "I love this product!"
print(f"'{sample}' → {predict_sentiment(sample)}")

sample = "This is the worst service ever."
print(f"'{sample}' → {predict_sentiment(sample)}")

sample = "Just a regular day."
print(f"'{sample}' → {predict_sentiment(sample)}")

sample = "I`m worried about you"
print(f"'{sample}' → {predict_sentiment(sample)}")

sample = "They will soon learn"
print(f"'{sample}' → {predict_sentiment(sample)}")

sample = "You made my day"
print(f"'{sample}' → {predict_sentiment(sample)}")