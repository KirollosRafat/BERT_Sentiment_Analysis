import streamlit as st
import torch
import joblib
import numpy as np
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download
from model import SentimentClassifier

# Load Model 
@st.cache_resource
def load_model_and_components():
    try:
        # Download from Hugging Face model repo
        MODEL_REPO = "KR5/sentiment-bert"

        # Label encoder
        label_encoder_path = hf_hub_download(repo_id=MODEL_REPO, filename="label_encoder.pkl")
        label_encoder = joblib.load(label_encoder_path)

        # Model weights
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename="bert_sentiment.pt")

        # Tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Initialize model
        num_labels = len(label_encoder.classes_)
        model = SentimentClassifier(num_labels=num_labels)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        return model, tokenizer, label_encoder

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None


# Make Predictions
def predict_sentiment(text, model, tokenizer, label_encoder):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(encoding["input_ids"], encoding["attention_mask"])
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label, confidence, probabilities[0].numpy()


# Streamlit UI
def main():
    st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊", layout="centered")
    st.title("🎭 Sentiment Analysis App")
    st.markdown("Enter text below to analyze its sentiment!")

    model, tokenizer, label_encoder = load_model_and_components()
    if model is None:
        st.stop()

    user_input = st.text_area("Enter text to analyze:", placeholder="Type something here...", height=100)

    if st.button("🔍 Analyze Sentiment"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                try:
                    predicted_label, confidence, all_probabilities = predict_sentiment(
                        user_input, model, tokenizer, label_encoder
                    )

                    st.success("✅ Analysis Complete!")

                    col1, col2 = st.columns(2)
                    col1.metric("Predicted Sentiment", predicted_label)
                    col2.metric("Confidence", f"{confidence:.2%}")

                    # Probabilities chart
                    st.subheader("Class Probabilities")
                    prob_data = {cls: float(all_probabilities[i]) for i, cls in enumerate(label_encoder.classes_)}
                    st.bar_chart(prob_data)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.warning("Please enter text first!")

    # Example texts
    st.markdown("---")
    st.subheader("Examples")
    examples = [
        "I absolutely love this product! It exceeded all my expectations.",
        "This movie was terrible. I want my money back.",
        "The service was okay, nothing special but not bad either.",
        "I'm feeling really happy today!",
        "This is the worst experience I've ever had."
    ]
    for i, ex in enumerate(examples):
        if st.button(f"Example {i+1}"):
            st.session_state["example_text"] = ex
            st.experimental_rerun()


if __name__ == "__main__":
    main()
