import streamlit as st
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import joblib

# --------------------------
# Load Model & Components
# --------------------------
@st.cache_resource
def load_model_and_components():
    try:
        LABEL_ENCODER_PATH = "label_encoder.pkl"
        label_encoder = joblib.load(LABEL_ENCODER_PATH)

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(label_encoder.classes_),
            torch_dtype=torch.float16
        )
        model.eval()

        return model, tokenizer, label_encoder

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

# --------------------------
# Prediction Function
# --------------------------
def predict_sentiment(text, model, tokenizer, label_encoder):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoding)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label, confidence, probabilities[0].cpu().numpy()

# --------------------------
# Streamlit UI
# --------------------------
def main():
    st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊", layout="centered")
    st.title("🎭 Sentiment Analysis App")
    st.markdown("Enter text below to analyze its sentiment!")

    # Initialize session_state for input
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    model, tokenizer, label_encoder = load_model_and_components()
    if model is None:
        st.stop()

    user_input = st.text_area(
        "Enter text to analyze:",
        value=st.session_state.user_input,
        placeholder="Type something here...",
        height=100
    )

    # Update session_state
    st.session_state.user_input = user_input

    if st.button("🔍 Analyze Sentiment"):
        if st.session_state.user_input.strip():
            with st.spinner("Analyzing..."):
                try:
                    predicted_label, confidence, all_probabilities = predict_sentiment(
                        st.session_state.user_input, model, tokenizer, label_encoder
                    )

                    st.success("✅ Analysis Complete!")
                    col1, col2 = st.columns(2)
                    col1.metric("Predicted Sentiment", predicted_label)
                    col2.metric("Confidence", f"{confidence:.2%}")

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
            # Instead of rerun, just set session_state and update text_area
            st.session_state.user_input = ex

# --------------------------
# Run App
# --------------------------
if __name__ == "__main__":
    main()
