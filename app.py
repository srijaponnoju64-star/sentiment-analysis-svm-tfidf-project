"""Streamlit web app for sentiment analysis - Compare SVM vs Pre-trained Transformer."""
import streamlit as st
import joblib
from scipy.special import softmax
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Sentiment Analyzer", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_svm_model():
    """Load traditional SVM + TF-IDF model."""
    try:
        model = joblib.load('models/svm_tfidf.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except:
        st.error("SVM model not found. Run: python train_all.py")
        return None, None

@st.cache_resource
def load_transformer_model():
    """Load pre-trained transformer model."""
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load both models
svm_model, svm_vectorizer = load_svm_model()
transformer = load_transformer_model()

# ============================================================
# APP LAYOUT
# ============================================================

# Title and description
st.title("🎯 Sentiment Analysis Platform")
st.markdown("Compare **Traditional ML vs Deep Learning** for Twitter sentiment classification")
st.divider()

# Sidebar: Model selection
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio(
    "Select Model:",
    ["SVM (Traditional ML)", "Pre-trained Transformer", "Compare Both"],
    help="Choose which model to use for predictions"
)

st.sidebar.divider()

st.sidebar.subheader("📊 Model Comparison")
st.sidebar.markdown("""
| Model | Accuracy | Speed |
|-------|----------|-------|
| **SVM + TF-IDF** | 64.12% | ⚡ Fast |
| **DistilBERT** | ~82-85% | 🚀 Medium |
""")

# ============================================================
# MAIN INPUT SECTION
# ============================================================

st.subheader("📝 Enter Text to Analyze")
user_input = st.text_area(
    "Your text:",
    height=120,
    placeholder="Type or paste a tweet, review, or any text here..."
)

# ============================================================
# PREDICTION LOGIC
# ============================================================

if st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        st.divider()
        
        # ============================================================
        # SVM PREDICTION
        # ============================================================
        
        if model_choice in ["SVM (Traditional ML)", "Compare Both"]:
            if svm_model is None:
                st.error("❌ SVM model not loaded. Run: python train_all.py")
            else:
                st.subheader("🤖 SVM (Traditional ML) - TF-IDF")
                
                text_vec = svm_vectorizer.transform([user_input])
                svm_pred = svm_model.predict(text_vec)[0]
                svm_scores = svm_model.decision_function(text_vec)[0]
                svm_prob = softmax(svm_scores)
                svm_confidence = max(svm_prob) * 100
                
                # Map probabilities to labels
                label_order = ["Negative", "Neutral", "Positive"]
                svm_probs_dict = {label: prob*100 for label, prob in zip(label_order, svm_prob)}
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    emoji = "🔴" if svm_pred == "Negative" else "🟡" if svm_pred == "Neutral" else "🟢"
                    st.metric("Prediction", f"{emoji} {svm_pred}")
                with col2:
                    st.metric("Confidence", f"{svm_confidence:.1f}%")
                with col3:
                    st.metric("Model Accuracy", "64.12%")
                with col4:
                    st.metric("Processing", "⚡ Fast")
                
                # Probability breakdown
                st.markdown("**Confidence Breakdown:**")
                st.bar_chart(svm_probs_dict)
        
        # ============================================================
        # TRANSFORMER PREDICTION
        # ============================================================
        
        if model_choice in ["Pre-trained Transformer", "Compare Both"]:
            st.divider()
            st.subheader("🧠 DistilBERT (Pre-trained Transformer)")
            
            transformer_result = transformer(user_input)[0]
            transformer_label = transformer_result['label']  # POSITIVE or NEGATIVE
            transformer_score = transformer_result['score'] * 100
            
            # Map to 3 classes
            if transformer_label == "POSITIVE":
                transformer_sentiment = "Positive"
            elif transformer_label == "NEGATIVE":
                transformer_sentiment = "Negative"
            else:
                transformer_sentiment = "Neutral"
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                emoji = "🔴" if transformer_sentiment == "Negative" else "🟡" if transformer_sentiment == "Neutral" else "🟢"
                st.metric("Prediction", f"{emoji} {transformer_sentiment}")
            with col2:
                st.metric("Confidence", f"{transformer_score:.1f}%")
            with col3:
                st.metric("Model Accuracy", "82-85%")
            with col4:
                st.metric("Processing", "🚀 Medium")
            
            # Confidence visualization
            st.markdown("**Confidence Score:**")
            st.progress(transformer_score / 100, text=f"{transformer_score:.1f}%")
        
        # ============================================================
        # COMPARISON
        # ============================================================
        
        if model_choice == "Compare Both":
            st.divider()
            st.subheader("📊 Side-by-Side Comparison")
            
            comparison_df = {
                "Metric": ["Prediction", "Confidence", "Model", "Accuracy", "Speed"],
                "SVM (TF-IDF)": [
                    svm_pred,
                    f"{svm_confidence:.1f}%",
                    "Traditional ML",
                    "64.12%",
                    "⚡ Fast"
                ],
                "DistilBERT": [
                    transformer_sentiment,
                    f"{transformer_score:.1f}%",
                    "Deep Learning",
                    "82-85%",
                    "🚀 Medium"
                ]
            }
            
            st.table(comparison_df)
            
            # Agreement check
            svm_matches = svm_pred == transformer_sentiment
            if svm_matches:
                st.success(f"✅ **Both models agree:** {svm_pred}")
            else:
                st.warning(f"⚠️ **Models disagree:** SVM says {svm_pred}, DistilBERT says {transformer_sentiment}")
        
        # ============================================================
        # INPUT DISPLAY
        # ============================================================
        
        st.divider()
        st.subheader("📌 Your Input")
        st.info(f'"{user_input}"')

# ============================================================
# SIDEBAR: INSTRUCTIONS & INFO
# ============================================================

st.sidebar.divider()
st.sidebar.subheader("📚 How to Use")
st.sidebar.markdown("""
1. **Enter text** in the main area
2. **Select a model** from the options above
3. **Click "Analyze Sentiment"** to get predictions

**Models Available:**
- **SVM + TF-IDF:** Fast, traditional machine learning
- **DistilBERT:** Accurate, pre-trained transformer
- **Compare Both:** See predictions from both models
""")

st.sidebar.divider()
st.sidebar.subheader("❓ FAQ")
with st.sidebar.expander("What's SVM?"):
    st.write("Support Vector Machine - Traditional ML algorithm. Fast & efficient.")

with st.sidebar.expander("What's DistilBERT?"):
    st.write("Pre-trained transformer model. Understands context better. More accurate.")

with st.sidebar.expander("Which model is better?"):
    st.write("DistilBERT (82-85%) is more accurate, but SVM is faster. Choose based on your needs.")

# ============================================================
# FOOTER
# ============================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🎯 Sentiment Classifier**
    - Dataset: 59,870 tweets
    - Classes: Negative, Neutral, Positive
    """)

with col2:
    st.markdown("""
    **📊 Model Details**
    - SVM: 64.12% accuracy
    - DistilBERT: 82-85% accuracy
    """)

with col3:
    st.markdown("""
    **🔧 Tech Stack**
    - Scikit-learn (SVM)
    - Transformers (DistilBERT)
    - Streamlit (UI)
    """)

st.markdown("---")
st.caption("Created by Srija | GRIET Data Science | Sept 2026")