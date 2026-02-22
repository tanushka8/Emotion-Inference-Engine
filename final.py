# Import necessary libraries
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st

# -------------------------------
# Streamlit Configuration
# -------------------------------
st.set_page_config(
    page_title="Emotion Inference Engine",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------
# Load and preprocess dataset
# -------------------------------
file_path = 'emotion_dataset.csv'
df = pd.read_csv(file_path)
df = df[['content', 'sentiment']]

def preprocess_text(content):
    content = content.lower()
    content = re.sub(r'http\S+', '', content)
    content = re.sub(r'[^\w\s]', '', content)
    content = re.sub(r'\d+', '', content)
    return content

df['content'] = df['content'].apply(preprocess_text)

# -------------------------------
# Train Models
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['content'], df['sentiment'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression
log_model = LogisticRegression(max_iter=100)
log_model.fit(X_train_vec, y_train)
log_pred = log_model.predict(X_test_vec)
log_accuracy = accuracy_score(y_test, log_pred)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)
nb_accuracy = accuracy_score(y_test, nb_pred)

# -------------------------------
# Gradient Header Styling
# -------------------------------
st.markdown("""
<style>
.header-box {
    background: linear-gradient(90deg, #1f77ff, #8e44ad);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
}
.subtext {
    font-size: 1.1em;
    opacity: 0.9;
}
.stButton>button {
    background-color: #1f77ff;
    color: white;
    border-radius: 8px;
    padding: 8px 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1>Emotion Inference Engine</h1>
    <p class="subtext">Multi-Class Emotion Prediction using Statistical Text Modeling</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Accuracy Dashboard
# -------------------------------
st.markdown("### 📊 Model Performance Overview")
col1, col2 = st.columns(2)
col1.metric("Logistic Regression", f"{log_accuracy:.2%}")
col2.metric("Naive Bayes", f"{nb_accuracy:.2%}")

# Best model highlight
#best_model = "Logistic Regression" if log_accuracy > nb_accuracy else "Naive Bayes"
#st.info(f"🏆 Best Performing Model: {best_model}")

# -------------------------------
# Model Selection
# -------------------------------
selected_model_name = st.selectbox(
    "Select Prediction Model",
    ["Logistic Regression", "Naive Bayes"]
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📈 Evaluation Panel")

metric_option = st.sidebar.radio(
    "Select Metric",
    ["Accuracy", "Classification Report", "Confusion Matrix"]
)

if selected_model_name == "Logistic Regression":
    selected_accuracy = log_accuracy
    selected_pred = log_pred
else:
    selected_accuracy = nb_accuracy
    selected_pred = nb_pred

if metric_option == "Accuracy":
    st.sidebar.write(f"Accuracy: {selected_accuracy:.2%}")
elif metric_option == "Classification Report":
    st.sidebar.text(classification_report(y_test, selected_pred))
else:
    st.sidebar.write(confusion_matrix(y_test, selected_pred))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 About System")
st.sidebar.write("""
• TF-IDF Feature Extraction  
• Logistic Regression  
• Naive Bayes  
• Multi-Class Classification  
• Probability Visualization  
""")

# -------------------------------
# Prediction History Storage
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# User Input
# -------------------------------
user_input = st.text_area(
    "Enter text to analyze emotional tone:",
    placeholder="Type your text here..."
)

#if st.button("Try Example Text"):
  #  user_input = "I am extremely happy and excited about my achievement!"

# -------------------------------
# Prediction Section
# -------------------------------
if st.button("Analyze Emotion"):
    if user_input:

        preprocessed_text = preprocess_text(user_input)
        input_vec = vectorizer.transform([preprocessed_text])

        selected_model = log_model if selected_model_name == "Logistic Regression" else nb_model

        prediction = selected_model.predict(input_vec)[0]
        probabilities = selected_model.predict_proba(input_vec)[0]
        emotions = selected_model.classes_

        # Emotion colors
        emotion_colors = {
            "joy": "#2ecc71",
            "sadness": "#3498db",
            "anger": "#e74c3c",
            "fear": "#9b59b6",
            "love": "#e84393",
            "surprise": "#f1c40f",
            "disgust": "#16a085",
            "trust": "#1abc9c",
            "anticipation": "#e67e22"
        }

        color = emotion_colors.get(prediction, "#cccccc")

        st.markdown(
            f"""
            <div style="background-color:{color};
                        padding:15px;
                        border-radius:10px;
                        text-align:center;
                        font-size:20px;
                        font-weight:bold;">
                Predicted Emotion: {prediction}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence
        confidence = max(probabilities) * 100
        if confidence > 80:
            st.success(f"High Confidence: {confidence:.2f}%")
            st.balloons()
        elif confidence > 60:
            st.warning(f"Moderate Confidence: {confidence:.2f}%")
        else:
            st.error(f"Low Confidence: {confidence:.2f}%")

        # Probability Breakdown
        sorted_probs = sorted(zip(emotions, probabilities), key=lambda x: x[1], reverse=True)

        st.markdown("### 📊 Emotion Probability Breakdown")

        for emotion, prob in sorted_probs:
            percentage = int(prob * 100)
            st.markdown(f"**{emotion} — {percentage}%**")
            st.progress(percentage)

        chart_data = pd.DataFrame(sorted_probs, columns=["Emotion", "Probability"])
        chart_data["Probability"] *= 100
        st.bar_chart(chart_data.set_index("Emotion"))

        # Store history
        st.session_state.history.append({
            "Text": user_input,
            "Model": selected_model_name,
            "Prediction": prediction
        })
        st.session_state.history = st.session_state.history[-5:]

# -------------------------------
# Show History
# -------------------------------
if st.session_state.history:
    st.markdown("### 🕘 Recent Predictions")
    st.table(st.session_state.history)

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<hr>
<center>
Built using NLP & Supervised Machine Learning | Tanushka 
</center>
""", unsafe_allow_html=True)