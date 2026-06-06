import streamlit as st
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load saved model
model = joblib.load("models/spam_model.pkl")

vectorizer = joblib.load("models/vectorizer.pkl")

stemmer = PorterStemmer()

nltk.download("stopwords")
danger_words = [
    "free",
    "winner",
    "offer",
    "click",
    "claim",
    "urgent",
    "verify",
    "reward",
    "bank",
    "password",
    "otp",
    "prize"
]

def preprocess(text):

    text = text.lower()

    words = text.split()

    clean_words = []

    for word in words:

        if word not in stopwords.words("english"):

            clean_words.append(
                stemmer.stem(word)
            )

    return " ".join(clean_words)

st.title("📧 Email Spam Detection System")

st.write(
    "Check whether an email is Spam or Legitimate"
)

email = st.text_area(
    "Paste Email Content"
)

if st.button("Analyze Email"):

    processed = preprocess(
        email
    )
    found_words = []

email_lower = email.lower()

for word in danger_words:

    if word in email_lower:

        found_words.append(word)

    vector = vectorizer.transform(
        [processed]
    )

    prediction = model.predict(
        vector
    )
    probability = model.predict_proba(vector)

    spam_score = round(
    probability[0][1] * 100,
    2
    )
    if spam_score >= 80:

        risk = "HIGH"

    elif spam_score >= 50:

        risk = "MEDIUM"

    else:

        risk = "LOW"

    if prediction[0] == 1:
       st.error("❌ Spam Email")
    st.metric(
    "Spam Probability",
    f"{spam_score}%"
    )

    st.metric(
    "Risk Level",
    risk
    )

    if found_words:

        st.subheader("⚠ Suspicious Words Found")

        for word in found_words:

            st.write("•", word)

    else:

        st.success(
            "✅ Legitimate Email"
        )