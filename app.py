import streamlit as st
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

model = joblib.load(
    "spam_model.pkl"
)

vectorizer = joblib.load(
    "vectorizer.pkl"
)

stemmer = PorterStemmer()

def preprocess(text):

    text = text.lower()

    words = text.split()

    clean_words = []

    for word in words:

        if word not in stopwords.words('english'):

            clean_words.append(
                stemmer.stem(word)
            )

    return " ".join(clean_words)

st.title("📧 Email Spam Detection")

email = st.text_area(
    "Paste Email Content"
)

if st.button("Analyze Email"):

    processed = preprocess(email)

    vector = vectorizer.transform(
        [processed]
    )

    prediction = model.predict(
        vector
    )

    if prediction[0] == 1:

        st.error(
            "❌ Spam Email"
        )

    else:

        st.success(
            "✅ Genuine Email"
        )
