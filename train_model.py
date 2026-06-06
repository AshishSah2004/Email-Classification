import pandas as pd
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download("stopwords")

# Load Dataset
data = pd.read_csv("spam.csv")

# Create Stemmer
stemmer = PorterStemmer()

# Text Cleaning Function
def preprocess(text):

    text = str(text).lower()

    words = text.split()

    clean_words = []

    for word in words:

        if word not in stopwords.words("english"):

            clean_words.append(
                stemmer.stem(word)
            )

    return " ".join(clean_words)

# Apply preprocessing
data["processed_text"] = data["text"].apply(preprocess)

# Convert labels
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

X = data["processed_text"]
y = data["label"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert words to numbers
vectorizer = CountVectorizer()

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Model
model = MultinomialNB()

# Train
model.fit(
    X_train_vector,
    y_train
)

# Prediction
predictions = model.predict(
    X_test_vector
)

# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy :", accuracy)

# Save model
joblib.dump(
    model,
    "spam_model.pkl"
)

# Save vectorizer
joblib.dump(
    vectorizer,
    "vectorizer.pkl"
)

print("Model Saved Successfully")