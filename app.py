from flask import Flask, render_template, request
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


app = Flask(__name__)


# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv("sentiment_dataset.csv")

df = df.dropna(subset=["text", "sentiment"])


# -----------------------------
# Text preprocessing
# -----------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


df["clean_text"] = df["text"].apply(preprocess_text)


# -----------------------------
# TF-IDF
# -----------------------------

X = df["clean_text"]
y = df["sentiment"]

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(X)


# -----------------------------
# Train model
# -----------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_tfidf, y)


# -----------------------------
# Website
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    user_text = ""

    if request.method == "POST":

        user_text = request.form["text"]

        if user_text.strip():

            cleaned_text = preprocess_text(user_text)

            text_tfidf = tfidf.transform([cleaned_text])

            prediction = model.predict(text_tfidf)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        user_text=user_text
    )


# -----------------------------
# Start server
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)