
# TASK 2: Intelligent Multi-Class Natural Language Text Sentiment Classifier

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ---------------------------------------------------------
# 1. Download required NLTK resources
# ---------------------------------------------------------

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------

# Your CSV should contain two columns:
# text      -> the sentence/review
# sentiment -> Positive, Negative, or Neutral

df = pd.read_csv("sentiment_dataset.csv")

print("Dataset:")
print(df.head())

print("\nClass distribution:")
print(df["sentiment"].value_counts())

# Remove missing values
df = df.dropna(subset=["text", "sentiment"])

# ---------------------------------------------------------
# 3. Text preprocessing
# ---------------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


df["clean_text"] = df["text"].apply(preprocess_text)

print("\nProcessed text:")
print(df[["text", "clean_text"]].head())

# ---------------------------------------------------------
# 4. Split data into training and testing sets
# ---------------------------------------------------------

X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# 5. Convert text to numerical data using TF-IDF
# ---------------------------------------------------------

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)

# ---------------------------------------------------------
# 6. Train the classification model
# ---------------------------------------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

# ---------------------------------------------------------
# 7. Make predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test_tfidf)

# ---------------------------------------------------------
# 8. Evaluate the model
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 9. F1-score for each class
# ---------------------------------------------------------

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

print("\nF1-Scores:")

for label in sorted(y.unique()):
    if label in report:
        print(
            f"{label}: "
            f"{report[label]['f1-score']:.4f}"
        )

print(
    "\nWeighted F1-score:",
    round(report["weighted avg"]["f1-score"], 4)
)

# ---------------------------------------------------------
# 10. Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot()
plt.title("Sentiment Classification Confusion Matrix")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 11. Test the classifier with new sentences
# ---------------------------------------------------------

new_sentences = [
    "I absolutely loved this product!",
    "This is the worst experience I have ever had.",
    "The product is okay, nothing special."
]

new_clean = [
    preprocess_text(sentence)
    for sentence in new_sentences
]

new_tfidf = tfidf.transform(new_clean)

predictions = model.predict(new_tfidf)

print("\nNew Sentence Predictions:")

for sentence, prediction in zip(new_sentences, predictions):
    print(f"{sentence} -> {prediction}")

