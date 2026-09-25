# AI Sentiment Classifier

An AI-powered web application that takes text entered by the user and determines whether the sentiment is **Positive, Negative, or Neutral**.

## Features

- User can enter their own text
- Analyzes the user's text using machine learning
- Classifies sentiment as Positive, Negative, or Neutral
- Simple and attractive web interface
- Real-time prediction through a Flask web application

## Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- NLTK
- HTML
- CSS

## Project Structure

```text
AI-Sentiment-Classifier/
│
├── templates/
│   └── index.html
│
├── app.py
├── sentiment_classifier.py
├── sentiment_dataset.csv
├── requirements.txt
└── .gitignore
```

## How It Works

1. The user enters text into the input box.
2. The application receives the user's text.
3. The sentiment classifier processes the text.
4. The machine learning model predicts the sentiment.
5. The result is displayed as **Positive**, **Negative**, or **Neutral**.

## Example

### Positive Sentiment

**Input:**
```text
I really enjoyed this product. It is amazing!
```

**Prediction:**
```text
Positive
```

### Negative Sentiment

**Input:**
```text
I am very disappointed with this service.
```

**Prediction:**
```text
Negative
```

### Neutral Sentiment

**Input:**
```text
The product was delivered today.
```

**Prediction:**
```text
Neutral
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Tuhintech1278/AI-Sentiment-Classifier.git
```

### 2. Open the project folder

```bash
cd AI-Sentiment-Classifier
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open the website

Open your browser and go to:

```text
http://127.0.0.1:5000
```

## Important

This project runs locally using Flask and is **not currently deployed online**.

## Future Improvements

- Improve model accuracy with a larger dataset
- Add more sentiment categories
- Add prediction confidence scores
- Improve the user interface
- Deploy the application online

## Author

**Tuhin Mondal**

GitHub: https://github.com/Tuhintech1278
