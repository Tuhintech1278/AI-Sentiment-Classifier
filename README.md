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
How It Works
The user enters a sentence or text.
The application receives the user's input.
The sentiment classifier analyzes the text.
The model predicts the sentiment.
The result is displayed as Positive, Negative, or Neutral.
Example
User Input
I really enjoyed this product. It is amazing!
Prediction
Positive

Another example:

User Input
I am very disappointed with this service.
Prediction
Negative
How to Run the Project
1. Clone the repository
git clone https://github.com/Tuhintech1278/AI-Sentiment-Classifier.git
2. Open the project folder
cd AI-Sentiment-Classifier
3. Install the required packages
pip install -r requirements.txt
4. Run the application
python app.py
5. Open the application

Open your browser and visit:

http://127.0.0.1:5000
Important

This project runs locally using Flask. It is not currently deployed online.

Author

Tuhin Mondal

GitHub: https://github.com/Tuhintech1278
