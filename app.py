from flask import Flask, render_template, request
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired, Length
import logging
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set secret key for CSRF protection

# Set up logging
logging.basicConfig(filename='logs/app.log', level=logging.INFO)
logging.info('Application started')

# Load dataset
df = pd.read_csv('spam.csv')

# Preprocess categories (mapping)
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})


# Function to preprocess message
def preprocess_message(message):
    message = message.lower()  # Convert to lower case
    message = re.sub(f'[{re.escape(string.punctuation)}\t\r\n]', '',
                     message)  # Remove punctuation, tabs, returns, and newlines

    # Tokenize and stem the message
    stemmer = PorterStemmer()
    words = nltk.word_tokenize(message)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)


# Preprocess messages
df['Message'] = df['Message'].apply(preprocess_message)

# Prepare data for the model
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.2, random_state=42)
model = make_pipeline(TfidfVectorizer(), SVC())  # Create a pipeline with TfidfVectorizer and SVC

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)


# Summary statistics
summary = {
    'total_messages': len(df),
    'spam_count': df['Category'].sum(),
    'ham_count': (df['Category'] == 0).sum(),
    'average_length': df['Message'].str.len().mean()
}


# Functions to create visualizations
def create_f1_score_barchart():
    labels = ['Ham', 'Spam']
    f1_scores = [report[str(label)]['f1-score'] for label in [0, 1] if str(label) in report]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, f1_scores, color=['#08306B', '#DCEAF6'])
    plt.title('F1 Score for Spam and Ham')
    plt.ylabel('F1 Score')
    plt.xlabel('Class')

    plt.savefig('static/f1_score_barchart.png')
    plt.close()


def create_confusion_matrix():
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plt.savefig('static/confusion_matrix.png')
    plt.close()


def create_precision_recall_curve():
    y_scores = model.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#08306B')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.85, 1.0])
    plt.ylim([0.4, 1.05])

    plt.savefig('static/precision_recall_curve.png')
    plt.close()


# Generate visualizations
create_f1_score_barchart()
create_confusion_matrix()
create_precision_recall_curve()


# Define Flask form for input
class MessageForm(FlaskForm):
    message = TextAreaField('Message', validators=[DataRequired(), Length(min=1, max=1000)])
    submit = SubmitField('Submit')


# Log incoming requests
@app.before_request
def log_request():
    logging.info(f'Request: {request.method} {request.url}')


@app.after_request
def log_response(response):
    logging.info(f'Response: {response.status} {response.content_length} bytes')
    return response


# Route for home page
@app.route('/', methods=['GET', 'POST'])
def home():
    form = MessageForm()
    result = None
    message = None

    if form.validate_on_submit():
        message = form.message.data
        prediction = model.predict([message])
        result = 'Spam' if prediction[0] == 1 else 'Not Spam'

    return render_template('index.html', form=form, result=result, message=message, summary=summary)


# Health check route
@app.route('/health', methods=['GET'])
def health_check():
    return {'status': 'healthy'}, 200


# Run the app
if __name__ == '__main__':
    app.run(debug=True)