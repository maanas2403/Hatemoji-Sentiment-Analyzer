import numpy as np
from textblob import TextBlob
from flask import Flask, request, render_template, jsonify
import re
import string
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM, Bidirectional, Embedding, GlobalMaxPool1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import Counter
import gdown
import os

app = Flask(__name__, template_folder='templates')

# Define the tokenizer
tokenizer = Tokenizer(num_words=80000)
maxlen = 40  # Ensure this matches what you used during training

# Define the model architecture
def create_model():
    model = Sequential([
        Embedding(input_dim=21613, output_dim=128),
        Bidirectional(LSTM(100, return_sequences=True)),
        Bidirectional(LSTM(100, return_sequences=True)),
        GlobalMaxPool1D(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(40, activation='softmax')  # Assuming 40 emojis in your label set
    ])
    return model

# Function to download model weights from Google Drive
def download_model_from_drive(file_id, local_filename='BTP_eval.weights.h5'):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, local_filename, quiet=False)

# Download the model weights from Google Drive if not already present
weights_file_path = 'BTP_eval.weights.h5'
if not os.path.exists(weights_file_path):
    google_drive_file_id = '1r--l8HrNN2p7ldXvAFeHVA_3njoQSYio'  # Replace with your file ID
    download_model_from_drive(google_drive_file_id, weights_file_path)

# Load model architecture and weights
model = create_model()
model.build(input_shape=(None, 40))
model.load_weights(weights_file_path)

# Emoji labels
emoji_labels = {0: '❤', 1: '🇧', 2: '🇮', 3: '🎉', 4: '🎧', 5: '🎵', 6: '🎶', 7: '👀', 8: '👇', 9: '👌',
                10: '👍', 11: '👏', 12: '💀', 13: '💔', 14: '💕', 15: '💖', 16: '💯', 17: '🔥', 18: '🕊',
                19: '🗿', 20: '😀', 21: '😁', 22: '😂', 23: '😅', 24: '😊', 25: '😌', 26: '😍', 27: '😎',
                28: '😔', 29: '😘', 30: '😢', 31: '😭', 32: '😮', 33: '😳', 34: '🙂', 35: '🙏', 36: '🚩',
                37: '🤣', 38: '🥰', 39: '🥺'}

# Emoji sentiment dictionary
emoji_sentiment = {
    "❤": 90, "🇧": 50, "🇮": 50, "🎉": 85, "🎧": 70, "🎵": 75,
    "🎶": 75, "👀": 50, "👇": 30, "👌": 80, "👍": 85, "👏": 88,
    "💀": 20, "💔": 10, "💕": 95, "💖": 95, "💯": 100, "🔥": 80,
    "🕊": 90, "🗿": 40, "😀": 95, "😁": 90, "😂": 85, "😅": 75,
    "😊": 80, "😌": 70, "😍": 95, "😎": 80, "😔": 40, "😘": 85,
    "😢": 30, "😭": 20, "😮": 50, "😳": 45, "🙂": 75, "🙏": 85,
    "🚩": 30, "🤣": 80, "🥰": 90, "🥺": 40
}

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
    text = re.sub(r'RT[\s]+', '', text)  # Remove RT
    text = re.sub(r'http\S+', '', text)  # Remove hyperlinks
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub('[0-9]+', '', text)  # Remove numbers
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuations
    return text

# Sentiment analysis function
def calculate_sentiment(text):
    text_sentiment = TextBlob(text).sentiment.polarity
    return text_sentiment

# Sentiment for emoji
def text_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    scaled_sentiment = (polarity + 1) * 50  # scale sentiment to 0-100 range
    return scaled_sentiment

def calculate_rms(text_sentiment, emoji_sentiment):
    return np.sqrt((text_sentiment ** 2 + emoji_sentiment ** 2) / 2)

# Predict function - run 10 times and return majority result
def run_predictions(user_input, num_runs=10):
    predictions = []
    processed_text = preprocess_text(user_input)
    tokenizer.fit_on_texts([processed_text])  # Fit tokenizer to the current input
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)

    for _ in range(num_runs):
        emoji_prediction = np.argmax(model.predict(padded_sequences), axis=-1)[0]
        predictions.append(emoji_labels[emoji_prediction])

    # Get the most frequent emoji
    most_common_emoji = Counter(predictions).most_common(1)[0][0]
    return most_common_emoji

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get JSON data from the request
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        user_input = data['text']  # Extract 'text' from the JSON payload
        if not user_input:
            return jsonify({'error': 'Empty text provided'}), 400

        # Run predictions 10 times and get majority result
        most_common_emoji = run_predictions(user_input)

        # Calculate sentiment for the input text
        processed_text = preprocess_text(user_input)
        sentiment_score = text_sentiment(processed_text)

        # Calculate emoji sentiment
        emoji_sentiment_value = emoji_sentiment.get(most_common_emoji, 50)  # Default to 50 if not found
        combined_sentiment = calculate_rms(sentiment_score, emoji_sentiment_value)
        
        response = {
            'input_text': user_input,
            'predicted_emoji': most_common_emoji,
            'sentiment_score': round(sentiment_score, 2),
            'emoji_sentiment': emoji_sentiment_value,
            'combined_sentiment': round(combined_sentiment, 2)
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
