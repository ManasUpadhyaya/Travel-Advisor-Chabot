import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import load_model

words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))
model = load_model('model.h5')

def preprocess_input(input_text):
    lemmatizer = WordNetLemmatizer()
    tokenized_words = nltk.word_tokenize(input_text)
    preprocessed_words = [lemmatizer.lemmatize(word.lower()) for word in tokenized_words]
    return preprocessed_words

def predict_intent(input_text):
    preprocessed_words = preprocess_input(input_text)
    bag = [0] * len(words)
    for w in preprocessed_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    input_data = np.array([bag])
    results = model.predict(input_data)[0]
    index = np.argmax(results)
    intent = classes[index]
    return intent

def test_accuracy(test_queries):
    correct_predictions = 0
    total_predictions = len(test_queries)

    for query, intent in test_queries:
        predicted_intent = predict_intent(query)
        if predicted_intent == intent:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

test_queries = [
    # Greeting intent
    ("Hello", "greeting"),
    ("Hi", "greeting"),
    ("Hey", "greeting"),
    ("Good morning", "greeting"),
    ("Good evening", "greeting"),

    # Goodbye intent
    ("Goodbye", "goodbye"),
    ("Bye", "goodbye"),
    ("See you later", "goodbye"),
    ("Take care", "goodbye"),

    # Thanks intent
    ("Thank you", "thanks"),
    ("Thanks a lot", "thanks"),
    ("Appreciate it", "thanks"),

    # Name intent
    ("What's your name?", "name"),
    ("Who are you?", "name"),
    ("May I know your name?", "name"),

    # Help intent
    ("Help me", "help"),
    ("I need assistance", "help"),
    ("Can you help?", "help"),
    ("What can you do?", "help"),

    # Weather intent
    ("What's the weather like today?", "weather"),
    ("How's the weather?", "weather"),
    ("Is it going to rain today?", "weather"),

    # Age intent
    ("How old are you?", "age"),
    ("When were you created?", "age"),
    ("What's your age?", "age"),

    # Chatbot intent
    ("What can you do?", "chatbot"),
    ("How do you work?", "chatbot"),
    ("Tell me about yourself", "chatbot"),

    # Creator intent
    ("Who created you?", "creator"),
    ("Who is your developer?", "creator"),
    ("Who made you?", "creator"),

    # Default intent (unrecognized queries)
    ("Where is the nearest coffee shop?", "default"),
    ("Can you bake a cake?", "default"),
    ("Tell me a joke", "default")
]

accuracy = test_accuracy(test_queries)
print("Accuracy:", accuracy)
