from flask import Flask, render_template, request
import random
import pickle
import json
import numpy as np
import tensorflow as tf
import requests
from googletrans import Translator
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("C:\Users\jnche\Documents\ICS PROJECT 2 YEAR4\CHATBOT\app\dataset3.jsonn").read())

data = pickle.load(open("C:\Users\jnche\Documents\ICS PROJECT 2 YEAR4\CHATBOT\app\data.pkl", 'rb'))
labels = pickle.load(open("C:\Users\jnche\Documents\ICS PROJECT 2 YEAR4\CHATBOT\app\labels.pkl", 'rb'))
model = tf.keras.models.load_model("C:\Users\jnche\Documents\ICS PROJECT 2 YEAR4\CHATBOT\app\chatbot_model.h5")

translator = Translator()

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    message = request.form['message']

    # Detect the language of the user's input
    lang = translator.detect(message).lang

    # Translate the input to English if it is not in English
    if lang != 'en':
        message = translator.translate(message).text

    message_words = nltk.word_tokenize(message)
    message_words = [lemmatizer.lemmatize(word.lower()) for word in message_words]

    bag = [0] * len(data)
    for word in message_words:
        for i, w in enumerate(data):
            if w == word:
                bag[i] = 1

    # Use the trained model to get a prediction for the user's message
    prediction = model.predict(np.array([bag]))[0]
    # Get the index of the predicted label
    predicted_label_index = np.argmax(prediction)
    # Get the corresponding label 
    predicted_label = labels[predicted_label_index]

    # Check if the predicted label has a corresponding response
    for intent in intents['intents']:
        if intent['tag'] == predicted_label:
            response = random.choice(intent['responses'])
            break

    # If the predicted label doesn't have a corresponding response, return a default message
    else:
        response = "I'm sorry, I don't understand, but will have the response shortly"

    # Translate the response back to the original language if it is not in English
    if lang != 'en':
        response = translator.translate(response, dest=lang).text

    return response

if __name__ == '__main__':
    app.run(debug=True)




