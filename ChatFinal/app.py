import nltk
# nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

import requests
import json

from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))




def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


def is_mixed_query(input_text, tags):
    matched_tags = []
    for tag in tags:
        for pattern in tag['patterns']:
            if pattern.lower() in input_text.lower():
                matched_tags.append(tag)
                break
    print(matched_tags)
    return len(matched_tags) > 1



ch = 0

def chatbot_response(msg):
    global ch
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    if is_mixed_query(msg, intents['intents']):
        return "I am sorry i dont have enough knowledge to answer that. [ERROR BECAUSE OF THE INPUT PATTERN THAT LIES IN TWO DIFFERENT TAGS AT THE SAME]"
    elif res == "hotels":
        ch = 1
        return "Please enter the city"
    elif res == "restaurants":
        ch = 2
        return "Please enter the city"
    elif res == "places":
        ch = 3
        return "Please enter the city"
    return res



from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    global ch
    if(ch==0):
        return chatbot_response(userText)
    elif(ch==1):
        ch = 0
        
        return hotels_rec(userText)
    elif(ch==2):
        ch = 0

        return restaurants_rec(userText)
    elif(ch==3):
        ch = 0

        return places_rec(userText)
        
def hotels_rec(city):
    city = city.lower()
    url = "https://travel-advisor.p.rapidapi.com/locations/v2/auto-complete"

    querystring = {"query":city,"lang":"en_US","units":"km"}

    headers = {
        "X-RapidAPI-Key": "081f6c55a0msh3a8ba627af7a2fap157913jsnddae3a1e4ef3",
        "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    res = json.loads(response.text)

    try:
        for x in res['data']['Typeahead_autocomplete']['results']:
            if(x['buCategory'] == "HOTELS"):
                return("https://www.tripadvisor.com"+  x['route']['url'])
  
    except:
        print("")


def restaurants_rec(city):
    city = city.lower()
    url = "https://travel-advisor.p.rapidapi.com/locations/v2/auto-complete"

    querystring = {"query":city,"lang":"en_US","units":"km"}

    headers = {
        "X-RapidAPI-Key": "081f6c55a0msh3a8ba627af7a2fap157913jsnddae3a1e4ef3",
        "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    res = json.loads(response.text)

    try:
        for x in res['data']['Typeahead_autocomplete']['results']:
            if(x['buCategory'] == "RESTAURANTS"):
                return("https://www.tripadvisor.com"+  x['route']['url'])
  
    except:
        print("")


    

def places_rec(city):
    city = city.lower()
    url = "https://travel-advisor.p.rapidapi.com/locations/v2/auto-complete"

    querystring = {"query":city,"lang":"en_US","units":"km"}

    headers = {
        "X-RapidAPI-Key": "081f6c55a0msh3a8ba627af7a2fap157913jsnddae3a1e4ef3",
        "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    res = json.loads(response.text)

    try:
        for x in res['data']['Typeahead_autocomplete']['results']:
            if(x['buCategory'] == "ATTRACTIONS"):
                return("https://www.tripadvisor.com"+  x['route']['url'])
    
    except:
        print("")






if __name__ == "__main__":
    app.run(port = 5500)


