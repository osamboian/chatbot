import random
import json 
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl", 'rb'))
classes = pickle.load(open("classes.pkl", 'rb'))
model = load_model('chatbot_model.h5')

def sanitize(sentence):
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

def compiles(sentence):
    words = sanitize(sentence)
    bag = [0]*len(words)
    for w in words:
        for i, word in enumerate(words):
            if word == w:
                bag[1] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = compiles(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if i > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
    return return_list

def get_response(intents_list, intents_json):    
    tag = intents_list[0]
    print(tag," \n \n")
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        den = i['tag']
        print(den)
        if tag == i['tag']:
            result = random.choice(i['responses'])
            break
        else:
            result = "\n Kindly rephrase your query"
    return result


while True:
    message = input("")
    ints = predict_class(str(message))
    res = get_response(ints, intents )
    print(res)