import random
import json
import pickle
import numpy as np
import nltk

from google.colab import files
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('all')

#lemaatizer for input words
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

#empty lists to hold the datasets after preprocessing has taken place
words = []
classes = []
documents = []

#punctuation marks to ignore
ignore_letters = ['?','!','.',',']

#loop through training data to create classes and words list
for intent in intents ['intents']:
    for pattern in intent['patterns']:
        #Tokenize sentence into seperate word variables
        word_list = nltk.tokenize.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#Lemmatize the words to return lemmatized form of the word
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

#Save the serialized data into pickle files which will be used for training
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#run the training sequence to get the h5 model files
training = []
output_empty = [0]*len(classes)
#Loop through all the training data in the document variable
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

        
#Shuffle the training data
random.shuffle(training)
#convert the data to a numpy array to be fed to the model
training = np.array(training, dtype='object')
train_x = list(training[:,0])
train_y = list(training[:,1])

#Code for creating the neural network for the chatbot
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Stochastic gradient descent will be used on this model for optimization
# sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9,nesterov=True)

#Train the Chatbot model and save trained model in h5 file
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])
modelled = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5", modelled)
print("Model successfully trained and saved to current working directory")


