from flask import Flask, request, render_template, jsonify, session
import os
import torch
import json
import random
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import torch.nn as nn
import torch.optim as optim
from flask_session import Session

app = Flask(__name__)
nltk.download('punkt')
# Configure session for chat history
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Load intents and prepare NLP functions
with open('intents.json', 'r') as f:
    intents = json.load(f)

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# Prepare training data
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = tokenize(pattern)
        all_words.extend(word_list)
        xy.append((word_list, intent['tag']))
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

all_words = sorted(set(stem(w) for w in all_words))
tags = sorted(tags)

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Model parameters
input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

# Load or train the model
model = NeuralNet(input_size, hidden_size, output_size)
model_path = "chatbot_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
else:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        inputs = torch.from_numpy(X_train).float()
        targets = torch.from_numpy(y_train).long()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), model_path)
    print("Model trained and saved!")

# Chatbot logic
def get_response(user_text):
    sentence = tokenize(user_text)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).float()

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."

@app.route('/')
def home():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    bot_response = get_response(user_message)
    chat_entry = {'user': user_message, 'bot': bot_response}
    
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    session['chat_history'].append(chat_entry)
    session.modified = True
    return jsonify(chat_entry)

if __name__ == "__main__":
    app.run(debug=True)
