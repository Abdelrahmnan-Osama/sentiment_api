# Load the model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os, pickle

def load_model():
    filename = "sentiment_model.pkl"

    if os.path.exists(filename):
        print("Loading model")
        with open(filename, 'rb') as file:
            sentiment_model = pickle.load(file)
    else:
        print("Downloading model")
        sentiment_model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment') 
        pickle.dump(sentiment_model, open(filename, 'wb'))
    
    return sentiment_model


def load_tokenizer():
    filename = "tokenizer.pkl"

    if os.path.exists(filename):
        print("Loading tokenizer")
        with open(filename, 'rb') as file:
            tokenizer = pickle.load(file)
    else:
        print("Downloading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        pickle.dump(tokenizer, open(filename, 'wb'))

    return tokenizer


def sentiment_score(review):
    tokenizer = load_tokenizer()
    sentiment_model = load_model()
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = sentiment_model(tokens)
    return int(torch.argmax(result.logits))+1

