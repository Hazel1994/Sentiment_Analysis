import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re, nltk
from evaluate import run_some_models


train_data_df = pd.read_csv('data.txt', header=None, delimiter="\t")
train_data_df.columns = [ "output","input"]

def tokenize(text):
    text = re.sub("[^A-Za-z0-9]", " ", text)
    tokens = nltk.word_tokenize(text)
    return tokens


vectorizer = CountVectorizer(
    analyzer='word',
    tokenizer=tokenize,
    lowercase=True,
    max_features=500
)

features = vectorizer.fit_transform(
    train_data_df.input.tolist())
features = features.toarray()

X_train, X_test, y_train, y_test = train_test_split(
    features,
    train_data_df.output,
    train_size=0.7,test_size=.3
)

run_some_models(X_train,X_test,y_train,y_test)