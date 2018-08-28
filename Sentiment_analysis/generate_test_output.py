import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re, nltk
from evaluate import run_some_models,predict_test_labels


train_data_df = pd.read_csv('train.txt', header=None, delimiter="\t")
train_data_df.columns = [ "input","output",]

test_data = pd.read_csv('test_without_label.txt', header=None,delimiter='\n')
test_data.columns = [ "input"]

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

corpus_data_features = vectorizer.fit_transform(
    train_data_df.input.tolist())
corpus_data_features_nd = corpus_data_features.toarray()

X_train=corpus_data_features_nd
y_train=np.asarray(train_data_df.output)

X_test=vectorizer.transform(test_data.input.tolist())
X_test=X_test.toarray()

predict_test_labels(X_train,X_test,y_train)