# Sentiment analyses in python Using bag of words features

This is a tutorial for beginners.
I am going to explain how to write a program for sentiment analysis
using python. I will try to be as comprehensive  as possible. 


## Installing python

You can easily download python from https://www.python.org/downloads/ , the version used in this project is python 3.5.2 64 bit.
After installing python you can run python files using CMD but you can also use Pycharm which is an IDE for python. Coding and debugging in Pycharm is really easy. You can download the fee (communit )version of it via the link below:
https://www.jetbrains.com/pycharm/download/#section=windows
It will usually recognize the python on your computer but you might have to go to Setting-interpreter and brows for the python.exe file.
2 Installing required libraries:
Unlike MATLAB, you have to install the required libraries before using them. Here we will use the “pip” method to get this done.
To install any library: 
•	First open the CMD as administrator 
•	Write pip install “LB
Where “LB” is the name of the library you want to install.
Sometimes your computer cant compile the necessary file to finish the installation , in that case you can go to the link below and search for the library you have trouble installing. http://www.lfd.uci.edu/~gohlke/pythonlibs/#vlfd 
After downloading the library, you can simply run the CMD as administrator in the current directory and use pip and the full name of the download library to install it..
The libraries you need to install to run this project are:
•	NLTK
•	sklearn
•	numpy
•	pandas

3	Dataset 
The dataset we are using in this project is the binary classification one, you can get it from https://www.kaggle.com/c/si650winter11.  It has 7086 reviews where label 1 indicates the review is positive and 0 means it is negative. A few samples is shown below
•	1	mission impossible 2 rocks!!....
•	1	i love being a sentry for mission impossible and a station for bonkers.
•	0	Brokeback Mountain was boring.
•	0	Oh, and Brokeback Mountain was a terrible movie.


## Feature extraction
Machine learning models works with numerical data, and hence we have to convert our review to numerical representation. There are many technique to achieve this goal.  Here we are using Bag of Word representation which you should already be familiar with , if youre not you can take a look at this: https://medium.com/greyatom/an-introduction-to-bag-of-words-in-nlp-ac967d43b428
So let’s dive right into it.
First we need to import the required library mentioned above 

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re, nltk
from evaluate import run_some_models

evaluate is a file we will use to evaluate our models. It’s not a built-in library.
Lets load our data using Pandas:
train_data_df = pd.read_csv(data.txt', header=None, delimiter="\t")

“delimiter” determines  a token used to separate the samples from their labels which is the tab in our case.
“Header=None” indicates that there is no header in the data file.

Now that we have our data stored as a dataframe, let’s name each column for better use.

train_data_df.columns = ["output","input"]

The first columns is the class of each review. 
The next step is to create a tokenizer to split the raw text into words and delete the unnecessary characters. So lets define a function that receives a raw review and gives out the tokenized version of it. 
def tokenize(text):
    text = re.sub("[^A-Za-z0-9]", " ", text)
    tokens = nltk.word_tokenize(text)
    return tokens

re.sub("[^A-Za-z0-9]", " ", text) simply removes everything from the text apart from English characters and numbers.

tokens = nltk.word_tokenize(text) uses nltk to split the text into its words.

Now that we preprocessed the raw data, let’s define a Bag of word vectorizer to convert the data into trainable numeric data. 

vectorizer = CountVectorizer(
    analyzer='word',
    tokenizer=tokenize,
    lowercase=True,
    max_features=500
)

analyzer='word', indicates that the smallest token to work with is word
tokenizer=tokenize introduces the defined tokenizer to use.
lowercase=True, simply lowercases all the text first.
max_features=500 specify the maximum numbe of features for each review, so here each review will be converted into a vector of length 500.

Now, let’s use the above function as store the result into features.

features = vectorizer.fit_transform(
    train_data_df.input.tolist())
features = features.toarray()

Splitting the data

To evaluate our models we need to split the data into test and train. Python has a function called: train_test_split() which can be used as follows:


X_train, X_test, y_train, y_test = train_test_split(
    features,
    train_data_df.output,
    train_size=0.7,test_size=.3
)

You can modify the portion for both train and test.
Alight, we are all prepare to use this data to train a classifier and evaluate it. There are numerous classifiers you can use but here we use SVM, Naive Bayes, and KNN. Run_some_models function takes the input data and print out the result for the three classifiers. 

I am going to illustrate the code for one classifier, the same explanation holds for the rest.

First we define the classifer

print("running Knn")
neigh = KNeighborsClassifier(n_neighbors=4)

train it on the train data
neigh.fit(X_train,y_train)

predict the train data and display accurary
y_pred=neigh.predict(X_train)
print('accuracy train : ', accuracy_score(y_train, y_pred))

predict the test data and display the accuracy

y_pred = neigh.predict(X_test)
print('test accuracy :', accuracy_score(y_test, y_pred))

display the classification report for test data, it will show recall, precision, and f-score for the predicted test labels.

print("classification report")
print(classification_report(y_test, y_pred))


## result 


Now let’s run the main.py file and see the results:


