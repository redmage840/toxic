# TO-DO
# gensim, spacy...
# Compare to results with different/no data cleaning
# Many classifiers are getting around 83%. Is this the result of overwhelming amount of zero scores?
# What is the accuracy of guessing all zeros?
# Get distribution of toxicity/toxicity_score
# Try other vectorizers/classifiers


import nltk
import pandas as pd

# Read and merge two files into df
comments=pd.read_csv('toxicity_annotated_comments.tsv', sep='\t')
scores=pd.read_csv('toxicity_annotations.tsv', sep='\t')
uniqueScores = scores[["rev_id", "toxicity_score", "toxicity"]].groupby("rev_id", as_index=False).first()
df = pd.merge(comments, uniqueScores, on="rev_id")

df['length'] = df.comment.str.len()

# Remove HTML elements and 'NEWLINE_TOKEN'
from bs4 import BeautifulSoup
df['cleaned_comments'] = df.comment.apply(lambda x: BeautifulSoup(x, 'html5lib').get_text())
df['cleaned_comment'] = df.cleaned_comments.apply(lambda x: x.replace('NEWLINE_TOKEN', ''))

# Remove non-(alpha|whitespace|apostrophe) chars, change to lowercase
import re

df['cleaned_comment'] = df.cleaned_comment.apply(lambda x: re.sub("[^a-zA-Z\s']", '', x))
df['cleaned_comment'] = df.cleaned_comment.apply(str.lower)

#Remove rows with blank comments
df = df[df['cleaned_comment'].str.len()>0]

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.3, random_state=666)
all_words_train = train_set.cleaned_comment
all_words_test = test_set.cleaned_comment


# TF-IDF Vectorizer (Try other alternatives as well)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2),  \
                             stop_words='english',  strip_accents='unicode',  norm='l2')

X_train = vectorizer.fit_transform(all_words_train)
X_test = vectorizer.transform(all_words_test)


# Classifier, Multinomial Naive Bayes # .51 on full dataset
# from sklearn.naive_bayes import MultinomialNB
# mNBclassifier = MultinomialNB()
# mNBclassifier.fit(X_train, train_set.toxicity_score)

#resultmNB = mNBclassifier.predict(X_test)
# print(mNBclassifier.score(X_test, test_set.toxicity_score))


# Classifier, Bernoulli Naive Bayes
# from sklearn.naive_bayes import BernoulliNB
# bernNBclassifier = BernoulliNB()
# bernNBclassifier.fit(X_train, train_set.toxicity_score)
# 
# resultBernoulliNB = bernNBclassifier.predict(X_test)
# 
# print(bernNBclassifier.score(X_test, test_set.toxicity_score))


# Classifier, linear model
# from sklearn import linear_model
# sgd_clsf = linear_model.SGDClassifier(max_iter=90)
# sgd_clsf.fit(X_train, train_set.toxicity_score)
# 
# resultSGD = sgd_clsf.predict(X_test)
# 
# print(sgd_clsf.score(X_test, test_set.toxicity_score))


# Classifier, Gaussian Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# gnbClassifier = GaussianNB()
# gnbClassifier.fit(X_train.toarray(), train_set.toxicity_score)

#resultGNB = gnbClassifier.predict(X_test.toarray())

# print(gnbClassifier.score(X_test.toarray(), test_set.toxicity_score))


# Classifier, Linear SVC
# from sklearn.svm import LinearSVC # .50 full dataset
# from nltk.classify.scikitlearn import SklearnClassifier
# linSVCclsf = LinearSVC()
# linSVCclsf.fit(X_train, train_set.toxicity_score)
# 
# #result_linearSVC= linSVCclsf.predict(X_test)
# 
# print(linSVCclsf.score(X_test, test_set.toxicity_score))


# Random Forest, best results so far
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(X_train, train_set.toxicity_score)
# resultForest = forest.predict(X_test)
print(forest.score(X_test, test_set.toxicity_score))


# from sklearn.ensemble import VotingClassifier
# all_clsf = VotingClassifier(estimators=[('multiNB', mNBclassifier), ('randForest', forest), ('linSVC', linSVCclsf),\
#                                        ('linModel', sgd_clsf)])
# all_clsf.fit(X_train, train_set.toxicity_score)
# print(all_clsf.score(X_test, test_set.toxicity_score))
