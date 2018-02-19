# bs4, remove non-alpha-apostrophe chars
# tfidf- max_df covers some domain words, min_df, ngram_range
# Compare to results with different/no data cleaning
# What is the accuracy of guessing all zeros? 49-full 89-unanimous
# Try other vectorizers/classifiers
# Derive features to show other corellations: length, num_exclaim, percent_caps, logged_in
# POS tagging features... number of verbs, nouns, etc
# use mean/median score for non-unanimous data set

# http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# Features to engineer: length, percent caps, percent alpha, number exclaims, logged_in, 

# Potentially cluster comments or other features to create groups/features

import nltk
import pandas as pd

# Read and merge two files into df
comments = pd.read_csv('toxicity_annotated_comments_unanimous.tsv', sep='\t')
scores = pd.read_csv('toxicity_annotations_unanimous.tsv', sep='\t')
uniqueScores = scores[["rev_id", "toxicity_score", "toxicity"]].groupby("rev_id", as_index=False).first()
df = pd.merge(comments, uniqueScores, on="rev_id")

df['length'] = df.comment.str.len()


# Open portion of non-unanimous data into df2
# Predict for non-unanimous data with models trained by unanimous data

comments2 = pd.read_csv('toxicity_annotated_comments.tsv', sep='\t')
# Mean scores rounded to nearest whole value
mean_scores = pd.read_csv('toxicity_annotations.tsv', sep='\t').groupby('rev_id', as_index=False)['toxicity_score'].mean().round()
df2 = pd.merge(comments2, mean_scores, on='rev_id')


# Remove HTML elements and 'NEWLINE_TOKEN'
from bs4 import BeautifulSoup
df['cleaned_comment'] = df.comment.apply(lambda x: BeautifulSoup(x, 'html5lib').get_text())

df['cleaned_comment'] = df.cleaned_comments.apply(lambda x: x.replace('NEWLINE_TOKEN', ''))


# Remove non-(alpha|whitespace|apostrophe) chars, change to lowercase
import re

df['cleaned_comment'] = df.cleaned_comment.apply(lambda x: re.sub("[^a-zA-Z\s']", '', x))
df['cleaned_comment'] = df.cleaned_comment.apply(str.lower)

#Remove rows with blank comments
df = df[df['cleaned_comment'].str.len()>0]

# Get percentage of zeroes
all_scores = df.toxicity_score
num_zeroes = df.toxicity_score[df.toxicity_score==0]
print(len(num_zeroes)/len(all_scores))

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df, test_size=0.3, random_state=666)
all_words_train = train_set.cleaned_comment
all_words_test = test_set.cleaned_comment


# all_words_train = df.cleaned_comment
# df2 = df2[:1000]
# all_words_test = df2.comment


# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=.5, ngram_range=(1, 3),  \
                             stop_words='english',  strip_accents='unicode',  norm='l2')

# print(tfidf_vectorizer.get_feature_names())


# Bag of Words Vectorizer
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(stop_words="english", min_df=2, max_df=.5)


from sklearn.pipeline import FeatureUnion

combined_features = FeatureUnion([("bagOwords", count_vectorizer), ("tfidf", tfidf_vectorizer)])

X_train = combined_features.fit_transform(all_words_train)
X_test = combined_features.transform(all_words_test)


# Classifier, Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
NBclassifier = MultinomialNB()
NBclassifier.fit(X_train, train_set.toxicity_score)

#resultNB = NBclassifier.predict(X_test)
    
print(NBclassifier.score(X_test, test_set.toxicity_score))


# Classifier, Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
bernNBclassifier = BernoulliNB()
bernNBclassifier.fit(X_train, train_set.toxicity_score)

#resultBernoulliNB = bernNBclassifier.predict(X_test)

print(bernNBclassifier.score(X_test, test_set.toxicity_score))


# Classifier, linear model
from sklearn import linear_model
sgd_clsf = linear_model.SGDClassifier(max_iter=1000)
sgd_clsf.fit(X_train, train_set.toxicity_score)

#resultSGD = sgd_clsf.predict(X_test)

print(sgd_clsf.score(X_test, test_set.toxicity_score))


# Classifier, Linear SVC, best results on unanimous dataset
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
linSVCclsf = LinearSVC()
linSVCclsf.fit(X_train, train_set.toxicity_score)

#result_linearSVC= linSVCclsf.predict(X_test)

print(linSVCclsf.score(X_test, test_set.toxicity_score))

# Classifier, Random Forest, best results when training with unanimous and predicting rounded mean toxicity score
#    on non-unanimous data
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 1000)

forest = forest.fit(X_train, train_set.toxicity_score)

#resultForest = forest.predict(X_test)

print(forest.score(X_test, test_set.toxicity_score))


from sklearn.ensemble import VotingClassifier
all_clsf = VotingClassifier(estimators=[('multiNB', NBclassifier), ('randForest', forest), ('linSVC', linSVCclsf),\
                                       ('linModel', sgd_clsf)])
all_clsf.fit(X_train, train_set.toxicity_score)
print(all_clsf.score(X_test, test_set.toxicity_score))