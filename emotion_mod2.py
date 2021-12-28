import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
from nltk.corpus import movie_reviews

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from collections import Counter

from nltk.classify.scikitlearn import SklearnClassifier

import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def get_all_modes(self, a):
        c = Counter(a)
        mode_count = max(c.values())
        mode = {key for key, count in c.items() if count == mode_count}
        return mode

    def classify(self, featuresets):
        votes = []
        for c in self._classifiers:
            v = c.classify(featuresets)
            votes.append(v)
        return self.get_all_modes(votes)

    def confidence(self, featuresets):
        votes = []
        for c in self._classifiers:
            v = c.classify(featuresets)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = float(choice_votes) /len(votes)
        return conf

documents_f =open("documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features_f =open("word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

def find_features(document):
   words = word_tokenize(document)
   features = {}
   for w in word_features:
      features[w] = (w in words)
   return features

featuresets_f =open("featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)

training_set = featuresets[:150]
testing_set = featuresets[150:]

open_file = open("original_naivebayes.pickle", "rb")
NB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("MNB.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("BNB.pickle", "rb")
BNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("KNN.pickle", "rb")
KNN_classifier = pickle.load(open_file)
open_file.close()

open_file = open("DT.pickle", "rb")
DT_classifier = pickle.load(open_file)
open_file.close()

open_file = open("LR.pickle", "rb")
LR_classifier = pickle.load(open_file)
open_file.close()

open_file = open("SVM.pickle", "rb")
SVM_classifier = pickle.load(open_file)
open_file.close()

open_file = open("BNB.pickle", "rb")
BNB_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(NB_classifier, MNB_classifier, DT_classifier, LR_classifier, BNB_classifier, SVM_classifier, KNN_classifier)

def emotion(text):
    feats = find_features(text)
    #print(feats)
    return voted_classifier.classify(feats)
