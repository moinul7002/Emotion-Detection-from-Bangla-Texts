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

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC,  NuSVC

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
        conf = choice_votes /len(votes)
        return conf

stop_words = set(stopwords.words("bangla"))

short_anger = open("F:/BACKUP/A.cuet/final project/corpus/anger.txt", "r", encoding="utf8").read()
short_happy = open("F:/BACKUP/A.cuet/final project/corpus/happy.txt", "r", encoding="utf8").read()
short_sad = open("F:/BACKUP/A.cuet/final project/corpus/sad.txt", "r", encoding="utf8").read()
short_fear = open("F:/BACKUP/A.cuet/final project/corpus/fear.txt", "r", encoding="utf8").read()
short_disgust = open("F:/BACKUP/A.cuet/final project/corpus/disgust.txt", "r", encoding="utf8").read()
short_surprise = open("F:/BACKUP/A.cuet/final project/corpus/surprise.txt", "r", encoding="utf8").read()

documents = []
all_words =[]

for r in short_anger.split("\n"):
    documents.append((r, "anger"))

for r in short_happy.split("\n"):
    documents.append((r, "happy"))

for r in short_sad.split("\n"):
    documents.append((r, "sad"))

for r in short_fear.split("\n"):
    documents.append((r, "fear"))

for r in short_disgust.split("\n"):
    documents.append((r, "disgust"))

for r in short_surprise.split("\n"):
    documents.append((r, "surprise"))


short_anger_words = word_tokenize(short_anger)
short_happy_words = word_tokenize(short_happy)
short_sad_words = word_tokenize(short_sad)
short_fear_words = word_tokenize(short_fear)
short_disgust_words = word_tokenize(short_disgust)
short_surprise_words = word_tokenize(short_surprise)

for w in short_anger_words:
    if w not in stop_words:
        all_words.append(w)

for w in short_happy_words:
    if w not in stop_words:
        all_words.append(w)

for w in short_sad_words:
    if w not in stop_words:
        all_words.append(w)

for w in short_fear_words:
    if w not in stop_words:
        all_words.append(w)

for w in short_disgust_words:
    if w not in stop_words:
        all_words.append(w)

for w in short_surprise_words:
    if w not in stop_words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))   prints most frequent 15 words
#print(all_words["stupid"])         prints how many times word 'stupid' appears

word_features = list(all_words.keys())[:200]
def find_features(document):
   words = word_tokenize(document)
   features = {}
   for w in word_features:
      features[w] = (w in words)
   return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents] 
random.shuffle(featuresets)
training_set = featuresets[:130]
testing_set = featuresets[130:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
#classifier_f =open("naivebayes.pickle", "rb")
#classifier = pickle.load(classifier_f)
#classifier_f.close()  ##pickle e save kore naive bayes use korlam eki data set e

print("Original Naive Bayes classifier accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

##save_classifier = open("naivebayes.pickle", "wb")
##pickle.dump(classifier, save_classifier)
##save_classifier.close()                            this saves the cfier in pickle

#from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernouliNB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)
#print("GaussianNB classifier accuracy percent: ", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

#BernouliNB_classifier = SklearnClassifier(BernouliNB())
#BernouliNB_classifier.train(training_set)
#print("BernouliNB classifier accuracy percent: ", (nltk.classify.accuracy(BernouliNB_classifier, testing_set))*100)

# LogisticRegression, SGDClassifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LR classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGD classifier accuracy percent: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)


#from sklearn.svm import SVC, linearSVC, NuSVC
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC classifier accuracy percent: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

#LinearSVC_classifier = SklearnClassifier(LinearSVC())
#LinearSVC_classifier.train(training_set)
#print("LinearSVC classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
#print((nltk.classify.SklearnClassifier(testing_set[0][0])))


voted_classifier = VoteClassifier(classifier, MNB_classifier, LogisticRegression_classifier, SGDClassifier_classifier, SVC_classifier)
print("Voted classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification: ", voted_classifier.classify(testing_set[0][0]), testing_set[:,1])
print("Classification: ", voted_classifier.classify(testing_set[1][0]))
print("Classification: ", voted_classifier.classify(testing_set[2][0]))
print("Classification: ", voted_classifier.classify(testing_set[3][0]))
print("Classification: ", voted_classifier.classify(testing_set[4][0]))
print("Classification: ", voted_classifier.classify(testing_set[5][0]))




WordDictAnger = dict.fromkeys(word_features, 0)
WordDictDisgust = dict.fromkeys(word_features, 0)
WordDictFear = dict.fromkeys(word_features, 0)
WordDictHappy = dict.fromkeys(word_features, 0)
WordDictSad = dict.fromkeys(word_features, 0)
WordDictSurprise = dict.fromkeys(word_features, 0)

for w in word_features:
    if w in short_anger_words:
        WordDictAnger[w] += 1

for w in word_features:
    if w in short_disgust_words:
        WordDictDisgust[w] += 1

for w in word_features:
    if w in short_fear_words:
        WordDictFear[w] += 1

for w in word_features:
    if w in short_happy_words:
        WordDictDisgust[w] += 1

for w in word_features:
    if w in short_sad_words:
        WordDictSad[w] += 1

for w in word_features:
    if w in short_surprise_words:
        WordDictSurprise[w] += 1



#dataframe = pd.DataFrame([WordDictAnger, WordDictDisgust, WordDictFear, WordDictHappy, WordDictSad, WordDictSurprise])

