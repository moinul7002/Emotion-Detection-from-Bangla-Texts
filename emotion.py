import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

import random
import gensim
from gensim.models import Word2Vec

import pandas as pd

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from nltk import precision
from nltk.metrics import *

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer

import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
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



stop_w = open("E:/final project/corpus/stop_words.txt", "r", encoding="utf8").read()
stop_words = word_tokenize(stop_w)
#for w in stop_words:
#    print(w)
#print(len(stop_words))

short_anger = open("E:/final project/corpus/anger.txt", "r", encoding="utf8").read()
short_happy = open("E:/final project/corpus/happy.txt", "r", encoding="utf8").read()
short_sad = open("E:/final project/corpus/sad.txt", "r", encoding="utf8").read()
short_fear = open("E:/final project/corpus/fear.txt", "r", encoding="utf8").read()
short_disgust = open("E:/final project/corpus/disgust.txt", "r", encoding="utf8").read()
short_surprise = open("E:/final project/corpus/surprise.txt", "r", encoding="utf8").read()

documents = []
sent = []
docs = []
#labels = []
for r in short_anger.split("\n"):
    documents.append((r, "anger"))
    docs.append(r)
    #labels.append('anger')

for r in short_happy.split("\n"):
    documents.append((r, "happy"))
    docs.append(r)
    #labels.append('happy')

for r in short_sad.split("\n"):
    documents.append((r, "sad"))
    docs.append(r)
    #labels.append('sad')

for r in short_fear.split("\n"):
    documents.append((r, "fear"))
    docs.append(r)
    #labels.append('fear')

for r in short_disgust.split("\n"):
    documents.append((r, "disgust"))
    docs.append(r)
    #labels.append('disgust')

for r in short_surprise.split("\n"):
    documents.append((r, "surprise"))
    docs.append(r)
    #labels.append('surprise')

#print(len(documents))

#for i in range(20):
#    print(docs[i], '\n')
#word_docs = word_tokenize(docs)
#print(word_docs[:10])
#print(dataset[1][:])

#random.shuffle(docs)


all_words =[]

short_anger_words = word_tokenize(short_anger)
short_happy_words = word_tokenize(short_happy)
short_sad_words = word_tokenize(short_sad)
short_fear_words = word_tokenize(short_fear)
short_disgust_words = word_tokenize(short_disgust)
short_surprise_words = word_tokenize(short_surprise)

all=[]
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


for w in range(40):
    print(all_words[w],'\n')

#all_words = list(set(all_words[:100]))
#vectorizer = CountVectorizer()
#vectorizer = vectorizer.fit(all_words)
#tfMatrix = vectorizer.transform(docs).toarray()
#pd.set_option('display.width', 300)
#print(pd.DataFrame(tfMatrix))

#no_order = list(set(all))
#print(len(no_order))
#for w in all_words:
#    print(w, '\n')

save_documents = open("documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

#all_words.pprint(100)
#all_words.plot(20)
#print(all_words.most_common(15))   prints most frequent 15 words
#print(all_words["stupid"])         prints how many times word 'stupid' appears
all_words = dict(all_words.most_common(800))
word_features = list(all_words.keys())
#model1 = gensim.models.Word2Vec(word_features, min_count = 1,size = 100, window = 5)
#print(word_features)

save_word_features = open("word_features.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
   words = word_tokenize(document)
   features = {}

   for w in word_features:
       features[w] = (w in words)
   #print(features)
   return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(r), category) for (r, category) in documents]
random.shuffle(featuresets)
#print(featuresets)

save_featuresets =open("featuresets.pickle", "wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

training_set = featuresets[:800]
testing_set = featuresets[800:]
#for i in range (20):
#    print(training_set[i], '\n')
classifier = nltk.NaiveBayesClassifier.train(training_set)

#classifier_f =open("naivebayes.pickle", "rb")
#classifier = pickle.load(classifier_f)
#classifier_f.close()  ##pickle e save kore naive bayes use korlam eki data set e

#classifier.show_most_informative_features(15)

save_classifier = open("original_naivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

#from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernouliNB
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

save_classifier = open("MNB.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

LR_classifier = SklearnClassifier(LogisticRegression())
LR_classifier.train(training_set)

save_classifier = open("LR.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)

BernouliNB_classifier = SklearnClassifier(BernoulliNB())
BernouliNB_classifier.train(training_set)

save_classifier = open("BNB.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
# LogisticRegression, SGDClassifier
KNN_classifier = SklearnClassifier(KNeighborsClassifier())
KNN_classifier.train(training_set)

save_classifier = open("KNN.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

DT_classifier = SklearnClassifier(DecisionTreeClassifier())
DT_classifier.train(training_set)

save_classifier = open("DT.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

#RandomForest_classifier = SklearnClassifier(RandomForestClassifier())
#RandomForestClassifier.train(training_set)

#from sklearn.svm import SVC, linearSVC, NuSVC
SVC_classifier = SklearnClassifier(SVC(kernel='linear', degree=3))
SVC_classifier.train(training_set)


save_classifier = open("SVM.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)


voted_classifier = VoteClassifier(classifier, MNB_classifier, KNN_classifier, DT_classifier)


refsets = defaultdict(set)
testsets = defaultdict(set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

refsets = defaultdict(set)
testsets = defaultdict(set)
labels = []
tests = []
for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)
    labels.append(label)
    tests.append(observed)

#print(metrics.confusion_matrix(labels, tests))
print(nltk.ConfusionMatrix(labels, tests))

print('anger')
print( 'Precision:', precision(refsets['anger'], testsets['anger']))
print( 'Recall:', recall(refsets['anger'], testsets['anger']))
print ('F-measure:', f_measure(refsets['anger'], testsets['anger']))
print('disgust')
print( 'Precision:', precision(refsets['disgust'], testsets['disgust']))
print( 'Recall:', recall(refsets['disgust'], testsets['disgust']))
print ('F-measure:', f_measure(refsets['disgust'], testsets['disgust']))
print('fear')
print( 'Precision:', precision(refsets['fear'], testsets['fear']))
print( 'Recall:', recall(refsets['fear'], testsets['fear']))
print ('F-measure:', f_measure(refsets['fear'], testsets['fear']))
print('happiness')
print( 'Precision:', precision(refsets['happy'], testsets['happy']))
print( 'Recall:', recall(refsets['happy'], testsets['happy']))
print ('F-measure:', f_measure(refsets['happy'], testsets['happy']))
print('sadness')
print( 'Precision:', precision(refsets['sad'], testsets['sad']))
print( 'Recall:', recall(refsets['sad'], testsets['sad']))
print ('F-measure:', f_measure(refsets['sad'], testsets['sad']))
print('surprise')
print( 'Precision:', precision(refsets['surprise'], testsets['surprise']))
print( 'Recall:', recall(refsets['surprise'], testsets['surprise']))
print ('F-measure:', f_measure(refsets['surprise'], testsets['surprise']))

def label2int(label):
    if label == 'anger':
        return 1
    elif label == 'disgust':
        return 2
    elif label == 'fear':
        return 3
    elif label == 'happy':
        return 4
    elif label == 'sad':
        return 5
    elif label == 'surprise':
        return 6

y_true, y_score = [], []

for i, (feats, label_true) in enumerate(testing_set):
    label_predicted = classifier.classify(feats)
    y_true.append(label2int(label_true))
    y_score.append(label2int(label_predicted))

pr_auc = []
roc_auc = []

precision = []
recall = []

# Precision-Recall AUC for anger
precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
pr_auc.append(auc(recall, precision))
print("Precision-Recall AUC: %.2f" % pr_auc[0])
# ROC AUC for anger
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
roc_auc.append(auc(fpr, tpr))
print("ROC AUC: %.2f" % roc_auc[0])

# Precision-Recall AUC for disgust
precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=2)
pr_auc.append(auc(recall, precision))
print("Precision-Recall AUC: %.2f" % pr_auc[1])
# ROC AUC for disgust
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=2)
roc_auc.append(auc(fpr, tpr))
print("ROC AUC: %.2f" % roc_auc[1])

# Precision-Recall AUC for fear
precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=3)
pr_auc.append(auc(recall, precision))
print("Precision-Recall AUC: %.2f" % pr_auc[2])
# ROC AUC for fear
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=3)
roc_auc.append(auc(fpr, tpr))
print("ROC AUC: %.2f" % roc_auc[2])

# Precision-Recall AUC for happiness
precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=4)
pr_auc.append(auc(recall, precision))
print("Precision-Recall AUC: %.2f" % pr_auc[3])
# ROC AUC for happiness
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=4)
roc_auc.append(auc(fpr, tpr))
print("ROC AUC: %.2f" % roc_auc[3])


# Precision-Recall AUC for sad
precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=5)
pr_auc.append(auc(recall, precision))
print("Precision-Recall AUC: %.2f" % pr_auc[4])
# ROC AUC for sad
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=5)
roc_auc.append(auc(fpr, tpr))
print("ROC AUC: %.2f" % roc_auc[4])


# Precision-Recall AUC for surprise
precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=6)
pr_auc.append(auc(recall, precision))
print("Precision-Recall AUC: %.2f" % pr_auc[5])
# ROC AUC for surprise
fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=6)
roc_auc.append(auc(fpr, tpr))
print("ROC AUC: %.2f" % roc_auc[5])

plt.plot(roc_curve(y_true, y_score, pos_label=4))
plt.xlabel('True Positive Rate')
plt.ylabel('False positive Rate')
plt.legend()
plt.show()



lines = []
labels = []
from itertools import cycle
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'gray'])
for i, color in zip(range(6), colors):
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=i+1)
    l, = plt.plot(recall, precision, color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, pr_auc[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=("upper center"), prop=dict(size=9), bbox_to_anchor=(0.5, -0.15), ncol = 2)
plt.show()

lines = []
labels = []
for i, color in zip(range(6), colors):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=i+1)
    l, = plt.plot(fpr, tpr, color=color, lw=2)
    lines.append(l)
    labels.append('ROC-AUC curve for class {0} (area = {1:0.2f})'
                  ''.format(i, roc_auc[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.ylabel('True Positive rate')
plt.xlabel('False Positive rate')
plt.title('Extension of ROC-AUC curve to multi-class')
plt.legend(lines, labels, loc=("upper center"), prop=dict(size=9), bbox_to_anchor=(0.5, -0.15), ncol = 2)


plt.show()

