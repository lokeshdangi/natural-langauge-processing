import nltk
import random
from nltk.corpus import stopwords,wordnet
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.svm import SVC,LinearSVC,NuSVC
from nltk.stem import PorterStemmer
import pickle

class VoteClassifier(ClassifierI):
    def __init__(self,*classifier):
        self._classifier = classifier
    
    def classify(self, features):
        votes = []
        for c in self._classifier:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self, features):
        votes = []
        for c in self._classifier:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf


load_pickle = open("Natural langauge processing/pickle/document.pickle","rb")
document = pickle.load(load_pickle)
load_pickle.close()

load_pickle = open("Natural langauge processing/pickle/features_set.pickle","rb")
features_set = pickle.load(load_pickle)
load_pickle.close()

load_pickle = open("Natural langauge processing/pickle/word_features.pickle","rb")
word_features = pickle.load(load_pickle)
load_pickle.close()



def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
    


def sentiment(text):
    fea = find_features(text)
    
    return my_clf.classify(fea),my_clf.confidence(fea)

load_pickle = open("Natural langauge processing/pickle/BNB_clf.pickle","rb")
BNB_clf = pickle.load(load_pickle)
load_pickle.close()

load_pickle = open("Natural langauge processing/pickle/MNB_clf.pickle","rb")
MNB_clf = pickle.load(load_pickle)
load_pickle.close()

load_pickle = open("Natural langauge processing/pickle/SVC_clf.pickle","rb")
SVC_clf = pickle.load(load_pickle)
load_pickle.close()


load_pickle = open("Natural langauge processing/pickle/LSVC_clf.pickle","rb")
LSVC_clf = pickle.load(load_pickle)
load_pickle.close()

load_pickle = open("Natural langauge processing/pickle/LR_clf.pickle","rb")
LR_clf = pickle.load(load_pickle)
load_pickle.close()


load_pickle = open("Natural langauge processing/pickle/NSVC_clf.pickle","rb")
NSVC_clf = pickle.load(load_pickle)
load_pickle.close()


load_pickle = open("Natural langauge processing/pickle/SGD_clf.pickle","rb")
SGD_clf = pickle.load(load_pickle)
load_pickle.close()



my_clf = VoteClassifier(BNB_clf,MNB_clf,SVC_clf,LSVC_clf,LR_clf,SGD_clf,MNB_clf)


