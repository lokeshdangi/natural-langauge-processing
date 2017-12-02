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

#stopwords,tokenizing,stemming,tagging,chunking(grouping),chinking,named entity,lemmatizing(change to other word),wordnet(similarity)

from nltk.corpus import movie_reviews
document = []
for cat in movie_reviews.categories():
    for fid in movie_reviews.fileids(cat):
        document.append([movie_reviews.words(fid),cat])
        
        
random.shuffle(document)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

feature_set = [(find_features(rev),category) for (rev,category) in document]

train_set = feature_set[:1900]
test_set = feature_set[1900:]

clf = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(clf,test_set))


clf.show_most_informative_features(15)

save_clf = open("my_classifier","wb")
pickle.dump(clf,save_clf)
save_clf.close()


clf_f = open("my_classifier","rb")
clf1 = pickle.load(clf_f)
clf_f.close()

MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(train_set)
print(nltk.classify.accuracy(MNB_clf,test_set))

BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(train_set)
print(nltk.classify.accuracy(BNB_clf,test_set))

SVC_clf = SklearnClassifier(SVC())
SVC_clf.train(train_set)
print(nltk.classify.accuracy(SVC_clf,test_set))

LSVC_clf = SklearnClassifier(LinearSVC())
LSVC_clf.train(train_set)
print(nltk.classify.accuracy(LSVC_clf,test_set))

NSVC_clf = SklearnClassifier(NuSVC())
NSVC_clf.train(train_set)
print(nltk.classify.accuracy(NSVC_clf,test_set))

LR_clf = SklearnClassifier(LogisticRegression())
LR_clf.train(train_set)
print(nltk.classify.accuracy(LR_clf,test_set))

SGD_clf = SklearnClassifier(SGDClassifier())
SGD_clf.train(train_set)
print(nltk.classify.accuracy(SGD_clf,test_set))

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
    
    
my_clf = VoteClassifier(MNB_clf,BNB_clf,MNB_clf,SVC_clf,LSVC_clf,LR_clf,SGD_clf)
        
nltk.classify.accuracy(my_clf,test_set)

print(my_clf.classify(test_set[0][0]),my_clf.confidence(test_set[0][0]))
print(my_clf.classify(test_set[1][0]),my_clf.confidence(test_set[1][0]))
print(my_clf.classify(test_set[2][0]),my_clf.confidence(test_set[2][0]))
print(my_clf.classify(test_set[3][0]),my_clf.confidence(test_set[4][0]))


#######################################################################################################


pos = open("Natural langauge processing/short_reviews/positive.txt","r").read()
neg = open("Natural langauge processing/short_reviews/negative.txt","r").read()

document = []

allowed_tags = ["J"]

for r in pos.split('\n'):
    document.append((r,"pos"))

    
for r in neg.split('\n'):
    document.append((r,"neg"))

all_words = []

pos_words = word_tokenize(pos)
neg_words = word_tokenize(neg)

all_words = []

for w in pos_words:
    all_words.append(w.lower())
    
for w in neg_words:
    all_words.append(w.lower())
    

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
    
    
feature_set = []
feature_set = [(find_features(rev),category) for (rev,category) in document]

random.shuffle(feature_set)

train_set = feature_set[:10000]
test_set = feature_set[10000:]
        
MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(train_set)
print(nltk.classify.accuracy(MNB_clf,test_set))

BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(train_set)
print(nltk.classify.accuracy(BNB_clf,test_set))

SVC_clf = SklearnClassifier(SVC())
SVC_clf.train(train_set)
print(nltk.classify.accuracy(SVC_clf,test_set))

LSVC_clf = SklearnClassifier(LinearSVC())
LSVC_clf.train(train_set)
print(nltk.classify.accuracy(LSVC_clf,test_set))

NSVC_clf = SklearnClassifier(NuSVC())
NSVC_clf.train(train_set)
print(nltk.classify.accuracy(NSVC_clf,test_set))

LR_clf = SklearnClassifier(LogisticRegression())
LR_clf.train(train_set)
print(nltk.classify.accuracy(LR_clf,test_set))

SGD_clf = SklearnClassifier(SGDClassifier())
SGD_clf.train(train_set)
print(nltk.classify.accuracy(SGD_clf,test_set))

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
    
    
my_clf = VoteClassifier(MNB_clf,BNB_clf,MNB_clf,SVC_clf,LSVC_clf,LR_clf,SGD_clf)
        
nltk.classify.accuracy(my_clf,test_set)
        

############################################################################################

    

pos = open("Natural langauge processing/short_reviews/positive.txt","r").read()
neg = open("Natural langauge processing/short_reviews/negative.txt","r").read()

document = []

allowed_tags = ["J"]
all_words = []

for r in pos.split('\n'):
    document.append((r,"pos"))
    words = word_tokenize(r)
    pos_tagged = nltk.pos_tag(words)
    for w in pos_tagged:
        if w[1][0] in allowed_tags:
            all_words.append(w[0].lower())
    
for r in neg.split('\n'):
    document.append((r,"neg"))
    words = word_tokenize(r)
    pos_tagged = nltk.pos_tag(words)
    for w in pos_tagged:
        if w[1][0] in allowed_tags:
            all_words.append(w[0].lower())
    

pickle_save = open("Natural langauge processing/pickle/document.pickle","wb")
pickle.dump(document,pickle_save)
pickle_save.close()


all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:4000]

pickle_save = open("Natural langauge processing/pickle/word_features.pickle","wb")
pickle.dump(word_features,pickle_save)
pickle_save.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
    
    
feature_set = []
feature_set = [(find_features(rev),category) for (rev,category) in document]

random.shuffle(feature_set)


pickle_save = open("Natural langauge processing/pickle/features_set.pickle","wb")
pickle.dump(feature_set,pickle_save)
pickle_save.close()

train_set = feature_set[:10000]
test_set = feature_set[10000:]
        
MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(train_set)
print(nltk.classify.accuracy(MNB_clf,test_set))

pickle_save = open("Natural langauge processing/pickle/MNB_clf.pickle","wb")
pickle.dump(MNB_clf,pickle_save)
pickle_save.close()


BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(train_set)
print(nltk.classify.accuracy(BNB_clf,test_set))

pickle_save = open("Natural langauge processing/pickle/BNB_clf.pickle","wb")
pickle.dump(BNB_clf,pickle_save)
pickle_save.close()

SVC_clf = SklearnClassifier(SVC())
SVC_clf.train(train_set)
print(nltk.classify.accuracy(SVC_clf,test_set))

pickle_save = open("Natural langauge processing/pickle/SVC_clf.pickle","wb")
pickle.dump(SVC_clf,pickle_save)
pickle_save.close()

LSVC_clf = SklearnClassifier(LinearSVC())
LSVC_clf.train(train_set)
print(nltk.classify.accuracy(LSVC_clf,test_set))

pickle_save = open("Natural langauge processing/pickle/LSVC_clf.pickle","wb")
pickle.dump(LSVC_clf,pickle_save)
pickle_save.close()


NSVC_clf = SklearnClassifier(NuSVC())
NSVC_clf.train(train_set)
print(nltk.classify.accuracy(NSVC_clf,test_set))

pickle_save = open("Natural langauge processing/pickle/NSVC_clf.pickle","wb")
pickle.dump(NSVC_clf,pickle_save)
pickle_save.close()


LR_clf = SklearnClassifier(LogisticRegression())
LR_clf.train(train_set)
print(nltk.classify.accuracy(LR_clf,test_set))

pickle_save = open("Natural langauge processing/pickle/LR_clf.pickle","wb")
pickle.dump(LR_clf,pickle_save)
pickle_save.close()

SGD_clf = SklearnClassifier(SGDClassifier())
SGD_clf.train(train_set)
print(nltk.classify.accuracy(SGD_clf,test_set))

pickle_save = open("Natural langauge processing/pickle/SGD_clf.pickle","wb")
pickle.dump(SGD_clf,pickle_save)
pickle_save.close()


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
    
    
my_clf = VoteClassifier(MNB_clf,BNB_clf,MNB_clf,SVC_clf,LSVC_clf,LR_clf,SGD_clf)
        
nltk.classify.accuracy(my_clf,test_set)


def sentiment(text):
    fea = find_features(text)
    
    return my_clf.confidence(fea)