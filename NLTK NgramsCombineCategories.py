import nltk
from nltk import bigrams
from nltk import word_tokenize
from nltk.collocations import *
from nltk import collections
import random
from nltk.corpus import stopwords
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import CategorizedPlaintextCorpusReader



dialect = LazyCorpusLoader('dialects1', CategorizedPlaintextCorpusReader, r'(?!\.).*\.txt', cat_pattern=r'(egyptian|gulf|levantine|standardArabic)/.*', encoding="utf-8")

sentences = [(list(dialect.words(fileid)[i:i+20]),category)
             for category in dialect.categories()
             for fileid in dialect.fileids(category)
             for i in range(0,len(dialect.words(fileid)),20)]

shuffled_sentences = random.sample(sentences,len(sentences))
print(len(sentences))

bgm = nltk.collocations.BigramAssocMeasures()
finder1 = nltk.collocations.BigramCollocationFinder.from_words(w for w in dialect.words(categories='egyptian') if not w in stopwords.words('arabic2'))
word_features1 = sorted(finder1.nbest(bgm.raw_freq, 500))
finder2 = nltk.collocations.BigramCollocationFinder.from_words(w for w in dialect.words(categories='gulf') if not w in stopwords.words('arabic2'))
word_features2 = sorted(finder2.nbest(bgm.raw_freq, 500))
finder3 = nltk.collocations.BigramCollocationFinder.from_words(w for w in dialect.words(categories='levantine') if not w in stopwords.words('arabic2'))
word_features3 = sorted(finder3.nbest(bgm.raw_freq, 500))
finder4 = nltk.collocations.BigramCollocationFinder.from_words(w for w in dialect.words(categories='standardArabic') if not w in stopwords.words('arabic2'))
word_features4 = sorted(finder4.nbest(bgm.raw_freq, 500))

topBigrams = word_features1 + word_features2 + word_features3 + word_features4
print(len(topBigrams))


def sentence_feature(sentence):
    features = {}
    bigm = list(bigrams(sentence))
    for word in topBigrams:
        features['contains({})'.format(word)] = word in bigm
    return features

print('entering feature set')
featuresets = [(sentence_feature(d), c) for (d, c) in shuffled_sentences]
print('finished feature set')
train_sets = featuresets[:int(len(featuresets)*0.8)]
test_sets =  featuresets[int(len(featuresets)*0.8):]
classifier = nltk.NaiveBayesClassifier.train(train_sets)

print(nltk.classify.accuracy(classifier,test_sets)*100 , '%')

classifier.show_most_informative_features(20)
