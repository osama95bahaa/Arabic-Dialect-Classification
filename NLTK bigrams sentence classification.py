import nltk
from nltk import bigrams
from nltk import word_tokenize
from nltk.collocations import *
from nltk import collections
import random
from nltk.corpus import stopwords
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import CategorizedPlaintextCorpusReader

# string = ['sadas','sdads','erwrw','rewhf']
# sbigram = list(bigrams(string))
# print(sbigram)


dialect = LazyCorpusLoader('dialects1', CategorizedPlaintextCorpusReader, r'(?!\.).*\.txt', cat_pattern=r'(egyptian|gulf|levantine|standardArabic)/.*', encoding="utf-8")

sentences = [(list(dialect.words(fileid)[i:i+20]),category)
             for category in dialect.categories()
             for fileid in dialect.fileids(category)
             for i in range(0,len(dialect.words(fileid)),20)]


shuffled_sentences = random.sample(sentences,len(sentences))
print(len(sentences))

# bigrams_measures = nltk.collocations.BigramAssocMeasures()
# finder = BigramCollocationFinder.from_words(dialect.words())
# finder.apply_freq_filter(10)
# print(finder.nbest(bigrams_measures.pmi,50))


bgm = nltk.collocations.BigramAssocMeasures()
finder = nltk.collocations.BigramCollocationFinder.from_words(w for w in dialect.words() if not w in stopwords.words('arabic'))
word_features = sorted(finder.nbest(bgm.raw_freq,2000))
print(len(word_features))
# print(sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[:15])
# scored = finder.score_ngrams(bgm.likelihood_ratio)
# print(scored[:15])


def sentence_feature(sentence):
    features = {}
    bigm = list(bigrams(sentence))
    for word in word_features:
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
