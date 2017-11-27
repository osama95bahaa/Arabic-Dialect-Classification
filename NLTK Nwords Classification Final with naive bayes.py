import nltk
import random
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from collections import Counter
from nltk.corpus import stopwords
from nltk.text import TextCollection
from operator import itemgetter


dialect = LazyCorpusLoader('dialects1', CategorizedPlaintextCorpusReader, r'(?!\.).*\.txt', cat_pattern=r'(egyptian|gulf|levantine|standardArabic)/.*', encoding="utf-8")

######Get Nwords in the corpus############
sentences = [(list(dialect.words(fileid)[i:i+100]),category)
             for category in dialect.categories()
             for fileid in dialect.fileids(category)
             for i in range(0,len(dialect.words(fileid)),100)]

print(sentences[40])

shuffled_sentences = random.sample(sentences, len(sentences))
print(len(shuffled_sentences))

text = dialect.words()
print(len(text))

x = TextCollection(dialect)

########### test with TopN per category then combine####################################################################
########### get top 100 words from each category and then add them in one list##########
# def topN():
#     egyWords = nltk.FreqDist(w for w in dialect.words(categories='egyptian') )
#     a = egyWords.most_common(1000)
#
#     gulfWords = nltk.FreqDist(w for w in dialect.words(categories='gulf') )
#     b = gulfWords.most_common(1000)
#
#     levWords = nltk.FreqDist(w for w in dialect.words(categories='levantine') )
#     c = levWords.most_common(1000)
#
#     MSAWords = nltk.FreqDist(w for w in dialect.words(categories='standardArabic'))
#     d = MSAWords.most_common(1000)
#
#     result = a + b + c + d
#     uniqueResult =[]
#     for i in result:
#         if not i in uniqueResult:
#             uniqueResult.append(i)
#     return uniqueResult
########################################################################################################################


######### Test with topN words from allover the corpus##################################################################
# all_words = nltk.FreqDist(w for w in dialect.words())
# topN = all_words.most_common(4000)
########################################################################################################################



########## Test with getting topN with highest idf #####################################################################
uniqueWords =[]
for i in text:
    if not i in uniqueWords:
        uniqueWords.append(i)
print('unique words count', len(uniqueWords))

wordsIDF = [(word,x.idf(word))
            for word in uniqueWords]
#print('finished getting idf for all unique words' , wordsIDF)

sortedidfs = sorted(wordsIDF,key=itemgetter(1), reverse = True)

#print('sortedidfs' , sortedidfs)
print('length of words with idfs' , len(sortedidfs))

eliminateBiggerThanOne = []
for i in sortedidfs:
    if i[1] < 1:
        eliminateBiggerThanOne.append(i)
print('eliminated bigger than one' , eliminateBiggerThanOne)
print('length of elimnation' , len(eliminateBiggerThanOne))

topN = []
for i in eliminateBiggerThanOne[:2000]:
    topN.append(i)
print('topN' , topN)
########################################################################################################################


word_features = [i[0] for i in topN]
print(len(topN))

def sentence_feature(sentence):
    sentence_word = set(sentence)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = word in sentence_word
    return features


featuresets = [(sentence_feature(d),c) for (d,c) in shuffled_sentences]
train_sets = featuresets[:int(len(featuresets)*0.8)]
test_sets =  featuresets[int(len(featuresets)*0.8):]

classifier = nltk.NaiveBayesClassifier.train(train_sets)

print(nltk.classify.accuracy(classifier,test_sets)*100 , '%')

classifier.show_most_informative_features(10)
