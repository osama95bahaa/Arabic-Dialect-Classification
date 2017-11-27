import nltk
import random
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.text import TextCollection
from collections import Counter
from nltk.tokenize import word_tokenize
from operator import itemgetter



dialect = LazyCorpusLoader('dialects1', CategorizedPlaintextCorpusReader, r'(?!\.).*\.txt', cat_pattern=r'(egyptian|gulf|levantine|standardArabic)/.*', encoding="utf-8")
x = TextCollection(dialect)

sentences = [(list(dialect.words(fileid)[i:i+40]),category)
             for category in dialect.categories()
             for fileid in dialect.fileids(category)
             for i in range(0,len(dialect.words(fileid)),40)]


shuffled_sentences = random.sample(sentences, len(sentences))
print('sentences count',len(sentences))

text = dialect.words()
print('words count',len(text))

#################### Test with getting topN ############################################################################
# all_words = nltk.FreqDist(w for w in dialect.words())
# Mcommon = all_words.most_common(4000)
# topN = [i[0] for i in Mcommon]
# print('finished topN')
########################################################################################################################


########### test with TopN per category then combine####################################################################
########### get top 100 words from each category and then add them in one list##########
# def topNwords():
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
#
# topN = [i[0] for i in topNwords()]
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
    topN.append(i[0])
print('topN' , topN)
########################################################################################################################



## Getting TF-IDF of the TopN words to get max and min##
s = [d for (d,c) in sentences]
tfList = []
for sen in s:
    for word in sen:
        if word in topN:
            tfList.append(x.tf_idf(word,sen))

print(len(tfList))
max= max(tfList)
print('max' ,max)
min= min(tfList)
print('min' , min)
res = max - min
print('res' , res)
half = res/2
print('half',half)
twoThird = (max+half)/2
print('twoThird' , twoThird)
quarter = half/2
print('quarter',quarter)

def sentence_feature(sentence):
    features = {}
    for word in sentence:
        if word in topN:
            t = x.tf_idf(word,sentence)
            if t >=min and t < quarter:
                result = 0
            elif t >= quarter and t < half:
                result = 1
            elif t >= half and t < twoThird:
                result = 2
            elif t >= twoThird and t <= max:
                result = 3
            else:
                result = 4

            features['contains({})'.format(word)] = result
    return features

## next time try in the feature to say for word in topN ##
## try to give the feature the t not the result##

featuresets = [(sentence_feature(d),c) for (d,c) in shuffled_sentences]
train_sets = featuresets[:int(len(featuresets)*0.8)]
test_sets =  featuresets[int(len(featuresets)*0.8):]

classifier = nltk.NaiveBayesClassifier.train(train_sets)

print(nltk.classify.accuracy(classifier,test_sets)*100 , '%')

classifier.show_most_informative_features(30)

