from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import sklearn
from sklearn import svm
import matplotlib.pyplot  as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest , chi2 , f_classif,SelectPercentile , RFE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.utils import shuffle


dialect = load_files('C:\\Users\\elmohandes tech\\Desktop\\dialects',shuffle = False,encoding="utf-8")

x = dialect.data
y = dialect.target

x_train ,x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=4)


########################## classify with gaussianNB and TF-IDf#####################
# ftr = TfidfVectorizer()
# clf = GaussianNB()

######################### classify with MultinomialNB and TF #####################
# ftr = TfidfVectorizer(use_idf=False)
# clf = MultinomialNB()


# x_trainftr  = ftr.fit_transform(x_train)
# x_testftr = ftr.transform(x_test)
#
# clf.fit(x_trainftr.toarray(), y_train)
# pred = clf.predict(x_testftr.toarray())
#
# print('Prediction:',pred)
# print('Actual:',y_test)
# print('Accuracy:',metrics.accuracy_score(y_test,pred) *100 , '%')
# # print(metrics.classification_report(y_test, pred,target_names=dialect.target_names))
# # print(metrics.confusion_matrix(y_test, pred))


######################## classify with BernoulliNB and binary feature ############
# ftr = CountVectorizer()
# clf = BernoulliNB()
# binarizer = Binarizer(threshold=1.1)
#
# x_trainftr  = ftr.fit_transform(x_train)
# x_testftr = ftr.transform(x_test)
# b = binarizer.transform(x_trainftr)
#
# clf.fit(b, y_train)
# pred = clf.predict(x_testftr.toarray())
#
# print('Prediction:',pred)
# print('Actual:',y_test)
# print('Accuracy:',metrics.accuracy_score(y_test,pred) *100 , '%')
# print(metrics.classification_report(y_test, pred,target_names=dialect.target_names))
# print(metrics.confusion_matrix(y_test, pred))



########## classify with removing low variance feature ################
# ftr = TfidfVectorizer(use_idf=False)
# clf = MultinomialNB()
# vt = VarianceThreshold(threshold=0)
#
# x_trainftr  = ftr.fit_transform(x_train)
# x_testftr = ftr.transform(x_test)
#
# #print(x_trainftr)
# #print(x_trainftr.toarray())
#
# x_vt_train = vt.fit_transform(x_trainftr)
# x_vt_test = vt.transform(x_testftr)
#
# print(x_vt_train.shape)
# print(x_vt_train)
# print(x_vt_train.toarray())
#
#
# clf.fit(x_vt_train.toarray(), y_train)
# pred = clf.predict(x_vt_test.toarray())
#
# print('Prediction:',pred)
# print('Actual:',y_test)
# print('Accuracy:',metrics.accuracy_score(y_test,pred) *100 , '%')
# # print(metrics.classification_report(y_test, pred,target_names=dialect.target_names))
# # print(metrics.confusion_matrix(y_test, pred))


########### classify with univariant feature #################################
# ftr = TfidfVectorizer(use_idf=False)
# clf = MultinomialNB()
# sbk = SelectKBest(k=50)
# #sbk = SelectPercentile(percentile=10)
#
# x_trainftr  = ftr.fit_transform(x_train)
# x_testftr = ftr.transform(x_test)
#
# #print(x_trainftr)
# #print(x_trainftr.toarray())
#
# x_sbk_train = sbk.fit_transform(x_trainftr,y_train)
# x_sbk_test = sbk.transform(x_testftr)
#
# print(x_sbk_train.shape)
# print(x_sbk_train)
#
#
# clf.fit(x_sbk_train.toarray(), y_train)
# pred = clf.predict(x_sbk_test.toarray())
#
# print('Prediction:',pred)
# print('Actual:',y_test)
# print('Accuracy:',metrics.accuracy_score(y_test,pred) *100 , '%')
# # print(metrics.classification_report(y_test, pred,target_names=dialect.target_names))
# # print(metrics.confusion_matrix(y_test, pred))


################ classify by recursive feature elimination #################
ftr = TfidfVectorizer(use_idf=False)
clf = MultinomialNB()
rec = RFE(clf,n_features_to_select=84000)

x_trainftr  = ftr.fit_transform(x_train)
x_testftr = ftr.transform(x_test)

#print(x_trainftr)
#print(x_trainftr.toarray())

x_rec_train = rec.fit_transform(x_trainftr, y_train)
x_rec_test = rec.transform(x_testftr)

print(x_rec_train.shape)
print(x_rec_train)


clf.fit(x_rec_train.toarray(), y_train)
pred = clf.predict(x_rec_test.toarray())

print('Prediction:',pred)
print('Actual:',y_test)
print('Accuracy:',metrics.accuracy_score(y_test,pred) *100 , '%')
# print(metrics.classification_report(y_test, pred,target_names=dialect.target_names))
# print(metrics.confusion_matrix(y_test, pred))







# cv = CountVectorizer()
# x_traincv = cv.fit_transform(["ايه يا معلم عامل ايه ", "كيف حالك يا خيي , بدك تنام؟", "ماذا انت بفاعل في مثل هذه الظروف"])
# print(x_traincv.toarray())
# print(cv.get_feature_names())



################################# trying to divie it to sentences ######################################################
# print(tokenize(x[0]))
# print(len(x[0]))

# s = []
# for i in x:
#     if i != '.':
#         s.append(tokenize(i))
# print(s)
# print(len(s))


#
# sentences = [(list(dialect.words(fileid)[i:i+100]),category)
#              for category in dialect.categories()
#              for fileid in dialect.fileids(category)
#              for i in range(0,len(dialect.words(fileid)),100)]


# sentences=[(list(fileid[t:t+10]),category)
#            for category in dialect.target
#            for fileid in dialect.data[category].split()
#            for t in range(0,len(dialect.data[category].split()),10)
#            ]
# print(sentences[0])

#x1 = dialect.data[0]

# s = [(list(x1.split()[t:t+100]))
#      for i in x1.split()
#      for t in range(0,len(x1.split()),100)]
# print(s[0])

# w = []
# for i in x1.split():
#     w.append(i[:10])
# print(w)

# senta = [(list(word_tokenize(i)[n:n+20]))
#         for i in x1.split()
#         for n in range(0,len(x1.split()),20)
#     ]
#
# print(senta[0])

# for i in x1.split():
#     for t in range(0,len(x1.split()),10):
#         print(word_tokenize(i)[t:t+10])




########################################################################################################################










# clf = svm.SVC(gamma=0.001,C=1.0)
# #clf = MultinomialNB()
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(dialect.data)
#
# x_train ,x_test , y_train , y_test = train_test_split(X_train_counts,y,test_size=0.2)
# clf.fit(x_train,y_train)
# print(clf.score(x_test,y_test))
# print(y_test)
# print(clf.predict(x_test))



# x,y = X_train_counts[:-1] , dialect.target[:-1]
# clf.fit(x,y)
#
# print('prediction:' , clf.predict(dialect.data[-1]))
# plt.imshow(dialect.images[-1] , cmap=plt.cm.gray_r , interpolation='nearest')
# plt.show()
