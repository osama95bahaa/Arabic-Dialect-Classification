import sklearn
from sklearn import svm
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest ,RFE , SelectPercentile
from sklearn import metrics


dialect = load_files('C:\\Users\\elmohandes tech\\Desktop\\dialect\\dialects20',shuffle = True,encoding="utf-8")

x_train ,x_test , y_train , y_test = train_test_split(dialect.data,dialect.target,test_size=0.2)

vectorizer = CountVectorizer(max_features=2000,binary=True)
clf = BernoulliNB()

#clf = svm.SVC()
#clf = RandomForestClassifier()
#clf = MLPClassifier()
# vt = VarianceThreshold(threshold=0.0001)
#sbk = SelectPercentile(percentile=100)
# rec = RFE(clf)


train_data_features = vectorizer.fit_transform(x_train)
test_data_features = vectorizer.transform(x_test)
print(train_data_features.shape)
# print(train_data_features)
# print(train_data_features.toarray())

# x_vt_train = sbk.fit_transform(train_data_features,y_train)
# x_vt_test = sbk.transform(test_data_features)
# print(x_vt_train.shape)

clf.fit(train_data_features,y_train)
pred = clf.predict(test_data_features.toarray())

print('Prediction:',pred)
print('Actual:',y_test)
print('Accuracy:',metrics.accuracy_score(y_test,pred) *100 , '%')
# print(metrics.classification_report(y_test, pred,target_names=dialect.target_names))
# print(metrics.confusion_matrix(y_test, pred))


vocab = vectorizer.get_feature_names()
print(vocab)
print(len(vocab))



######## trying to give some eamples and test the output##########
# docs_new = ['هذا الشخص سعيد في حياته','بس يا خيي انا بدي انام','بقولك عايز العب ','والله لو دفعنا مبلغ بسيط يمكن يرد عنا وعن أهلنا نصايب كبيرة']
# X_new_counts = vectorizer.transform(docs_new)
# predicted = clf.predict(X_new_counts)
# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, dialect.target_names[category]))

