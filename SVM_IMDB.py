import numpy as np
import os
import string
import sys
import time
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
import random as rn
#All this for reproducibility
np.random.seed(1)
rn.seed(1)
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk_stopw = stopwords.words('english')

def tokenize (text):        #   no punctuation & starts with a letter & between 2-15 characters in length
    tokens = [word.strip(string.punctuation) for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    return  [f.lower() for f in tokens if f and f.lower() not in nltk_stopw]

def getMovies():
    X, labels, labelToName  = [], [], { 0 : 'neg', 1: 'pos' }
    for dataset in ['train', 'test']:
        for classIndex, directory in enumerate(['neg', 'pos']):
            dirName = './data/' + dataset + "/" + directory
            for reviewFile in os.listdir(dirName):
                with open (dirName + '/' + reviewFile, 'r') as f:
                    tokens = tokenize (f.read())
                    if (len(tokens) == 0):
                        continue
                X.append(tokens)
                labels.append(classIndex)
    nTokens = [len(x) for x in X]
    return X, np.array(labels), labelToName, nTokens

X, labels, labelToName, nTokens = getMovies()
print ('Token Summary. min/avg/median/std/85/86/87/88/89/90/95/99/max:',)
print (np.amin(nTokens), np.mean(nTokens),np.median(nTokens),np.std(nTokens),np.percentile(nTokens,85),np.percentile(nTokens,86),np.percentile(nTokens,87),np.percentile(nTokens,88),np.percentile(nTokens,89),np.percentile(nTokens,90),np.percentile(nTokens,95),np.percentile(nTokens,99),np.amax(nTokens))
labelToNameSortedByLabel = sorted(labelToName.items(), key=lambda kv: kv[0]) # List of tuples sorted by the label number [ (0, ''), (1, ''), .. ]
namesInLabelOrder = [item[1] for item in labelToNameSortedByLabel]
numClasses = len(namesInLabelOrder)
print ('X, labels #classes classes {} {} {} {}'.format(len(X), str(labels.shape), numClasses, namesInLabelOrder))

X=np.array([np.array(xi) for xi in X])          #   rows: Docs. columns: words
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1).fit(X)
word_index = vectorizer.vocabulary_
Xencoded = vectorizer.transform(X)
print ('Vocab sparse-Xencoded {} {}'.format(len(word_index), str(Xencoded.shape)))

# Test & Train Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(Xencoded, labels)
train_indices, test_indices = next(sss)
train_x, test_x = Xencoded[train_indices], Xencoded[test_indices]
train_labels, test_labels = labels[train_indices], labels[test_indices]
start_time = time.time()
model = LinearSVC(tol=1.0e-6,max_iter=20000,verbose=1)
model.fit(train_x, train_labels)
predicted_labels = model.predict(test_x)
elapsed_time = time.time() - start_time
results = {}
results['confusion_matrix'] = confusion_matrix(test_labels, predicted_labels).tolist()
results['classification_report'] = classification_report(test_labels, predicted_labels, digits=4, target_names=namesInLabelOrder, output_dict=True)

print (confusion_matrix(labels[test_indices], predicted_labels))
print (classification_report(labels[test_indices], predicted_labels, digits=4, target_names=namesInLabelOrder))
print ('Time Taken:', elapsed_time)
results['elapsed_time'] = elapsed_time        # seconds

f = open ('svm.json','w')
out = json.dumps(results, ensure_ascii=True)
f.write(out)
f.close()
