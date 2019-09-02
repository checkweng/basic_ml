# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
# Any results you write to the current directory are saved as output.


train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")

train = train[['listing_id','description','interest_level']]
test = test[['listing_id','description']]

train['flag'] = 'train'
test['flag'] = 'test'
full_data = pd.concat([train,test])


from nltk.stem import PorterStemmer
import re


# Removes symbols, numbers and stem the words to reduce dimentional space
stemmer = PorterStemmer()

def clean(x):
    regex = re.compile('[^a-zA-Z ]')
    # For user clarity, broken it into three steps
    i = regex.sub(' ', x).lower()
    i = i.split(" ") 
    i= [stemmer.stem(l) for l in i]
    i= " ".join([l.strip() for l in i if (len(l)>2) ]) # Keeping words that have length greater than 2
    return i
    
# This takes some time to run. It would be helpful if someone can help me optimize clean() function.
full_data['description_new'] = full_data.description.apply(lambda x: clean(x))


full_data[['description','description_new']].head()

from sklearn.feature_extraction.text import CountVectorizer #Can use tfidffvectorizer as well

cvect_desc = CountVectorizer(stop_words='english', max_features=200)
full_sparse = cvect_desc.fit_transform(full_data.description_new)
 # Renaming words to avoid collisions with other feature names in the model
col_desc = ['desc_'+ i for i in cvect_desc.get_feature_names()] 
count_vect_df = pd.DataFrame(full_sparse.todense(), columns=col_desc)
full_data = pd.concat([full_data.reset_index(),count_vect_df],axis=1)

train =(full_data[full_data.flag=='train'])
test =(full_data[full_data.flag=='test'])

labels = {'high':0, 'medium':1, 'low':2}
train['interest_level'] = train.interest_level.apply(lambda x: labels[x])

feat = train.drop(['interest_level','flag','listing_id','description','index','description_new'],axis=1).columns.values

from sklearn.ensemble import GradientBoostingClassifier  as GBM
from sklearn.ensemble import RandomForestClassifier  as RF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss



###一下为重点

def run_mod(train_X, test_X,train_Y):
    reg = GBM(max_features = 'auto',n_estimators=200,random_state=1)
    reg.fit(train_X,train_Y)
    pred = reg.predict_proba(test_X)
    imp = reg.feature_importances_
    return pred,imp

def cross_val(train,feat,split):
    cv_scores = []
    importances = []
    # Cross Validation preprocessing
    train_X = train[feat]
    train_Y = train['interest_level']

    train_X = train_X.as_matrix()
    train_Y = train_Y.as_matrix()

    test_X = test[feat]
    test_X = test_X.as_matrix()

    kf = StratifiedKFold(n_splits=split, shuffle=True, random_state=1)
    for dev_index, val_index in kf.split(train_X,train_Y):
            train_X_X, test_X_X = train_X[dev_index,:], train_X[val_index,:]
            train_Y_Y, test_Y_Y = train_Y[dev_index,], train_Y[val_index,]
            pred,imp = run_mod(train_X_X, test_X_X,train_Y_Y)
            cv_scores.append(log_loss(test_Y_Y, pred))
            importances.append(imp)
    return np.mean(cv_scores),importances
#print np.average(importances,axis=0)

cv_score,imp = cross_val(train,feat,3)


# Lets chaeck the importance of words
importances = list(np.average(imp,axis=0))
features = cvect_desc.get_feature_names()
df = pd.DataFrame({'words':features,'imp':importances}).sort_values(by='imp',ascending=False).head(30)
plt.figure(figsize=(12,15))
sns.barplot(y=df.words,x=df.imp)
# Remember, these are stemmed words
