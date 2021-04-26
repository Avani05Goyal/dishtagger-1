#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv('data.csv', usecols = ['item_name','category_new'])


# In[3]:


data.info()


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data['item_name'].value_counts()


# In[7]:


data.category_new.value_counts()


# In[8]:


# data['category_new'] = data['category_new'].map({'soup':0,'rice':1,'main course':2,'bread':3,'starters':4,'dessert':5,'beverages':6}).astype(np.int)


# In[9]:


y = data['category_new']
y.head()


# In[10]:


data.category_new.value_counts()


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['item_name'],y, test_size=0.2, random_state=0,stratify=y)


# In[33]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_train)
XTest=vectorizer.transform(X_test)


# In[13]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X,y_train)


# In[14]:


y_pred = nb.predict(XTest)


# In[15]:


print(y_pred.shape)
print(X_test.shape)
print(y_test.shape)


# In[29]:


final=np.vstack((X_test, y_pred))
print(final)


# In[30]:


print(X_test)


# In[32]:


print(y_test)


# In[17]:


from sklearn.metrics import confusion_matrix


# In[18]:


y_test.value_counts()


# In[19]:


y_test.value_counts().head(7) / len(y_test)


# In[20]:


print('True:', y_test.values[0:25])
print('False:', y_pred[0:25])





# In[21]:


print(confusion_matrix(y_test, y_pred))


# In[22]:


confusion = confusion_matrix(y_test, y_pred)
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]


# In[23]:


from sklearn.metrics import accuracy_score


# print((TP + TN) / float(TP + TN + FP + FN))
print(accuracy_score(y_test, y_pred))


# In[24]:


classification_error = (FP + FN) / float(TP + TN + FP + FN)

print(classification_error)
print(1 - accuracy_score(y_test, y_pred))


# In[25]:


import sklearn.metrics as metrics

from sklearn.metrics import recall_score 
sensitivity = TP / float(FN + TP)

print(sensitivity)
# print(recall_score(y_test, y_pred))


# In[26]:


specificity = TN / (TN + FP)

print(specificity)





# In[27]:


false_positive_rate = FP / float(TN + FP)

print(false_positive_rate)
print(1 - specificity)


# In[28]:


precision = TP / float(TP + FP)

print(precision)
# print(metrics.precision_score(y_test, y_pred))


# In[ ]:




