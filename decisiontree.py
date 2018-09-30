
# coding: utf-8

# In[2]:


#!/usr/bin/env python
from datasets import load_training_data,load_test_data
from preprocessing import fill_missing_values

""" get training data """
Xtr,Ytr=load_training_data("training-and-test/training.json")
Xtr=fill_missing_values(Xtr)

""" get test data """
Xte,Yte=load_test_data("training-and-test/sample-test.in.json","training-and-test/sample-test.out.json")
Xte=fill_missing_values(Xte)

"""training"""
from sklearn.tree import DecisionTreeClassifier
learner = DecisionTreeClassifier(max_depth=7,random_state=0).fit(Xtr, Ytr)

"""predicting"""
label_test = learner.predict(Xte)
print(label_test)

print('Accuracy is ', accuracy_score(Yte,label_test)*100)
print(learner.score(Xtr,Ytr))
print(learner.score(Xte,Yte))

