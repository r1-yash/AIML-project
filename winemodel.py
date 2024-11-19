#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# In[30]:


wine_dataset = pd.read_csv('winequality-red.csv')


# In[31]:


wine_dataset.shape


# In[32]:


wine_dataset.head()


# In[33]:


#find missing values 
wine_dataset.isnull().sum()


# In[34]:


wine_dataset.describe()


# In[35]:


# number of values for each quality
sns.catplot(x='quality', data = wine_dataset, kind = 'count')


# In[36]:


# volatile acidity vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'volatile acidity', data = wine_dataset)


# In[37]:


# citric acid vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'citric acid', data = wine_dataset)


# In[38]:


correlation = wine_dataset.corr()


# In[39]:


# constructing a heatmap to understand the correlation between the columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt = '.1f', annot = True, annot_kws={'size':8}, cmap = 'Blues')


# In[40]:


# separate the data and Label
X = wine_dataset.drop('quality',axis=1)


# In[41]:


print(X)


# In[42]:


Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)


# In[43]:


print(Y)


# In[44]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[45]:


print(Y.shape, Y_train.shape, Y_test.shape)


# In[46]:


model = RandomForestClassifier()
model.fit(X_train, Y_train)


# In[47]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[49]:


input_data = (7.8,0.58,0.02,2.0,0.073,9.0,18.0,0.9968,3.36,0.57,9.5)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
 print('Good Quality Wine')
else:
  print('Bad Quality Wine')


# In[ ]:


import pickle

# Assuming 'model' is your trained machine learning model
with open('winemodel.pkl', 'wb') as file:
    pickle.dump(model, file)

