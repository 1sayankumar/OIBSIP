#!/usr/bin/env python
# coding: utf-8

# # Sayan Kumar
# # Task 1 - Iris flower classification
# # OASIS INFOBYTE

# In[121]:


#import librarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#read the datasets
df = pd.read_csv('Iris.csv')


# In[3]:


df.head()


# In[4]:


df=df.drop(columns=['Id'])
df.head()


# In[5]:


#to display stats about data
df.describe()


# In[6]:


# this is the basic info of the datatype
df.info()


# In[7]:


# to display the no of each specices
df['Species'].value_counts()


# In[8]:


#checking null values
df.isnull().sum()


# In[9]:


df['SepalLengthCm'].hist()


# In[10]:


df['SepalWidthCm'].hist()


# In[11]:


df['PetalLengthCm'].hist()


# In[12]:


df['PetalWidthCm'].hist()


# # EDA PROCESS

# In[13]:


color = ['red','blue','green']
species = ['Iris-setosa','Iris-versicolor','Iris-virginica']


# In[15]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'],c=color[i],label = species[i])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()


# In[18]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'],c=color[i],label = species[i])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()


# In[17]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'],c=color[i],label = species[i])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend()


# In[19]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'],c=color[i],label = species[i])
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')
plt.legend()


# In[20]:


df.corr()


# In[21]:


corr = df.corr()
fig,ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr,annot = True,ax=ax)


# In[25]:


#label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[26]:


#transform the string value to int value
df['Species'] = le.fit_transform(df['Species'])
df.head()


# In[110]:


#model training using train test split method
from sklearn.model_selection import train_test_split
X = df.drop(columns = ['Species'])
Y = df['Species']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.30)


# In[111]:


#logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[112]:


model.fit(x_train,y_train)


# In[113]:


print("Accuracy:",model.score(x_test,y_test)*100)


# In[114]:


#knn classifier
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[115]:


model.fit(x_train,y_train)


# In[116]:


print("Accuracy:",model.score(x_test,y_test)*100)


# In[117]:


# desicion tree classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[118]:


model.fit(x_train,y_train)


# In[119]:


print("Accuracy:",model.score(x_test,y_test)*100)


# In[ ]:




