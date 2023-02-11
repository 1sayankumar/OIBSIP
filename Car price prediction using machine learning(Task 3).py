#!/usr/bin/env python
# coding: utf-8

# In[52]:


#importing librarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns


# In[53]:


# this is the path of car price dataset
path = "https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv"


# In[54]:


# read the datasets using pandas library
df = pd.read_csv(path)


# In[55]:


#here is the first five columns of the car 
df.head()


# In[56]:


df=df.drop(columns=['car_ID'])
df.head()


# In[57]:


df.describe()


# In[58]:


df.info()


# In[59]:


#checking null values 
df.isnull().sum()


# # EDA process

# In[60]:


# this graph is showing variours price of cars 
plt.figure(figsize=(20,15))
plt.plot(df['price'])
plt.title('price of cars',fontsize=20)
plt.ylabel("car price in doller")
plt.show()


# In[61]:


#display the no of fuel type cars
df['fueltype'].value_counts()


# In[62]:


df['aspiration'].value_counts()


# In[63]:


df['doornumber'].value_counts()


# In[64]:


df['carbody'].value_counts()


# In[65]:


df['drivewheel'].value_counts()


# In[66]:


df['enginelocation'].value_counts()


# In[67]:


df['fuelsystem'].value_counts()


# In[68]:


# here is the correlation
df.corr()


# In[69]:


# here is showing correlation coefficients between variables using sns library
corr=df.corr()
fig,ax=plt.subplots(figsize=(30,20))
sns.heatmap(corr,annot=True,ax=ax)


# In[70]:


#label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[71]:


df['CarName'] = le.fit_transform(df['CarName'])
df['fueltype'] = le.fit_transform(df['fueltype'])
df['aspiration'] = le.fit_transform(df['aspiration'])
df['doornumber'] = le.fit_transform(df['doornumber'])
df['carbody'] = le.fit_transform(df['carbody'])
df['drivewheel'] = le.fit_transform(df['drivewheel'])
df['enginelocation'] = le.fit_transform(df['enginelocation'])
df['fuelsystem'] = le.fit_transform(df['fuelsystem'])
df['enginetype'] = le.fit_transform(df['enginetype'])
df['cylindernumber'] = le.fit_transform(df['cylindernumber'])
df['fuelsystem'] = le.fit_transform(df['fuelsystem'])


# In[72]:


df.head()


# In[73]:


X = df.drop(columns=['CarName','price'])
Y = df['price']


# In[74]:


#train test split method
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)


# In[75]:


# prediction of the model using linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[76]:


model.fit(x_train,y_train)


# In[77]:


training_data_prediction = model.predict(x_train)


# In[78]:


#i have calculated the r2 score of the model
from sklearn import metrics
error_score = metrics.r2_score(y_train,training_data_prediction)
print("R2 square of the model:",error_score*100)


# In[79]:


# i have visualise the actual and predicted price through this graph
plt.scatter(y_train,training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()


# In[80]:


test_data_prediction = model.predict(x_test)


# In[81]:


error_score = metrics.r2_score(y_test,test_data_prediction)
print("R2 square of the model:",error_score*100)


# In[82]:


plt.scatter(y_test,test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()


# In[ ]:




