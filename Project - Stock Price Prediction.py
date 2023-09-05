#!/usr/bin/env python
# coding: utf-8

# # Stock Price Prediction Project

# In[1]:


from platform import python_version
print(python_version())


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
import seaborn as sb
 
import warnings
warnings.filterwarnings('ignore')


# ## Importing Data

# In[3]:


stock_df=pd.read_csv('Stock_Price_data_set.csv')


# In[4]:


stock_df


# ## Exploring Data

# In[5]:


stock_df.shape


# In[6]:


stock_df.info()


# In[7]:


stock_df.columns


# In[8]:


stock_df.describe


# In[9]:


plt.figure(figsize=(15,5))
plt.plot(stock_df['Low'],color="red",label='Low')
plt.plot(stock_df['High'],color="green",label='High')
plt.title('Stock Price', fontsize=15)

plt.ylabel('Price')
plt.show()


# ## Missing Values

# In[10]:


stock_df.isna().any()


# ## Duplicates

# In[11]:


stock_df.duplicated().sum()


# ## Column Data Type

# In[12]:


stock_df.dtypes


# ## Outliers

# In[13]:


plt.subplot(2,3,1)
stock_df['Open'].plot(kind='box') 

plt.subplot(2,3,2)
stock_df['Close'].plot(kind='box')

plt.subplot(2,3,3)
stock_df['Adj Close'].plot(kind='box')

plt.subplot(2,3,4)
stock_df['High'].plot(kind='box')

plt.subplot(2,3,5)
stock_df['Low'].plot(kind='box')

plt.subplot(2,3,6)
stock_df['Volume'].plot(kind='box')

plt.tight_layout()


# In[14]:


stock_df['Volume'].plot(kind='box')


# In[15]:


def find_outlier_limits(col_name):
    Q1,Q3=stock_df[col_name].quantile([.25,.75])
    IQR=Q3-Q1
    low=Q1-(2* IQR)
    high=Q3+(1* IQR)
    return (high,low)

high_vol,low_vol=find_outlier_limits('Volume')
print('Volume: ','upper limit: ',high_vol,' lower limit: ',low_vol)


# In[16]:


low_limit = 0
print('Volume: ','upper limit: ',high_vol,'lower limit: ',low_limit)


# In[17]:


#replacing outliers value
stock_df.loc[stock_df['Volume'] > high_vol,'Volume'] = high_vol

stock_df.loc[stock_df['Volume']>high_vol,'Volume']=high_vol


# In[18]:


plt.subplot(2,3,1)
stock_df['Open'].plot(kind='box') 

plt.subplot(2,3,2)
stock_df['Close'].plot(kind='box')

plt.subplot(2,3,3)
stock_df['Adj Close'].plot(kind='box')

plt.subplot(2,3,4)
stock_df['High'].plot(kind='box')

plt.subplot(2,3,5)
stock_df['Low'].plot(kind='box')

plt.subplot(2,3,6)
stock_df['Volume'].plot(kind='box')

plt.tight_layout()


# In[19]:


stock_df['Volume'].plot(kind='box')


# In[20]:


outliers = [stock_df['Volume'] > high_vol,'Volume']
outliers[True]


# ## ML MODELING

# In[21]:


stock_df


# In[22]:


X = stock_df.iloc[:, 1:8]
X = pd.get_dummies(X)
X


# In[23]:


features = ['Open', 'High', 'Low', 'Close', 'Volume']
 
plt.subplots(figsize=(20,10))
 
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(stock_df[col])
plt.show()


# In[24]:


splitted = stock_df['Date'].str.split('-', expand=True)
 
stock_df['month'] = splitted[1].astype('int')
stock_df['year'] = splitted[0].astype('int')
stock_df['date'] = splitted[2].astype('int')
 
stock_df.head()


# In[25]:


stock_df['is_quarter_end'] = np.where(stock_df['month']%3==0,1,0)
stock_df.head()


# In[26]:


df=stock_df


# In[27]:


data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20,10))
 
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
plt.show()


# In[28]:


df.groupby('is_quarter_end').mean()


# In[29]:


df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


# In[30]:


plt.pie(df['target'].value_counts().values,labels=[0, 1], autopct='%1.1f%%')
plt.show()


# ### Correlation

# In[31]:


plt.figure(figsize=(15, 10))
sb.heatmap(df.corr(), annot=True)
plt.show()


# ## DATA SPLITTING AND NORMALISING

# In[32]:


features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']
 
scaler = StandardScaler()
features = scaler.fit_transform(features)
 
X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=20)
print(X_train.shape, X_valid.shape)


# In[33]:


models = [LogisticRegression(), SVC(kernel='poly', probability=True)]
 
for i in range(2):
  models[i].fit(X_train, Y_train)
 
  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()


# In[34]:


metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()


# We can observe that the accuracy achieved by the ML model is no better than simply guessing with a probability of 50%. Possible reasons for this may be the lack of data or using a very simple model to perform such a complex task as Stock Market prediction.

# In[ ]:




