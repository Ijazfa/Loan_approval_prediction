#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import seaborn as sns
from sklearn import svm


# In[3]:


df = pd.read_csv("C:/Users/Ijaz khan/OneDrive/Desktop/lone.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.tail()


# In[8]:


df.describe()


# In[9]:


df.isnull().sum()


# In[10]:


df['loanamount_log'] = np.log(df['LoanAmount'])
df['loanamount_log'].hist(bins=20)


# In[11]:


df.isnull().sum()


# In[12]:


df['Totalincome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Totalincome_log'] = np.log(df['Totalincome'])
df['Totalincome_log'].hist(bins=20)


# In[13]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace = True)
df['Married'].fillna(df['Married'].mode()[0],inplace = True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace = True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace = True)

df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
df.loanamount_log = df.loanamount_log.fillna(df.loanamount_log.mean())

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace = True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace = True)

df.isnull().sum()


# In[14]:


x = df.iloc[:,np.r_[1:5,9:11,13:15]].values
y = df.iloc[:,12].values

x


# In[15]:


y


# In[16]:


print("Number of peoples are eligable as per the gender :")
print(df['Gender'].value_counts())
sns.countplot(x='Gender',data=df, palette = 'Set1')


# In[17]:


print("Number of peoples are eligable as per the Married :")
print(df['Married'].value_counts())
sns.countplot(x = 'Married', data = df, palette = 'Set2')


# In[18]:


print("Number of peoples are eligable as per the Dependents :")
print(df['Dependents'].value_counts())
sns.countplot(x = 'Dependents', data = df, palette = 'Set3')


# In[21]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 34)

from sklearn.preprocessing import LabelEncoder
LabelEncoder_x = LabelEncoder()


# In[22]:


for i in range(0,5):
    x_train[:,i] = LabelEncoder_x.fit_transform(x_train[:,i])
    x_train[:,7] = LabelEncoder_x.fit_transform(x_train[:,7])

x_train


# In[27]:


LabelEncoder_y = LabelEncoder()

y_train = LabelEncoder_y.fit_transform(y_train)

y_train


# In[28]:


for i in range(0,5):
    x_test[:,i] = LabelEncoder_x.fit_transform(x_test[:,i])
    x_test[:,7] = LabelEncoder_x.fit_transform(x_test[:,7])
x_test


# In[29]:


y_test = LabelEncoder_y.fit_transform(y_test)

y_test


# In[33]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)


# In[35]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
rf_clf.fit(x_train,y_train)


# In[37]:


from sklearn import metrics
y_pred = rf_clf.predict(x_test)

print('acc score of the rf_clf :', metrics.accuracy_score(y_pred,y_test))

y_pred


# In[39]:


from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB()

nb_clf.fit(x_train,y_train)


# In[42]:


y_pred = nb_clf.predict(x_test)
print("acc of the navie bayes is :", metrics.accuracy_score(y_pred, y_test))


# In[43]:


y_pred


# In[47]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)


# In[48]:


y_pred = dt.predict(x_test)
print("acc of the decision tree is %", metrics.accuracy_score(y_pred,y_test))


# In[49]:


y_pred


# In[51]:


from sklearn.neighbors import KNeighborsClassifier

nk_clf = KNeighborsClassifier()

nk_clf.fit(x_train,y_train)


# In[52]:


y_pred = nk_clf.predict(x_test)

print("acc of the neighbors is % ", metrics.accuracy_score(y_pred,y_test))


# In[53]:


y_pred


# In[ ]:





# In[ ]:




