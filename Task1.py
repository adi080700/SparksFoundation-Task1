#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


df = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")

df.head()


# In[6]:


df.plot(x='Hours', y='Scores', style='.', color='blue')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.grid()
plt.show()

df.describe()


# In[7]:


X = df.iloc[:, :-1].values  
X

y = df.iloc[:, 1:].values  
y

# Splitting data into training data and testing data
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)


# In[8]:


from sklearn.linear_model import LinearRegression  

l = LinearRegression()  
l.fit(X_train, y_train)

# print coefficient and intercepts model
l.coef_

l.intercept_


# In[9]:


line = l.coef_*X+l.intercept_


# In[10]:


plt.show()
plt.scatter(X_train, y_train, color='black')
plt.plot(X, line, color='blue');
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.grid()
plt.show()


# In[11]:


print(X_test)
y_pred = l.predict(X_test)


# In[12]:


comp = pd.DataFrame({ 'Actual':[y_test],'Predicted':[y_pred] })
comp


# In[13]:


hours = 9.25
own_pred = l.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[14]:



from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




