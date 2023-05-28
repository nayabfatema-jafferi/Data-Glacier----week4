#!/usr/bin/env python
# coding: utf-8

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn import metrics
# from sklearn.metrics import accuracy_score

# In[2]:


file = pd.read_csv("insurance.csv")
file.head(5)


# In[3]:


file.info()


# In[4]:


file.isnull().sum()


# In[5]:


file["sex"].replace({"male":1, "female":0}, inplace = True)


# In[6]:


file["children"].replace({"yes":1, "no":0}, inplace = True)


# In[7]:


file["smoker"].replace({"yes":1, "no":0}, inplace = True)


# In[8]:


file["region"].unique()


# In[14]:


file["region"] = file["region"].replace({"northeast":0,"northwest":1,"southeast":2,"southwest":3})


# In[15]:


file.info()


# In[16]:


#spliiting the dataset
X = file.drop("charges", axis = 1)
y = file["charges"]


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# In[18]:


#scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# In[19]:


#fitting the model
from sklearn.ensemble import RandomForestRegressor
RC = RandomForestRegressor()
model = RC.fit(X_train_sc, y_train)


# In[20]:


y_pred = RC.predict(X_test_sc)
from sklearn.metrics import r2_score

# R2 Score 
print('R2 Score:', r2_score(y_test, y_pred))


# In[21]:


#### Save, Load and Used this Model


# In[22]:


# import pickle library 
import pickle 
# save the model
RandomForest = open("model.pkl","wb")          # open the file for writing
pickle.dump(RC, RandomForest)           # dumps an object to a file object
RandomForest.close()                           # here we close the fileObject


# In[23]:


# Load the model
res_model = open("model.pkl","rb")           # open the file for reading
new_model = pickle.load(res_model)           # load the object from the file into new_model
new_model


# In[27]:


print(new_model.predict([[30,1,28,0,0,3]]))


# In[ ]:




