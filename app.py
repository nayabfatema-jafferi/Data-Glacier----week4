#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request,render_template
import pickle



# In[2]:


app = Flask(__name__)
res_model = open("model.pkl","rb")           # open the file for reading
new_model = pickle.load(res_model)           # load the object from the file into new_model
new_model


# In[3]:


@app.route('/')
def home():
    return render_template('index.html')


# In[4]:


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = new_model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Cost of Insurance will be $ {}'.format(output))


# In[7]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




