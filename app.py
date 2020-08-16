#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from flask import Flask, request, jsonify,render_template
import pickle


# In[4]:


app=Flask(__name__)
model=pickle.load(open('deployment.pkl','rb'))


# In[5]:


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict_proba(final_features)
    output=round(prediction[0],3)
    return render_template('index.html',prediction_text='MPG of car is{}'.format(output))
if __name__=='__main__':
    app.run(debug=True)


# In[ ]:




